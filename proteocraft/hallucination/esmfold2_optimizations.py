"""Explicit, instance-scoped adapters for uplifting-biomolecular-modeling.

These are ProteoCraft compositions, not the upstream kits' complete cookbook /
SDK compositions. Source hashes, installed levers and runtime counters are part
of the experiment record. Nothing is imported from the kit in ``off`` mode.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from types import MethodType

import torch

from .esmfold2_memory import ModelResidencyManager


COMPOSITION = "proteocraft-uplifting-v2"
# Adapter derived from smiles724/ProteoCraft commit 98aacad9c421718c1925b18d71aa3975969c26ab.
PROPOSAL_MODES = ("off", "exact", "fast", "big")
VERIFIER_MODES = PROPOSAL_MODES
ESMC_MODES = ("off", "exact")
_PROCESS_ROOT: Path | None = None


def measured(phase):
    """Bind the execution device and optionally measure CUDA wall time."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            with self.optimizations.execution_context(), self.optimizations.measure(phase):
                context = (self.optimizations.memory.saved_tensors_context()
                           if phase in {"proposal_forward", "esmc_prior_forward", "proposal_diffusion_gradient"}
                           else nullcontext())
                with context:
                    return fn(self, *args, **kwargs)
        return wrapped
    return decorate


class OptimizationManager:
    def __init__(self, config, device):
        self.config, self.device = config, torch.device(device)
        self.root = None
        self.modules = {}
        self.handles = {}
        self.records = {}
        self.events = Counter()
        self.report_path = Path(config.optimization_report_path) if config.optimization_report_path else None
        self.source_hashes = {}
        self.source_revision = None
        self._verified = set()
        # Resolving a module filename can require one filesystem lookup per path
        # component.  Keep both the resolved result and the set of filenames
        # already considered for provenance so repeated snapshots never redo
        # that work on network filesystems.
        self._resolved_source_paths = {}
        self._source_files_scanned = set()
        self.memory = ModelResidencyManager(config, self.device, self.release_model_caches)
        # An autoloaded kit can contaminate even the nominal off arm.
        for key in ("ESMFOLD2_OPT", "ESMC_OPT", "EF2_FAST_KIT", "ESMC_KIT"):
            if os.environ.get(key, "").lower() not in ("", "off", "0"):
                raise RuntimeError(f"Unset {key}: select optimizations through the ProteoCraft config")

    def _source(self):
        global _PROCESS_ROOT
        if self.root is not None:
            return self.root
        root = self.config.optimization_root or os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
        if not root:
            raise ValueError("Optimization requires optimization_root / --esmfold2_optimization_root")
        root = Path(root).expanduser().resolve()
        directories = [
            root / "ef2inv/opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k",
            root / "esmfold2/opt/forward/fast_inference/driver",
            root / "esmc/opt",
            root / "common/opt_core",
        ]
        for path in directories:
            if not path.is_dir():
                raise FileNotFoundError(path)
        if _PROCESS_ROOT is not None and root != _PROCESS_ROOT:
            raise RuntimeError("Use a fresh process when switching uplifting source trees")
        for path in reversed(directories):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        _PROCESS_ROOT = self.root = root
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        self.source_revision = revision.stdout.strip()
        return root

    def _check_core_pin(self, kit):
        root = self._source()
        project = root / kit / "opt/pyproject.toml"
        with project.open("rb") as handle:
            minimum = tomllib.load(handle)["tool"]["opt_core"]["version"]
        core = importlib.import_module("opt_core")
        source = self._resolve_source_path(core.__file__)
        expected_root = (root / "common/opt_core/opt_core").resolve()
        if not source.is_relative_to(expected_root):
            raise RuntimeError(f"opt_core loaded from unexpected source: {source}")
        found = str(core.__version__)
        if tuple(map(int, found.split("."))) < tuple(map(int, minimum.split("."))):
            raise RuntimeError(f"opt_core {found} is older than {kit} requires ({minimum})")
        self.source_hashes["common/opt_core/opt_core/__init__.py"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        return {"version": found, "minimum": minimum, "source": str(source)}

    def _import(self, name):
        root = self._source()
        module = importlib.import_module(name)
        source = self._resolve_source_path(module.__file__)
        if not source.is_relative_to(root):
            raise RuntimeError(f"{name} loaded from unexpected source: {source}")
        self.source_hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        self._refresh_source_hashes()
        self.modules[name] = module
        return module

    def _resolve_source_path(self, filename):
        key = os.fspath(filename)
        path = self._resolved_source_paths.get(key)
        if path is None:
            path = Path(key).resolve()
            self._resolved_source_paths[key] = path
        return path

    def _refresh_source_hashes(self):
        """Discover and hash newly loaded Python sources from the optimization tree."""
        if self.root is None:
            return
        for dependency in tuple(sys.modules.values()):
            filename = getattr(dependency, "__file__", None)
            if not filename:
                continue
            filename = os.fspath(filename)
            if filename in self._source_files_scanned:
                continue
            path = self._resolve_source_path(filename)
            self._source_files_scanned.add(filename)
            if path.suffix == ".py" and path.is_relative_to(self.root):
                key = str(path.relative_to(self.root))
                if key not in self.source_hashes:
                    self.source_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    def _check_fork(self, *names):
        """Check imported modeling files against the kit's immutable fork pins."""
        root = self._source()
        with (root / "ef2inv/stock/PINS.json").open() as handle:
            pins = json.load(handle)["transformers_fork"]
        # Resolve pin entries by their source filename; schemas differ by release.
        def entries(obj, prefix=""):
            if isinstance(obj, dict):
                if "sha256" in obj:
                    yield prefix, obj["sha256"]
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) == 64:
                        yield str(key), value
                    elif isinstance(value, (dict, list)):
                        yield from entries(value, str(key))
            elif isinstance(obj, list):
                for value in obj:
                    yield from entries(value, prefix)
        hashes = list(entries(pins))
        for name in names:
            if name in self._verified:
                continue
            module = sys.modules.get(name)
            if module is not None:
                source = self._resolve_source_path(module.__file__)
            else:
                spec = importlib.util.find_spec(name)
                if spec is None or spec.origin is None:
                    raise RuntimeError(f"Cannot locate pinned upstream source for {name}")
                source = self._resolve_source_path(spec.origin)
            expected = [digest for key, digest in hashes if key.endswith(source.name)]
            if len(expected) != 1:
                raise RuntimeError(f"Missing or ambiguous upstream source pin for {name}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != expected[0]:
                raise RuntimeError(f"Upstream source mismatch: {name}; expected {expected[0]}, got {digest}")
            self.source_hashes[name] = digest
            self._verified.add(name)

    def _require_cuda(self, model):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Uplifting kernels require a CUDA device")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Optimization supports frozen model weights and input gradients only")

    def _require_fast_gpu(self, label):
        capability = torch.cuda.get_device_capability(self.device)
        if capability not in ((8, 0), (9, 0)):
            raise RuntimeError(
                f"{label} fast/big kernels are validated for compute capability "
                f"8.0/9.0; got {capability[0]}.{capability[1]}"
            )

    def proposal(self, model):
        mode = self.config.proposal_optimization
        if "proposal" in self.records or mode == "off":
            return
        self._require_cuda(model)
        self._check_fork("transformers.models.esmfold2.modeling_esmfold2_common",
                         "transformers.models.esmfold2.modeling_esmfold2_experimental")
        if mode in ("fast", "big"):
            self._require_fast_gpu("proposal")
            language = importlib.import_module("triton.language")
            if not hasattr(language, "make_tensor_descriptor"):
                raise RuntimeError(
                    "Proposal fast/big requires Triton tensor descriptors; "
                    "use the isolated optimization environment (Triton 3.6.0), "
                    "or select proposal exact/off in the baseline environment"
                )
        agk = self._import("ef2_autograd_kernels")
        agk.enable_skip_unused_confidence(model)
        levers = ["skip_unused_confidence"]
        if mode in ("fast", "big"):
            # The grad-enabled fast path replaces the pair trunk and controls
            # checkpointing.  Exact deliberately leaves that trunk untouched:
            # its adapter-level parity contract is bitwise, including gradients.
            agk.enable(model, trimul=None, transition="refround_lean",
                       checkpoint="block", trunks="main")
            trimul = self._import("ef2_trimul")
            self.handles["proposal_trimul"] = trimul.enable(model, variant="fused")
            levers += ["block_checkpointing", "fused_trimul_input_backward",
                       "transition_refround_lean"]
        if mode == "big":
            model.set_chunk_size(32)
            levers += ["chunk_size_32"]
        self.records["proposal"] = {"mode": mode, "source": "ef2inv", "levers": levers}
        self.write_report()

    def release_model_caches(self, model):
        """Invalidate address-dependent caches at a model-placement boundary."""
        if "ef2_opt" in self.modules:
            # The kit shares graph pools: release the whole generation, not
            # individual graphs whose addresses can still be referenced.
            self.modules["ef2_opt"].clear_graphs(reason="phase_offload")
        seen = set()
        for module in model.modules():
            for name in ("_agk_cfg", "_ef2_trimul_handle"):
                state = getattr(module, name, None)
                if state is not None and id(state.cache) not in seen:
                    state.cache.clear()
                    seen.add(id(state.cache))
            cache = getattr(module, "_ef2_trimul_cache", None)
            if cache is not None:
                cache.clear()
        self.events["memory.cache_releases"] += 1

    def verifier(self, model, builder):
        mode = self.config.verifier_optimization
        if mode == "off":
            return
        cap = self.config.optimization_graph_max_tokens
        if self.memory.active:
            limit = self.config.memory_token_threshold
            cap = min(cap, limit) if cap is not None and cap > 0 else limit
        if "verifier" in self.records:
            if self.memory.active and self.records["verifier"]["graph_budget_tokens"] != cap:
                opt = self.modules["ef2_opt"]
                opt.clear_graphs(reason="memory_policy")
                self._set_graph_budget(opt, cap)
                self.records["verifier"].update(
                    graph_budget_tokens=cap, memory_graph_cap_applied=True,
                    graph_site_budgets={site: getattr(opt.CFG, f"graph_budget_tokens_{site}", None)
                                       for site in ("trunk", "encoder", "sampler")})
                self.write_report()
            return
        self._require_cuda(model)
        self._check_fork("transformers.models.esmfold2.modeling_esmfold2_common",
                         "transformers.models.esmfold2.modeling_esmfold2")
        opt = self._import("ef2_opt")
        core = self._check_core_pin("esmfold2")
        # Explicit install is essential: aligned_common bypasses InputBuilder.fold.
        # Avoid upstream cross-input feature/LM caches; search changes sequences.
        if mode in ("fast", "big"):
            self._require_fast_gpu("verifier")
            model.set_kernel_backend("fused")
            model.set_chunk_size(32 if mode == "big" else None)
        # Preserve upstream graph budgets unless the caller explicitly overrides
        # them. Zero means uncapped and can increase long-complex memory usage.
        if cap is not None:
            self._set_graph_budget(opt, cap)
        installed = opt.install(
            model, trunk_graphs=mode != "big", sampler_graphs=mode != "big",
            fuse_msa_trimul=mode in ("fast", "big"), swa_mask_cache=True,
            pair_bias_cache=True, esmc_cache=False, builder=None,
        )
        self.records["verifier"] = {
            "mode": mode, "source": "esmfold2", "levers": installed,
            "opt_core": core,
            "kernel_backend": "fused" if mode in ("fast", "big") else "unchanged",
            "graph_max_tokens_requested": self.config.optimization_graph_max_tokens,
            "memory_graph_cap_applied": self.memory.active,
            "graph_budget_tokens": opt.CFG.graph_budget_tokens,
            "graph_site_budgets": {site: getattr(opt.CFG, f"graph_budget_tokens_{site}", None)
                                   for site in ("trunk", "encoder", "sampler")},
        }
        self.write_report()

    @staticmethod
    def _set_graph_budget(opt, cap):
        opt.CFG.graph_budget_tokens = cap
        for site in ("trunk", "encoder", "sampler"):
            attribute = f"graph_budget_tokens_{site}"
            if hasattr(opt.CFG, attribute):
                setattr(opt.CFG, attribute, cap)

    def esmc(self, model, label):
        if self.config.esmc_optimization == "off" or label in self.records:
            return
        self._require_cuda(model)
        self._check_fork("transformers.models.esmc.modeling_esmc")
        modeling = importlib.import_module("transformers.models.esmc.modeling_esmc")
        kernel = self._import("esmc_opt.kits.qk_rotary.qk_ln_rope")
        count = 0
        for attention in model.modules():
            if type(attention) is not modeling.MultiHeadAttention:
                continue
            if getattr(attention, "_proteocraft_esmc_exact", False):
                continue
            self._patch_esmc_attention(attention, modeling, kernel, label)
            count += 1
        if not count:
            raise RuntimeError(f"{label}: no supported Transformers ESMC attention modules; refusing an inactive optimization")
        # The separate ESMC SDK's kernels are inference-only. Its adapted no-grad
        # path cannot replace the sequence prior's autograd path. Reuse ef2inv's
        # exact differentiable rotary there, with its own per-call guards.
        rope = self._import("ef2_esmc_rope")
        self.handles[label + "_rope"] = rope.enable(model)
        self.records[label] = {
            "mode": "exact", "source": "esmc", "attention_modules": count,
            "levers": ["qk_layernorm_no_grad", "ef2inv_exact_differentiable_rotary"],
            "contract": "first-call bitwise QK check per shape; gradient path uses original attention",
        }
        self.write_report()

    def _patch_esmc_attention(self, attention, modeling, kernel, label):
        original = attention.forward
        checked = set()
        buffers = {}
        manager = self

        def forward(this, x, seq_id, output_attentions=False):
            if torch.is_grad_enabled() or output_attentions or not x.is_cuda or x.ndim != 3:
                manager.events[label + ".stock_attention"] += 1
                return original(x, seq_id, output_attentions=output_attentions)
            qkv = this.layernorm_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            if q.dtype != torch.bfloat16 or this.q_ln.bias is not None or this.k_ln.bias is not None:
                # Do not execute layernorm_qkv twice on unsupported input.
                qn, kn = this.q_ln(q).to(q.dtype), this.k_ln(k).to(q.dtype)
                manager.events[label + ".stock_qk"] += 1
            else:
                batch, length, width = q.shape
                key = (batch, length, width, str(q.device), q.dtype)
                if key not in buffers:
                    # rotate=False uses the LN kernel only. Retain upstream RoPE
                    # arithmetic (the SDK's rotary rounding differs from torch).
                    cos = torch.ones((1, this.d_head // 2), device=q.device, dtype=q.dtype)
                    cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * length
                    buffers.clear()
                    buffers[key] = (cos, cu)
                cos, cu = buffers[key]
                qn = kernel.qk_ln_rope(q.reshape(-1, width), this.q_ln.weight, cos, cos, cu,
                                       head_dim=this.d_head, rotate=False, eps=this.q_ln.eps).reshape_as(q)
                kn = kernel.qk_ln_rope(k.reshape(-1, width), this.k_ln.weight, cos, cos, cu,
                                       head_dim=this.d_head, rotate=False, eps=this.k_ln.eps).reshape_as(k)
                if key not in checked:
                    qr, kr = this.q_ln(q).to(q.dtype), this.k_ln(k).to(q.dtype)
                    if not torch.equal(qn, qr) or not torch.equal(kn, kr):
                        raise RuntimeError(f"{label}: ESMC exact QK parity failed; this stack is unsupported")
                    checked.add(key)
                    manager.events[label + ".parity_checks"] += 1
                manager.events[label + ".fused_qk"] += 1
            qn, kn = this._apply_rotary(qn, kn)
            context = modeling._scaled_dot_product_attention(
                qn, kn, v, n_heads=this.n_heads, d_head=this.d_head, seq_id=seq_id)
            return this.out_proj(context), None

        attention.forward = MethodType(forward, attention)
        attention._proteocraft_esmc_exact = True

    def report(self, *, refresh_sources=False):
        if refresh_sources:
            self._refresh_source_hashes()
        counters = dict(self.events)
        for name, handle in self.handles.items():
            counters[name] = dict(getattr(handle, "stats", {}))
        if "ef2_opt" in self.modules:
            counters["verifier"] = self.modules["ef2_opt"].stats()
        versions = {}
        for name in ("torch", "triton", "esm", "transformers"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {"composition": COMPOSITION, "upstream_complete_modes": False,
                "requested": {"proposal": self.config.proposal_optimization,
                              "verifier": self.config.verifier_optimization,
                              "esmc": self.config.esmc_optimization},
                "source_root": str(self.root) if self.root else None, "source_revision": self.source_revision,
                "versions": versions, "source_hashes": self.source_hashes,
                "applied": self.records, "counters": counters,
                "memory": self.memory.report()}

    def write_report(self, *, refresh_sources=False):
        if self.report_path is None:
            return
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(
            self.report(refresh_sources=refresh_sources), indent=2, default=str
        ) + "\n")
        temporary.replace(self.report_path)

    def finalize(self):
        """Write the final counters and a complete, cached provenance snapshot."""
        self.write_report(refresh_sources=True)

    @contextmanager
    def execution_context(self):
        # Tensor/model placement does not select the current CUDA device. The
        # upstream kit creates streams and captures graphs on that device.
        with torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext():
            if self.device.type == "cuda" and self.config.verifier_optimization in ("exact", "fast"):
                # PyTorch reuses this stream process-wide even inside a device
                # context. Refuse a foreign stream rather than capture an empty
                # graph and silently replay stale tensors on another GPU.
                stream = getattr(torch.cuda.graph, "default_capture_stream", None)
                device = torch.device("cuda", torch.cuda.current_device())
                if stream is not None and stream.device != device:
                    raise RuntimeError(
                        f"CUDA graph capture stream belongs to {stream.device}, not {device}; "
                        "run ESMFold2 optimization on one device per fresh process"
                    )
            yield

    @contextmanager
    def measure(self, phase):
        if not self.config.optimization_timing:
            self.events[phase + ".calls"] += 1
            try:
                yield
            except BaseException:
                self.events[phase + ".errors"] += 1
                raise
            else:
                self.events[phase + ".successes"] += 1
            return
        cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        status = "error"
        try:
            yield
            status = "ok"
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            event = {"phase": phase, "elapsed_s": time.perf_counter() - start, "status": status}
            if cuda:
                event.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            self.events[phase + ".calls"] += 1
            self.events[phase + (".successes" if status == "ok" else ".errors")] += 1
            self.events[phase + ".elapsed_s"] += event["elapsed_s"]
            if self.report_path is not None:
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                with self.report_path.with_suffix(".timing.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
