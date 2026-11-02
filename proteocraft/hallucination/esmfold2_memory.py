"""Phase-scoped model residency for long ESMFold2 design complexes.

This changes placement only: weights, dtypes, RNG, objectives and budgets stay
unchanged. Decisions are made at public runtime boundaries, never inside a
live autograd graph. An OOM is still an error, not an implicit retry.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import time

import torch


logger = logging.getLogger(__name__)
GIB = 1024 ** 3
PHASE_MODELS = {
    "verifier": frozenset({"verifier"}),
    "proposal": frozenset({"proposal"}),
    # Keep the proposal model: its captured graph may be needed by diffusion
    # after the independent prior's own backward has completed.
    "prior": frozenset({"proposal", "prior"}),
    "diffusion": frozenset({"proposal"}),
}


class ModelResidencyManager:
    def __init__(self, config, device, release_model_caches):
        self.config = config
        self.device = torch.device(device)
        self.release_model_caches = release_model_caches
        self.active = False
        self.reasons = []
        self.phase = None
        self.total_tokens = 0
        self.events = Counter()
        self.last_snapshot = None
        report = config.optimization_report_path
        self.trace_path = Path(report).with_suffix(".memory.jsonl") if report else None

    def snapshot(self):
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return {"free_bytes": free, "total_bytes": total,
                "allocated_bytes": allocated, "reserved_bytes": reserved,
                "available_with_allocator_cache_bytes": free + reserved - allocated,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}

    def _record(self, event, **values):
        record = {"event": event, "phase": self.phase, "active": self.active,
                  "time_unix": time.time(), **values, **self.snapshot()}
        self.last_snapshot = record
        self.events[event] += 1
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _needs_move(model, destination):
        destination = torch.device(destination)
        if destination.type == "cuda" and destination.index is None:
            destination = torch.device("cuda", torch.cuda.current_device())
        # Real runtime models are nn.Modules; inspect buffers as well as weights.
        return any(t.device != destination for t in (*model.parameters(), *model.buffers()))

    def prepare(self, phase, total_tokens, models):
        if phase not in PHASE_MODELS:
            raise ValueError(f"Unknown ESMFold2 memory phase: {phase}")
        self.total_tokens = total_tokens
        if self.device.type != "cuda" or self.config.memory_mode == "resident":
            return
        with torch.cuda.device(self.device):
            if not self.active:
                if self.config.memory_mode == "phase_offload":
                    self.reasons.append("explicit_phase_offload")
                else:
                    if total_tokens >= self.config.memory_token_threshold:
                        self.reasons.append("complex_token_threshold")
                    minimum = self.config.memory_min_free_gib
                    if minimum is not None and self.snapshot()["available_with_allocator_cache_bytes"] < minimum * GIB:
                        self.reasons.append("available_memory_threshold")
                self.active = bool(self.reasons)
                if self.active:
                    logger.info("ESMFold2 phase offload enabled: tokens=%d, reasons=%s",
                                total_tokens, ",".join(self.reasons))
            self.phase = phase
            if not self.active:
                self._record("resident", total_tokens=total_tokens)
                return
            self._record("prepare", total_tokens=total_tokens)
            active_names = self.models_for_phase(phase)
            active_ids = {id(model) for name, model in models.items()
                          if name in active_names and model is not None}
            for name, model in models.items():
                if model is None or id(model) in active_ids or not self._needs_move(model, "cpu"):
                    continue
                self.offload_model(name, model)
            for name, model in models.items():
                if name in active_names and model is not None:
                    self.place(name, model)
            self._record("prepared", total_tokens=total_tokens)

    def models_for_phase(self, phase):
        """Residency only: never infer task identity or change model arithmetic."""
        active = PHASE_MODELS[phase]
        policy = self.config.memory_prior_residency
        if phase == "proposal" and policy != "per_step":
            active = active | {"prior"}
        if phase == "diffusion" and policy == "proposal":
            active = active | {"prior"}
        return active

    def offload_model(self, name, model):
        """Offload only after this model's own backward has completed."""
        if model is None or not self.active or self.device.type != "cuda":
            return
        with torch.cuda.device(self.device):
            if not self._needs_move(model, "cpu"):
                return
            torch.cuda.synchronize(self.device)
            self.release_model_caches(model)
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self._record("offload", model=name)

    def place(self, name, model):
        """Restore an already loaded model before the runtime uses it."""
        if not self.active or self.device.type != "cuda":
            return
        if name not in self.models_for_phase(self.phase):
            raise RuntimeError(f"Cannot activate {name} during {self.phase}")
        with torch.cuda.device(self.device):
            if self._needs_move(model, self.device):
                model.to(self.device)
                self._record("restore", model=name)

    def ready(self):
        if self.device.type == "cuda" and self.config.memory_mode != "resident":
            self._record("ready")

    def saved_tensors_context(self):
        """Keep backward saves on host RAM; autograd restores their exact dtype/device.

        Used only around differentiable forward operations. Checkpoint inputs
        otherwise retain a full long-complex pair plane for every trunk block.
        """
        if not self.active or self.device.type != "cuda":
            return nullcontext()
        self.events["saved_tensor_offload_contexts"] += 1
        if self.config.memory_saved_tensor_pin_memory:
            return torch.autograd.graph.save_on_cpu(pin_memory=True)

        # Match the existing pinned path's fresh, contiguous snapshot exactly;
        # only the CPU allocation type and copy synchrony differ. Native
        # save_on_cpu(False) preserves strided layouts and can change GEMM
        # rounding. Never share saved snapshots: upstream custom kernels may
        # update storage without advancing PyTorch's tensor version counter.
        def pack(tensor):
            packed = torch.empty(tensor.size(), dtype=tensor.dtype,
                                 layout=tensor.layout, device="cpu")
            packed.copy_(tensor)
            return tensor.device, packed

        def unpack(packed):
            device, tensor = packed
            return tensor.to(device)

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def report(self):
        return {"requested": self.config.memory_mode, "active": self.active,
                "saved_tensor_offload": self.active,
                "saved_tensor_pin_memory": self.config.memory_saved_tensor_pin_memory,
                "prior_residency": self.config.memory_prior_residency,
                "token_threshold": self.config.memory_token_threshold,
                "min_free_gib": self.config.memory_min_free_gib,
                "activation_reasons": self.reasons, "phase": self.phase,
                "total_tokens": self.total_tokens,
                "counters": dict(self.events), "last_snapshot": self.last_snapshot,
                "trace_path": str(self.trace_path) if self.trace_path else None}
