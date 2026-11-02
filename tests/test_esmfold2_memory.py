from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
from proteocraft.hallucination.esmfold2_memory import GIB, ModelResidencyManager
from proteocraft.hallucination.esmfold2_optimizations import OptimizationManager


class FakeModel:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.tensor = SimpleNamespace(device=torch.device("cuda:0"))

    def parameters(self):
        return iter([self.tensor])

    def buffers(self):
        return iter([])

    def to(self, device):
        self.events.append(("move", self.name, str(device)))
        self.tensor.device = torch.device(device)
        return self


def manager(monkeypatch, **kwargs):
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append(("synchronize",)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cfg = ESMFold2ProposalConfig(**kwargs)
    obj = ModelResidencyManager(cfg, "cuda:0", lambda model: events.append(("clear", model.name)))
    monkeypatch.setattr(obj, "snapshot", lambda: {
        "free_bytes": 60 * GIB, "total_bytes": 140 * GIB,
        "allocated_bytes": 70 * GIB, "reserved_bytes": 80 * GIB,
        "available_with_allocator_cache_bytes": 70 * GIB})
    models = {name: FakeModel(name, events) for name in ("verifier", "proposal", "prior")}
    return obj, models, events


@pytest.mark.parametrize("kwargs", [
    {"memory_mode": "unknown"}, {"memory_token_threshold": 0},
    {"memory_token_threshold": True}, {"memory_token_threshold": 1.5},
    {"memory_min_free_gib": 0}, {"memory_min_free_gib": float("nan")},
    {"memory_min_free_gib": True},
    {"memory_saved_tensor_pin_memory": "false"},
    {"memory_saved_tensor_pin_memory": 0},
    {"memory_prior_residency": "always"},
    {"memory_prior_residency": None},
    {"memory_prior_residency": False},
])
def test_config_rejects_invalid_memory_policy(kwargs):
    with pytest.raises(ValueError, match="memory_"):
        ESMFold2ProposalConfig(**kwargs)


def test_default_keeps_models_and_does_not_touch_cuda_memory(monkeypatch):
    obj, models, events = manager(monkeypatch)
    monkeypatch.setattr(obj, "snapshot", lambda: pytest.fail("resident must not inspect GPU memory"))
    obj.prepare("proposal", 4000, models)
    obj.place("proposal", models["proposal"])
    obj.ready()
    assert events == []
    assert not obj.active


@pytest.mark.parametrize("pin_memory", [True, False])
def test_saved_tensor_hook_is_used_only_for_active_memory_mode(monkeypatch, pin_memory):
    obj, _, _ = manager(monkeypatch, memory_mode="auto",
                        memory_saved_tensor_pin_memory=pin_memory)
    with obj.saved_tensors_context():
        pass
    assert not obj.events["saved_tensor_offload_contexts"]
    obj.active = True
    with obj.saved_tensors_context():
        pass
    assert obj.events["saved_tensor_offload_contexts"] == 1
    assert obj.report()["saved_tensor_pin_memory"] is pin_memory


def test_auto_triggers_on_whole_complex_and_stays_active(monkeypatch):
    obj, models, events = manager(monkeypatch, memory_mode="auto")
    obj.prepare("verifier", 600, models)
    assert not obj.active and events == []
    obj.prepare("proposal", 1523, models)
    assert obj.reasons == ["complex_token_threshold"]
    assert events.index(("clear", "verifier")) < events.index(("move", "verifier", "cpu"))
    assert models["verifier"].tensor.device.type == "cpu"
    assert models["proposal"].tensor.device.type == "cuda"
    obj.prepare("verifier", 600, models)
    assert obj.active
    assert models["proposal"].tensor.device.type == "cpu"
    assert models["prior"].tensor.device.type == "cpu"
    assert models["verifier"].tensor.device == torch.device("cuda:0")
    assert events.index(("move", "prior", "cpu")) < events.index(("move", "verifier", "cuda:0"))
    before = list(events)
    obj.prepare("verifier", 600, models)
    assert events == before  # Human/mouse verification does not swap models again.


def test_low_memory_trigger_and_report(monkeypatch, tmp_path):
    obj, models, _ = manager(monkeypatch, memory_mode="auto", memory_min_free_gib=80,
                            optimization_report_path=str(tmp_path / "optimization_report.json"))
    obj.prepare("proposal", 500, models)
    assert obj.reasons == ["available_memory_threshold"]
    obj.ready()
    assert obj.report()["counters"]["offload"] == 2
    assert obj.trace_path.read_text().count('"event"') >= 4


def test_shared_model_is_not_offloaded_while_active(monkeypatch):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload")
    models["verifier"] = models["proposal"]
    obj.prepare("proposal", 200, models)
    assert not any(event[:2] == ("move", "proposal") for event in events)


def test_prior_is_transient_while_live_proposal_stays_on_gpu(monkeypatch):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload")
    obj.prepare("proposal", 1523, models)
    assert models["prior"].tensor.device.type == "cpu"
    obj.prepare("prior", obj.total_tokens, models)
    assert models["prior"].tensor.device.type == "cuda"
    assert models["proposal"].tensor.device.type == "cuda"
    obj.prepare("proposal", 1523, models)
    assert models["prior"].tensor.device.type == "cpu"
    assert not any(event == ("move", "proposal", "cpu") for event in events)


@pytest.mark.parametrize("policy,roundtrips", [
    ("per_step", 60), ("proposal", 1), ("between_diffusion", 12),
])
def test_two_proposal_chunks_reuse_prior_without_changing_diffusion_budget(monkeypatch, policy, roundtrips):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload",
                                  memory_prior_residency=policy)
    for model in models.values():
        model.tensor.device = torch.device("cpu")
    for _ in range(2):
        obj.prepare("proposal", 1523, models)
        for step in range(1, 31):
            obj.prepare("prior", 1523, models)
            obj.prepare("proposal", 1523, models)
            if step % 5 == 0 and policy == "between_diffusion":
                obj.prepare("diffusion", 1523, models)
            assert models["proposal"].tensor.device.type == "cuda"
    obj.prepare("verifier", 1523, models)
    assert events.count(("move", "prior", "cuda:0")) == roundtrips
    assert events.count(("move", "prior", "cpu")) == roundtrips
    assert models["prior"].tensor.device.type == "cpu"
    assert obj.report()["prior_residency"] == policy


@pytest.mark.parametrize("policy,kept", [("per_step", False), ("proposal", True), ("between_diffusion", False)])
def test_diffusion_boundary_keeps_live_proposal_and_only_optional_prior(monkeypatch, policy, kept):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload",
                                  memory_prior_residency=policy)
    obj.prepare("prior", 1523, models)
    before = len(events)
    obj.prepare("diffusion", 1523, models)
    assert models["proposal"].tensor.device.type == "cuda"
    assert (models["prior"].tensor.device.type == "cuda") is kept
    assert ("move", "proposal", "cpu") not in events[before:]


@pytest.mark.parametrize("policy", ["per_step", "proposal", "between_diffusion"])
def test_model_loading_respects_prior_residency_policy(monkeypatch, policy):
    from proteocraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
    backend = ESMFold2RuntimeBackend(config=ESMFold2ProposalConfig(
        memory_prior_residency=policy), device="cpu")
    backend.experimental_model = torch.nn.Linear(2, 2)
    backend.esmc_model = torch.nn.Linear(2, 2)
    backend.optimizations.memory.active = True
    calls = []
    monkeypatch.setattr(backend.optimizations.memory, "place", lambda *args: None)
    monkeypatch.setattr(backend.optimizations.memory, "offload_model", lambda name, _: calls.append(name))
    monkeypatch.setattr(backend.optimizations, "proposal", lambda _: None)
    monkeypatch.setattr(backend.optimizations, "esmc", lambda *args: None)
    monkeypatch.setattr(backend, "_record_memory_ready", lambda: None)
    backend._ensure_proposal_models_loaded()
    assert calls == (["prior"] if policy == "per_step" else [])


@pytest.mark.parametrize("policy", ["per_step", "proposal", "between_diffusion"])
@pytest.mark.parametrize("target_chains", [1, 2, 3])
def test_runtime_enters_optional_diffusion_phase_before_gradient(monkeypatch, policy, target_chains):
    from proteocraft.hallucination import esmfold2_runtime as runtime
    backend = runtime.ESMFold2RuntimeBackend(config=ESMFold2ProposalConfig(
        memory_prior_residency=policy), device="cpu")
    calls = []
    monkeypatch.setattr(backend, "_prepare_memory_phase",
                        lambda phase, targets, length: calls.append((phase, len(targets), length)))
    def stop(**kwargs):
        raise RuntimeError("gradient entry")
    monkeypatch.setattr(runtime, "run_esmfold2_truncated_diffusion", stop)
    with pytest.raises(RuntimeError, match="gradient entry"):
        backend._one_step_diffusion_gradient(
            logits=torch.zeros(1, 4, 20), fold_result={"_captured": None, "_features": {}},
            targets=("AAAA",) * target_chains, current_seq="AAAA", mutable_mask=None,
            target_chain_roles=("target",) * target_chains,
            reference_coordinates=None, reference_features=None, diffusion_seed=43,
            hotspot_params=None, effective_hotspot_weight=0.0)
    assert calls == ([("diffusion", target_chains, 4)] if policy == "between_diffusion" else [])


def test_cache_cleanup_failure_is_not_swallowed(monkeypatch):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload")
    def fail(_):
        raise RuntimeError("cache release failed")
    obj.release_model_caches = fail
    with pytest.raises(RuntimeError, match="cache release failed"):
        obj.prepare("proposal", 1523, models)
    assert not any(event[0] == "move" for event in events)


def test_implicit_cuda_device_does_not_move_already_active_model(monkeypatch):
    obj, models, events = manager(monkeypatch, memory_mode="phase_offload")
    obj.device = torch.device("cuda")
    obj.prepare("proposal", 200, models)
    assert not any(event[:2] == ("move", "proposal") for event in events)


def test_clear_graph_generation_and_cast_weight_caches_together():
    obj = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    calls = []
    obj.modules["ef2_opt"] = SimpleNamespace(clear_graphs=lambda **kw: calls.append(kw))
    model = torch.nn.Linear(2, 2)
    model._agk_cfg = SimpleNamespace(cache={"old_gpu_address": torch.ones(2)})
    model._ef2_trimul_handle = SimpleNamespace(cache={"old_weight_copy": torch.ones(2)})
    obj.release_model_caches(model)
    assert calls == [{"reason": "phase_offload"}]
    assert model._agk_cfg.cache == model._ef2_trimul_handle.cache == {}


@pytest.mark.parametrize("requested,expected", [(None, 1024), (0, 1024), (2048, 1024), (512, 512)])
def test_late_memory_activation_caps_existing_graphs_without_reinstall(requested, expected):
    config = ESMFold2ProposalConfig(memory_mode="auto", verifier_optimization="fast",
                                    optimization_graph_max_tokens=requested)
    obj = OptimizationManager(config, "cpu")
    obj.memory.active = True
    calls = []
    cfg = SimpleNamespace(graph_budget_tokens=0, graph_budget_tokens_trunk=0,
                          graph_budget_tokens_encoder=0, graph_budget_tokens_sampler=0)
    obj.modules["ef2_opt"] = SimpleNamespace(CFG=cfg, clear_graphs=lambda **kw: calls.append(kw))
    obj.records["verifier"] = {"graph_budget_tokens": 0}
    obj.verifier(None, None)
    assert cfg.graph_budget_tokens == cfg.graph_budget_tokens_trunk == expected
    assert calls == [{"reason": "memory_policy"}]
    assert obj.records["verifier"]["memory_graph_cap_applied"]


@pytest.mark.parametrize("phase", ["verify", "preview", "proposal", "reference"])
def test_runtime_selects_phase_before_loading_and_counts_all_chains(monkeypatch, phase):
    from proteocraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
    backend = ESMFold2RuntimeBackend(config=ESMFold2ProposalConfig(), device="cpu")
    calls = []
    monkeypatch.setattr(backend.optimizations.memory, "prepare",
                        lambda kind, tokens, models: calls.append((kind, tokens)))
    def stop():
        raise RuntimeError("stop before model loading")
    monkeypatch.setattr(backend, "_ensure_verifier_loaded", stop)
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", stop)
    targets = ("A" * 640, "A" * 640)
    with pytest.raises(RuntimeError, match="stop before model loading"):
        if phase == "verify":
            backend.verify(binder_seq="A" * 243, target_seq=targets)
        elif phase == "preview":
            backend.preview_loss_dict(binder_seq="A" * 243, target_seq=targets)
        elif phase == "reference":
            backend.prepare_verifier_reference_features(binder_seq="A" * 243, target_seq=targets)
        else:
            backend._run_soft_update(current_seq="A" * 243, target_seq=targets,
                global_step=0, seed=43, mutable_mask=None, action_group=None, proposal_logits=None)
    assert calls == [("verifier" if phase in ("verify", "reference") else "proposal", 1523)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for residency parity")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_cuda_residency_preserves_weights_dtype_rng_outputs_and_input_gradient(pin_memory):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(43)
    proposal = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 8))
    proposal.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    verifier = torch.nn.Linear(16, 8).to(device).eval().requires_grad_(False)
    baseline_weights = {key: value.detach().cpu().clone() for key, value in proposal.state_dict().items()}
    x = torch.randn(3, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = proposal(x)
    grad = torch.autograd.grad(y.float().square().sum(), x)[0]
    obj = ModelResidencyManager(ESMFold2ProposalConfig(
        memory_mode="phase_offload", memory_saved_tensor_pin_memory=pin_memory), device, lambda _: None)
    models = {"verifier": verifier, "proposal": proposal, "prior": None}
    rng = torch.cuda.get_rng_state(device).clone()
    obj.prepare("verifier", 32, models)
    assert next(proposal.parameters()).device.type == "cpu"
    obj.prepare("proposal", 32, models)
    with obj.saved_tensors_context():
        y2 = proposal(x)
    grad2 = torch.autograd.grad(y2.float().square().sum(), x)[0]
    assert torch.equal(rng, torch.cuda.get_rng_state(device))
    assert torch.equal(y, y2) and torch.equal(grad, grad2)
    assert all(torch.equal(baseline_weights[key], value.cpu()) for key, value in proposal.state_dict().items())
    assert next(proposal.parameters()).dtype == torch.bfloat16
    assert obj.events["saved_tensor_offload_contexts"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for activation memory check")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_cuda_saved_tensor_offload_reduces_peak_and_preserves_gradient(pin_memory):
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randn(2048, 2048, dtype=torch.float32)
    obj = ModelResidencyManager(ESMFold2ProposalConfig(
        memory_mode="phase_offload", memory_saved_tensor_pin_memory=pin_memory), device, lambda _: None)
    obj.active = True
    def run(offload):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = source.to(device).requires_grad_()
        with obj.saved_tensors_context() if offload else nullcontext():
            y = x
            for _ in range(8):
                y = torch.sin(y) + 0.1
        grad = torch.autograd.grad(y.sum(), x)[0]
        torch.cuda.synchronize(device)
        return y.detach().cpu(), grad.cpu(), torch.cuda.max_memory_allocated(device)
    baseline, gradient, peak = run(False)
    actual, actual_gradient, offload_peak = run(True)
    assert torch.equal(baseline, actual)
    assert torch.equal(gradient, actual_gradient)
    assert offload_peak < peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for transient-prior parity")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_cuda_prior_swap_preserves_retained_proposal_graph(pin_memory):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(43)
    proposal = torch.nn.Linear(16, 8).to(device).eval().requires_grad_(False)
    prior = torch.nn.Linear(16, 4).to(device).eval().requires_grad_(False)
    source = torch.randn(3, 16, device=device)
    x = source.clone().requires_grad_()
    y = proposal(x)
    expected_structure = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    expected_prior = torch.autograd.grad(prior(x).square().sum(), x)[0]
    expected_diffusion = torch.autograd.grad(y.sum(), x)[0]
    manager = ModelResidencyManager(ESMFold2ProposalConfig(
        memory_mode="phase_offload", memory_saved_tensor_pin_memory=pin_memory), device, lambda _: None)
    models = {"proposal": proposal, "prior": prior, "verifier": None}
    manager.prepare("proposal", 64, models)
    x = source.clone().requires_grad_()
    with manager.saved_tensors_context():
        y = proposal(x)
    structure = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    manager.prepare("prior", 64, models)
    with manager.saved_tensors_context():
        prior_loss = prior(x).square().sum()
    prior_gradient = torch.autograd.grad(prior_loss, x)[0]
    manager.prepare("proposal", 64, models)
    diffusion = torch.autograd.grad(y.sum(), x)[0]
    assert next(prior.parameters()).device.type == "cpu"
    # Pinned save_on_cpu materializes strided saved views. A GEMM may then
    # choose a different reduction layout; compare at float32 rounding scale.
    for name, expected, actual in [("structure", expected_structure, structure),
                                   ("prior", expected_prior, prior_gradient),
                                   ("diffusion", expected_diffusion, diffusion)]:
        print({"gradient": name, "max_abs_error": (expected - actual).abs().max().item(),
               "max_reference": expected.abs().max().item()})
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned/pageable numerical match")
def test_cuda_pageable_keeps_pinned_snapshot_layout_values_outputs_and_gradient():
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(43)
    source = torch.randn(32, 16, device=device)
    weight = torch.randn(8, 16, device=device)
    outputs = []
    for pinned in (True, False):
        obj = ModelResidencyManager(ESMFold2ProposalConfig(memory_mode="phase_offload",
            memory_saved_tensor_pin_memory=pinned), device, lambda _: None)
        obj.active = True
        context = obj.saved_tensors_context()
        snapshot = context.pack_hook(weight.T)
        assert snapshot[1].is_contiguous()
        assert snapshot[1].is_pinned() is pinned
        assert torch.equal(snapshot[1], weight.T.cpu())
        x = source.clone().requires_grad_()
        with obj.saved_tensors_context():
            y = (x @ weight.T).sin()
        grad = torch.autograd.grad(y.square().sum(), x)[0]
        outputs.append((y.detach(), grad))
    for a, b in zip(*outputs):
        assert torch.equal(a, b)
