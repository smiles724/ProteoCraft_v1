import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import subprocess
import textwrap
from types import SimpleNamespace

import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
from proteocraft.hallucination.esmfold2_optimizations import OptimizationManager, measured


@pytest.mark.parametrize("field,value", [
    ("proposal_optimization", "auto"), ("verifier_optimization", "unknown"),
    ("esmc_optimization", "fast"),
])
def test_invalid_optimization_config(field, value):
    with pytest.raises(ValueError):
        ESMFold2ProposalConfig(**{field: value})


def test_off_does_not_import_external_kit(monkeypatch):
    manager = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    monkeypatch.setattr(manager, "_source", lambda: pytest.fail("off imported kit"))
    model = torch.nn.Linear(2, 2)
    manager.proposal(model)
    manager.verifier(model, None)
    manager.esmc(model, "prior")
    assert manager.records == {}


@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_runtime_call_binds_device_and_restores_caller(monkeypatch, timing, fail):
    current = [0]

    @contextmanager
    def device_context(device):
        previous = current[0]
        current[0] = device.index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manager = OptimizationManager(ESMFold2ProposalConfig(
        verifier_optimization="fast", optimization_timing=timing), "cuda:1")

    @measured("verifier")
    def run(backend):
        assert current[0] == 1
        if fail:
            raise ValueError("model failed")
        return 42

    if fail:
        with pytest.raises(ValueError, match="model failed"):
            run(SimpleNamespace(optimizations=manager))
    else:
        assert run(SimpleNamespace(optimizations=manager)) == 42
    assert current[0] == 0
    assert manager.events["verifier.errors" if fail else "verifier.successes"] == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("timing", [False, True])
def test_upstream_graph_replay_on_nondefault_device(timing):
    # A fresh process isolates the kit's global graph pools and PyTorch stream.
    root = kit_root()
    script = textwrap.dedent('''
        import sys
        from collections import OrderedDict
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        from proteocraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
        from proteocraft.hallucination.esmfold2_optimizations import OptimizationManager, measured

        sys.path.insert(0, str(Path(sys.argv[1]) / "esmfold2/opt/forward/fast_inference/driver"))
        import ef2_opt
        torch.cuda.set_device(0)
        manager = OptimizationManager(ESMFold2ProposalConfig(
            verifier_optimization="fast", optimization_timing=sys.argv[2] == "True"), "cuda:1")
        trunk = SimpleNamespace(blocks=[object()], _ef2opt_name="device_test",
            _ef2opt_graphs=OrderedDict(),
            _ef2opt_eager_forward=lambda x, pair_attention_mask=None: x * 2)

        @measured("verifier")
        def run(backend, x):
            with torch.no_grad():
                return ef2_opt._graphed_trunk_forward(trunk, x)

        backend = SimpleNamespace(optimizations=manager)
        for value in (1., 3., 1.):
            x = torch.full((1, 2, 2, 2), value, device="cuda:1")
            actual = run(backend, x)
            torch.testing.assert_close(actual, x * 2, rtol=0, atol=0)
            assert torch.cuda.current_device() == 0
        assert ef2_opt.STATS["trunk_graph_captures"] == 1
        assert ef2_opt.STATS["trunk_graph_replays"] == 3

        # A second device must fail clearly rather than reuse the process-wide
        # capture stream left on cuda:1 by the preceding calls.
        other = SimpleNamespace(optimizations=OptimizationManager(
            ESMFold2ProposalConfig(verifier_optimization="exact"), "cuda:0"))
        try:
            run(other, torch.ones((1, 2, 2, 2), device="cuda:0"))
        except RuntimeError as error:
            assert "one device per fresh process" in str(error)
        else:
            raise AssertionError("Foreign capture stream was accepted")
        assert torch.cuda.current_device() == 0
    ''')
    result = subprocess.run([sys.executable, "-c", script, root, str(timing)],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_off_rejects_autoload_contamination(monkeypatch):
    monkeypatch.setenv("ESMFOLD2_OPT", "fast")
    with pytest.raises(RuntimeError, match="Unset ESMFOLD2_OPT"):
        OptimizationManager(ESMFold2ProposalConfig(), "cpu")


def test_timing_records_failure_without_claiming_success(tmp_path):
    path = tmp_path / "report.json"
    manager = OptimizationManager(ESMFold2ProposalConfig(
        optimization_timing=True, optimization_report_path=str(path)), "cpu")
    with pytest.raises(ValueError):
        with manager.measure("verifier"):
            raise ValueError("failed")
    event = json.loads(path.with_suffix(".timing.jsonl").read_text())
    assert event["status"] == "error"
    assert event["elapsed_s"] >= 0
    assert not path.exists()
    manager.finalize()
    assert json.loads(path.read_text())["counters"]["verifier.calls"] == 1


def test_source_refresh_resolves_each_module_filename_once(tmp_path, monkeypatch):
    source = tmp_path / "cached_source.py"
    source.write_text("VALUE = 1\n")
    manager = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    manager.root = tmp_path.resolve()
    for dependency in tuple(sys.modules.values()):
        filename = getattr(dependency, "__file__", None)
        if filename:
            manager._source_files_scanned.add(os.fspath(filename))
    monkeypatch.setitem(
        sys.modules, "_proteocraft_cached_source_test",
        SimpleNamespace(__file__=str(source)),
    )
    original = manager._resolve_source_path
    calls = []

    def resolve(filename):
        calls.append(os.fspath(filename))
        return original(filename)

    monkeypatch.setattr(manager, "_resolve_source_path", resolve)
    manager._refresh_source_hashes()
    manager._refresh_source_hashes()

    assert calls == [str(source)]
    assert "cached_source.py" in manager.source_hashes


@pytest.mark.parametrize("cap", [None, 0, 256])
def test_verifier_installs_explicitly_without_builder_fold(monkeypatch, cap):
    manager = OptimizationManager(ESMFold2ProposalConfig(verifier_optimization="exact", optimization_graph_max_tokens=cap), "cpu")
    calls = []
    module = SimpleNamespace(
        CFG=SimpleNamespace(
            graph_budget_tokens=512,
            graph_budget_tokens_trunk=-1,
            graph_budget_tokens_encoder=-1,
            graph_budget_tokens_sampler=-1,
        ),
        install=lambda model, **kw: calls.append((model, kw)) or {"trunk_graphs": 1},
    )
    monkeypatch.setattr(manager, "_require_cuda", lambda model: None)
    monkeypatch.setattr(manager, "_check_fork", lambda *names: None)
    monkeypatch.setattr(manager, "_check_core_pin", lambda kit: {"version": "test"})
    monkeypatch.setattr(manager, "_import", lambda name: module)
    model = object()
    manager.verifier(model, object())
    manager.verifier(model, object())
    assert len(calls) == 1
    assert calls[0][1]["builder"] is None
    assert calls[0][1]["esmc_cache"] is False
    assert manager.records["verifier"]["levers"]["trunk_graphs"] == 1
    assert module.CFG.graph_budget_tokens == (512 if cap is None else cap)
    for site in ("trunk", "encoder", "sampler"):
        assert getattr(module.CFG, "graph_budget_tokens_" + site) == (-1 if cap is None else cap)


def test_proposal_exact_does_not_replace_gradient_trunk(monkeypatch):
    manager = OptimizationManager(ESMFold2ProposalConfig(proposal_optimization="exact"), "cpu")
    calls = []
    module = SimpleNamespace(
        enable=lambda *args, **kwargs: pytest.fail("exact replaced the gradient trunk"),
        enable_skip_unused_confidence=lambda model: calls.append(model),
    )
    monkeypatch.setattr(manager, "_require_cuda", lambda model: None)
    monkeypatch.setattr(manager, "_check_fork", lambda *names: None)
    monkeypatch.setattr(manager, "_import", lambda name: module)
    model = object()
    manager.proposal(model)
    assert calls == [model]
    assert manager.records["proposal"]["levers"] == ["skip_unused_confidence"]


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_proposal_rejects_missing_tensor_descriptors_before_patching(monkeypatch, mode):
    manager = OptimizationManager(ESMFold2ProposalConfig(proposal_optimization=mode), "cpu")
    monkeypatch.setattr(manager, "_require_cuda", lambda model: None)
    monkeypatch.setattr(manager, "_require_fast_gpu", lambda label: None)
    monkeypatch.setattr(manager, "_check_fork", lambda *names: None)
    monkeypatch.setattr(manager, "_import", lambda name: pytest.fail("patched unsupported stack"))
    monkeypatch.setattr("proteocraft.hallucination.esmfold2_optimizations.importlib.import_module",
                        lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="Triton tensor descriptors"):
        manager.proposal(object())


def kit_root():
    root = os.environ.get("PROTEOCRAFT_UPLIFT_ROOT")
    if not root:
        pytest.skip("Set PROTEOCRAFT_UPLIFT_ROOT for real upstream integration tests")
    return root


def test_real_upstream_source_pins():
    manager = OptimizationManager(ESMFold2ProposalConfig(optimization_root=kit_root()), "cpu")
    manager._check_fork("transformers.models.esmc.modeling_esmc",
                        "transformers.models.esmfold2.modeling_esmfold2_common",
                        "transformers.models.esmfold2.modeling_esmfold2_experimental",
                        "transformers.models.esmfold2.modeling_esmfold2")
    assert len(manager._verified) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_real_esmc_attention_forward_and_input_gradient():
    from transformers.models.esmc.modeling_esmc import MultiHeadAttention
    torch.manual_seed(13)
    original = MultiHeadAttention(2560, 40).cuda().eval().requires_grad_(False)
    optimized = copy.deepcopy(original)
    manager = OptimizationManager(ESMFold2ProposalConfig(
        optimization_root=kit_root(), esmc_optimization="exact"), "cuda")
    manager.esmc(optimized, "test_esmc")
    for length in (8, 13, 8):
        x = torch.randn(1, length, 2560, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = original(x, None)[0]
            actual = optimized(x, None)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    x = torch.randn(1, 8, 2560, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = original(x, None)[0].float().square().mean()
        b = optimized(x, None)[0].float().square().mean()
    ga = torch.autograd.grad(a, x)[0]
    gb = torch.autograd.grad(b, x)[0]
    torch.testing.assert_close(gb, ga, rtol=0, atol=0)
    assert manager.events["test_esmc.fused_qk"] == 3
    assert manager.events["test_esmc.stock_attention"] == 1


@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_invalid_graph_cap(cap):
    with pytest.raises(ValueError, match="graph_max_tokens"):
        ESMFold2ProposalConfig(optimization_graph_max_tokens=cap)


def test_untimed_calls_count_success_and_failure(tmp_path):
    manager = OptimizationManager(ESMFold2ProposalConfig(
        optimization_report_path=str(tmp_path / "report.json")), "cpu")
    with manager.measure("verifier"):
        pass
    with pytest.raises(ValueError):
        with manager.measure("verifier"):
            raise ValueError("failed")
    manager.finalize()
    counters = json.loads((tmp_path / "report.json").read_text())["counters"]
    assert counters == {"verifier.calls": 2, "verifier.successes": 1, "verifier.errors": 1}
    assert not (tmp_path / "report.timing.jsonl").exists()


@pytest.mark.parametrize("fail_finalizer", [False, True])
def test_cli_finalizes_without_masking_search_failure(monkeypatch, fail_finalizer):
    from proteocraft import cli
    finalized = []
    def finalize():
        finalized.append(True)
        if fail_finalizer:
            raise OSError("report write failure")
    def run(argv, *, optimization_managers):
        optimization_managers.append(SimpleNamespace(finalize=finalize))
        raise ValueError("original search failure")
    monkeypatch.setattr(cli, "_main", run)
    with pytest.raises(ValueError, match="original search failure"):
        cli.main([])
    assert finalized == [True]


def _search_backend(args):
    from proteocraft import cli
    cli.resolve_search_defaults(args)
    cli.resolve_esmfold2_verifier_precision_defaults(args)
    cli.resolve_esmfold2_hotspot_defaults(args, model_backend="esmfold2", hotspot_params=None)
    return cli.load_esmfold2_backend(args, device="cpu")


@pytest.mark.parametrize("profile", [[], ["--profile", "esmfold2_d"]])
def test_cli_optimization_defaults_and_graph_override(profile):
    from proteocraft.cli import parse_args_with_profile
    args, _ = parse_args_with_profile(["--target", "AAAA",   *profile])
    assert (args.esmfold2_proposal_optimization, args.esmfold2_verifier_optimization,
            args.esmfold2_esmc_optimization) == ("fast", "fast", "exact")
    backend = _search_backend(args)
    assert (backend.config.proposal_optimization, backend.config.verifier_optimization,
            backend.config.esmc_optimization) == ("fast", "fast", "exact")
    if profile:
        assert args.esmfold2_stagea_candidate_strategy == "positive_score_guarded"
    assert args.esmfold2_optimization_graph_max_tokens is None
    args, _ = parse_args_with_profile(["--target", "AAAA",
                                      "--esmfold2_optimization_graph_max_tokens", "256"])
    assert args.esmfold2_optimization_graph_max_tokens == 256


@pytest.mark.parametrize("profile", [[], ["--profile", "esmfold2_d"]])
def test_search_stock_override_does_not_require_optimization_sources(profile, monkeypatch):
    from proteocraft.cli import parse_args_with_profile
    monkeypatch.delenv("PROTEOCRAFT_UPLIFT_ROOT", raising=False)
    args, _ = parse_args_with_profile([
        "--target", "AAAA",   *profile,
        "--esmfold2_proposal_optimization", "off", "--esmfold2_verifier_optimization", "off",
        "--esmfold2_esmc_optimization", "off",
    ])
    backend = _search_backend(args)
    monkeypatch.setattr(backend.optimizations, "_source", lambda: pytest.fail("stock imported kit"))
    backend.optimizations.proposal(object())
    backend.optimizations.verifier(object(), None)
    backend.optimizations.esmc(object(), "prior")
    assert backend.optimizations.records == {}


@pytest.mark.parametrize("kind", ["denovo", "tcr"])
def test_panel_search_launchers_default_to_fast(kind, tmp_path):
    from benchmark.script.run_esmfold2_59 import PreparedCase, build_hallucination_command
    from proteocraft.cli import parse_args_with_profile
    case = PreparedCase(case_id="example", task_type=kind, target_name="target",
                        source_run_id="source", target_sequence="AAAA", binder_length=4)
    command = build_hallucination_command(case, profile="esmfold2_d", device="cpu", case_dir=tmp_path)
    args, _ = parse_args_with_profile(command[2:])
    backend = _search_backend(args)
    assert (backend.config.proposal_optimization, backend.config.verifier_optimization,
            backend.config.esmc_optimization) == ("fast", "fast", "exact")


@pytest.mark.parametrize("kind", ["cdr", "vhh"])
@pytest.mark.parametrize("explicit", [False, True])
def test_task_launchers_forward_optimization_settings(kind, tmp_path, explicit):
    from proteocraft.cli import parse_args_with_profile
    config = {"launch_profile": "esmfold2_d", "esmfold2_optimization_root": "/kit",
              "esmfold2_proposal_optimization": "fast", "esmfold2_verifier_optimization": "exact",
              "esmfold2_esmc_optimization": "off", "esmfold2_optimization_graph_max_tokens": 256,
              "esmfold2_optimization_timing": True}
    if not explicit:
        config = {"launch_profile": "esmfold2_d"}
    if kind == "vhh":
        from tests.test_esmfold2_vhh import _vhh_case
        from proteocraft.hallucination.esmfold2_vhh import prepare_esmfold2_vhh_case
        from benchmark.script.run_esmfold2_vhh_redesign import build_hallucination_command
        case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    else:
        import yaml
        from proteocraft.hallucination.esmfold2_cdr import prepare_esmfold2_cdr_case
        from benchmark.script.run_esmfold2_cdr_redesign import build_hallucination_command
        root = Path(__file__).resolve().parents[1]
        original = yaml.safe_load((root / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml").read_text())
        case = prepare_esmfold2_cdr_case(original["cases"][0], input_dir=root / original["input_dir"])
    command = build_hallucination_command(case, config, case_dir=tmp_path / "out")
    args, _ = parse_args_with_profile(command[2:])
    for name, value in config.items():
        if name != "launch_profile":
            assert getattr(args, name) == value
    assert args.esmfold2_stagea_candidate_strategy == "positive_score_guarded"
    if not explicit:
        assert (args.esmfold2_proposal_optimization, args.esmfold2_verifier_optimization,
                args.esmfold2_esmc_optimization) == ("fast", "fast", "exact")
