from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from proteocraft.evaluation import esmfold2_rescore as rescore
from proteocraft.evaluation import esmfold2_runner


def candidate(name="c1", **kwargs):
    return {"case_id": "test", "candidate_id": name, "binder_seq": "ACDE",
            "target_sequences": ["AAAA", "CCCC"], "target_chain_roles": ["target", "scaffold"],
            "target_hotspot_concat_indices": [0, 3], "search_seed": 43, **kwargs}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def prepared(tmp_path, rows=None):
    manifest = jsonl(tmp_path / "input.jsonl", rows or [candidate()])
    output = tmp_path / "evaluation"
    contract = rescore.freeze(manifest, rescore.DEFAULT_CONFIG, output)
    return output, contract, manifest


def save_score(output, contract, row, seed, value=0.8, status="ok"):
    path = rescore.score_path(output, row, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cif = path.parent / "esmfold2.cif"
    cif.write_text("data_test\n#\n")
    rescore.write_json(output / "runtime_environment.json", {"test": True})
    score = {"case_id": row["case_id"], "candidate_id": row["candidate_id"], "seed": seed,
             "candidate_hash": rescore.digest(row), "contract_hash": contract["contract_hash"],
             "runtime_hash": rescore.digest({"test": True}), "status": status,
             "structure_sha256": rescore.file_hash(cif),
             "metrics": {"pair_chain_iptm": value, "total_loss": 0.2, "binder_plddt": 0.9} if status == "ok" else {},
             "runtime_s": 1.0}
    rescore.write_json(path, score)
    return path


def test_defaults_are_explicit_stock_and_fixed_protocol():
    config = rescore.load_config(rescore.DEFAULT_CONFIG)
    assert [config[k] for k in rescore.SWITCHES] == ["off"] * 3
    assert (config["num_loops"], config["num_sampling_steps"]) == (20, 200)
    assert config["seeds"] == [1043, 1044, 1045]


@pytest.mark.parametrize("change", [
    {"verifier_optimization": "fast"}, {"seeds": [1, 1]}, {"seeds": [-1]},
    {"model_revision": "main"}, {"structure_mode": "scaffold_consistency"},
    {"unexpected": True},
])
def test_reject_invalid_or_ambiguous_protocol(tmp_path, change):
    config = rescore.load_config(rescore.DEFAULT_CONFIG)
    config.update(change)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        rescore.load_config(path)


@pytest.mark.parametrize("change", [
    {"candidate_id": "../../escape"}, {"target_chain_roles": ["target"]},
    {"target_hotspot_concat_indices": [8]}, {"target_hotspot_concat_indices": [1.5]},
    {"binder_seq": "AC*E"}, {"target_hotspot_concat_indices": None},
])
def test_invalid_input_rejected(change):
    with pytest.raises(ValueError):
        rescore.normalize_candidate(candidate(**change))


def test_freeze_is_idempotent_and_rejects_changed_input(tmp_path):
    output, contract, manifest = prepared(tmp_path)
    assert rescore.freeze(manifest, rescore.DEFAULT_CONFIG, output) == contract
    jsonl(manifest, [candidate(binder_seq="CCDE")])
    with pytest.raises(ValueError, match="contract changed"):
        rescore.freeze(manifest, rescore.DEFAULT_CONFIG, output)
    assert rescore.load_contract(output) == contract


def test_duplicate_context_or_seed_leak_rejected(tmp_path):
    for rows, message in [([candidate(), candidate()], "Duplicate"),
                          ([candidate(), candidate("c2", target_sequences=["CCCC", "AAAA"])], "context"),
                          ([candidate(search_seed=1043)], "differ")]:
        with pytest.raises(ValueError, match=message):
            rescore.freeze(jsonl(tmp_path / "m.jsonl", rows), rescore.DEFAULT_CONFIG, tmp_path / "out")


def test_top_k_deduplicates_and_requires_completion(tmp_path):
    run = tmp_path / "search"
    artifacts = run / "esmfold2_artifacts"
    artifacts.mkdir(parents=True)
    jsonl(artifacts / "candidates.jsonl", [
        {"sequence": "AAAA", "total_loss": 0.7},
        {"sequence": "AAAA", "total_loss": 0.2},
        {"sequence": "CCCC", "total_loss": 0.3},
    ])
    item = candidate(run_dir="search", candidate_prefix="seed43")
    manifest = jsonl(tmp_path / "runs.jsonl", [item])
    with pytest.raises(ValueError, match="not successfully complete"):
        rescore.freeze(manifest, rescore.DEFAULT_CONFIG, tmp_path / "out", from_search=True)
    rescore.write_json(artifacts / "run_summary.json", {"done": True})
    rescore.write_json(run / "experiment_run.json", {"returncode": 0})
    result = rescore.freeze(manifest, rescore.DEFAULT_CONFIG, tmp_path / "out", from_search=True)
    assert [(r["binder_seq"], r["search_loss"]) for r in result["candidates"]] == [("AAAA", 0.2), ("CCCC", 0.3)]
    assert result["sources"][0]["selected"] == 2
    rescore.write_json(run / "experiment_run.json", {"returncode": 1})
    with pytest.raises(ValueError, match="not successfully complete"):
        rescore.freeze(manifest, rescore.DEFAULT_CONFIG, tmp_path / "other", from_search=True)


def test_ranking_requires_all_seeds_and_keeps_failure_denominator(tmp_path):
    output, contract, _ = prepared(tmp_path, [candidate(), candidate("c2", binder_seq="CCCC")])
    a, b = contract["candidates"]
    for seed in contract["config"]["seeds"]:
        save_score(output, contract, a, seed, value=0.7)
    save_score(output, contract, b, 1043, value=0.99)
    save_score(output, contract, b, 1044, status="failed")
    status = rescore.report(output)
    assert (status["expected_scores"], status["successful_scores"], status["failed_scores"], status["missing_scores"]) == (6, 4, 1, 1)
    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert rows[0]["rank"] == "1"
    assert rows[1]["rank"] == rows[1]["pair_chain_iptm_mean"] == ""
    assert not status["complete"]


@pytest.mark.parametrize("tamper", ["sequence", "structure", "runtime", "unexpected_score"])
def test_resume_and_report_reject_tampering(tmp_path, tamper):
    output, contract, _ = prepared(tmp_path)
    row = contract["candidates"][0]
    path = save_score(output, contract, row, 1043)
    if tamper == "sequence":
        jsonl(output / "candidates.jsonl", [candidate(binder_seq="CCCC")])
    elif tamper == "structure":
        (path.parent / "esmfold2.cif").write_text("corrupted")
    elif tamper == "runtime":
        rescore.write_json(output / "runtime_environment.json", {"changed": True})
    else:
        rescore.write_json(output / "structures/extra/score.json", {})
    with pytest.raises(ValueError):
        rescore.report(output)


def test_stock_process_rejects_autoload_or_previous_patches(monkeypatch):
    monkeypatch.setenv("ESMFOLD2_OPT", "fast")
    with pytest.raises(RuntimeError, match="Unset"):
        rescore.assert_clean_process()
    monkeypatch.delenv("ESMFOLD2_OPT")
    monkeypatch.setitem(sys.modules, "ef2_opt", SimpleNamespace())
    with pytest.raises(RuntimeError, match="fresh"):
        rescore.assert_clean_process()


def test_worker_failure_resume_and_explicit_retry(tmp_path, monkeypatch):
    output, contract, _ = prepared(tmp_path)
    calls = []
    fake = SimpleNamespace(
        config=SimpleNamespace(**dict.fromkeys(rescore.SWITCHES, "off")),
        optimizations=SimpleNamespace(finalize=lambda: None, report=lambda: {"applied": {}}))
    monkeypatch.setattr(rescore, "assert_clean_process", lambda: None)
    monkeypatch.setattr(rescore, "stock_backend", lambda *a: fake)
    monkeypatch.setattr(rescore, "runtime_identity", lambda *a: {"test": True})
    def evaluate(backend, **kw):
        calls.append(kw["seed"])
        if kw["seed"] == 1044 and calls.count(1044) == 1:
            raise RuntimeError("test failure")
        path = kw["design_dir"] / "esmfold2.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_test\n#\n")
        return SimpleNamespace(runtime_s=1.0, mmcif_path=str(path), verifier_result=SimpleNamespace(
            total_loss=0.2, summary_confidence={"pair_chain_iptm": 0.8, "binder_plddt": 0.9}))
    monkeypatch.setattr(esmfold2_runner, "evaluate", evaluate)
    assert rescore.run_worker(output, "cpu") == 1
    assert rescore.run_worker(output, "cpu") == 1
    assert calls == [1043, 1044, 1045]
    assert rescore.run_worker(output, "cpu", retry_failed=True) == 0
    assert calls == [1043, 1044, 1045, 1044]
    failed_dir = rescore.score_path(output, contract["candidates"][0], 1044).parent / "failed_attempts"
    assert len(list(failed_dir.glob("*.json"))) == 1
    assert rescore.run_worker(output, "cpu") == 0


def test_prepare_cli_needs_no_model_and_run_dispatches_fresh_process(tmp_path, monkeypatch):
    from proteocraft.evaluation import cli as module
    output, _, _ = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0))
    assert module.main(["run", "--out-dir", str(output)]) == 0
    assert calls[0][:4] == [sys.executable, "-m", "proteocraft.evaluation.cli", "_worker"]


def test_evaluator_explicitly_overrides_future_fast_defaults(monkeypatch):
    # Actual builder configuration, with model loading replaced by a capture.
    from proteocraft.hallucination import esmfold2_backend, esmfold2_runtime
    actual = esmfold2_backend.ESMFold2ProposalConfig
    def config(**kwargs):
        return actual(**{**dict.fromkeys(rescore.SWITCHES, "fast"), **kwargs})
    monkeypatch.setattr(esmfold2_backend, "ESMFold2ProposalConfig", config)
    monkeypatch.setattr(esmfold2_runtime, "ESMFold2RuntimeBackend", lambda **kwargs: SimpleNamespace(**kwargs))
    backend = esmfold2_runner.build_backend(device="cpu")
    assert [getattr(backend.config, key) for key in rescore.SWITCHES] == ["off"] * 3


@pytest.mark.parametrize("change", ["bounded_source", "geometry_source", "bounded_weights", "raw_weights"])
def test_resume_rejects_changed_scoring_implementation(tmp_path, monkeypatch, change):
    import torch
    from proteocraft.hallucination import esmfold2_one_step

    output, contract, _ = prepared(tmp_path)
    source_root = tmp_path / "source"
    for name in rescore.LOCAL_EVALUATOR_SOURCES:
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rescore.ROOT / name, destination)
    monkeypatch.setattr(rescore, "ROOT", source_root)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    ccd = model / "ccd.pkl"
    ccd.write_bytes(b"test CCD fixture")
    monkeypatch.setenv("ESMCFOLD_CCD_PATH", str(ccd))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test-device")
    backend = SimpleNamespace(
        device="cuda:0", config=SimpleNamespace(verifier_model_name=str(model),
                                               **dict.fromkeys(rescore.SWITCHES, "off")),
        verifier_model=SimpleNamespace(config=SimpleNamespace(esmc_id=str(model))))
    before = rescore.runtime_identity(backend)
    candidate = contract["candidates"][0]
    path = save_score(output, contract, candidate, contract["config"]["seeds"][0])
    score = json.loads(path.read_text())
    score["runtime_hash"] = rescore.digest(before)
    rescore.write_json(path, score)
    rescore.write_json(output / "runtime_environment.json", before)
    assert not rescore.report(output)["complete"]
    if change.endswith("source"):
        name = ("proteocraft/hallucination/esmfold2_one_step.py" if change == "bounded_source"
                else "proteocraft/losses/geometry.py")
        source = source_root / name
        old = source.read_text()
        if change == "bounded_source":
            new = old.replace('"pair_chain_iptm_loss": 1.0,', '"pair_chain_iptm_loss": 0.5,', 1)
        else:
            new = old.replace("rg_th = 2.38", "rg_th = 3.0", 1)
        assert old != new
        source.write_text(new)
    elif change == "bounded_weights":
        monkeypatch.setitem(esmfold2_one_step.ESMFOLD2_VERIFIER_LOSS_WEIGHTS, "pair_chain_iptm_loss", 0.5)
    else:
        monkeypatch.setitem(esmfold2_one_step.ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS, "clash", 0.5)
    assert rescore.runtime_identity(backend) != before
    monkeypatch.setattr(rescore, "assert_clean_process", lambda: None)
    monkeypatch.setattr(rescore, "stock_backend", lambda *args: backend)
    monkeypatch.setattr(esmfold2_runner, "evaluate", lambda *args, **kwargs: pytest.fail("Mixed scoring versions"))
    with pytest.raises(ValueError, match="Runtime environment changed"):
        rescore.run_worker(output, "cuda:0")
    assert json.loads((output / "runtime_environment.json").read_text()) == before
    assert len(list((output / "structures").rglob("score.json"))) == 1
