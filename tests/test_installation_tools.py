"""Installation failures must be reported before a costly model search starts."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from environment.esmfold2 import check_environment, download_models
from proteocraft import cli, model_sources
from proteocraft.hallucination.profile_loader import write_case_profile
from scripts.smoke_test import SMOKE_OVERRIDES, check_outputs
from tests.test_esmfold2_pipeline import FakeESMFold2Backend


def snapshot(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"esmfold2"}')
    weights = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2)}, weights)
    monkeypatch.setitem(model_sources.MODEL_FILE_SIZES, "biohub/ESMFold2", {
        weights.name: weights.stat().st_size,
    })
    return weights


def test_snapshot_checks_real_safetensors_without_loading_tensors(tmp_path, monkeypatch):
    snapshot(tmp_path, monkeypatch)
    model_sources.validate_model_snapshot("biohub/ESMFold2", tmp_path)


@pytest.mark.parametrize("damage", ["missing", "truncated", "corrupt"])
def test_snapshot_rejects_incomplete_or_corrupt_weights(tmp_path, monkeypatch, damage):
    weights = snapshot(tmp_path, monkeypatch)
    if damage == "missing":
        weights.unlink()
    elif damage == "truncated":
        weights.write_bytes(weights.read_bytes()[:-1])
    else:
        weights.write_bytes(b"\0" * weights.stat().st_size)
    expected_error = {"missing": FileNotFoundError, "truncated": ValueError, "corrupt": SafetensorError}[damage]
    with pytest.raises(expected_error):
        model_sources.validate_model_snapshot("biohub/ESMFold2", tmp_path)


def test_empty_snapshot_directory_does_not_pass_model_check(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        model_sources.validate_model_snapshot("biohub/ESMFold2", tmp_path)


def test_esmc_index_must_match_all_shards_and_tensor_names(tmp_path, monkeypatch):
    sizes, weight_map = {}, {}
    (tmp_path / "config.json").write_text('{"model_type":"esmc"}')
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (tmp_path / name).write_text('{"tokenizer":"test"}')
    for i in range(6):
        name = f"model-{i + 1:05d}-of-00006.safetensors"
        save_file({f"weight_{i}": torch.ones(2)}, tmp_path / name)
        sizes[name] = (tmp_path / name).stat().st_size
        weight_map[f"weight_{i}"] = name
    monkeypatch.setitem(model_sources.MODEL_FILE_SIZES, "biohub/ESMC-6B", sizes)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    model_sources.validate_model_snapshot("biohub/ESMC-6B", tmp_path)
    weight_map["weight_0"], weight_map["weight_1"] = weight_map["weight_1"], weight_map["weight_0"]
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="shard/index mismatch"):
        model_sources.validate_model_snapshot("biohub/ESMC-6B", tmp_path)


def test_environment_checker_defaults_to_accelerated_with_explicit_stock_option():
    assert check_environment.parse_args([]).stack == "accelerated"
    assert check_environment.parse_args(["--stack", "stock"]).stack == "stock"


def test_source_check_reads_editable_checkout_commit(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source").write_text("test")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture",
    ], check=True)
    expected = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    direct_url = json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(check_environment.metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda file: direct_url))
    assert check_environment.direct_url_commit("esm") == expected


def test_saved_activation_restores_all_paths_with_spaces_and_quotes(tmp_path):
    environment = tmp_path / "env's directory"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/activate").write_text(f"export VIRTUAL_ENV={shlex.quote(str(environment))}\n")
    script = tmp_path / "activate.sh"
    values = {
        "PROTEOCRAFT_UPLIFT_ROOT": str(tmp_path / "optimization's source"),
        "HF_HOME": str(tmp_path / "model cache"),
        "ESMCFOLD_CCD_PATH": str(tmp_path / "model cache/ccd.pkl"),
    }
    download_models.write_activation_script(
        script, environment=environment, uplift_root=Path(values["PROTEOCRAFT_UPLIFT_ROOT"]),
        hf_home=Path(values["HF_HOME"]), ccd_path=Path(values["ESMCFOLD_CCD_PATH"]),
    )
    code = 'import json,os; print(json.dumps(dict(os.environ)))'
    shell = f"source {shlex.quote(str(script))}\n" + shlex.join([sys.executable, "-c", code])
    restored = json.loads(subprocess.check_output(["bash", "-c", shell], text=True))
    assert all(restored[key] == value for key, value in values.items())
    assert restored["VIRTUAL_ENV"] == str(environment)
    assert restored["HF_HUB_OFFLINE"] == restored["TRANSFORMERS_OFFLINE"] == "1"


def test_short_profile_reaches_stageb_and_retains_matching_output(tmp_path, monkeypatch):
    class RecordingBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            self.propose_calls.append(kwargs)
            sequence = kwargs["current_seq"]
            residue = "Y" if sequence[0] != "Y" else "F"
            return [residue + sequence[1:]]

    backend = RecordingBackend()
    monkeypatch.setattr(cli, "load_esmfold2_backend", lambda *args, **kwargs: backend)
    profile = write_case_profile("esmfold2_d", SMOKE_OVERRIDES, tmp_path / "profile.yaml")
    cli.main([
        "--profile", str(profile), "--target", "TTTT", "--binder_length", "3",
        "--init_binder_seq", "AAA", "--device", "cpu", "--output_dir", str(tmp_path),
    ])
    artifacts = tmp_path / "esmfold2_artifacts"
    result = json.loads((artifacts / "run_summary.json").read_text())
    assert result["n_steps"] == 4
    assert "stageb_selection_surfaces" in result
    assert (artifacts / "mcts_handoff.jsonl").read_text().strip()
    assert any(call["action_group"] is not None for call in backend.propose_calls)
    metadata_path = artifacts / "checkpoints/final/step_0004/step_0004_best_meta.json"
    saved = json.loads(metadata_path.read_text())
    assert saved["binder_seq"] == result["best_sequence"]
    assert saved["total_loss"] == result["best_loss"]
    saved["binder_seq"] = "WRONG"
    metadata_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="does not match"):
        check_outputs(tmp_path)


def test_ready_vhh_example_uses_complete_original_input(tmp_path):
    import yaml
    from proteocraft.hallucination.esmfold2_vhh import prepare_esmfold2_vhh_case
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml").read_text())
    case = prepare_esmfold2_vhh_case(config["cases"][0], input_dir=tmp_path)
    assert sum(map(len, case.target_sequences)) == 217
    assert len(case.initial_vhh_sequence) == 128
    assert sum(case.mutable_mask) == 37
    assert case.target_hotspots_1based == (53, 81, 83, 85, 97, 99)


@pytest.mark.parametrize("damage", [None, "missing_gradient", "disabled_acceleration"])
def test_smoke_output_check_requires_completed_compute_and_acceleration(tmp_path, damage):
    artifacts = tmp_path / "esmfold2_artifacts"
    checkpoint = artifacts / "checkpoints/final/step_0004"
    checkpoint.mkdir(parents=True)
    (artifacts / "run_summary.json").write_text(json.dumps({
        "n_steps": 4, "strategy": "two_stage", "best_sequence": "AAA", "best_loss": 0.5,
        "stageb_selection_surfaces": {},
    }))
    (artifacts / "mcts_handoff.jsonl").write_text('{"target_stage":"stage_b"}\n')
    (checkpoint / "step_0004_best_meta.json").write_text(json.dumps({
        "binder_seq": "AAA", "total_loss": 0.5, "structure_status": "ok",
        "structure_file": "step_0004_best.cif",
    }))
    (checkpoint / "step_0004_best.cif").write_text(
        "data_fixture\nloop_\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n0.0 1.0 2.0\n"
    )
    report = {
        "requested": {"proposal": "fast", "verifier": "fast", "esmc": "exact"},
        "applied": {"proposal": {"mode": "fast"}, "verifier": {"mode": "fast"},
                    **{label: {"mode": "exact"} for label in ("proposal_esmc", "verifier_esmc", "prior_esmc")}},
        "counters": {phase + ".successes": 1 for phase in (
            "proposal_update", "proposal_forward", "proposal_diffusion_gradient", "esmc_prior_forward", "verifier",
        )},
    }
    if damage == "missing_gradient":
        del report["counters"]["proposal_diffusion_gradient.successes"]
    if damage == "disabled_acceleration":
        report["applied"]["proposal"]["mode"] = "off"
    (tmp_path / "optimization_report.json").write_text(json.dumps(report))
    if damage:
        with pytest.raises(ValueError, match="acceleration|proposal_diffusion_gradient"):
            check_outputs(tmp_path)
    else:
        check_outputs(tmp_path)
