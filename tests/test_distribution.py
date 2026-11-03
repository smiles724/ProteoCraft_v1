"""User-facing installation and dependency boundaries of this distribution."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from proteocraft.cli import parse_args_with_profile, resolve_search_defaults
from proteocraft.hallucination.profile_loader import PROFILE_DIRECTORY
from proteocraft.evaluation import esmfold2_rescore


def test_esmfold2_imports_without_any_legacy_or_model_dependency():
    script = """
import importlib.abc, sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'protenix', 'bindcraft', 'esm', 'transformers'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
from proteocraft.losses import compute_rg_loss
from proteocraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
from proteocraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination
from proteocraft.evaluation.cli import main
from proteocraft.cli import parse_args_with_profile, resolve_search_defaults
args, parser = parse_args_with_profile(['--target', 'ACDEFGHIK'])
assert resolve_search_defaults(args, parser) == 'esmfold2'
assert args.n_steps == 110
assert args.esmfold2_stagea_candidate_strategy == 'positive_score_guarded'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_cli_matches_every_mainline_profile_option():
    args, parser = parse_args_with_profile(["--target", "ACDEFGHIK"])
    resolve_search_defaults(args, parser)
    defaults = yaml.safe_load((PROFILE_DIRECTORY / "esmfold2_d.yaml").read_text())[
        "defaults"
    ]
    for name, expected in defaults.items():
        actual = getattr(args, name)
        # lambda_esmc's validated CLI representation is a string.
        assert actual == expected or (
            name == "esmfold2_lambda_esmc" and str(actual) == str(expected)
        ), name


@pytest.mark.parametrize(
    "option,value",
    [
        ("--model_backend", "protenix"),
        ("--mode", "hybrid"),
        ("--model_name", "protenix-v2"),
        ("--w_pde", "1.0"),
    ],
)
def test_removed_protenix_options_fail_explicitly(option, value):
    with pytest.raises(SystemExit):
        parse_args_with_profile(["--target", "ACDE", option, value])


def test_documented_rescore_candidate_freezes_without_models(tmp_path):
    candidate = {
        "case_id": "demo",
        "candidate_id": "demo_1",
        "target_sequences": ["GIVEQCCTSICSLYQLENYCN"],
        "binder_seq": "AAAAAAAAAA",
        "target_chain_roles": ["target"],
        "target_hotspot_concat_indices": [],
    }
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text(json.dumps(candidate) + "\n")
    output = tmp_path / "rescore"
    esmfold2_rescore.freeze(manifest, esmfold2_rescore.DEFAULT_CONFIG, output)
    result = esmfold2_rescore.report(output)
    assert result["expected_scores"] == 3
    assert result["successful_scores"] == 0


def test_design_cli_writes_result_after_completed_search(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from proteocraft import cli
    from proteocraft.hallucination import esmfold2_pipeline

    monkeypatch.setattr(cli, "load_esmfold2_backend", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        esmfold2_pipeline, "run_esmfold2_mcts_hallucination",
        lambda **kwargs: ("AA", "AG", [1.0, 0.9]),
    )
    cli.main(["--target", "ACDE", "--binder_length", "2",
              "--output_dir", str(tmp_path), "--device", "cpu"])
    result = (tmp_path / "hallucination_result.txt").read_text()
    assert "Best Sequence: AA" in result
    assert "Final Sequence: AG" in result
