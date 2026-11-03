"""Public input and maintainer configuration contracts for the mainline search."""
import pytest
import yaml

from proteocraft.cli import parse_args_with_profile, resolve_search_defaults
from proteocraft.hallucination.profile_loader import LaunchProfileError, write_case_profile


@pytest.mark.parametrize("option", ["--n_steps", "--esmfold2_stagea_soft_steps", "--save_checkpoint_dir"])
def test_algorithm_and_removed_settings_are_not_public_cli_options(option):
    with pytest.raises(SystemExit):
        parse_args_with_profile(["--target", "ACDE", option, "1"])


def test_profile_budget_and_cli_task_inputs_resolve_together(tmp_path):
    profile = write_case_profile("esmfold2_d", {
        "n_steps": 2, "mcts_bootstrap_steps": 1, "mcts_bootstrap_max_steps": 1,
        "esmfold2_stagea_soft_steps": 1, "esmfold2_stageb_soft_steps": 1,
    }, tmp_path / "profile.yaml")
    args, parser = parse_args_with_profile([
        "--target", "ACDE", "--profile", str(profile), "--binder_length", "80", "--seed", "7"])
    resolve_search_defaults(args, parser)
    assert (args.n_steps, args.mcts_bootstrap_steps, args.mcts_strategy) == (2, 1, "two_stage")
    assert (args.esmfold2_stagea_soft_steps, args.esmfold2_stageb_soft_steps) == (1, 1)
    assert (args.binder_length, args.seed) == (80, 7)


@pytest.mark.parametrize("setting,value", [
    ("mcts_strategy", "best_of_n"), ("mcts_strategy", "stage_a_only"),
    ("esmfold2_best_of_n", 8), ("n_step", 110),
])
def test_removed_strategies_and_unknown_settings_fail_before_model_loading(tmp_path, setting, value):
    with pytest.raises(LaunchProfileError):
        write_case_profile("esmfold2_d", {setting: value}, tmp_path / "profile.yaml")


def test_invalid_soft_budget_fails_before_model_loading(tmp_path):
    profile = write_case_profile("esmfold2_d", {"esmfold2_stagea_soft_steps": 0}, tmp_path / "profile.yaml")
    args, parser = parse_args_with_profile(["--target", "ACDE", "--profile", str(profile)])
    with pytest.raises(SystemExit):
        resolve_search_defaults(args, parser)


def test_unknown_key_in_user_supplied_yaml_is_rejected(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"schema_version": 1, "name": "bad",
                                     "defaults": {"n_step": 110}}))
    with pytest.raises(SystemExit):
        parse_args_with_profile(["--target", "ACDE", "--profile", str(profile)])


def test_quoted_false_in_profile_disables_boolean_settings(tmp_path):
    settings = {
        "esmfold2_stagea_preserve_soft_state": "false",
        "esmfold2_memory_saved_tensor_pin_memory": "false",
        "esmfold2_optimization_timing": "false",
    }
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "booleans", "defaults": settings,
    }))
    args, _ = parse_args_with_profile(["--target", "ACDE", "--profile", str(profile)])
    assert all(getattr(args, name) is False for name in settings)


@pytest.mark.parametrize("setting", [
    "esmfold2_stagea_preserve_soft_state",
    "esmfold2_memory_saved_tensor_pin_memory",
    "esmfold2_optimization_timing",
])
def test_invalid_boolean_in_profile_is_rejected(tmp_path, setting):
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "bad", "defaults": {setting: "not-a-boolean"},
    }))
    with pytest.raises(SystemExit):
        parse_args_with_profile(["--target", "ACDE", "--profile", str(profile)])


def test_invalid_boolean_in_case_override_is_rejected(tmp_path):
    with pytest.raises(LaunchProfileError):
        write_case_profile("esmfold2_d", {
            "esmfold2_stagea_preserve_soft_state": "not-a-boolean",
        }, tmp_path / "profile.yaml")


def test_invalid_boolean_on_cli_is_rejected():
    with pytest.raises(SystemExit):
        parse_args_with_profile([
            "--target", "ACDE", "--esmfold2_resume_reverify_roots", "not-a-boolean",
        ])


@pytest.mark.parametrize("kind", ["cdr", "vhh"])
@pytest.mark.parametrize("option,value", [
    ("esmfold2_final_polish", True), ("checkpoint_save_policy", "all"),
])
def test_removed_options_in_batch_configs_are_not_silently_ignored(tmp_path, kind, option, value):
    if kind == "vhh":
        from tests.test_esmfold2_vhh import _vhh_case
        from proteocraft.hallucination.esmfold2_vhh import prepare_esmfold2_vhh_case
        from benchmark.script.run_esmfold2_vhh_redesign import build_hallucination_command
        case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    else:
        from tests.test_esmfold2_cdr import _write_case_fasta
        from proteocraft.hallucination.esmfold2_cdr import prepare_esmfold2_cdr_case
        from benchmark.script.run_esmfold2_cdr_redesign import build_hallucination_command
        fasta = tmp_path / "case.fasta"
        _write_case_fasta(fasta)
        case = prepare_esmfold2_cdr_case({
            "id": "case", "fasta_path": fasta.name, "binder_chain": "C",
            "target_chains": ["A"], "scaffold_chains": ["B"], "cdr_motif": "ACDEFG",
        }, input_dir=tmp_path)
    with pytest.raises(ValueError, match=option):
        build_hallucination_command(case, {option: value}, case_dir=tmp_path / "out")
