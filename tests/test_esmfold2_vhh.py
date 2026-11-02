import json
from pathlib import Path

import pytest
import yaml

from benchmark.script.run_esmfold2_vhh_redesign import (
    build_hallucination_command,
    effective_hotspot_settings,
    write_case_record,
)
from proteocraft.hallucination.esmfold2_vhh import prepare_esmfold2_vhh_case
from proteocraft.cli import parse_args_with_profile, resolve_search_defaults


VHH_7EOW_SEQUENCE = (
    "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSV"
    "EGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"
)
VHH_7EOW_MASK = "".join(
    (
        "0" * 25,
        "1" * 8,
        "0" * 17,
        "1" * 8,
        "0" * 38,
        "1" * 21,
        "0" * 11,
    )
)


def _vhh_case() -> dict:
    return {
        "id": "7EOW_all_CDR",
        "target_sequences": ["ACDEFGHIK"],
        "initial_vhh_sequence": VHH_7EOW_SEQUENCE,
        "numbering": {
            "scheme": "imgt",
            "tool": "anarcii",
            "tool_version": "2.0.7",
        },
        "cdr_regions": {
            "CDR1": {"sequence_indices_1based": list(range(26, 34))},
            "CDR2": {"sequence_indices_1based": list(range(51, 59))},
            "CDR3": {"sequence_indices_1based": list(range(97, 118))},
        },
        "mutable_mask": VHH_7EOW_MASK,
        "scaffold": {"pdb_id": "7EOW", "chain_id": "B"},
    }


def test_prepare_vhh_case_builds_imgt_all_cdr_mask(tmp_path):
    case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)

    assert len(case.initial_vhh_sequence) == 128
    assert sum(case.mutable_mask) == 37
    assert case.mutable_mask_argument == VHH_7EOW_MASK
    assert case.mutable_positions_1based == (
        *range(26, 34),
        *range(51, 59),
        *range(97, 118),
    )
    assert dict(case.cdr_regions_1based)["CDR3"] == tuple(range(97, 118))
    assert case.target_chain_roles == ("target",)
    assert case.scaffold_pdb_id == "7EOW"


@pytest.mark.parametrize("missing_region", ("CDR1", "CDR2", "CDR3"))
def test_vhh_case_requires_all_three_imgt_cdrs(tmp_path, missing_region):
    raw_case = _vhh_case()
    del raw_case["cdr_regions"][missing_region]

    with pytest.raises(ValueError, match="requires exactly CDR1, CDR2 and CDR3"):
        prepare_esmfold2_vhh_case(raw_case, input_dir=tmp_path)


def test_vhh_case_rejects_overlapping_cdr_regions(tmp_path):
    raw_case = _vhh_case()
    raw_case.pop("mutable_mask")
    raw_case["cdr_regions"]["CDR2"]["sequence_indices_1based"][0] = 33

    with pytest.raises(ValueError, match="CDR regions overlap"):
        prepare_esmfold2_vhh_case(raw_case, input_dir=tmp_path)


def test_vhh_case_rejects_non_imgt_numbering(tmp_path):
    raw_case = _vhh_case()
    raw_case["numbering"]["scheme"] = "kabat"

    with pytest.raises(ValueError, match="requires IMGT numbering"):
        prepare_esmfold2_vhh_case(raw_case, input_dir=tmp_path)


def test_vhh_case_rejects_mask_that_disagrees_with_cdr_regions(tmp_path):
    raw_case = _vhh_case()
    raw_case["mutable_mask"] = "0" * len(VHH_7EOW_SEQUENCE)

    with pytest.raises(ValueError, match="does not match the union"):
        prepare_esmfold2_vhh_case(raw_case, input_dir=tmp_path)


def test_vhh_launcher_uses_recommended_profile_without_changing_core_modes(tmp_path):
    case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    command = build_hallucination_command(
        case,
        {
            "launch_profile": "esmfold2_d",
            "seed": 43,
            "esmfold2_save_verified_candidate_cif": True,
        },
        case_dir=tmp_path / "output",
    )

    args, parser = parse_args_with_profile(command[2:])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_task_type == "vhh_redesign"
    assert args.init_binder_seq == VHH_7EOW_SEQUENCE
    assert args.esmfold2_mutable_mask == VHH_7EOW_MASK
    assert args.mcts_strategy == "two_stage"
    assert args.mcts_bootstrap_steps == 70
    assert args.mcts_prior_mode == "no_group"
    assert args.seed == 43
    assert args.esmfold2_stagea_candidate_strategy == "positive_score_guarded"
    assert args.esmfold2_stagea_dominance_gap == pytest.approx(0.5)
    assert args.esmfold2_stagea_protected_mutation_candidates == 2
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.5)
    assert args.esmfold2_stagea_include_argmax_candidate is True
    assert args.esmfold2_one_step_diffusion_mode == "truncated_trajectory"
    assert args.esmfold2_one_step_diffusion_tau == pytest.approx(1.0)
    assert args.esmfold2_one_step_recycle_coordinates is False
    assert args.esmfold2_verifier_num_loops == 4
    assert args.esmfold2_verifier_num_sampling_steps == 50
    assert args.esmfold2_save_verified_candidate_cif is True


def test_vhh_launcher_forwards_dense_truncated_schedule(tmp_path):
    case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    command = build_hallucination_command(
        case,
        {
            "launch_profile": "esmfold2_d",
            "esmfold2_one_step_truncated_schedule_mode": "dense",
            "esmfold2_one_step_truncated_dense_num_sampling_steps": 41,
        },
        case_dir=tmp_path / "output",
    )

    args, parser = parse_args_with_profile(command[2:])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_one_step_truncated_schedule_mode == "dense"
    assert args.esmfold2_one_step_truncated_dense_num_sampling_steps == 41


def test_7eow_vhh_yaml_template_matches_adapter_contract(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "yaml"
        / "vhh"
        / "esmfold2_vhh_redesign_7eow_template.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_case = config["cases"][0]
    raw_case["target_sequences"] = ["ACDEFGHIK"]

    case = prepare_esmfold2_vhh_case(raw_case, input_dir=tmp_path)

    assert config["launch_profile"] == "esmfold2_d"
    assert case.initial_vhh_sequence == VHH_7EOW_SEQUENCE
    assert case.mutable_mask_argument == VHH_7EOW_MASK
    assert sum(case.mutable_mask) == 37


def test_vhh_case_record_marks_reference_coordinates_as_record_only(tmp_path):
    case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    case_dir = tmp_path / "output"
    command = build_hallucination_command(
        case,
        {"launch_profile": "esmfold2_d"},
        case_dir=case_dir,
    )
    write_case_record(
        case,
        {
            "launch_profile": "esmfold2_d",
            "config_path": str(tmp_path / "config.yaml"),
        },
        case_dir=case_dir,
        command=command,
    )

    provenance = json.loads(
        (case_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["task_type"] == "vhh_redesign"
    assert provenance["redesign_region"] == "imgt_all_cdr"
    assert provenance["framework_sequence_fixed"] is True
    assert provenance["framework_coordinates_fixed"] is False
    assert provenance["reference_structure_usage"] == (
        "recorded_only_not_model_input"
    )


def test_vhh_hotspot_launcher_preserves_explicit_overrides_and_records_numbering(tmp_path):
    raw = _vhh_case()
    raw["target_hotspots_1based"] = [1, 9]
    case = prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)
    config = {
        "launch_profile": "esmfold2_d",
        "config_path": str(tmp_path / "config.yaml"),
        "esmfold2_hotspot_weight": 0.7,
        "esmfold2_hotspot_cutoff": 7.0,
        "esmfold2_hotspot_satisfied_proposal_weight": 0.1,
        "esmfold2_hotspot_interface_mode": "localized",
        "esmfold2_hotspot_verify_weight": 0.25,
        "esmfold2_hotspot_verify_near_cutoff": 5.0,
        "esmfold2_hotspot_verify_far_cutoff": 18.0,
        "esmfold2_hotspot_verify_min_distance": 2.0,
        "esmfold2_hotspot_verify_too_close_weight": 0.0,
        "esmfold2_distogram_hotspot": False,
        "esmfold2_one_step_diffusion_hotspot": True,
    }
    command = build_hallucination_command(case, config, case_dir=tmp_path / "out")
    write_case_record(case, config, case_dir=tmp_path / "out", command=command)
    record = json.loads((tmp_path / "out/provenance.json").read_text())["hotspot"]
    assert record["target_hotspots_1based"] == [1, 9]
    assert record["target_hotspot_concat_indices_0based"] == [0, 8]
    for key, value in record["effective_options"].items():
        assert value == config[key]
    assert record["binder_contact_scope"] == "vhh_cdr_mutable_mask"


def test_vhh_hotspot_profile_zero_is_not_silently_reenabled(tmp_path):
    raw = _vhh_case()
    raw["target_hotspots_1based"] = [2]
    case = prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)
    command = build_hallucination_command(case, {"launch_profile": "esmfold2_d"}, case_dir=tmp_path)
    effective = effective_hotspot_settings(command)["effective_options"]
    assert effective["esmfold2_hotspot_weight"] == pytest.approx(0.5)
    assert effective["esmfold2_hotspot_verify_weight"] == 0.0
    assert effective["esmfold2_hotspot_verify_too_close_weight"] == 0.0
    assert effective["esmfold2_distogram_hotspot"] is True


@pytest.mark.parametrize("position", [0, 10, True, 1.5, float("nan"), float("inf")])
def test_vhh_hotspot_rejects_invalid_input_numbering(tmp_path, position):
    raw = _vhh_case()
    raw["target_hotspots_1based"] = [position]
    with pytest.raises(ValueError):
        prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)


def test_vhh_hotspot_multichain_mapping_excludes_scaffold_chains(tmp_path):
    raw = _vhh_case()
    raw["target_sequences"] = ["ACD", "EFG", "HIK"]
    raw["target_chain_roles"] = ["target", "scaffold", "target"]
    raw["target_hotspots_1based"] = [3, 7]
    case = prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)
    command = build_hallucination_command(case, {}, case_dir=tmp_path)
    assert effective_hotspot_settings(command)["target_hotspot_concat_indices_0based"] == [2, 6]
    raw["target_hotspots_1based"] = [4]
    with pytest.raises(ValueError, match="role=target"):
        prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)


def test_vhh_hotspot_rejects_duplicate_positions(tmp_path):
    raw = _vhh_case()
    raw["target_hotspots_1based"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_esmfold2_vhh_case(raw, input_dir=tmp_path)
