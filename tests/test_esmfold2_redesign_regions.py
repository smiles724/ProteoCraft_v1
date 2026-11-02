from pathlib import Path

import pytest
import yaml

from benchmark.script.run_esmfold2_cdr_redesign import build_hallucination_command
from proteocraft.hallucination.esmfold2_cdr import (
    framework_unchanged,
    make_redesign_mutable_mask,
    prepare_esmfold2_cdr_case,
)
from proteocraft.hallucination.esmfold2_template import validate_structure_regeneration_mask


@pytest.fixture
def input_case(tmp_path):
    (tmp_path / "case.fasta").write_text(
        ">case|Chain A|target\nAAAA\n>case|Chain B|scaffold\nCCCC\n"
        ">case|Chain C|binder\nACDEFGHIKLMNPQRSTVWY\n"
    )
    (tmp_path / "case.cif").write_text("data_case\n")
    return {"id": "case", "fasta_path": "case.fasta", "binder_chain": "C",
            "target_chains": ["A"], "scaffold_chains": ["B"]}


def test_three_disjoint_regions_preserve_gaps_and_record_real_intervals(input_case, tmp_path):
    case = prepare_esmfold2_cdr_case(
        dict(input_case, redesign_regions=[[14, 16], [3, 4], [8, 9]]), input_dir=tmp_path
    )
    expected = {3, 4, 8, 9, 14, 15, 16}
    assert {i + 1 for i, enabled in enumerate(case.mutable_mask) if enabled} == expected
    assert case.partner_sequences == ("AAAA", "CCCC")
    assert case.target_chain_roles == ("target", "scaffold")
    assert case.cdr_motif is None and case.cdr_start_0based is None
    assert case.cdr_end_0based_exclusive is None
    record = case.to_dict()
    assert record["cdr_start_1based"] is None and record["cdr_end_1based"] is None
    assert record["redesign_regions_1based"] == ((3, 4), (8, 9), (14, 16))
    assert record["redesign_region_sequences"] == ("DE", "IK", "QRS")
    for i, aa in enumerate(case.binder_sequence):
        changed = case.binder_sequence[:i] + ("A" if aa != "A" else "G") + case.binder_sequence[i + 1:]
        assert framework_unchanged(case.binder_sequence, changed, case.mutable_mask) == (i + 1 in expected)


@pytest.mark.parametrize("regions", [
    None, [], "3-4", [[3]], [[3, 4, 5]], ["34"], [[True, 4]], [[3.0, 4]],
    [["3", 4]], [[0, 4]], [[-1, 4]], [[4, 3]], [[3, 21]],
    [[3, 5], [5, 7]], [[3, 5], [3, 5]], {"start": 3, "end": 4},
])
def test_reject_ambiguous_or_invalid_regions(regions):
    with pytest.raises(ValueError, match="redesign_regions"):
        make_redesign_mutable_mask("ACDEFGHIKLMNPQRSTVWY", regions)


@pytest.mark.parametrize("selectors", [{}, {"cdr_motif": "DE", "redesign_regions": [[3, 4]]}])
def test_exactly_one_selector_is_required(input_case, tmp_path, selectors):
    with pytest.raises(ValueError, match="exactly one"):
        prepare_esmfold2_cdr_case(dict(input_case, **selectors), input_dir=tmp_path)


def test_singleton_and_terminal_regions_are_legal_without_template(input_case, tmp_path):
    case = prepare_esmfold2_cdr_case(
        dict(input_case, redesign_regions=[[1, 1], [20, 20]]), input_dir=tmp_path
    )
    assert sum(case.mutable_mask) == 2
    command = build_hallucination_command(case, {}, case_dir=tmp_path / "out")
    assert "--target_template_path" not in command


def test_motif_and_equivalent_region_generate_identical_launch_arguments(input_case, tmp_path):
    legacy = prepare_esmfold2_cdr_case(dict(input_case, cdr_motif="DEFG"), input_dir=tmp_path)
    regions = prepare_esmfold2_cdr_case(dict(input_case, redesign_regions=[[3, 6]]), input_dir=tmp_path)
    assert legacy.mutable_mask == regions.mutable_mask
    assert legacy.cdr_start_0based == regions.cdr_start_0based == 2
    assert legacy.cdr_end_0based_exclusive == regions.cdr_end_0based_exclusive == 6
    assert "redesign_regions_1based" not in legacy.to_dict()
    assert legacy.to_dict()["cdr_motif"] == "DEFG"
    config = {"launch_profile": "esmfold2_d"}
    assert build_hallucination_command(legacy, config, case_dir=tmp_path) == build_hallucination_command(
        regions, config, case_dir=tmp_path
    )


def test_regions_template_forwarding_and_trace_options(input_case, tmp_path):
    case = prepare_esmfold2_cdr_case(
        dict(input_case, template_path="case.cif", redesign_regions=[[3, 4], [8, 9], [14, 16]]),
        input_dir=tmp_path,
    )
    command = build_hallucination_command(case, {
        "esmfold2_one_step_diffusion_mode": "template_anchored_truncated_trajectory",
        "esmfold2_save_verified_candidate_cif": True, "esmfold2_save_visualization_trace": True,
    }, case_dir=tmp_path / "out")
    options = dict(zip(command[2::2], command[3::2]))
    assert options["--esmfold2_structure_regeneration_mask"] == case.mutable_mask_argument
    assert options["--esmfold2_template_chain_ids"] == "A,B,C"
    assert options["--esmfold2_save_verified_candidate_cif"] == "true"
    assert options["--esmfold2_save_visualization_trace"] == "true"


def test_nearby_intervals_need_two_actual_fixed_anchors(input_case, tmp_path):
    # One fixed position between the two loops does not provide two anchors.
    case = prepare_esmfold2_cdr_case(
        dict(input_case, template_path="case.cif", redesign_regions=[[3, 4], [6, 7]]),
        input_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="fixed framework"):
        validate_structure_regeneration_mask(case.mutable_mask)
    with pytest.raises(ValueError, match="fixed framework"):
        build_hallucination_command(case, {
            "esmfold2_one_step_diffusion_mode": "template_anchored_truncated_trajectory",
        }, case_dir=tmp_path / "out")
    assert validate_structure_regeneration_mask(
        make_redesign_mutable_mask(case.binder_sequence, [[3, 4], [7, 8]])[0]
    )


def test_multi_region_real_template_clamp_does_not_free_intervening_framework():
    from tests.test_esmfold2_template import _backbone_features
    from proteocraft.hallucination.esmfold2_template import build_template_clamp

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml").read_text())
    raw = dict(config["cases"][0])
    old = prepare_esmfold2_cdr_case(raw, input_dir=root / config["input_dir"])
    start = old.cdr_start_0based + 1
    raw.pop("cdr_motif")
    raw["redesign_regions"] = [[start, start + 1], [start + 4, start + 5], [start + 8, start + 9]]
    case = prepare_esmfold2_cdr_case(raw, input_dir=root / config["input_dir"])
    sequences = (*case.partner_sequences, case.binder_sequence)
    clamp = build_template_clamp(features=_backbone_features(sequences), template_path=case.template_path,
        template_chain_ids=case.template_chain_ids, template_chain_sequences=sequences,
        binder_structure_regeneration_mask=case.mutable_mask)
    binder_start = sum(map(len, case.partner_sequences)) * 4
    fixed = clamp.fixed_atom_mask[0, binder_start:].reshape(-1, 4)
    for index, mutable in enumerate(case.mutable_mask):
        assert bool(fixed[index].all()) != mutable


@pytest.mark.parametrize("case_index", range(4))
def test_all_original_benchmark_cases_keep_the_same_command(case_index, tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml").read_text())
    raw = dict(config["cases"][case_index])
    old = prepare_esmfold2_cdr_case(raw, input_dir=root / config["input_dir"])
    raw.pop("cdr_motif")
    raw["redesign_regions"] = [[old.cdr_start_0based + 1, old.cdr_end_0based_exclusive]]
    new = prepare_esmfold2_cdr_case(raw, input_dir=root / config["input_dir"])
    assert old.mutable_mask == new.mutable_mask
    assert build_hallucination_command(old, config, case_dir=tmp_path) == build_hallucination_command(
        new, config, case_dir=tmp_path
    )
