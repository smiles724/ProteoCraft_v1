from pathlib import Path

import pytest
import yaml

from proteocraft.hallucination.esmfold2_cdr import (
    framework_unchanged,
    prepare_esmfold2_cdr_case,
)
from benchmark.script.run_esmfold2_cdr_redesign import build_hallucination_command
from proteocraft.cli import parse_args_with_profile, resolve_search_defaults


def _write_case_fasta(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                ">example|Chain A|target",
                "AAAA",
                ">example|Chain B|scaffold",
                "CCCC",
                ">example|Chain C|binder",
                "GGACDEFGHIIK",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_esmfold2_cdr_case_preserves_roles_and_cdr_mask(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    template_path = tmp_path / "case.cif"
    _write_case_fasta(fasta_path)
    template_path.write_text("data_case\n", encoding="utf-8")

    case = prepare_esmfold2_cdr_case(
        {
            "id": "case_alpha",
            "fasta_path": fasta_path.name,
            "template_path": template_path.name,
            "binder_chain": "C",
            "target_chains": ["A"],
            "scaffold_chains": ["B"],
            "cdr_motif": "ACDEFG",
        },
        input_dir=tmp_path,
    )

    assert case.partner_sequences == ("AAAA", "CCCC")
    assert case.target_chain_roles == ("target", "scaffold")
    assert case.target_sequence_argument == "AAAA|CCCC"
    assert case.mutable_mask_argument == "001111110000"
    assert case.template_chain_ids == ("A", "B", "C")
    assert case.template_chain_ids_argument == "A,B,C"
    assert case.cdr_start_0based == 2
    assert case.cdr_end_0based_exclusive == 8


def test_framework_guard_rejects_changes_outside_cdr():
    reference = "AAACCC"
    mutable_mask = (False, False, False, True, True, True)

    assert framework_unchanged(reference, "AAAGGG", mutable_mask)
    assert not framework_unchanged(reference, "TAAGGG", mutable_mask)


def test_prepare_esmfold2_cdr_case_rejects_missing_motif(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    _write_case_fasta(fasta_path)

    with pytest.raises(ValueError, match="CDR motif not found"):
        prepare_esmfold2_cdr_case(
            {
                "id": "bad_case",
                "fasta_path": fasta_path.name,
                "binder_chain": "C",
                "target_chains": ["A"],
                "scaffold_chains": ["B"],
                "cdr_motif": "YYYY",
            },
            input_dir=tmp_path,
        )


def test_cdr_launcher_forwards_one_step_diffusion_options(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    template_path = tmp_path / "case.cif"
    _write_case_fasta(fasta_path)
    template_path.write_text("data_case\n", encoding="utf-8")
    case = prepare_esmfold2_cdr_case(
        {
            "id": "case_alpha",
            "fasta_path": fasta_path.name,
            "template_path": template_path.name,
            "binder_chain": "C",
            "target_chains": ["A"],
            "scaffold_chains": ["B"],
            "cdr_motif": "ACDEFG",
        },
        input_dir=tmp_path,
    )

    command = build_hallucination_command(
        case,
        {
            "launch_profile": "esmfold2_d",
            "esmfold2_one_step_diffusion": True,
            "esmfold2_stagea_root_selection_mode": "hierarchical_ucb",
            "esmfold2_stagea_min_root_expansions": 2,
            "esmfold2_stagea_cumulative_annealing": False,
            "esmfold2_one_step_diffusion_tau": 2.0,
            "esmfold2_one_step_truncated_schedule_mode": "dense",
            "esmfold2_one_step_truncated_dense_num_sampling_steps": 41,
            "esmfold2_one_step_diffusion_weight": 0.5,
            "esmfold2_one_step_diffusion_rg_weight": 0.1,
            "esmfold2_one_step_diffusion_interval": 5,
            "esmfold2_one_step_recycle_coordinates": True,
            "esmfold2_one_step_diffusion_hotspot": False,
            "esmfold2_distogram_hotspot": True,
            "esmfold2_verifier_loss_mode": "aligned_common",
            "mcts_handoff_top_k": 2,
            "esmfold2_stageb_discrete_rerank": True,
            "esmfold2_stageb_rerank_pool_size": 20,
            "esmfold2_stageb_exploration_actions": 1,
            "esmfold2_stageb_verifier_candidates": 8,
            "esmfold2_stageb_paired_verifier_seed": True,
            "checkpoint_save_policy": "all",
        },
        case_dir=tmp_path / "output",
    )
    options = dict(zip(command[::2], command[1::2]))

    assert options["--profile"] == "esmfold2_d"
    assert options["--esmfold2_one_step_diffusion"] == "true"
    assert options["--esmfold2_stagea_root_selection_mode"] == "hierarchical_ucb"
    assert options["--esmfold2_stagea_min_root_expansions"] == "2"
    assert options["--esmfold2_stagea_cumulative_annealing"] == "false"
    assert options["--esmfold2_one_step_diffusion_tau"] == "2.0"
    assert options["--esmfold2_one_step_truncated_schedule_mode"] == "dense"
    assert (
        options["--esmfold2_one_step_truncated_dense_num_sampling_steps"]
        == "41"
    )
    assert options["--esmfold2_one_step_diffusion_weight"] == "0.5"
    assert options["--esmfold2_one_step_diffusion_rg_weight"] == "0.1"
    assert options["--esmfold2_one_step_diffusion_interval"] == "5"
    assert options["--esmfold2_one_step_recycle_coordinates"] == "true"
    assert options["--esmfold2_one_step_diffusion_hotspot"] == "false"
    assert options["--esmfold2_distogram_hotspot"] == "true"
    assert options["--esmfold2_verifier_loss_mode"] == "aligned_common"
    assert options["--mcts_handoff_top_k"] == "2"
    assert options["--esmfold2_stageb_discrete_rerank"] == "true"
    assert options["--esmfold2_stageb_rerank_pool_size"] == "20"
    assert options["--esmfold2_stageb_exploration_actions"] == "1"
    assert options["--esmfold2_stageb_verifier_candidates"] == "8"
    assert options["--esmfold2_stageb_paired_verifier_seed"] == "true"
    assert options["--save_checkpoint_dir"] == str(tmp_path / "output" / "checkpoints")
    assert options["--checkpoint_save_policy"] == "all"


@pytest.mark.parametrize(
    "config_name",
    (
        "esmfold2_cdr_redesign_d_4cases.yaml",
    ),
)
def test_formal_cdr_configs_inherit_recommended_two_stage_profile(
    tmp_path,
    config_name,
):
    fasta_path = tmp_path / "case.fasta"
    _write_case_fasta(fasta_path)
    case = prepare_esmfold2_cdr_case(
        {
            "id": "case_alpha",
            "fasta_path": fasta_path.name,
            "binder_chain": "C",
            "target_chains": ["A"],
            "scaffold_chains": ["B"],
            "cdr_motif": "ACDEFG",
        },
        input_dir=tmp_path,
    )
    config_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "yaml"
        / "cdr_tcr"
        / config_name
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command = build_hallucination_command(
        case,
        config,
        case_dir=tmp_path / "output",
    )

    args, parser = parse_args_with_profile(command[2:])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_task_type == "cdr_redesign"
    assert args.mcts_strategy == "two_stage"
    assert args.n_steps == 110
    assert args.mcts_bootstrap_steps == 70
    assert args.mcts_bootstrap_max_steps == 70
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


def test_template_anchored_cdr_config_forwards_template_and_structure_mask(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    template_path = tmp_path / "case.cif"
    _write_case_fasta(fasta_path)
    template_path.write_text("data_case\n", encoding="utf-8")
    case = prepare_esmfold2_cdr_case(
        {
            "id": "case_alpha",
            "fasta_path": fasta_path.name,
            "template_path": template_path.name,
            "binder_chain": "C",
            "target_chains": ["A"],
            "scaffold_chains": ["B"],
            "cdr_motif": "ACDEFG",
        },
        input_dir=tmp_path,
    )
    config_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["esmfold2_one_step_diffusion_mode"] = "template_anchored_truncated_trajectory"
    command = build_hallucination_command(
        case,
        config,
        case_dir=tmp_path / "output",
    )
    args, parser = parse_args_with_profile(command[2:])

    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_one_step_diffusion_mode == (
        "template_anchored_truncated_trajectory"
    )
    assert args.target_template_path == str(template_path)
    assert args.esmfold2_template_chain_ids == "A,B,C"
    assert args.esmfold2_structure_regeneration_mask == case.mutable_mask_argument
