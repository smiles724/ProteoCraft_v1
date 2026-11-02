#!/usr/bin/env python3
"""Run single-chain ESMFold2 redesign from a motif or explicit sequence regions."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from proteocraft.hallucination.esmfold2_cdr import (  # noqa: E402
    ESMFold2CDRCase,
    framework_unchanged,
    prepare_esmfold2_cdr_case,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "benchmark" / "yaml" / "cdr_tcr" / "esmfold2_cdr_redesign_d_4cases.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--case", default=None)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str, *, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    config["config_path"] = str(config_path)
    if args.input_dir is not None:
        config["input_dir"] = args.input_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    config["prepare_only"] = bool(args.prepare_only)
    config["case"] = args.case
    config["case_index"] = args.case_index
    return config


def select_cases(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = list(config.get("cases") or [])
    if not cases:
        raise ValueError("Config must define a non-empty cases list")
    if config.get("case") is not None:
        cases = [case for case in cases if str(case.get("id")) == str(config["case"])]
        if not cases:
            raise ValueError(f"Case id not found: {config['case']}")
    if config.get("case_index") is not None:
        index = int(config["case_index"])
        if index < 0 or index >= len(cases):
            raise ValueError(f"case_index out of range: {index}")
        cases = [cases[index]]
    return cases


def _append_cli_argument(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = str(value).lower()
    command.extend((f"--{name}", str(value)))


def build_hallucination_command(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_hallucination.py"),
    ]
    launch_profile = config.get("launch_profile")
    if launch_profile:
        command.extend(("--profile", str(launch_profile)))
    command.extend(
        [
            "--esmfold2_task_type",
            "cdr_redesign",
            "--target",
            case.target_sequence_argument,
            "--esmfold2_target_chain_roles",
            case.target_chain_roles_argument,
            "--binder_length",
            str(len(case.binder_sequence)),
            "--init_binder_seq",
            case.binder_sequence,
            "--esmfold2_mutable_mask",
            case.mutable_mask_argument,
            "--output_dir",
            str(case_dir),
            "--esmfold2_artifact_dir",
            str(case_dir / "esmfold2_artifacts"),
            "--save_checkpoint_dir",
            str(case_dir / "checkpoints"),
            "--log_file",
            str(case_dir / "run.log"),
        ]
    )
    one_step_mode = str(
        config.get("esmfold2_one_step_diffusion_mode", "truncated_trajectory")
    ).strip().lower()
    if one_step_mode == "template_anchored_truncated_trajectory":
        from proteocraft.hallucination.esmfold2_template import validate_structure_regeneration_mask

        if case.template_path is None:
            raise ValueError(
                f"{case.case_id}: template-anchored mode requires template_path"
            )
        validate_structure_regeneration_mask(case.mutable_mask)
        command.extend(
            [
                "--target_template_path",
                case.template_path,
                "--esmfold2_template_chain_ids",
                case.template_chain_ids_argument,
                "--esmfold2_structure_regeneration_mask",
                case.mutable_mask_argument,
            ]
        )
    option_map = {
        "esmfold2_optimization_root": "esmfold2_optimization_root",
        "esmfold2_proposal_optimization": "esmfold2_proposal_optimization",
        "esmfold2_verifier_optimization": "esmfold2_verifier_optimization",
        "esmfold2_esmc_optimization": "esmfold2_esmc_optimization",
        "esmfold2_optimization_graph_max_tokens": "esmfold2_optimization_graph_max_tokens",
        "esmfold2_stagea_dominance_gap": "esmfold2_stagea_dominance_gap",
        "esmfold2_stagea_protected_mutation_candidates": "esmfold2_stagea_protected_mutation_candidates",
        "device": "device",
        "n_steps": "n_steps",
        "learning_rate": "learning_rate",
        "seed": "seed",
        "k_top": "k_top",
        "b_candidates": "b_candidates",
        "t_sample": "t_sample",
        "max_mutations_per_seq": "max_mutations_per_seq",
        "stagea_max_mutations_per_seq": "stagea_max_mutations_per_seq",
        "esmfold2_stagea_mutation_fraction": (
            "esmfold2_stagea_mutation_fraction"
        ),
        "esmfold2_stagea_candidate_strategy": (
            "esmfold2_stagea_candidate_strategy"
        ),
        "esmfold2_stagea_include_argmax_candidate": (
            "esmfold2_stagea_include_argmax_candidate"
        ),
        "stageb_max_mutations_per_seq": "stageb_max_mutations_per_seq",
        "mcts_num_beams": "mcts_num_beams",
        "mcts_strategy": "mcts_strategy",
        "mcts_bootstrap_steps": "mcts_bootstrap_steps",
        "mcts_bootstrap_patience": "mcts_bootstrap_patience",
        "mcts_bootstrap_min_delta": "mcts_bootstrap_min_delta",
        "mcts_bootstrap_max_steps": "mcts_bootstrap_max_steps",
        "mcts_handoff_top_k": "mcts_handoff_top_k",
        "mcts_expand_actions_per_step": "mcts_expand_actions_per_step",
        "mcts_prior_mode": "mcts_prior_mode",
        "esmfold2_stagea_soft_steps": "esmfold2_stagea_soft_steps",
        "esmfold2_stagea_tree_children": "esmfold2_stagea_tree_children",
        "esmfold2_stagea_root_selection_mode": (
            "esmfold2_stagea_root_selection_mode"
        ),
        "esmfold2_stagea_min_root_expansions": (
            "esmfold2_stagea_min_root_expansions"
        ),
        "esmfold2_stagea_preserve_soft_state": (
            "esmfold2_stagea_preserve_soft_state"
        ),
        "esmfold2_stagea_cumulative_annealing": "esmfold2_stagea_cumulative_annealing",
        "esmfold2_stagea_paired_verifier_seed": (
            "esmfold2_stagea_paired_verifier_seed"
        ),
        "esmfold2_stageb_soft_steps": "esmfold2_stageb_soft_steps",
        "esmfold2_stageb_discrete_rerank": (
            "esmfold2_stageb_discrete_rerank"
        ),
        "esmfold2_stageb_rerank_pool_size": (
            "esmfold2_stageb_rerank_pool_size"
        ),
        "esmfold2_stageb_exploration_actions": (
            "esmfold2_stageb_exploration_actions"
        ),
        "esmfold2_stageb_verifier_candidates": (
            "esmfold2_stageb_verifier_candidates"
        ),
        "esmfold2_stageb_paired_verifier_seed": (
            "esmfold2_stageb_paired_verifier_seed"
        ),
        "esmfold2_stageb_verifier_seed": "esmfold2_stageb_verifier_seed",
        "esmfold2_experimental_model": "esmfold2_experimental_model",
        "esmfold2_verifier_model": "esmfold2_verifier_model",
        "esmfold2_verifier_num_loops": "esmfold2_verifier_num_loops",
        "esmfold2_verifier_num_sampling_steps": "esmfold2_verifier_num_sampling_steps",
        "esmfold2_proposal_num_loops": "esmfold2_proposal_num_loops",
        "esmfold2_proposal_num_sampling_steps": "esmfold2_proposal_num_sampling_steps",
        "esmfold2_proposal_load_internal_esmc": "esmfold2_proposal_load_internal_esmc",
        "esmfold2_proposal_lm_dropout": "esmfold2_proposal_lm_dropout",
        "esmfold2_force_proposal_lm_dropout": "esmfold2_force_proposal_lm_dropout",
        "esmfold2_one_step_diffusion": "esmfold2_one_step_diffusion",
        "esmfold2_one_step_diffusion_tau": "esmfold2_one_step_diffusion_tau",
        "esmfold2_one_step_diffusion_mode": "esmfold2_one_step_diffusion_mode",
        "esmfold2_one_step_truncated_num_sampling_steps": (
            "esmfold2_one_step_truncated_num_sampling_steps"
        ),
        "esmfold2_one_step_truncated_schedule_mode": (
            "esmfold2_one_step_truncated_schedule_mode"
        ),
        "esmfold2_one_step_truncated_dense_num_sampling_steps": (
            "esmfold2_one_step_truncated_dense_num_sampling_steps"
        ),
        "esmfold2_one_step_diffusion_weight": "esmfold2_one_step_diffusion_weight",
        "esmfold2_one_step_diffusion_rg_weight": "esmfold2_one_step_diffusion_rg_weight",
        "esmfold2_one_step_diffusion_interval": "esmfold2_one_step_diffusion_interval",
        "esmfold2_one_step_recycle_coordinates": "esmfold2_one_step_recycle_coordinates",
        "esmfold2_one_step_diffusion_hotspot": "esmfold2_one_step_diffusion_hotspot",
        "esmfold2_distogram_hotspot": "esmfold2_distogram_hotspot",
        "esmfold2_verifier_loss_mode": "esmfold2_verifier_loss_mode",
        "esmfold2_lambda_esmc": "esmfold2_lambda_esmc",
        "esmfold2_logit_init_mode": "esmfold2_logit_init_mode",
        "esmfold2_initial_logits_checkpoint": "esmfold2_initial_logits_checkpoint",
        "esmfold2_track_proposal_confidence": (
            "esmfold2_track_proposal_confidence"
        ),
        "esmfold2_final_polish": "esmfold2_final_polish",
        "esmfold2_stagea_handoff_checkpoint_interval": "esmfold2_stagea_handoff_checkpoint_interval",
        "esmfold2_hotspot_cutoff": "esmfold2_hotspot_cutoff",
        "esmfold2_hotspot_satisfied_proposal_weight": (
            "esmfold2_hotspot_satisfied_proposal_weight"
        ),
        "esmfold2_hotspot_interface_mode": "esmfold2_hotspot_interface_mode",
        "esmfold2_hotspot_verify_weight": "esmfold2_hotspot_verify_weight",
        "esmfold2_hotspot_verify_near_cutoff": (
            "esmfold2_hotspot_verify_near_cutoff"
        ),
        "esmfold2_hotspot_verify_far_cutoff": (
            "esmfold2_hotspot_verify_far_cutoff"
        ),
        "esmfold2_hotspot_verify_min_distance": (
            "esmfold2_hotspot_verify_min_distance"
        ),
        "esmfold2_hotspot_verify_too_close_weight": (
            "esmfold2_hotspot_verify_too_close_weight"
        ),
        "checkpoint_save_policy": "checkpoint_save_policy",
        "esmfold2_save_verified_candidate_cif": "esmfold2_save_verified_candidate_cif",
        "esmfold2_save_visualization_trace": "esmfold2_save_visualization_trace",
    }
    for config_key, cli_name in option_map.items():
        _append_cli_argument(command, cli_name, config.get(config_key))
    if config.get("esmfold2_optimization_timing") is True:
        command.append("--esmfold2_optimization_timing")
    return command


def parse_hallucination_result(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key.strip()] = value.strip()
    return values


def write_case_record(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
    command: Iterable[str],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_spec.json").write_text(
        json.dumps(case.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "command.txt").write_text(
        shlex.join(list(command)) + "\n",
        encoding="utf-8",
    )
    one_step_mode = str(
        config.get("esmfold2_one_step_diffusion_mode", "truncated_trajectory")
    ).strip().lower()
    template_anchored = one_step_mode == (
        "template_anchored_truncated_trajectory"
    )
    provenance = {
        "model_backend": "esmfold2",
        "task_type": "cdr_redesign",
        "launch_profile": config.get("launch_profile"),
        "structure_condition": (
            "external_complex_template_noise_consistent_clamp"
            if template_anchored
            else "sequence_only"
        ),
        "template_path": case.template_path,
        "template_usage": (
            "target_and_partner_all_atom_binder_framework_backbone"
            if template_anchored
            else "recorded_only_not_consumed_by_esmfold2"
        ),
        "template_chain_ids": list(case.template_chain_ids),
        "structure_regeneration_mask": case.mutable_mask_argument,
        "config_path": config["config_path"],
        "target_chain_roles": list(case.target_chain_roles),
        "mutable_residue_count": sum(case.mutable_mask),
    }
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_case(
    case: ESMFold2CDRCase,
    config: Mapping[str, Any],
    *,
    output_root: Path,
) -> int:
    case_dir = output_root / case.case_id
    command = build_hallucination_command(case, config, case_dir=case_dir)
    write_case_record(case, config, case_dir=case_dir, command=command)
    if bool(config.get("prepare_only")):
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    result_path = case_dir / "hallucination_result.txt"
    summary: Dict[str, Any] = {
        "case_id": case.case_id,
        "returncode": int(completed.returncode),
        "status": "failed",
    }
    if completed.returncode == 0 and result_path.is_file():
        result = parse_hallucination_result(result_path)
        best_sequence = result.get("Best Sequence", "")
        final_sequence = result.get("Final Sequence", "")
        summary.update(
            {
                "status": "ok",
                "best_sequence": best_sequence,
                "final_sequence": final_sequence,
                "best_framework_unchanged": framework_unchanged(
                    case.binder_sequence,
                    best_sequence,
                    case.mutable_mask,
                ),
                "final_framework_unchanged": framework_unchanged(
                    case.binder_sequence,
                    final_sequence,
                    case.mutable_mask,
                ),
            }
        )
        if not summary["best_framework_unchanged"] or not summary[
            "final_framework_unchanged"
        ]:
            summary["status"] = "framework_changed"
            completed = subprocess.CompletedProcess(
                completed.args,
                returncode=2,
            )
    (case_dir / "cdr_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    input_dir = resolve_path(str(config["input_dir"]))
    output_root = resolve_path(str(config["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        prepare_esmfold2_cdr_case(case, input_dir=input_dir)
        for case in select_cases(config)
    ]
    return max(
        (run_case(case, config, output_root=output_root) for case in cases),
        default=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
