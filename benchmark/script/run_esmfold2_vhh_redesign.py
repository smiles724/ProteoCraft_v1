#!/usr/bin/env python3
"""Run sequence-only ESMFold2 VHH all-CDR redesign from a YAML config."""

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

from proteocraft.hallucination.esmfold2_cdr import framework_unchanged  # noqa: E402
from proteocraft.hallucination.esmfold2_vhh import (  # noqa: E402
    ESMFold2VHHCase,
    prepare_esmfold2_vhh_case,
)


HOTSPOT_OPTIONS = (
    "esmfold2_hotspot_weight",
    "esmfold2_hotspot_cutoff",
    "esmfold2_hotspot_satisfied_proposal_weight",
    "esmfold2_hotspot_interface_mode",
    "esmfold2_hotspot_verify_weight",
    "esmfold2_hotspot_verify_near_cutoff",
    "esmfold2_hotspot_verify_far_cutoff",
    "esmfold2_hotspot_verify_min_distance",
    "esmfold2_hotspot_verify_too_close_weight",
    "esmfold2_distogram_hotspot",
    "esmfold2_one_step_diffusion_hotspot",
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
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
    config["input_dir"] = (
        args.input_dir
        if args.input_dir is not None
        else config.get("input_dir", str(config_path.parent))
    )
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
        cases = [
            case
            for case in cases
            if str(case.get("id")) == str(config["case"])
        ]
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
    if name == "esmfold2_memory_saved_tensor_pin_memory" and isinstance(value, bool):
        command.append(f"--{name}" if value else f"--no-{name}")
        return
    if isinstance(value, bool):
        value = str(value).lower()
    command.extend((f"--{name}", str(value)))


def build_hallucination_command(
    case: ESMFold2VHHCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_hallucination.py"),
    ]
    from proteocraft.cli import build_arg_parser
    from proteocraft.hallucination.search_config import build_search_config_parser
    from proteocraft.hallucination.profile_loader import write_case_profile
    launch_profile = config.get("launch_profile") or "esmfold2_d"
    schema_names = {action.dest for action in build_search_config_parser()._actions}
    public_names = {action.dest for action in build_arg_parser()._actions}
    unsupported = sorted(
        key for key in config
        if (key.startswith(("esmfold2_", "mcts_")) or key in {"save_checkpoint_dir", "checkpoint_save_policy"})
        and key not in schema_names | public_names
    )
    if unsupported:
        raise ValueError("Unsupported search options in task config: " + ", ".join(unsupported))
    settings = {key: value for key, value in config.items() if key in schema_names and value is not None}
    if settings:
        launch_profile = str(write_case_profile(launch_profile, settings, case_dir / "search_profile.yaml"))
    command.extend(("--profile", str(launch_profile)))
    command.extend(
        [
            "--esmfold2_task_type",
            "vhh_redesign",
            "--target",
            case.target_sequence_argument,
            "--esmfold2_target_chain_roles",
            case.target_chain_roles_argument,
            "--binder_length",
            str(len(case.initial_vhh_sequence)),
            "--init_binder_seq",
            case.initial_vhh_sequence,
            "--esmfold2_mutable_mask",
            case.mutable_mask_argument,
            "--output_dir",
            str(case_dir),
            "--esmfold2_artifact_dir",
            str(case_dir / "esmfold2_artifacts"),
            "--log_file",
            str(case_dir / "run.log"),
        ]
    )
    if case.target_hotspots_argument is not None:
        command.extend(("--target_hotspots", case.target_hotspots_argument))
    task_inputs = {"target", "binder_length", "esmfold2_task_type", "init_binder_seq",
                   "esmfold2_mutable_mask", "esmfold2_target_chain_roles", "target_hotspots", "output_dir", "profile"}
    for name in sorted(public_names - task_inputs - {"help", "esmfold2_optimization_timing"}):
        _append_cli_argument(command, name, config.get(name))
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


def effective_hotspot_settings(command: Iterable[str]) -> Dict[str, Any]:
    """Resolve the same profile/CLI/default precedence as the actual runner."""
    from proteocraft.cli import (
        _build_hotspot_params,
        parse_args_with_profile,
        resolve_esmfold2_hotspot_defaults,
    )

    args, parser = parse_args_with_profile(list(command)[2:])
    params = _build_hotspot_params(args, parser)
    resolve_esmfold2_hotspot_defaults(
        args, model_backend="esmfold2", hotspot_params=params
    )
    return {
        "target_hotspots_1based": [] if params is None else params["target_hotspot_residues_1based"],
        "target_hotspot_concat_indices_0based": [] if params is None else params["target_hotspot_concat_indices"],
        "numbering": "1-based positions in concatenated input target chains, not PDB or UniProt numbering",
        "binder_contact_scope": "vhh_cdr_mutable_mask",
        "effective_options": {name: getattr(args, name) for name in HOTSPOT_OPTIONS},
    }


def write_case_record(
    case: ESMFold2VHHCase,
    config: Mapping[str, Any],
    *,
    case_dir: Path,
    command: Iterable[str],
) -> None:
    command = list(command)
    hotspot_settings = effective_hotspot_settings(command)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_spec.json").write_text(
        json.dumps(case.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "command.txt").write_text(
        shlex.join(list(command)) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "model_backend": "esmfold2",
        "task_type": "vhh_redesign",
        "redesign_region": "imgt_all_cdr",
        "launch_profile": config.get("launch_profile"),
        "structure_condition": "sequence_only",
        "framework_sequence_fixed": True,
        "framework_coordinates_fixed": False,
        "reference_structure_path": case.reference_structure_path,
        "reference_structure_usage": "recorded_only_not_model_input",
        "scaffold_pdb_id": case.scaffold_pdb_id,
        "scaffold_chain_id": case.scaffold_chain_id,
        "numbering_scheme": case.numbering_scheme,
        "numbering_tool": case.numbering_tool,
        "numbering_tool_version": case.numbering_tool_version,
        "config_path": config["config_path"],
        "target_chain_roles": list(case.target_chain_roles),
        "mutable_residue_count": sum(case.mutable_mask),
        "hotspot": hotspot_settings,
    }
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_case(
    case: ESMFold2VHHCase,
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
                    case.initial_vhh_sequence,
                    best_sequence,
                    case.mutable_mask,
                ),
                "final_framework_unchanged": framework_unchanged(
                    case.initial_vhh_sequence,
                    final_sequence,
                    case.mutable_mask,
                ),
            }
        )
        if not summary["best_framework_unchanged"] or not summary[
            "final_framework_unchanged"
        ]:
            summary["status"] = "framework_changed"
            completed = subprocess.CompletedProcess(completed.args, returncode=2)
    (case_dir / "vhh_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    if not config.get("output_dir"):
        raise ValueError("Config must define output_dir or pass --output-dir")
    input_dir = resolve_path(str(config["input_dir"]))
    output_root = resolve_path(str(config["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        prepare_esmfold2_vhh_case(case, input_dir=input_dir)
        for case in select_cases(config)
    ]
    return max(
        (run_case(case, config, output_root=output_root) for case in cases),
        default=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
