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
    if name == "esmfold2_memory_saved_tensor_pin_memory" and isinstance(value, bool):
        command.append(f"--{name}" if value else f"--no-{name}")
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
    if settings.get("esmfold2_one_step_diffusion_mode") == "template_anchored_truncated_trajectory":
        from proteocraft.hallucination.esmfold2_template import validate_structure_regeneration_mask
        if case.template_path is None:
            raise ValueError(f"{case.case_id}: template-anchored mode requires template_path")
        validate_structure_regeneration_mask(case.mutable_mask)
        settings.update(target_template_path=case.template_path,
                        esmfold2_template_chain_ids=case.template_chain_ids_argument,
                        esmfold2_structure_regeneration_mask=case.mutable_mask_argument)
    if settings:
        launch_profile = str(write_case_profile(launch_profile, settings, case_dir / "search_profile.yaml"))
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
            "--log_file",
            str(case_dir / "run.log"),
        ]
    )
    task_inputs = {"target", "binder_length", "esmfold2_task_type", "init_binder_seq",
                   "esmfold2_mutable_mask", "esmfold2_target_chain_roles", "output_dir", "profile"}
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
