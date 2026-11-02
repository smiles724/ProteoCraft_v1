#!/usr/bin/env python3
"""Validate and run the portable 59-case ProteoCraft ESMFold2 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from proteocraft.hallucination.esmfold2_cdr import (  # noqa: E402
    prepare_esmfold2_cdr_case,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "benchmark"
    / "datasets"
    / "proteocraft_59_v1"
    / "benchmark.yaml"
)
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
TASK_TYPES = ("denovo", "tcr", "cdr_redesign")
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "case_id",
        "task_type",
        "target_name",
        "sequence_file",
        "structure_file",
        "alpha_chain",
        "beta_chain",
        "hotspot_chain",
        "hotspot_start",
        "hotspot_end",
        "target_hotspots",
        "binder_chain",
        "target_chains",
        "scaffold_chains",
        "cdr_motif",
        "binder_length",
        "source_run_id",
    }
)
PROTECTED_OVERRIDES = frozenset(
    {
        "profile",
        "model_backend",
        "mode",
        "esmfold2_task_type",
        "target",
        "target_hotspots",
        "esmfold2_target_chain_roles",
        "binder_length",
        "init_binder_seq",
        "esmfold2_mutable_mask",
        "output_dir",
        "esmfold2_artifact_dir",
        "save_checkpoint_dir",
        "log_file",
        "device",
    }
)


@dataclass(frozen=True)
class PDBResidue:
    residue_number: int
    insertion_code: str
    amino_acid: str


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    task_type: str
    target_name: str
    source_run_id: str
    target_sequence: str
    binder_length: int
    target_chain_roles: tuple[str, ...] = ()
    target_hotspots_1based: tuple[int, ...] = ()
    initial_binder_sequence: str | None = None
    mutable_mask: str | None = None
    sequence_file: str | None = None
    structure_file: str | None = None
    source_chain_hotspot: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--task-type",
        action="append",
        choices=TASK_TYPES,
        default=[],
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forward an additional non-input option to run_hallucination.py.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_dataset_path(dataset_root: Path, raw_path: str, field: str) -> Path:
    relative = Path(str(raw_path).strip())
    if not str(relative):
        raise ValueError(f"{field} must not be empty")
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the dataset: {relative}")
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the dataset root: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} not found: {resolved}")
    return resolved


def load_benchmark_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark config must be a mapping: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark config schema_version must be 1")
    for key in ("name", "manifest", "launch_profile", "output_dir"):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"benchmark config requires {key}")
    expected_counts = payload.get("expected_counts")
    if not isinstance(expected_counts, dict):
        raise ValueError("benchmark config requires expected_counts")
    payload["config_path"] = str(path)
    payload["dataset_root"] = str(path.parent.resolve())
    payload["manifest_path"] = str(
        _resolve_dataset_path(path.parent, payload["manifest"], "manifest")
    )
    return payload


def load_manifest(config: Mapping[str, Any]) -> list[dict[str, str]]:
    manifest_path = Path(str(config["manifest_path"]))
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - columns)
        if missing:
            raise ValueError(
                "manifest is missing required columns: " + ", ".join(missing)
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("manifest must contain at least one case")

    case_ids = [row["case_id"].strip() for row in rows]
    duplicates = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError("duplicate case_id values: " + ", ".join(duplicates))
    if any(not case_id for case_id in case_ids):
        raise ValueError("manifest case_id must not be empty")

    actual_counts = Counter(row["task_type"].strip() for row in rows)
    expected_counts = {
        str(task): int(count)
        for task, count in dict(config["expected_counts"]).items()
    }
    if dict(actual_counts) != expected_counts:
        raise ValueError(
            f"manifest task counts {dict(actual_counts)} do not match "
            f"expected_counts {expected_counts}"
        )
    return rows


def _validate_sequence(sequence: str, *, label: str) -> str:
    normalized = str(sequence).replace(" ", "").strip().upper()
    invalid = sorted(set(normalized) - STANDARD_AMINO_ACIDS)
    if not normalized:
        raise ValueError(f"{label} sequence must not be empty")
    if invalid:
        raise ValueError(f"{label} contains unsupported amino acids: {invalid}")
    return normalized


def read_single_fasta(path: Path) -> str:
    records: list[str] = []
    current: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records.append("".join(current))
            current = []
        else:
            if current is None:
                current = []
            current.append(line)
    if current is not None:
        records.append("".join(current))
    if len(records) != 1:
        raise ValueError(f"expected exactly one FASTA record in {path}")
    return _validate_sequence(records[0], label=str(path))


def extract_pdb_chain(path: Path, chain_id: str) -> tuple[PDBResidue, ...]:
    residues: list[PDBResidue] = []
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if (line[21].strip() or "_") != chain_id:
            continue
        key = (line[22:26].strip(), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        amino_acid = AA3_TO_1.get(line[17:20].strip())
        if amino_acid is None:
            raise ValueError(
                f"{path} chain {chain_id} has unsupported residue "
                f"{line[17:20].strip()} at {key[0]}{key[1]}"
            )
        residues.append(
            PDBResidue(
                residue_number=int(key[0]),
                insertion_code=key[1],
                amino_acid=amino_acid,
            )
        )
    if not residues:
        raise ValueError(f"no CA residues found for chain {chain_id} in {path}")
    return tuple(residues)


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _positive_int(raw: str, *, field: str) -> int:
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _parse_direct_hotspots(raw: str, *, target_length: int) -> tuple[int, ...]:
    positions = tuple(
        _positive_int(token, field="target_hotspots") for token in _split_csv(raw)
    )
    if not positions:
        return ()
    if len(set(positions)) != len(positions):
        raise ValueError("target_hotspots contains duplicate positions")
    if max(positions) > target_length:
        raise ValueError(
            f"target hotspot {max(positions)} exceeds target length {target_length}"
        )
    return positions


def _map_tcr_hotspots(
    row: Mapping[str, str],
    alpha: Sequence[PDBResidue],
    beta: Sequence[PDBResidue],
) -> tuple[tuple[int, ...], str | None]:
    chain_name = row["hotspot_chain"].strip().lower()
    start_raw = row["hotspot_start"].strip()
    end_raw = row["hotspot_end"].strip()
    if not chain_name:
        if start_raw or end_raw:
            raise ValueError(f"{row['case_id']}: hotspot range requires a chain")
        return (), None
    if chain_name not in {"alpha", "beta"}:
        raise ValueError(f"{row['case_id']}: hotspot_chain must be alpha or beta")
    start = _positive_int(start_raw, field="hotspot_start")
    end = _positive_int(end_raw, field="hotspot_end")
    if end < start:
        raise ValueError(f"{row['case_id']}: hotspot_end must be >= hotspot_start")
    residues = alpha if chain_name == "alpha" else beta
    offset = 0 if chain_name == "alpha" else len(alpha)
    positions = tuple(
        offset + index + 1
        for index, residue in enumerate(residues)
        if start <= residue.residue_number <= end
    )
    if len(positions) != end - start + 1:
        raise ValueError(
            f"{row['case_id']}: mapped {len(positions)} hotspot residues, "
            f"expected {end - start + 1}"
        )
    return positions, f"{chain_name}:{start}-{end}"


def prepare_manifest_case(
    row: Mapping[str, str],
    *,
    dataset_root: Path,
) -> PreparedCase:
    case_id = row["case_id"].strip()
    task_type = row["task_type"].strip()
    if task_type not in TASK_TYPES:
        raise ValueError(f"{case_id}: unsupported task_type {task_type!r}")
    target_name = row["target_name"].strip()
    source_run_id = row["source_run_id"].strip()
    if not target_name or not source_run_id:
        raise ValueError(f"{case_id}: target_name and source_run_id are required")

    if task_type == "denovo":
        sequence_path = _resolve_dataset_path(
            dataset_root, row["sequence_file"], "sequence_file"
        )
        structure_path = _resolve_dataset_path(
            dataset_root, row["structure_file"], "structure_file"
        )
        target_sequence = read_single_fasta(sequence_path)
        hotspots = _parse_direct_hotspots(
            row["target_hotspots"], target_length=len(target_sequence)
        )
        return PreparedCase(
            case_id=case_id,
            task_type=task_type,
            target_name=target_name,
            source_run_id=source_run_id,
            target_sequence=target_sequence,
            binder_length=_positive_int(
                row["binder_length"], field="binder_length"
            ),
            target_hotspots_1based=hotspots,
            sequence_file=str(sequence_path),
            structure_file=str(structure_path),
        )

    if task_type == "tcr":
        structure_path = _resolve_dataset_path(
            dataset_root, row["structure_file"], "structure_file"
        )
        alpha_chain = row["alpha_chain"].strip()
        beta_chain = row["beta_chain"].strip()
        if not alpha_chain or not beta_chain or alpha_chain == beta_chain:
            raise ValueError(f"{case_id}: distinct alpha_chain/beta_chain required")
        alpha = extract_pdb_chain(structure_path, alpha_chain)
        beta = extract_pdb_chain(structure_path, beta_chain)
        hotspots, source_hotspot = _map_tcr_hotspots(row, alpha, beta)
        target_sequence = "|".join(
            (
                "".join(residue.amino_acid for residue in alpha),
                "".join(residue.amino_acid for residue in beta),
            )
        )
        return PreparedCase(
            case_id=case_id,
            task_type=task_type,
            target_name=target_name,
            source_run_id=source_run_id,
            target_sequence=target_sequence,
            binder_length=_positive_int(
                row["binder_length"], field="binder_length"
            ),
            target_chain_roles=("target", "target"),
            target_hotspots_1based=hotspots,
            structure_file=str(structure_path),
            source_chain_hotspot=source_hotspot,
        )

    sequence_path = _resolve_dataset_path(
        dataset_root, row["sequence_file"], "sequence_file"
    )
    structure_path = _resolve_dataset_path(
        dataset_root, row["structure_file"], "structure_file"
    )
    cdr_case = prepare_esmfold2_cdr_case(
        {
            "id": case_id,
            "fasta_path": str(sequence_path),
            "template_path": str(structure_path),
            "binder_chain": row["binder_chain"].strip(),
            "target_chains": _split_csv(row["target_chains"]),
            "scaffold_chains": _split_csv(row["scaffold_chains"]),
            "cdr_motif": row["cdr_motif"].strip(),
        },
        input_dir=dataset_root,
    )
    return PreparedCase(
        case_id=case_id,
        task_type=task_type,
        target_name=target_name,
        source_run_id=source_run_id,
        target_sequence=cdr_case.target_sequence_argument,
        binder_length=len(cdr_case.binder_sequence),
        target_chain_roles=cdr_case.target_chain_roles,
        initial_binder_sequence=cdr_case.binder_sequence,
        mutable_mask=cdr_case.mutable_mask_argument,
        sequence_file=str(sequence_path),
        structure_file=str(structure_path),
    )


def load_prepared_cases(
    config: Mapping[str, Any],
) -> list[PreparedCase]:
    dataset_root = Path(str(config["dataset_root"]))
    return [
        prepare_manifest_case(row, dataset_root=dataset_root)
        for row in load_manifest(config)
    ]


def select_cases(
    cases: Sequence[PreparedCase],
    *,
    case_ids: Sequence[str] = (),
    task_types: Sequence[str] = (),
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[PreparedCase]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    requested_ids = set(case_ids)
    available_ids = {case.case_id for case in cases}
    missing_ids = sorted(requested_ids - available_ids)
    if missing_ids:
        raise ValueError("unknown case ids: " + ", ".join(missing_ids))
    requested_tasks = set(task_types)
    filtered = [
        case
        for case in cases
        if (not requested_ids or case.case_id in requested_ids)
        and (not requested_tasks or case.task_type in requested_tasks)
    ]
    return [
        case
        for index, case in enumerate(filtered)
        if index % num_shards == shard_index
    ]


def parse_overrides(raw_overrides: Sequence[str]) -> list[tuple[str, str]]:
    overrides: list[tuple[str, str]] = []
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"--set requires KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if not key or not value:
            raise ValueError(f"--set requires non-empty KEY=VALUE, got {raw!r}")
        if key in PROTECTED_OVERRIDES:
            raise ValueError(f"--set cannot override benchmark input option {key}")
        overrides.append((key, value))
    return overrides


def build_hallucination_command(
    case: PreparedCase,
    *,
    profile: str,
    device: str,
    case_dir: Path,
    overrides: Sequence[tuple[str, str]] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_hallucination.py"),
        "--profile",
        profile,
        "--esmfold2_task_type",
        case.task_type,
        "--target",
        case.target_sequence,
        "--binder_length",
        str(case.binder_length),
        "--output_dir",
        str(case_dir),
        "--esmfold2_artifact_dir",
        str(case_dir / "esmfold2_artifacts"),
        "--save_checkpoint_dir",
        str(case_dir / "checkpoints"),
        "--log_file",
        str(case_dir / "run.log"),
        "--device",
        device,
    ]
    if case.target_chain_roles:
        command.extend(
            ("--esmfold2_target_chain_roles", ",".join(case.target_chain_roles))
        )
    if case.target_hotspots_1based:
        command.extend(
            (
                "--target_hotspots",
                ",".join(str(value) for value in case.target_hotspots_1based),
            )
        )
    if case.initial_binder_sequence is not None:
        command.extend(("--init_binder_seq", case.initial_binder_sequence))
    if case.mutable_mask is not None:
        command.extend(("--esmfold2_mutable_mask", case.mutable_mask))
    for key, value in overrides:
        command.extend((f"--{key}", value))
    return command


def _write_case_record(
    case: PreparedCase,
    *,
    case_dir: Path,
    command: Sequence[str],
    config_path: Path,
    manifest_path: Path,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(case)
    payload["structure_usage"] = "provenance_and_evaluation_only"
    payload["esmfold2_input"] = "sequence_only"
    payload["benchmark_config"] = str(config_path)
    payload["benchmark_manifest"] = str(manifest_path)
    (case_dir / "case_spec.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (case_dir / "command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )


def _is_complete(case_dir: Path) -> bool:
    return (case_dir / "esmfold2_artifacts" / "run_summary.json").is_file()


def benchmark_summary_filename(*, num_shards: int, shard_index: int) -> str:
    if num_shards == 1:
        return "benchmark_run_summary.json"
    return (
        f"benchmark_run_summary.shard-{shard_index:02d}"
        f"-of-{num_shards:02d}.json"
    )


def run_cases(
    cases: Sequence[PreparedCase],
    *,
    config: Mapping[str, Any],
    profile: str,
    output_root: Path,
    device: str,
    overrides: Sequence[tuple[str, str]],
    prepare_only: bool,
    skip_existing: bool,
    continue_on_error: bool,
    num_shards: int,
    shard_index: int,
) -> int:
    records: list[dict[str, Any]] = []
    manifest_path = Path(str(config["manifest_path"]))
    config_path = Path(str(config["config_path"]))
    for case in cases:
        case_dir = output_root / case.case_id
        command = build_hallucination_command(
            case,
            profile=profile,
            device=device,
            case_dir=case_dir,
            overrides=overrides,
        )
        if prepare_only:
            print(shlex.join(command))
            records.append({"case_id": case.case_id, "status": "prepared"})
            continue
        if skip_existing and _is_complete(case_dir):
            print(f"[skip] {case.case_id}: existing run_summary.json")
            records.append({"case_id": case.case_id, "status": "skipped"})
            continue
        _write_case_record(
            case,
            case_dir=case_dir,
            command=command,
            config_path=config_path,
            manifest_path=manifest_path,
        )
        print(f"[run] {case.case_id}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        records.append(
            {
                "case_id": case.case_id,
                "status": status,
                "returncode": int(completed.returncode),
            }
        )
        if completed.returncode != 0 and not continue_on_error:
            break

    if not prepare_only:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "benchmark": config["name"],
            "profile": profile,
            "device": device,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "selected_cases": len(cases),
            "records": records,
        }
        summary_path = output_root / benchmark_summary_filename(
            num_shards=num_shards,
            shard_index=shard_index,
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 1 if any(record["status"] == "failed" for record in records) else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve_config_path(args.config)
    config = load_benchmark_config(config_path)
    cases = load_prepared_cases(config)
    selected = select_cases(
        cases,
        case_ids=args.case,
        task_types=args.task_type,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not selected:
        raise ValueError("case selection is empty")
    profile = str(args.profile or config["launch_profile"])
    output_root = Path(str(args.output_dir or config["output_dir"])).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    overrides = parse_overrides(args.overrides)
    counts = Counter(case.task_type for case in selected)
    print(
        f"Validated {len(cases)} cases; selected {len(selected)} "
        f"for shard {args.shard_index}/{args.num_shards}: {dict(counts)}"
    )
    return run_cases(
        selected,
        config=config,
        profile=profile,
        output_root=output_root.resolve(),
        device=args.device,
        overrides=overrides,
        prepare_only=bool(args.prepare_only or args.dry_run),
        skip_existing=bool(args.skip_existing),
        continue_on_error=bool(args.continue_on_error),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
