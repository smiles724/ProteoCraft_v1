#!/usr/bin/env python3
"""Download the exact model revisions used by the ESMFold2 pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proteocraft.model_sources import MODEL_REVISIONS, validate_model_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--write-env", type=Path, help="Save a reusable activation script for the accelerated venv")
    return parser.parse_args()


def write_activation_script(destination: Path, *, environment: Path, uplift_root: Path,
                            hf_home: Path, ccd_path: Path) -> None:
    lines = ["#!/usr/bin/env bash", f"source {shlex.quote(str(environment / 'bin/activate'))}"]
    variables = {
        "PROTEOCRAFT_UPLIFT_ROOT": str(uplift_root), "HF_HOME": str(hf_home),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "ESMCFOLD_CCD_PATH": str(ccd_path),
    }
    lines.extend(f"export {key}={shlex.quote(value)}" for key, value in variables.items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    from huggingface_hub import snapshot_download

    if args.write_env is not None:
        if not (Path(sys.prefix) / "bin/activate").is_file():
            raise ValueError("--write-env requires the accelerated venv; activate it first")
        if not os.environ.get("PROTEOCRAFT_UPLIFT_ROOT"):
            raise ValueError("Set PROTEOCRAFT_UPLIFT_ROOT before writing the activation script")
    hf_home = args.cache_dir.expanduser().resolve()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, Path] = {}
    for repo_id, revision in MODEL_REVISIONS.items():
        print(f"Resolving {repo_id}@{revision}")
        snapshots[repo_id] = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=hub_cache,
                local_files_only=args.local_files_only,
            )
        ).resolve()
        validate_model_snapshot(repo_id, snapshots[repo_id])
        print(f"[OK] {repo_id}: required files, sizes and safetensors headers")

    ccd_path = snapshots["biohub/ESMFold2"] / "ccd.pkl"
    if args.write_env is not None:
        destination = args.write_env.expanduser().resolve()
        write_activation_script(
            destination, environment=Path(sys.prefix),
            uplift_root=Path(os.environ["PROTEOCRAFT_UPLIFT_ROOT"]).expanduser().resolve(),
            hf_home=hf_home, ccd_path=ccd_path,
        )
        print(f"\nActivate this installation in each new shell:\nsource {shlex.quote(str(destination))}")
        return 0

    print("\nExport these variables before running ProteoCraft:")
    print(f"export HF_HOME={shlex.quote(str(hf_home))}")
    print("export HF_HUB_OFFLINE=1")
    print("export TRANSFORMERS_OFFLINE=1")
    print(f"export ESMCFOLD_CCD_PATH={shlex.quote(str(ccd_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
