#!/usr/bin/env python3
"""Download the exact model revisions used by the ESMFold2 pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "biohub/ESMFold2": "1ebf0e3481a5184eb6171d40615c79e384b48796",
    "biohub/ESMFold2-Experimental-Fast": "04dec820ede9283c9893e318e5ca5a9ac2ab93bc",
    "biohub/ESMC-6B": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
}


def parse_args() -> argparse.Namespace:
    default_cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_home = args.cache_dir.expanduser().resolve()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, Path] = {}
    for repo_id, revision in MODELS.items():
        print(f"Resolving {repo_id}@{revision}")
        snapshots[repo_id] = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=hub_cache,
                local_files_only=args.local_files_only,
            )
        ).resolve()

    ccd_path = snapshots["biohub/ESMFold2"] / "ccd.pkl"
    if not ccd_path.is_file():
        raise FileNotFoundError(f"ESMFold2 CCD cache is missing: {ccd_path}")

    print("\nExport these variables before running ProteoCraft:")
    print(f"export HF_HOME={shlex.quote(str(hf_home))}")
    print("export HF_HUB_OFFLINE=1")
    print("export TRANSFORMERS_OFFLINE=1")
    print(f"export ESMCFOLD_CCD_PATH={shlex.quote(str(ccd_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
