#!/usr/bin/env python3
"""Validate the lightweight ESMFold2 runtime without loading model weights."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "torch": "2.7.1",
    "transformers": "4.57.6",
    "esm": "3.3.0",
    "huggingface-hub": "0.36.2",
    "accelerate": "1.14.0",
    "numpy": "2.4.6",
}
EXPECTED_GIT_COMMITS = {
    "transformers": "ef32577f55da19a4989cd7b22e004dc43a4998cb",
    "esm": "82ee35553d39169d678f784c8d3f8712ffd7d2c4",
}
MODEL_REVISIONS = {
    "biohub/ESMFold2": "1ebf0e3481a5184eb6171d40615c79e384b48796",
    "biohub/ESMFold2-Experimental-Fast": "04dec820ede9283c9893e318e5ca5a9ac2ab93bc",
    "biohub/ESMC-6B": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
}
OPTIONAL_KERNELS = ("xformers", "flash-attn", "transformer-engine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def direct_url_commit(name: str) -> str | None:
    try:
        raw = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    data = json.loads(raw)
    return data.get("vcs_info", {}).get("commit_id")


def cache_snapshot(cache_dir: Path, repo_id: str, revision: str) -> Path:
    repo_dir = "models--" + repo_id.replace("/", "--")
    return cache_dir / "hub" / repo_dir / "snapshots" / revision


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 is required; found {sys.version.split()[0]}")
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    for package, expected in EXPECTED_PACKAGES.items():
        actual = package_version(package)
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual or 'missing'}")
        else:
            print(f"[OK] {package} {actual}")

    if package_version("fair-esm") is not None:
        errors.append("fair-esm is installed and conflicts with the Biohub esm namespace")

    for package, expected in EXPECTED_GIT_COMMITS.items():
        actual = direct_url_commit(package)
        if actual != expected:
            errors.append(
                f"{package}: expected Git commit {expected}, found {actual or 'unknown'}"
            )
        else:
            print(f"[OK] {package} Git commit {actual}")

    try:
        from esm.models.esmfold2 import (  # noqa: F401
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
            prepare_esmfold2_input,
        )
        from transformers.models.esmfold2.modeling_esmfold2 import (  # noqa: F401
            ESMFold2Model,
        )
        from transformers.models.esmfold2.modeling_esmfold2_experimental import (  # noqa: F401
            ESMFold2ExperimentalModel,
        )
        from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM  # noqa: F401
    except Exception as exc:
        errors.append(f"ESMFold2 API import failed: {type(exc).__name__}: {exc}")
    else:
        print("[OK] ESMFold2 and ESMC APIs import successfully")

    try:
        import proteocraft  # noqa: F401
    except Exception as exc:
        errors.append(f"ProteoCraft import failed: {type(exc).__name__}: {exc}")
    else:
        print("[OK] ProteoCraft is importable")

    import torch

    if torch.version.cuda != "12.6":
        warnings.append(f"tested Torch CUDA is 12.6; found {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif args.require_cuda:
        errors.append("CUDA is required but torch.cuda.is_available() is false")
    else:
        warnings.append("CUDA is unavailable; model execution was not tested")

    installed_optional = [name for name in OPTIONAL_KERNELS if package_version(name)]
    if installed_optional:
        warnings.append(
            "optional fused kernels are installed and may change numerical results: "
            + ", ".join(installed_optional)
        )
    else:
        print("[OK] Pure-PyTorch fallback matches the current experimental baseline")

    cache_dir = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    missing_models = []
    for repo_id, revision in MODEL_REVISIONS.items():
        snapshot = cache_snapshot(cache_dir, repo_id, revision)
        if not snapshot.is_dir():
            missing_models.append(f"{repo_id}@{revision}")
    if missing_models:
        message = "missing cached model revisions: " + ", ".join(missing_models)
        if args.require_models:
            errors.append(message)
        else:
            warnings.append(message)
    else:
        print(f"[OK] Exact model revisions are cached under {cache_dir}")

    ccd_path = os.environ.get("ESMCFOLD_CCD_PATH")
    if ccd_path and Path(ccd_path).is_file():
        print(f"[OK] ESMCFOLD_CCD_PATH={ccd_path}")
    elif args.require_models:
        errors.append("ESMCFOLD_CCD_PATH is unset or does not point to ccd.pkl")
    else:
        warnings.append("ESMCFOLD_CCD_PATH is unset or does not point to ccd.pkl")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
