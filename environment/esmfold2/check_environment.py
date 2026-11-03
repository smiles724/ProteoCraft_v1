#!/usr/bin/env python3
"""Check the selected ESMFold2 installation without loading model weights."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from proteocraft.model_sources import MODEL_REVISIONS, validate_model_snapshot  # noqa: E402

COMMON_PACKAGES = {
    "transformers": "4.57.6", "huggingface-hub": "0.36.2",
    "accelerate": "1.14.0", "numpy": "2.4.6", "zstd": "1.5.7.2",
}
STACK_PACKAGES = {
    "accelerated": {"torch": "2.11.0+cu128", "triton": "3.6.0", "esm": "3.4.0",
                    "xformers": "0.0.35", "regex": "2026.7.19"},
    "stock": {"torch": "2.7.1", "triton": "3.3.1", "esm": "3.3.0"},
}
ESM_COMMITS = {
    "accelerated": "d0207ea3c8cbf072679ece3bb332787d25e06852",
    "stock": "82ee35553d39169d678f784c8d3f8712ffd7d2c4",
}
UPLIFT_COMMIT = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
TRANSFORMERS_COMMIT = "ef32577f55da19a4989cd7b22e004dc43a4998cb"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=tuple(STACK_PACKAGES), default="accelerated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args(argv)


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def direct_url_commit(name):
    raw = metadata.distribution(name).read_text("direct_url.json")
    if not raw:
        return None
    data = json.loads(raw)
    if "vcs_info" in data:
        return data["vcs_info"].get("commit_id")
    location = urlparse(data.get("url", ""))
    if data.get("dir_info", {}).get("editable") and location.scheme == "file":
        return subprocess.run(
            ["git", "-C", unquote(location.path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    return None


def check_sources(stack):
    actual = direct_url_commit("esm")
    if actual != ESM_COMMITS[stack]:
        raise ValueError(f"esm: expected source {ESM_COMMITS[stack]}, found {actual or 'unknown'}")
    if stack == "stock":
        if direct_url_commit("transformers") != TRANSFORMERS_COMMIT:
            raise ValueError(f"transformers must use source {TRANSFORMERS_COMMIT}")
        return
    # Reuse the source-directory and modeling-file hash guards used by search.
    from proteocraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from proteocraft.hallucination.esmfold2_optimizations import OptimizationManager
    manager = OptimizationManager(ESMFold2ProposalConfig(), "cpu")
    manager._source()
    if manager.source_revision != UPLIFT_COMMIT:
        raise ValueError(f"Optimization checkout must use {UPLIFT_COMMIT}")
    manager._check_core_pin("ef2inv")
    manager._check_core_pin("esmfold2")
    manager._check_fork(
        "transformers.models.esmfold2.modeling_esmfold2",
        "transformers.models.esmfold2.modeling_esmfold2_common",
        "transformers.models.esmfold2.modeling_esmfold2_experimental",
        "transformers.models.esmc.modeling_esmc",
    )


def main(argv=None):
    args = parse_args(argv)
    errors, warnings = [], []
    print(f"Checking {args.stack} environment")
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 is required; found {sys.version.split()[0]}")
    if args.stack == "accelerated" and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        errors.append("The accelerated environment requires Linux x86-64")
    for name, expected in (COMMON_PACKAGES | STACK_PACKAGES[args.stack]).items():
        actual = package_version(name)
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual or 'missing'}")
    if package_version("fair-esm") is not None:
        errors.append("fair-esm conflicts with the Biohub esm namespace; use an isolated environment")
    try:
        from esm.models.esmfold2 import (  # noqa: F401
            ESMFold2InputBuilder, ProteinInput, StructurePredictionInput, prepare_esmfold2_input,
        )
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: F401
        from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel  # noqa: F401
        from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM  # noqa: F401
    except Exception as exc:
        errors.append(f"Model API import failed: {type(exc).__name__}: {exc}")
    else:
        print("[OK] ESMFold2 and ESMC APIs")
    try:
        check_sources(args.stack)
    except Exception as exc:
        errors.append(f"Source check failed: {type(exc).__name__}: {exc}")
    else:
        print("[OK] Pinned model sources and optimization sources")
    try:
        import torch
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must select a CUDA GPU, for example cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to Torch")
        properties = torch.cuda.get_device_properties(device)
        if args.stack == "accelerated" and (properties.major, properties.minor) not in ((8, 0), (9, 0)):
            raise RuntimeError(f"Default fast kernels require SM80/SM90; found SM{properties.major}{properties.minor}")
        print(f"[OK] {properties.name}: {properties.total_memory / 2**30:.1f} GiB, Torch CUDA {torch.version.cuda}")
    except Exception as exc:
        (errors if args.require_cuda else warnings).append(f"GPU: {type(exc).__name__}: {exc}")
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    models_not_downloaded = []
    for repo_id, revision in MODEL_REVISIONS.items():
        snapshot = cache_dir / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        if not args.require_models and not snapshot.exists():
            models_not_downloaded.append(repo_id)
            continue
        try:
            validate_model_snapshot(repo_id, snapshot)
        except Exception as exc:
            (errors if args.require_models else warnings).append(f"{repo_id}: {type(exc).__name__}: {exc}")
        else:
            print(f"[OK] {repo_id}: complete files and valid safetensors headers")
    ccd_path = os.environ.get("ESMCFOLD_CCD_PATH")
    expected_ccd = cache_dir / "hub/models--biohub--ESMFold2/snapshots" / MODEL_REVISIONS["biohub/ESMFold2"] / "ccd.pkl"
    if models_not_downloaded:
        print("[INFO] Models not downloaded yet; next run environment/esmfold2/download_models.py")
    if (args.require_models or not models_not_downloaded) and (
        not ccd_path or Path(ccd_path).expanduser().resolve() != expected_ccd.resolve() or not expected_ccd.is_file()
    ):
        (errors if args.require_models else warnings).append("ESMCFOLD_CCD_PATH must point to ccd.pkl in the pinned ESMFold2 snapshot")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    if not errors:
        print("Environment check passed")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
