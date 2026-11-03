"""Resolve official models to the immutable snapshots downloaded by our helper."""
from __future__ import annotations

import json
from pathlib import Path


MODEL_REVISIONS = {
    "biohub/ESMFold2": "1ebf0e3481a5184eb6171d40615c79e384b48796",
    "biohub/ESMFold2-Experimental-Fast": "04dec820ede9283c9893e318e5ca5a9ac2ab93bc",
    "biohub/ESMC-6B": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
}

# Byte lengths of the immutable Hub revisions above. No model tensors are read
# into memory by the installation check.
MODEL_FILE_SIZES = {
    "biohub/ESMFold2": {"model.safetensors": 939505228, "ccd.pkl": 417306584},
    "biohub/ESMFold2-Experimental-Fast": {"model.safetensors": 717341848},
    "biohub/ESMC-6B": {
        "model-00001-of-00006.safetensors": 4864457920,
        "model-00002-of-00006.safetensors": 4971211344,
        "model-00003-of-00006.safetensors": 4863752992,
        "model-00004-of-00006.safetensors": 4971211344,
        "model-00005-of-00006.safetensors": 4863752992,
        "model-00006-of-00006.safetensors": 873762296,
    },
}


def validate_model_snapshot(repo_id: str, snapshot: str | Path) -> None:
    """Check required pinned files, lengths, JSON and safetensors headers offline."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    required = MODEL_FILE_SIZES[repo_id]
    config = json.loads((snapshot / "config.json").read_text())
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Invalid model config: {snapshot / 'config.json'}")
    for filename, expected_size in required.items():
        path = snapshot / filename
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Incomplete model file {path}: expected {expected_size} bytes, found {actual_size}"
            )
        if filename.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as weights:
                if not weights.keys():
                    raise ValueError(f"No tensors in {path}")
    if repo_id == "biohub/ESMC-6B":
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if not isinstance(json.loads((snapshot / filename).read_text()), dict):
                raise ValueError(f"Invalid tokenizer file: {snapshot / filename}")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        if not weight_map or set(weight_map.values()) != set(required):
            raise ValueError(f"ESMC weight index does not reference the six pinned shards: {snapshot}")
        for filename in required:
            with safe_open(snapshot / filename, framework="pt", device="cpu") as weights:
                expected_keys = {key for key, shard in weight_map.items() if shard == filename}
                if set(weights.keys()) != expected_keys:
                    raise ValueError(f"ESMC shard/index mismatch: {snapshot / filename}")


def resolve_model_source(name: str) -> str:
    """Use cached official revisions; retain explicit custom model locations."""
    if name not in MODEL_REVISIONS:
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=name, revision=MODEL_REVISIONS[name], local_files_only=True,
    )
