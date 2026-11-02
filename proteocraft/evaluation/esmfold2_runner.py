"""Current ProteoCraft ESMFold2 full-verifier adapter.

This module deliberately reuses the verifier implementation used by the
ProteoCraft search pipeline.  The evaluation pipeline only owns input parsing,
artifact persistence, and runtime bookkeeping.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


DEFAULT_MODEL_NAME = "biohub/ESMFold2"
DEFAULT_NUM_LOOPS = 20
DEFAULT_NUM_SAMPLING_STEPS = 200
DEFAULT_SEED = 43
DEFAULT_LOSS_MODE = "aligned_common"


@dataclass
class ESMFold2EvalOutput:
    verifier_result: Any
    runtime_s: float
    mmcif_path: str


def parse_target_sequences(value: str | Sequence[str]) -> list[str]:
    """Parse the manifest's colon-separated target chains for ESMFold2."""

    if isinstance(value, str):
        parts = re.split(r"[:|]", value.strip())
    else:
        parts = [str(item).strip() for item in value]
    sequences = [part.strip() for part in parts if part.strip()]
    if not sequences:
        raise ValueError("target_seq must contain at least one non-empty chain")
    return sequences


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def parse_target_chain_roles(value: Any, n_target_chains: int) -> tuple[str, ...]:
    """Parse optional target/scaffold roles; absent values mean all targets."""

    if _is_missing(value) or str(value).strip() == "":
        return ("target",) * int(n_target_chains)
    if isinstance(value, str):
        roles = tuple(
            part.strip().lower()
            for part in re.split(r"[,:+|;\s]+", value.strip())
            if part.strip()
        )
    else:
        roles = tuple(str(part).strip().lower() for part in value)
    if len(roles) != int(n_target_chains):
        raise ValueError(
            "target_chain_roles length must match target chains: "
            f"{len(roles)} != {n_target_chains}"
        )
    invalid = sorted(set(roles) - {"target", "scaffold"})
    if invalid:
        raise ValueError(f"invalid target_chain_roles: {invalid}")
    if "target" not in roles:
        raise ValueError("target_chain_roles must contain at least one target")
    return roles


def parse_hotspot_indices(value: Any) -> list[int]:
    """Parse optional zero-based indices over concatenated target chains."""

    if _is_missing(value) or str(value).strip() == "":
        return []
    raw: Iterable[Any]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            raw = json.loads(text)
        else:
            raw = [part for part in re.split(r"[,;|\s]+", text) if part]
    elif isinstance(value, (int, float)):
        raw = [value]
    else:
        raw = value
    indices = [int(item) for item in raw]
    if any(index < 0 for index in indices):
        raise ValueError("target_hotspot_concat_indices must be zero-based and non-negative")
    return list(dict.fromkeys(indices))


def build_backend(
    *,
    device: str,
    model_name: str = DEFAULT_MODEL_NAME,
    num_loops: int = DEFAULT_NUM_LOOPS,
    num_sampling_steps: int = DEFAULT_NUM_SAMPLING_STEPS,
    loss_mode: str = DEFAULT_LOSS_MODE,
):
    """Build the current ProteoCraft verifier without loading proposal models."""

    from proteocraft.hallucination.esmfold2_backend import ESMFold2ProposalConfig
    from proteocraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend

    config = ESMFold2ProposalConfig(
        verifier_model_name=str(model_name),
        verifier_num_loops=int(num_loops),
        verifier_num_sampling_steps=int(num_sampling_steps),
        verifier_loss_mode=str(loss_mode),
        # Evaluation must remain stock even if search defaults change to fast.
        proposal_optimization="off",
        verifier_optimization="off",
        esmc_optimization="off",
        verifier_structure_mode="sequence_only",
        hotspot_cutoff=8.0,
        hotspot_verify_weight=0.0,
        hotspot_verify_too_close_weight=0.0,
    )
    return ESMFold2RuntimeBackend(config=config, device=device, task_type="evaluation")


def evaluate(
    backend: Any,
    *,
    binder_seq: str,
    target_seq: str | Sequence[str],
    design_dir: Path,
    seed: int = DEFAULT_SEED,
    target_chain_roles: Any = None,
    hotspot_indices: Any = None,
) -> ESMFold2EvalOutput:
    """Run one full verification and persist its decoded mmCIF structure."""

    targets = parse_target_sequences(target_seq)
    roles = parse_target_chain_roles(target_chain_roles, len(targets))
    hotspots = parse_hotspot_indices(hotspot_indices)
    previous_config = backend.config
    backend.config = replace(previous_config, target_chain_roles=roles)
    try:
        started = time.perf_counter()
        result = backend.verify(
            binder_seq=str(binder_seq),
            target_seq=targets,
            seed=int(seed),
            hotspot_params=(
                {"target_hotspot_concat_indices": hotspots} if hotspots else None
            ),
        )
        runtime_s = time.perf_counter() - started
    finally:
        backend.config = previous_config

    mmcif = result.summary_confidence.get("mmcif")
    if not isinstance(mmcif, str) or not mmcif.strip():
        raise ValueError("ESMFold2 verifier did not return a decoded mmCIF structure")
    design_dir.mkdir(parents=True, exist_ok=True)
    mmcif_path = design_dir / "esmfold2.cif"
    mmcif_path.write_text(mmcif)
    return ESMFold2EvalOutput(
        verifier_result=result,
        runtime_s=float(runtime_s),
        mmcif_path=str(mmcif_path),
    )
