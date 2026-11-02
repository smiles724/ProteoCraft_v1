"""Reproducibility helpers for hallucination experiments."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
from typing import Iterator, Optional

import numpy as np
import torch


def stable_int_seed(base_seed: Optional[int], *parts: object) -> Optional[int]:
    """Derive a stable 31-bit seed from a base seed and identifying parts."""
    if base_seed is None:
        return None
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def set_global_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs for one process."""
    if seed is None:
        return
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


@contextlib.contextmanager
def temporary_seed(
    seed: Optional[int],
    device: Optional[torch.device] = None,
) -> Iterator[None]:
    """Temporarily seed RNGs, then restore their previous states."""
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except RuntimeError:
            cuda_states = None

    try:
        set_global_seed(int(seed), deterministic=False)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except RuntimeError:
                pass
