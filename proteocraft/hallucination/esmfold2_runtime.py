"""Biohub ESMFold2 model runtime for differentiable proposals and full verification."""

from __future__ import annotations

import logging
import math
import string
import inspect
import random
from contextlib import contextmanager
from functools import lru_cache
from types import MethodType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim as optim

from proteocraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESMFold2ProposalBatch,
    ESMFold2ProposalConfig,
    ESMFOLD2_ACTION_GROUP_PROPOSAL_LOSS_KEYS,
    ESM_AA_ORDER,
    anchor_binder_logits_to_sequence,
    build_logit_delta_candidate_batch,
    build_esm_verifier_result,
    build_gradient_mask,
    combine_normalized_gradients,
    esm_temperature,
    init_binder_logits,
    logit_delta_mutation_scores,
    normalized_gradient_tensor,
    resolve_lambda_esmc,
    sequence_from_logits_argmax,
)
from proteocraft.hallucination.esmfold2_one_step import (
    ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS,
    ESMFOLD2_VERIFIER_LOSS_WEIGHTS,
    capture_experimental_trunk_state,
    compute_esmfold2_bounded_verifier_losses,
    compute_esmfold2_common_losses,
    remap_reference_coordinates,
    run_esmfold2_one_step_diffusion,
    run_esmfold2_scaffold_consistency_diffusion,
    run_esmfold2_template_anchored_truncated_diffusion,
    run_esmfold2_truncated_diffusion,
)
from proteocraft.hallucination.esmfold2_template import build_template_clamp
from proteocraft.hallucination.esmfold2_interface import (
    interface_distogram_bins, native_interface_summary,
    select_coordinate_interface, select_distogram_interface,
)
from proteocraft.hallucination.esmfold2_optimizations import OptimizationManager, measured
from proteocraft.model_sources import resolve_model_source


logger = logging.getLogger(__name__)

CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits
ESMC_MASK_FRACTION = 0.15


@contextmanager
def _seed_context(seed: Optional[int]):
    """Temporarily set Python/Torch RNG state like the official ESMFold2 flow."""

    if seed is None:
        yield
        return
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _scheduled_one_step_soft_steps(
    *,
    soft_steps: int,
    interval: int,
) -> tuple[int, ...]:
    """Return one-based soft steps that receive a joint diffusion update."""

    if int(soft_steps) <= 0:
        raise ValueError("soft_steps must be positive")
    if int(interval) <= 0:
        return ()
    return tuple(
        step
        for step in range(1, int(soft_steps) + 1)
        if step % int(interval) == 0 or step == int(soft_steps)
    )


def _first_coordinate_sample(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim == 4:
        return coordinates[0, 0]
    if coordinates.ndim == 3:
        return coordinates[0]
    if coordinates.ndim == 2:
        return coordinates
    raise ValueError(
        "coordinates must have shape [A,3], [B,A,3], or [B,S,A,3]"
    )


def _fold_result_coordinates(fold_result: Any) -> Optional[torch.Tensor]:
    complex_obj = getattr(fold_result, "complex", None)
    atom_positions = getattr(complex_obj, "atom_positions", None)
    if atom_positions is None:
        return None
    coordinates = torch.as_tensor(atom_positions, dtype=torch.float32)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("ESMFold2 decoded atom positions must have shape [A, 3]")
    return coordinates.unsqueeze(0)


def _validate_decoded_coordinate_layout(
    coordinate: Optional[torch.Tensor],
    atom_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Ensure decoded coordinates follow the valid-atom order of raw features."""

    if coordinate is None:
        raise RuntimeError("ESMFold2 verifier did not return decoded coordinates")
    atom_mask = atom_attention_mask.bool()
    if atom_mask.ndim == 2:
        atom_mask = atom_mask[0]
    if atom_mask.ndim != 1:
        raise ValueError("atom_attention_mask must have shape [A] or [1,A]")
    expected_real_atoms = int(atom_mask.sum().item())
    if coordinate.ndim != 3 or coordinate.shape[0] != 1:
        raise ValueError("decoded coordinates must have shape [1,A_valid,3]")
    if coordinate.shape[1] != expected_real_atoms:
        raise RuntimeError(
            "ESMFold2 decoded coordinate layout is inconsistent with "
            f"the verifier atom mask: decoded={coordinate.shape[1]}, "
            f"expected_valid={expected_real_atoms}"
        )
    return coordinate


def _tensor_loss_dict_to_float(
    losses: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in losses.items():
        scalar = _as_float(value)
        if scalar is not None:
            result[str(key)] = scalar
    return result


def _callable_accepts_argument(func: Callable[..., Any], name: str) -> bool:
    """Return True when a callable can accept a keyword argument."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def _target_sequences(target_seq: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(target_seq, str):
        text = target_seq.strip()
        if "|" in text:
            return [part for part in text.split("|") if part]
        return [text]
    return [str(seq).strip() for seq in target_seq if str(seq).strip()]


def _normalize_target_chain_roles(
    target_chain_roles: Optional[Sequence[str]],
    *,
    n_target_chains: int,
) -> tuple[str, ...]:
    if n_target_chains <= 0:
        raise ValueError("ESMFold2 requires at least one target chain")
    if target_chain_roles is None:
        return ("target",) * int(n_target_chains)
    roles = tuple(str(role).strip().lower() for role in target_chain_roles)
    if len(roles) != int(n_target_chains):
        raise ValueError(
            "target_chain_roles length must match the number of non-binder chains: "
            f"{len(roles)} != {n_target_chains}"
        )
    invalid_roles = sorted(set(roles) - {"target", "scaffold"})
    if invalid_roles:
        raise ValueError(
            "target_chain_roles only supports target/scaffold; "
            f"got {invalid_roles}"
        )
    if "target" not in roles:
        raise ValueError("target_chain_roles must contain at least one target")
    return roles


def _mean_target_to_binder_pair_iptm(
    pair_chains_iptm: Any,
    *,
    n_target_chains: int,
    target_chain_roles: Optional[Sequence[str]] = None,
    selected_role: str = "target",
) -> Optional[float]:
    if pair_chains_iptm is None:
        return None
    roles = _normalize_target_chain_roles(
        target_chain_roles,
        n_target_chains=n_target_chains,
    )
    selected_indices = [
        index for index, role in enumerate(roles) if role == str(selected_role)
    ]
    if not selected_indices:
        return None
    tensor = torch.as_tensor(pair_chains_iptm).detach().float()
    if tensor.ndim == 3:
        if tensor.shape[1] <= n_target_chains or tensor.shape[2] <= n_target_chains:
            return None
        binder_idx = n_target_chains
        values = tensor[:, selected_indices, binder_idx]
        if values.numel() == 0:
            return None
        return _as_float(values)
    if tensor.ndim != 2 or tensor.shape[0] <= n_target_chains or tensor.shape[1] <= n_target_chains:
        return None
    binder_idx = n_target_chains
    values = tensor[selected_indices, binder_idx]
    if values.numel() == 0:
        return None
    return _as_float(values)


def _midpoint_bins(
    device: torch.device,
    *,
    num_bins: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if int(num_bins) < 2:
        raise ValueError("num_bins must be at least 2")
    boundaries = torch.linspace(
        2.0,
        52.0,
        int(num_bins) - 1,
        device=device,
        dtype=dtype,
    )
    lower = torch.tensor([1.0], device=device, dtype=dtype)
    upper = torch.tensor([57.0], device=device, dtype=dtype)
    expanded = torch.cat((lower, boundaries, upper))
    return (expanded[:-1] + expanded[1:]) / 2.0


def _binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1.0e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def _contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    *,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    con_loss = _binned_entropy(distogram_logits, bin_distance, cutoff)
    positions = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    p_dist = positions[:, None] - positions[None, :]
    if min_sep > 0:
        separation_mask = torch.abs(p_dist) >= min_sep
        binder_mask = torch.logical_and(separation_mask, binder_mask.bool())
    per_residue = _masked_min_k(con_loss, mask=binder_mask, k=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _normalize_target_hotspot_indices(
    hotspot_params: Optional[Dict[str, Any]],
    *,
    target_length: int,
) -> list[int]:
    """Validate target-relative concatenated hotspot indices."""

    raw_indices: Any = (hotspot_params or {}).get("target_hotspot_concat_indices")
    if raw_indices is None:
        return []
    if isinstance(raw_indices, torch.Tensor):
        values = raw_indices.detach().cpu().flatten().tolist()
    elif isinstance(raw_indices, (int, str)):
        values = [raw_indices]
    else:
        values = list(raw_indices)

    indices: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError("hotspot indices must be integers, not booleans")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid hotspot index: {value!r}") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"invalid non-integer hotspot index: {value!r}")
        if index < 0 or index >= int(target_length):
            raise ValueError(
                f"hotspot target concat index {index} is out of range for target_length={target_length}"
            )
        if index not in seen:
            seen.add(index)
            indices.append(index)
    return indices


def _binder_structure_metrics(
    fold_result: Any,
    *,
    binder_length: int,
) -> dict[str, float]:
    """Extract binder-local confidence and compactness from a full prediction."""

    binder_length = int(binder_length)
    if binder_length <= 0:
        raise ValueError("binder_length must be positive")

    metrics: dict[str, float] = {}
    plddt = getattr(fold_result, "plddt", None)
    if plddt is not None:
        values = torch.as_tensor(plddt).detach().float().reshape(-1)
        if values.numel() >= binder_length:
            binder_values = values[-binder_length:]
            binder_values = binder_values[torch.isfinite(binder_values)]
            if binder_values.numel() == binder_length:
                metrics["binder_plddt"] = float(binder_values.mean().item())

    complex_obj = getattr(fold_result, "complex", None)
    residue_centers: list[torch.Tensor] = []
    if complex_obj is not None and all(
        hasattr(complex_obj, name)
        for name in ("sequence", "atom_positions", "atom_elements", "token_to_atoms")
    ):
        token_count = len(complex_obj.sequence)
        if token_count >= binder_length:
            positions = torch.as_tensor(complex_obj.atom_positions, dtype=torch.float32)
            token_to_atoms = torch.as_tensor(complex_obj.token_to_atoms, dtype=torch.long)
            elements = list(complex_obj.atom_elements)
            for token_index in range(token_count - binder_length, token_count):
                atom_start, atom_end = map(int, token_to_atoms[token_index].tolist())
                heavy_mask = []
                for element in elements[atom_start:atom_end]:
                    if isinstance(element, bytes):
                        element = element.decode(errors="ignore")
                    heavy_mask.append(str(element).strip().upper() != "H")
                token_positions = positions[atom_start:atom_end]
                if heavy_mask:
                    token_positions = token_positions[
                        torch.as_tensor(heavy_mask, dtype=torch.bool)
                    ]
                if token_positions.numel() > 0:
                    residue_centers.append(token_positions.mean(dim=0))
    elif complex_obj is not None and all(
        hasattr(complex_obj, name) for name in ("atom37_positions", "atom37_mask")
    ):
        positions = torch.as_tensor(complex_obj.atom37_positions, dtype=torch.float32)
        atom_mask = torch.as_tensor(complex_obj.atom37_mask, dtype=torch.bool)
        if positions.shape[0] >= binder_length:
            for residue_positions, residue_mask in zip(
                positions[-binder_length:], atom_mask[-binder_length:]
            ):
                if residue_mask.numel() > 1 and bool(residue_mask[1].item()):
                    residue_centers.append(residue_positions[1])
                elif residue_mask.any():
                    residue_centers.append(residue_positions[residue_mask].mean(dim=0))

    if len(residue_centers) == binder_length:
        centers = torch.stack(residue_centers, dim=0)
        centroid = centers.mean(dim=0)
        rg = torch.sqrt(torch.square(centers - centroid).sum(dim=-1).mean())
        rg_threshold = 2.38 * (binder_length**0.365)
        rg_excess = max(float(rg.item()) / max(float(rg_threshold), 1.0e-8) - 1.0, 0.0)
        metrics.update(
            {
                "binder_radius_gyration": float(rg.item()),
                "binder_radius_gyration_threshold": float(rg_threshold),
                "binder_rg_excess": float(rg_excess),
            }
        )
    return metrics


def _hotspot_structure_metrics(
    complex_obj: Any,
    *,
    target_sequences: Sequence[str],
    hotspot_params: Optional[Dict[str, Any]],
    near_cutoff: float,
    far_cutoff: float,
    min_safe_distance: float = 0.0,
    binder_contact_mask: Optional[Sequence[bool]] = None,
) -> dict[str, Any]:
    """Measure hotspot distances to all binder residues or an explicit subset."""

    if float(near_cutoff) < 0.0:
        raise ValueError("hotspot verifier near cutoff must be non-negative")
    if float(far_cutoff) <= float(near_cutoff):
        raise ValueError("hotspot verifier far cutoff must exceed near cutoff")
    if float(min_safe_distance) < 0.0:
        raise ValueError(
            "hotspot verifier minimum safe distance must be non-negative"
        )
    if float(min_safe_distance) > float(near_cutoff):
        raise ValueError(
            "hotspot verifier minimum safe distance must not exceed near cutoff"
        )
    n_target_chains = len(target_sequences)
    expected_target_lengths = [len(sequence) for sequence in target_sequences]
    expected_target_length = sum(expected_target_lengths)
    hotspot_indices = _normalize_target_hotspot_indices(
        hotspot_params,
        target_length=expected_target_length,
    )
    if not hotspot_indices:
        raise ValueError(
            "hotspot-aware verifier requires target_hotspot_concat_indices"
        )

    def selected_binder_indices(start: int, end: int) -> list[int]:
        if binder_contact_mask is None:
            return list(range(start, end))
        mask = torch.as_tensor(binder_contact_mask)
        if mask.ndim != 1 or mask.numel() != end - start:
            raise ValueError("binder_contact_mask length must match the decoded binder chain")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("binder_contact_mask must contain only boolean or 0/1 values")
        indices = [start + i for i, selected in enumerate(mask.tolist()) if selected]
        if not indices:
            raise ValueError("binder_contact_mask must select at least one residue")
        return indices

    molecular_fields = (
        "atom_positions",
        "atom_elements",
        "token_to_atoms",
        "chain_id",
        "sequence",
    )
    protein_fields = ("chain_boundaries", "atom37_positions", "atom37_mask")
    if all(hasattr(complex_obj, name) for name in molecular_fields):
        token_count = len(complex_obj.sequence)
        chain_ids = list(complex_obj.chain_id)
        if len(chain_ids) != token_count:
            raise ValueError("ESMFold2 MolecularComplex chain_id length mismatch")
        boundaries: list[tuple[int, int]] = []
        if chain_ids:
            chain_start = 0
            for token_index in range(1, token_count):
                if chain_ids[token_index] != chain_ids[token_index - 1]:
                    boundaries.append((chain_start, token_index))
                    chain_start = token_index
            boundaries.append((chain_start, token_count))
        if len(boundaries) <= n_target_chains:
            raise ValueError(
                "ESMFold2 MolecularComplex does not contain the expected binder chain"
            )
        observed_target_lengths = [
            boundaries[index][1] - boundaries[index][0]
            for index in range(n_target_chains)
        ]
        if observed_target_lengths != expected_target_lengths:
            raise ValueError(
                "ESMFold2 verifier target chain lengths do not match input: "
                f"expected={expected_target_lengths}, observed={observed_target_lengths}"
            )

        positions = torch.as_tensor(complex_obj.atom_positions, dtype=torch.float32)
        token_to_atoms = torch.as_tensor(complex_obj.token_to_atoms, dtype=torch.long)
        elements = list(complex_obj.atom_elements)

        def token_heavy_atoms(token_index: int) -> torch.Tensor:
            atom_start, atom_end = map(int, token_to_atoms[token_index].tolist())
            heavy_mask = []
            for element in elements[atom_start:atom_end]:
                if isinstance(element, bytes):
                    element = element.decode(errors="ignore")
                heavy_mask.append(str(element).strip().upper() != "H")
            return positions[atom_start:atom_end][
                torch.as_tensor(heavy_mask, dtype=torch.bool)
            ]

        target_residue_indices = [
            residue_index
            for chain_index in range(n_target_chains)
            for residue_index in range(*boundaries[chain_index])
        ]
        binder_start, binder_end = boundaries[n_target_chains]
        contact_indices = selected_binder_indices(binder_start, binder_end)
        binder_atom_groups = [
            token_heavy_atoms(token_index)
            for token_index in contact_indices
        ]
        nonempty_binder_atoms = [
            atoms for atoms in binder_atom_groups if atoms.numel() > 0
        ]
        binder_atoms = (
            torch.cat(nonempty_binder_atoms, dim=0)
            if nonempty_binder_atoms
            else torch.zeros((0, 3), dtype=torch.float32)
        )

        def hotspot_atoms(hotspot_index: int) -> torch.Tensor:
            return token_heavy_atoms(target_residue_indices[hotspot_index])

    elif all(hasattr(complex_obj, name) for name in protein_fields):
        boundaries = [tuple(map(int, bounds)) for bounds in complex_obj.chain_boundaries]
        if len(boundaries) <= n_target_chains:
            raise ValueError(
                "ESMFold2 ProteinComplex does not contain the expected binder chain"
            )
        observed_target_lengths = [
            boundaries[index][1] - boundaries[index][0]
            for index in range(n_target_chains)
        ]
        if observed_target_lengths != expected_target_lengths:
            raise ValueError(
                "ESMFold2 verifier target chain lengths do not match input: "
                f"expected={expected_target_lengths}, observed={observed_target_lengths}"
            )
        target_residue_indices = [
            residue_index
            for chain_index in range(n_target_chains)
            for residue_index in range(*boundaries[chain_index])
        ]
        positions = torch.as_tensor(complex_obj.atom37_positions, dtype=torch.float32)
        atom_mask = torch.as_tensor(complex_obj.atom37_mask, dtype=torch.bool)
        binder_start, binder_end = boundaries[n_target_chains]
        contact_indices = selected_binder_indices(binder_start, binder_end)
        binder_positions = positions[contact_indices]
        binder_mask = atom_mask[contact_indices]
        binder_atoms = binder_positions[binder_mask]

        def hotspot_atoms(hotspot_index: int) -> torch.Tensor:
            residue_index = target_residue_indices[hotspot_index]
            return positions[residue_index][atom_mask[residue_index]]

    else:
        raise ValueError(
            "ESMFold2 verifier structure is neither a supported MolecularComplex "
            "nor ProteinComplex"
        )

    if binder_atoms.numel() == 0:
        raise ValueError("ESMFold2 verifier binder chain has no predicted atoms")

    minimum_distances: list[float] = []
    for hotspot_index in hotspot_indices:
        residue_atoms = hotspot_atoms(hotspot_index)
        if residue_atoms.numel() == 0:
            raise ValueError(
                f"ESMFold2 verifier hotspot residue {hotspot_index} has no predicted atoms"
            )
        minimum_distances.append(
            float(torch.cdist(residue_atoms, binder_atoms).min().item())
        )

    distance_tensor = torch.tensor(minimum_distances, dtype=torch.float32)
    distance_loss = torch.clamp(
        (distance_tensor - float(near_cutoff))
        / (float(far_cutoff) - float(near_cutoff)),
        min=0.0,
        max=1.0,
    ).mean()
    too_close_loss = torch.clamp(
        (float(min_safe_distance) - distance_tensor)
        / max(float(min_safe_distance), 1.0e-8),
        min=0.0,
        max=1.0,
    ).max()
    covered_count = int((distance_tensor <= float(near_cutoff)).sum().item())
    strict_covered_count = int((distance_tensor <= 5.0).sum().item())
    hotspot_count = len(hotspot_indices)
    return {
        "hotspot_verify_distance_loss": float(distance_loss.item()),
        "hotspot_too_close_loss": float(too_close_loss.item()),
        "hotspot_mean_min_heavy_atom_distance": float(distance_tensor.mean().item()),
        "hotspot_heavy_atom_coverage": covered_count / hotspot_count,
        "hotspot_heavy_atom_covered_count": covered_count,
        "hotspot_heavy_atom_coverage_5a": strict_covered_count / hotspot_count,
        "hotspot_heavy_atom_covered_count_5a": strict_covered_count,
        "hotspot_count": hotspot_count,
        "hotspot_binder_contact_residue_count": len(contact_indices),
        "hotspot_binder_contact_positions_1based": [i - binder_start + 1 for i in contact_indices],
        "hotspot_verify_near_cutoff": float(near_cutoff),
        "hotspot_verify_far_cutoff": float(far_cutoff),
        "hotspot_verify_min_distance": float(min_safe_distance),
        "hotspot_min_heavy_atom_distances": minimum_distances,
    }


def _structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    *,
    target_chain_lengths: Optional[Sequence[int]] = None,
    target_chain_roles: Optional[Sequence[str]] = None,
    target_interface_mode: str = "all",
    target_interface_contact_cutoff: float = 5.0,
    target_interface_min_contacts: int = 5,
    binder_mutable_mask: Optional[Sequence[bool]] = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    hotspot_weight: float = 0.0,
    hotspot_cutoff: float = 8.0,
    hotspot_interface_mode: str = "additive",
) -> dict[str, Any]:
    if target_interface_mode == "contacted":
        bin_distance = interface_distogram_bins(
            distogram_logits.shape[-1], device=distogram_logits.device,
            dtype=distogram_logits.dtype,
        )
    else:
        bin_distance = _midpoint_bins(
            distogram_logits.device,
            num_bins=distogram_logits.shape[-1],
            dtype=distogram_logits.dtype,
        )
    full_len = distogram_logits.shape[1]
    target_length = int(full_len) - int(binder_length)
    if target_length <= 0:
        raise ValueError(
            f"distogram length {full_len} must exceed binder_length {binder_length}"
        )
    if float(hotspot_weight) < 0.0:
        raise ValueError("hotspot_weight must be non-negative")
    if float(hotspot_cutoff) <= 0.0:
        raise ValueError("hotspot_cutoff must be positive")
    hotspot_interface_mode = str(hotspot_interface_mode).strip().lower()
    if hotspot_interface_mode not in {"additive", "localized"}:
        raise ValueError(
            "hotspot_interface_mode must be 'additive' or 'localized'"
        )
    if hotspot_interface_mode == "localized" and float(hotspot_weight) > 1.0:
        raise ValueError(
            "localized hotspot_interface_mode requires hotspot_weight in [0, 1]"
        )
    if target_chain_lengths is None:
        target_chain_lengths = (target_length,)
    chain_lengths = tuple(int(length) for length in target_chain_lengths)
    if any(length <= 0 for length in chain_lengths):
        raise ValueError("target_chain_lengths must contain positive lengths")
    if sum(chain_lengths) != target_length:
        raise ValueError(
            "target_chain_lengths must sum to the non-binder residue count: "
            f"{sum(chain_lengths)} != {target_length}"
        )
    roles = _normalize_target_chain_roles(
        target_chain_roles,
        n_target_chains=len(chain_lengths),
    )

    is_binder = torch.zeros(full_len, device=distogram_logits.device)
    is_binder[-binder_length:] = 1.0
    is_design = is_binder.clone()
    if binder_mutable_mask is not None:
        mutable = torch.as_tensor(
            binder_mutable_mask,
            dtype=torch.bool,
            device=distogram_logits.device,
        ).reshape(-1)
        if mutable.numel() != int(binder_length):
            raise ValueError(
                "binder_mutable_mask length must match binder_length: "
                f"{mutable.numel()} != {binder_length}"
            )
        if not bool(mutable.any().item()):
            raise ValueError("binder_mutable_mask must select at least one residue")
        is_design[-binder_length:] = mutable.to(dtype=is_design.dtype)

    is_true_target = torch.zeros(full_len, device=distogram_logits.device)
    target_masks = []
    offset = 0
    for chain_length, role in zip(chain_lengths, roles):
        if role == "target":
            is_true_target[offset : offset + chain_length] = 1.0
            mask = torch.zeros(full_len, device=distogram_logits.device, dtype=torch.bool)
            mask[offset : offset + chain_length] = True
            target_masks.append(mask)
        offset += chain_length
    interface_target = is_true_target
    selection = None
    if target_interface_mode == "contacted":
        selection = select_distogram_interface(
            distogram_logits, chain_masks=target_masks, binder_mask=is_binder.bool(),
            cutoff=target_interface_contact_cutoff, min_contacts=target_interface_min_contacts,
        )
        interface_target = torch.zeros_like(is_true_target)
        for index in selection["selected_indices"]:
            interface_target[target_masks[index]] = 1.0
    elif target_interface_mode != "all":
        raise ValueError("target_interface_mode must be all or contacted")

    compact_mask = is_binder

    intra = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=compact_mask,
        binder_mask=compact_mask,
    )
    inter = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=interface_target,
        binder_mask=is_design,
    )

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    capped_bins = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(capped_bins), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    glob = F.elu(rg_term - rg_th)

    base_total = 0.5 * intra + 0.5 * inter + 0.2 * glob
    losses = {
        "intra_contact_loss": intra,
        "inter_contact_loss": inter,
        "glob_loss": glob,
        "total_loss": base_total,
    }
    if selection is not None:
        losses["_interface_selection"] = selection
        losses["interface_selected_chain_count"] = inter.new_tensor(len(selection["selected_indices"]))
        losses["interface_selection_fallback"] = inter.new_tensor(float(selection["fallback"]))
    hotspot_indices = _normalize_target_hotspot_indices(
        hotspot_params,
        target_length=target_length,
    )
    if float(hotspot_weight) > 0.0 and not hotspot_indices:
        raise ValueError(
            "positive hotspot_weight requires target_hotspot_concat_indices"
        )
    if hotspot_indices:
        invalid_hotspots = [
            index
            for index in hotspot_indices
            if not bool(is_true_target[index].item())
        ]
        if invalid_hotspots:
            raise ValueError(
                "hotspot indices must belong to chains with role=target; "
                f"invalid indices={invalid_hotspots}"
            )
        hotspot_mask = torch.zeros(full_len, device=distogram_logits.device)
        hotspot_mask[torch.as_tensor(hotspot_indices, device=distogram_logits.device)] = 1.0
        hotspot_contact = _contact_loss(
            distogram_logits,
            bin_distance,
            num_contacts=1,
            min_sep=0,
            cutoff=float(hotspot_cutoff),
            chain_mask=hotspot_mask,
            binder_mask=is_design,
        )
        weighted_hotspot = float(hotspot_weight) * hotspot_contact
        if hotspot_interface_mode == "localized":
            effective_inter = (
                (1.0 - float(hotspot_weight)) * inter
                + float(hotspot_weight) * hotspot_contact
            )
            total_loss = 0.5 * intra + 0.5 * effective_inter + 0.2 * glob
        else:
            effective_inter = inter
            total_loss = base_total + weighted_hotspot
        losses.update(
            {
                "base_structure_loss": base_total,
                "hotspot_contact_loss": hotspot_contact,
                "hotspot_weighted_contribution": weighted_hotspot,
                "hotspot_objective_adjustment": total_loss - base_total,
                "effective_inter_contact_loss": effective_inter,
                "total_loss": total_loss,
            }
        )
    return losses


def _stageb_structure_loss(
    losses: dict[str, torch.Tensor],
    action_group: Optional[str],
    *,
    group_weight: float = 1.0,
    hotspot_interface_mode: str = "additive",
) -> tuple[torch.Tensor, str]:
    base_loss = losses["total_loss"]
    if action_group is None:
        return base_loss, "total_loss"

    weights = ESMFOLD2_ACTION_GROUP_PROPOSAL_LOSS_KEYS.get(str(action_group), {})
    if not weights:
        return base_loss, "total_loss"
    selected = []
    proposal_loss = None
    for key, weight in weights.items():
        loss_key = key
        if (
            key == "inter_contact_loss"
            and str(hotspot_interface_mode).strip().lower() == "localized"
            and "effective_inter_contact_loss" in losses
        ):
            loss_key = "effective_inter_contact_loss"
        if loss_key not in losses:
            continue
        term = float(weight) * losses[loss_key]
        proposal_loss = term if proposal_loss is None else proposal_loss + term
        selected.append(loss_key)
    if proposal_loss is None:
        raise ValueError(f"No proposal loss is available for action group {action_group!r}")

    proposal_loss = float(group_weight) * proposal_loss
    hotspot_is_already_in_interface = (
        str(action_group) == "interface_formation"
        and str(hotspot_interface_mode).strip().lower() == "localized"
    )
    if not hotspot_is_already_in_interface and "hotspot_weighted_contribution" in losses:
        proposal_loss = proposal_loss + losses["hotspot_weighted_contribution"]
        selected.append("hotspot_weighted_contribution")
    return proposal_loss, "+".join(selected)


class ESMFold2RuntimeBackend:
    """ESMFold2 verifier + short proposal-gradient adapter."""

    def _prepare_memory_phase(self, phase, targets, binder_length):
        self.optimizations.memory.prepare(
            phase, sum(map(len, targets)) + binder_length,
            {"verifier": self.verifier_model, "proposal": self.experimental_model,
             "prior": self.esmc_model},
        )

    def _record_memory_ready(self):
        self.optimizations.memory.ready()
        if self.config.memory_mode != "resident":
            self.optimizations.write_report()

    def _interface_options(self) -> dict[str, Any]:
        return {
            "target_interface_mode": self.config.target_interface_mode,
            "target_interface_contact_cutoff": self.config.target_interface_contact_cutoff,
            "target_interface_min_contacts": self.config.target_interface_min_contacts,
        }

    def _validate_target_assembly(self, targets: Sequence[str]) -> None:
        if self.config.target_assembly == "homotrimer":
            if len(targets) != 3 or len(set(targets)) != 1:
                raise ValueError("homotrimer requires exactly three identical target sequences")
            if any(not sequence or set(sequence) - set(ESM_AA_ORDER) for sequence in targets):
                raise ValueError("homotrimer requires nonempty canonical amino-acid sequences")

    def _contacted_summary_from_raw(self, output: Mapping[str, Any],
            features: Mapping[str, torch.Tensor], targets: Sequence[str],
            binder_length: int) -> dict[str, Any]:
        """Use raw feature identities, never cumulated decoded atom positions."""
        self._validate_target_assembly(targets)
        def vector(name):
            value = features[name].detach()
            if value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 1:
                raise ValueError(f"Contacted interface expects a single feature vector: {name}")
            return value
        valid = vector("token_attention_mask").bool()
        asym = vector("asym_id").long()
        if valid.shape != asym.shape:
            raise ValueError("Contacted interface token/asym layout mismatch")
        # Native chain confidence axes are model asym_id, not arbitrary labels.
        ids = torch.unique_consecutive(asym[valid]).tolist()
        expected_ids = list(range(len(targets) + 1))
        if ids != expected_ids:
            raise ValueError(f"Contacted interface requires contiguous native chain axes {expected_ids}, got {ids}")
        masks = [valid & (asym == i) for i in expected_ids]
        observed = [int(mask.sum()) for mask in masks]
        expected = [len(sequence) for sequence in targets] + [binder_length]
        if observed != expected:
            raise ValueError(f"Contacted interface chain token lengths mismatch: {observed} != {expected}")
        atom_mask = vector("atom_attention_mask").bool().clone()
        # Canonical ESMFold2 features contain heavy atoms. Apply atomic number
        # when supplied by the input builder so future hydrogen padding is safe.
        if "ref_element" in features:
            atom_mask &= vector("ref_element") > 1
        selection = select_coordinate_interface(
            output["sample_atom_coords"], atom_to_token=vector("atom_to_token"),
            atom_mask=atom_mask, chain_masks=masks[:-1], binder_mask=masks[-1],
            cutoff=self.config.target_interface_contact_cutoff,
            min_contacts=self.config.target_interface_min_contacts,
        )
        summary = native_interface_summary(output.get("pair_chains_iptm"), selection,
                                           n_target_chains=len(targets))
        summary["pair_chain_iptm_axis_order"] = list(CHAIN_IDS[:len(targets) + 1])
        summary["target_interface_contact_cutoff"] = self.config.target_interface_contact_cutoff
        summary["target_interface_min_contacts"] = self.config.target_interface_min_contacts
        summary["target_interface_chain_token_lengths"] = observed
        return summary

    def __init__(
        self,
        *,
        config: ESMFold2ProposalConfig,
        device: Union[str, torch.device] = "cuda",
        verifier_model: Any = None,
        experimental_model: Any = None,
        esmc_model: Any = None,
        input_builder: Any = None,
        protein_input_cls: Any = None,
        structure_prediction_input_cls: Any = None,
        soft_update_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
        task_type: str = "denovo",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.verifier_model = verifier_model
        self.experimental_model = experimental_model
        self.esmc_model = esmc_model
        self.input_builder = input_builder
        self.protein_input_cls = protein_input_cls
        self.structure_prediction_input_cls = structure_prediction_input_cls
        self.soft_update_fn = soft_update_fn
        self.task_type = task_type
        self.verifier_num_loops = int(config.verifier_num_loops)
        self.verifier_num_sampling_steps = int(config.verifier_num_sampling_steps)
        self.optimizations = OptimizationManager(config, self.device)

    def _fold_verifier_with_raw_output(
        self,
        input_spec: Any,
        *,
        seed: Optional[int],
    ) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Mirror the official input-builder fold while retaining raw tensors."""

        from esm.models.esmfold2.processor import _lm_dropout_context

        features, chain_infos = self.input_builder.prepare_input(
            input_spec,
            seed=seed,
            device=self.verifier_model.device,
        )
        with torch.no_grad(), _seed_context(seed), _lm_dropout_context(
            self.verifier_model,
            0.3,
        ):
            output = self.verifier_model(
                **features,
                num_loops=self.verifier_num_loops,
                num_sampling_steps=self.verifier_num_sampling_steps,
                num_diffusion_samples=1,
                early_exit=False,
                msa_max_depth=1024,
                msa_column_mask_rate=0.1,
                msa_subsample_at_inference=True,
            )
        fold_result = self.input_builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=1,
            complex_id="esmfold2_verify",
        )
        return fold_result, features, output

    def prepare_verifier_reference_features(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Build the atom identity layout paired with verifier coordinates."""

        targets = _target_sequences(target_seq)
        self._prepare_memory_phase("verifier", targets, len(binder_seq))
        self._ensure_verifier_loaded()
        input_spec = self._build_structure_input(targets, binder_seq)
        features, _ = self.input_builder.prepare_input(
            input_spec,
            seed=seed,
            device=self.verifier_model.device,
        )
        required = (
            "atom_attention_mask",
            "atom_to_token",
            "ref_atom_name_chars",
            "ref_element",
            "ref_pos",
        )
        missing = [name for name in required if name not in features]
        if missing:
            raise ValueError(
                "ESMFold2 verifier reference features are missing: "
                + ", ".join(missing)
            )
        return {
            name: features[name].detach().clone()
            for name in required
        }

    @contextmanager
    def _temporary_scaffold_consistency_sampler(
        self,
        *,
        template_coordinates: torch.Tensor,
        fixed_atom_mask: torch.Tensor,
    ):
        """Temporarily replace only the standard verifier sampling call.

        ``ESMFold2Model.forward`` owns trunk and confidence evaluation.  This
        narrow override changes only its diffusion sampler for the duration of
        one explicit scaffold-consistency verification and is restored even if
        the model raises.
        """

        structure_head = self.verifier_model.structure_head
        original_sample = structure_head.sample
        sample_trace: dict[str, torch.Tensor] = {}

        def scaffold_sample(
            _head: Any,
            **kwargs: Any,
        ) -> dict[str, torch.Tensor | None]:
            unsupported = {
                "return_atom_repr",
                "denoising_early_exit_rmsd",
                "use_inference_cache",
                "noise_scale",
                "step_scale",
            }
            sample_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in unsupported
            }
            result = run_esmfold2_scaffold_consistency_diffusion(
                structure_head=_head,
                template_coordinates=template_coordinates,
                fixed_atom_mask=fixed_atom_mask,
                **sample_kwargs,
            )
            sample_trace["template_coordinates"] = result[
                "scaffold_template_coordinates"
            ].detach()
            sample_trace["fixed_atom_mask"] = result["fixed_atom_mask"].detach()
            return result

        structure_head.sample = MethodType(scaffold_sample, structure_head)
        try:
            yield sample_trace
        finally:
            structure_head.sample = original_sample

    def _fold_scaffold_consistency_verifier_with_raw_output(
        self,
        input_spec: Any,
        *,
        seed: Optional[int],
    ) -> tuple[
        Any,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        Mapping[str, Any],
    ]:
        """Run an explicit full verifier with a template-consistent scaffold."""

        from esm.models.esmfold2.processor import _lm_dropout_context

        features, chain_infos = self.input_builder.prepare_input(
            input_spec,
            seed=seed,
            device=self.verifier_model.device,
        )
        template_clamp = build_template_clamp(
            features=features,
            template_path=str(self.config.template_structure_path),
            template_chain_ids=self.config.template_chain_ids or (),
            template_chain_sequences=self.config.template_chain_sequences or (),
            binder_structure_regeneration_mask=(
                self.config.binder_structure_regeneration_mask or ()
            ),
        )
        with torch.no_grad(), _seed_context(seed), _lm_dropout_context(
            self.verifier_model,
            0.3,
        ), self._temporary_scaffold_consistency_sampler(
            template_coordinates=template_clamp.coordinates,
            fixed_atom_mask=template_clamp.fixed_atom_mask,
        ) as sample_trace:
            output = self.verifier_model(
                **features,
                num_loops=self.verifier_num_loops,
                num_sampling_steps=self.verifier_num_sampling_steps,
                num_diffusion_samples=1,
                early_exit=False,
                msa_max_depth=1024,
                msa_column_mask_rate=0.1,
                msa_subsample_at_inference=True,
            )
        if not sample_trace:
            raise RuntimeError("scaffold-consistency sampler did not run")
        predicted = _first_coordinate_sample(output["sample_atom_coords"])
        template_final = _first_coordinate_sample(
            sample_trace["template_coordinates"]
        ).to(device=predicted.device, dtype=predicted.dtype)
        fixed_mask = sample_trace["fixed_atom_mask"]
        if fixed_mask.ndim == 2:
            fixed_mask = fixed_mask[0]
        fixed_mask = fixed_mask.to(device=predicted.device, dtype=torch.bool)
        fixed_delta = predicted[fixed_mask] - template_final[fixed_mask]
        template_report = dict(template_clamp.report)
        template_report["final_fixed_atom_rmsd"] = float(
            fixed_delta.pow(2).sum(dim=-1).mean().sqrt().item()
        )
        fold_result = self.input_builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=1,
            complex_id="esmfold2_scaffold_consistency",
        )
        return fold_result, features, output, template_report

    def _build_verifier_result_from_fold(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        targets: Sequence[str],
        target_chain_roles: Sequence[str],
        hotspot_params: Optional[Dict[str, Any]],
        fold_result: Any,
        raw_features: Optional[dict[str, torch.Tensor]] = None,
        raw_output: Optional[dict[str, torch.Tensor]] = None,
        structure_mode: str = "sequence_only",
        template_report: Optional[Mapping[str, Any]] = None,
    ):
        """Convert one decoded full-fold result into the stable verifier contract."""

        summary = self._summary_from_fold_result(
            fold_result,
            n_target_chains=len(targets),
            target_chain_roles=target_chain_roles,
        )
        if self.config.target_interface_mode == "contacted":
            if raw_features is None or raw_output is None:
                raise ValueError("Contacted interface verification requires raw model features and output")
            summary["pair_chain_iptm_legacy_all_target_mean"] = summary.get("pair_chain_iptm")
            summary.update(self._contacted_summary_from_raw(raw_output, raw_features, targets, len(binder_seq)))
        summary.update(
            _binder_structure_metrics(
                fold_result,
                binder_length=len(binder_seq),
            )
        )
        if template_report is not None:
            summary["verifier_structure_mode"] = str(structure_mode)
            summary["scaffold_consistency_template_report"] = dict(template_report)
            summary["scaffold_consistency_fixed_atom_count"] = int(
                template_report.get("fixed_atom_count", 0)
            )
            if "final_fixed_atom_rmsd" in template_report:
                summary["scaffold_consistency_final_fixed_atom_rmsd"] = float(
                    template_report["final_fixed_atom_rmsd"]
                )
            for metric_name in (
                "pair_chain_iptm",
                "iptm",
                "ptm",
                "plddt",
                "binder_plddt",
            ):
                if metric_name in summary:
                    summary[f"scaffold_consistency_{metric_name}"] = summary[
                        metric_name
                    ]

        hotspot_verify_weight = float(self.config.hotspot_verify_weight)
        hotspot_too_close_weight = float(
            self.config.hotspot_verify_too_close_weight
        )
        hotspot_metrics: dict[str, Any] = {}
        hotspot_indices = _normalize_target_hotspot_indices(
            hotspot_params,
            target_length=sum(len(sequence) for sequence in targets),
        )
        if (
            hotspot_indices
            or hotspot_verify_weight > 0.0
            or hotspot_too_close_weight > 0.0
        ):
            binder_contact_mask = None
            if self.task_type == "vhh_redesign":
                binder_contact_mask = self.config.binder_mutable_mask
                if binder_contact_mask is None:
                    raise ValueError("VHH hotspot verification requires the configured CDR mutable mask")
            hotspot_metrics = _hotspot_structure_metrics(
                fold_result.complex,
                target_sequences=targets,
                hotspot_params=hotspot_params,
                near_cutoff=float(self.config.hotspot_verify_near_cutoff),
                far_cutoff=float(self.config.hotspot_verify_far_cutoff),
                min_safe_distance=float(self.config.hotspot_verify_min_distance),
                binder_contact_mask=binder_contact_mask,
            )
            summary.update(hotspot_metrics)
            summary["hotspot_binder_contact_scope"] = (
                "vhh_cdr" if binder_contact_mask is not None else "all_binder"
            )

        coordinate = _fold_result_coordinates(fold_result)
        aligned_loss_dict = None
        raw_common_loss_dict: dict[str, float] = {}
        plddt_logits = None
        if raw_features is not None and raw_output is not None:
            plddt_logits = raw_output["plddt_logits"].detach().float().cpu()
            coordinate = _validate_decoded_coordinate_layout(
                coordinate,
                raw_features["atom_attention_mask"],
            )
            distogram_losses = _structure_losses(
                raw_output["distogram_logits"],
                len(binder_seq),
                **self._interface_options(),
                target_chain_lengths=[len(sequence) for sequence in targets],
                target_chain_roles=target_chain_roles,
                binder_mutable_mask=self.config.binder_mutable_mask,
                hotspot_params=None,
                hotspot_weight=0.0,
                hotspot_cutoff=float(self.config.hotspot_cutoff),
                hotspot_interface_mode="additive",
            )
            _, raw_common_losses = compute_esmfold2_common_losses(
                **self._interface_options(),
                coordinates=raw_output["sample_atom_coords"],
                plddt_logits=raw_output["plddt_logits"],
                pae_logits=raw_output["pae_logits"],
                distogram_logits=raw_output["distogram_logits"],
                features=raw_features,
                binder_length=len(binder_seq),
                task_type=self.task_type,
                target_chain_roles=target_chain_roles,
                binder_mutable_mask=self.config.binder_mutable_mask,
                distogram_losses=distogram_losses,
                weights=ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS,
            )
            raw_common_loss_dict = _tensor_loss_dict_to_float(raw_common_losses)
            scoring_summary = dict(summary)
            _, aligned_loss_dict = compute_esmfold2_bounded_verifier_losses(
                scoring_summary,
                weights=ESMFOLD2_VERIFIER_LOSS_WEIGHTS,
            )
            summary["esm_verify_loss_weights"] = dict(ESMFOLD2_VERIFIER_LOSS_WEIGHTS)
            summary["esm_verify_score_version"] = "bounded_pair_v1"
            if self.config.target_interface_mode == "contacted":
                summary["esm_verify_score_version"] = "bounded_contacted_bidirectional_min_mean_v1"
            summary["esm_verify_raw_common_loss"] = raw_common_loss_dict["total"]
            summary["esm_verify_raw_common_loss_weights"] = dict(
                ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS
            )
            summary["esm_verify_bounded_loss"] = aligned_loss_dict["total"]
        raw_common_diagnostics = {
            f"raw_common_{key}": value
            for key, value in raw_common_loss_dict.items()
        }
        return build_esm_verifier_result(
            binder_seq=binder_seq,
            target_seq=target_seq,
            summary_confidence=summary,
            coordinate=coordinate,
            plddt_logits=plddt_logits,
            extra_loss_dict={
                **raw_common_diagnostics,
                **{
                    key: value
                    for key, value in hotspot_metrics.items()
                    if not isinstance(value, list)
                },
            },
            hotspot_verify_loss=hotspot_metrics.get("hotspot_verify_distance_loss"),
            hotspot_verify_weight=hotspot_verify_weight,
            hotspot_too_close_loss=hotspot_metrics.get("hotspot_too_close_loss"),
            hotspot_too_close_weight=hotspot_too_close_weight,
            aligned_loss_dict=aligned_loss_dict,
        )

    @measured("verifier")
    def verify(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
    ):
        targets = _target_sequences(target_seq)
        self._prepare_memory_phase("verifier", targets, len(binder_seq))
        self._ensure_verifier_loaded()
        target_chain_roles = _normalize_target_chain_roles(
            self.config.target_chain_roles,
            n_target_chains=len(targets),
        )
        input_spec = self._build_structure_input(targets, binder_seq)
        verifier_loss_mode = str(self.config.verifier_loss_mode).strip().lower()
        raw_features: Optional[dict[str, torch.Tensor]] = None
        raw_output: Optional[dict[str, torch.Tensor]] = None
        if verifier_loss_mode == "aligned_common":
            fold_result, raw_features, raw_output = (
                self._fold_verifier_with_raw_output(
                    input_spec,
                    seed=seed,
                )
            )
        else:
            fold_result = self.input_builder.fold(
                self.verifier_model,
                input_spec,
                num_loops=self.verifier_num_loops,
                num_sampling_steps=self.verifier_num_sampling_steps,
                num_diffusion_samples=1,
                seed=seed,
                complex_id="esmfold2_verify",
            )
        return self._build_verifier_result_from_fold(
            binder_seq=binder_seq,
            target_seq=target_seq,
            targets=targets,
            target_chain_roles=target_chain_roles,
            hotspot_params=hotspot_params,
            fold_result=fold_result,
            raw_features=raw_features,
            raw_output=raw_output,
        )

    @measured("scaffold_verifier")
    def verify_scaffold_consistency(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
    ):
        """Verify a local-redesign candidate with an anchored scaffold.

        This is deliberately separate from :meth:`verify`: callers must opt in
        through ``verifier_structure_mode='scaffold_consistency'`` and invoke
        this method explicitly.  It is intended for post-hoc CDR/local-redesign
        assessment, never for the default MCTS search cache.
        """

        if str(self.config.verifier_structure_mode).strip().lower() != (
            "scaffold_consistency"
        ):
            raise ValueError(
                "verify_scaffold_consistency requires "
                "verifier_structure_mode='scaffold_consistency'"
            )
        targets = _target_sequences(target_seq)
        self._prepare_memory_phase("verifier", targets, len(binder_seq))
        self._ensure_verifier_loaded()
        target_chain_roles = _normalize_target_chain_roles(
            self.config.target_chain_roles,
            n_target_chains=len(targets),
        )
        input_spec = self._build_structure_input(targets, binder_seq)
        fold_result, raw_features, raw_output, template_report = (
            self._fold_scaffold_consistency_verifier_with_raw_output(
                input_spec,
                seed=seed,
            )
        )
        return self._build_verifier_result_from_fold(
            binder_seq=binder_seq,
            target_seq=target_seq,
            targets=targets,
            target_chain_roles=target_chain_roles,
            hotspot_params=hotspot_params,
            fold_result=fold_result,
            raw_features=raw_features,
            raw_output=raw_output,
            structure_mode="scaffold_consistency",
            template_report=template_report,
        )

    def propose(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        K_top: int,
        B_candidates: int,
        T_sample: float,
        max_mutations_per_seq: int,
        mutation_fraction: Optional[float] = None,
        candidate_strategy: str = "legacy",
        dominance_gap: float = 0.5,
        protected_mutation_candidates: int = 2,
        include_argmax_candidate: Optional[bool] = None,
        seed: Optional[int] = None,
        mutable_mask: Optional[Sequence[bool]] = None,
        action_group: Optional[str] = None,
        proposal_logits: Optional[torch.Tensor] = None,
        reference_coordinates: Optional[torch.Tensor] = None,
        reference_features: Optional[Mapping[str, torch.Tensor]] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
        hotspot_weight_override: Optional[float] = None,
        track_confidence_override: Optional[bool] = None,
        soft_steps_override: Optional[int] = None,
        temperature_step_offset: int = 0,
        temperature_total_steps: Optional[int] = None,
        preserve_proposal_logits: bool = False,
    ) -> ESMFold2ProposalBatch:
        logits_before, logits_after, confidence_candidates, soft_step_metrics = self._run_soft_update_with_tracking(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=global_step,
            seed=seed,
            mutable_mask=mutable_mask,
            action_group=action_group,
            proposal_logits=proposal_logits,
            reference_coordinates=reference_coordinates,
            reference_features=reference_features,
            hotspot_params=hotspot_params,
            hotspot_weight_override=hotspot_weight_override,
            track_confidence_override=track_confidence_override,
            soft_steps_override=soft_steps_override,
            temperature_step_offset=temperature_step_offset,
            temperature_total_steps=temperature_total_steps,
            preserve_proposal_logits=preserve_proposal_logits,
        )
        confidence_sequences = self._top_confidence_sequences(confidence_candidates)
        include_logit_candidates = (
            action_group is None
            if include_argmax_candidate is None
            else bool(include_argmax_candidate) and action_group is None
        )
        with _seed_context(seed):
            discrete_batch = build_logit_delta_candidate_batch(
                logits_before,
                logits_after,
                current_seq=current_seq,
                mutable_mask=mutable_mask,
                K_top=K_top,
                B_candidates=B_candidates,
                T_sample=T_sample,
                max_mutations_per_seq=max_mutations_per_seq,
                mutation_fraction=mutation_fraction,
                include_logit_candidates=include_logit_candidates,
                candidate_strategy=candidate_strategy,
                dominance_gap=dominance_gap,
                protected_mutation_candidates=protected_mutation_candidates,
            )
        # The positive-score experiment requires every returned sequence to
        # obey the same mutation-count contract. Confidence snapshots are
        # retained for diagnostics, but cannot enter this candidate pool.
        returned_confidence_sequences = (
            [] if str(candidate_strategy).strip().lower() in {"positive_score", "positive_score_guarded"}
            else confidence_sequences
        )
        candidates = list(
            dict.fromkeys(
                [*returned_confidence_sequences, *discrete_batch.sequences]
            )
        )
        metadata_by_sequence = {
            str(item.get("sequence")): dict(item)
            for item in discrete_batch.candidate_metadata
            if item.get("sequence")
        }
        for sequence in returned_confidence_sequences:
            metadata_by_sequence.setdefault(
                str(sequence),
                {
                    "sequence": str(sequence),
                    "source": "proposal_confidence",
                },
            )
        candidate_metadata = [
            metadata_by_sequence.get(
                sequence,
                {"sequence": sequence, "source": "unknown"},
            )
            for sequence in candidates
        ]
        proposal_diagnostics = dict(discrete_batch.diagnostics)
        proposal_diagnostics["confidence_candidate_count"] = len(
            confidence_sequences
        )
        proposal_diagnostics["returned_confidence_candidate_count"] = len(
            returned_confidence_sequences
        )
        proposal_diagnostics["returned_candidate_count"] = len(candidates)
        return ESMFold2ProposalBatch(
            sequences=candidates,
            logits_before=logits_before,
            logits_after=logits_after,
            confidence_candidates=confidence_candidates,
            soft_step_metrics=soft_step_metrics,
            candidate_metadata=candidate_metadata,
            proposal_diagnostics=proposal_diagnostics,
        )

    def continuous_design(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        seed: Optional[int] = None,
        mutable_mask: Optional[Sequence[bool]] = None,
        proposal_logits: Optional[torch.Tensor] = None,
        reference_coordinates: Optional[torch.Tensor] = None,
        reference_features: Optional[Mapping[str, torch.Tensor]] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
        hotspot_weight_override: Optional[float] = None,
        track_confidence_override: Optional[bool] = None,
        soft_steps_override: Optional[int] = None,
        temperature_step_offset: int = 0,
        temperature_total_steps: Optional[int] = None,
        preserve_proposal_logits: bool = False,
    ) -> ESMFold2ProposalBatch:
        """Run one continuous ESMFold2 soft-logit trajectory and discretize it.

        This diagnostic path is intentionally separate from ``propose`` so the
        normal MCTS loop can keep its verify-and-branch behavior unchanged.
        """

        logits_before, logits_after, confidence_candidates, soft_step_metrics = self._run_soft_update_with_tracking(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=global_step,
            seed=seed,
            mutable_mask=mutable_mask,
            action_group=None,
            proposal_logits=proposal_logits,
            reference_coordinates=reference_coordinates,
            reference_features=reference_features,
            hotspot_params=hotspot_params,
            hotspot_weight_override=hotspot_weight_override,
            track_confidence_override=track_confidence_override,
            soft_steps_override=soft_steps_override,
            temperature_step_offset=temperature_step_offset,
            temperature_total_steps=temperature_total_steps,
            preserve_proposal_logits=preserve_proposal_logits,
        )
        sequence = sequence_from_logits_argmax(
            logits_after,
            current_seq=current_seq,
            mutable_mask=mutable_mask,
        )
        confidence_sequences = self._top_confidence_sequences(confidence_candidates)
        return ESMFold2ProposalBatch(
            sequences=list(dict.fromkeys([*confidence_sequences, sequence])),
            logits_before=logits_before,
            logits_after=logits_after,
            confidence_candidates=confidence_candidates,
            soft_step_metrics=soft_step_metrics,
        )

    def preview_loss_dict(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
    ) -> dict[str, float]:
        """Compute optional lightweight distogram diagnostics for a binder.

        Stage-B routing uses full-verifier metrics.  This method remains
        available for explicit diagnostics and does not mutate verifier state.
        """

        targets = _target_sequences(target_seq)
        self._prepare_memory_phase("proposal", targets, len(binder_seq))
        self._ensure_proposal_models_loaded()
        explicit_target_chain_roles = self.config.target_chain_roles
        design = self._sequence_to_design_one_hot(binder_seq)
        fold_result = self._fold_distogram(
            target_seq=target_seq,
            design=design,
            seed=seed,
        )
        losses = _structure_losses(
            fold_result["distogram_logits"],
            len(binder_seq),
            **self._interface_options(),
            target_chain_lengths=(
                [len(sequence) for sequence in targets]
                if explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted"
                else None
            ),
            target_chain_roles=explicit_target_chain_roles,
            binder_mutable_mask=(
                self.config.binder_mutable_mask
                if explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted"
                else None
            ),
            hotspot_params=hotspot_params,
            hotspot_weight=(
                float(self.config.hotspot_weight)
                if self.config.distogram_hotspot_enabled
                else 0.0
            ),
            hotspot_cutoff=float(self.config.hotspot_cutoff),
            hotspot_interface_mode=str(self.config.hotspot_interface_mode),
        )
        output: dict[str, float] = {}
        for key in (
            "intra_contact_loss",
            "inter_contact_loss",
            "effective_inter_contact_loss",
            "glob_loss",
            "hotspot_contact_loss",
        ):
            value = losses.get(key)
            scalar = _as_float(value)
            if scalar is not None:
                output[key] = scalar
        return output

    def _ensure_verifier_loaded(self) -> None:
        if self.input_builder is None or self.protein_input_cls is None or self.structure_prediction_input_cls is None:
            from esm.models.esmfold2 import (
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )

            self.input_builder = self.input_builder or ESMFold2InputBuilder()
            self.protein_input_cls = self.protein_input_cls or ProteinInput
            self.structure_prediction_input_cls = (
                self.structure_prediction_input_cls or StructurePredictionInput
            )

        if self.verifier_model is None:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

            logger.info("Loading ESMFold2 verifier model: %s", self.config.verifier_model_name)
            verifier_model = ESMFold2Model.from_pretrained(
                resolve_model_source(self.config.verifier_model_name),
                load_esmc=False,
                esmc_precision="bf16",
            )
            verifier_model.load_esmc(
                resolve_model_source(verifier_model.config.esmc_id), precision="bf16",
            )
            self.verifier_model = verifier_model.to(self.device).eval().requires_grad_(False)
        self.optimizations.memory.place("verifier", self.verifier_model)
        self.optimizations.verifier(self.verifier_model, self.input_builder)
        if self.config.esmc_optimization != "off":
            internal = getattr(self.verifier_model, "_esmc", None)
            if internal is not None:
                self.optimizations.esmc(internal, "verifier_esmc")
        self._record_memory_ready()

    def _ensure_proposal_models_loaded(self) -> None:
        if self.experimental_model is None:
            from transformers.models.esmfold2.modeling_esmfold2_experimental import (
                ESMFold2ExperimentalModel,
            )

            logger.info(
                "Loading ESMFold2 experimental proposal model: %s",
                self.config.experimental_model_name,
            )
            experimental_model = ESMFold2ExperimentalModel.from_pretrained(
                resolve_model_source(self.config.experimental_model_name),
                load_esmc=False,
            )
            if bool(getattr(self.config, "proposal_load_internal_esmc", True)):
                if not hasattr(experimental_model, "load_esmc"):
                    raise AttributeError(
                        "ESMFold2 experimental proposal model does not expose load_esmc"
                    )
                esmc_id = getattr(getattr(experimental_model, "config", None), "esmc_id", None)
                if not esmc_id:
                    raise ValueError(
                        "ESMFold2 experimental proposal model config does not define esmc_id"
                    )
                logger.info("Loading ESMFold2 experimental internal ESMC: %s", esmc_id)
                experimental_model.load_esmc(resolve_model_source(esmc_id))
            proposal_lm_dropout = getattr(self.config, "proposal_lm_dropout", None)
            if proposal_lm_dropout is not None:
                if not hasattr(experimental_model, "configure_lm_dropout"):
                    raise AttributeError(
                        "ESMFold2 experimental proposal model does not expose "
                        "configure_lm_dropout"
                    )
                experimental_model.configure_lm_dropout(
                    float(proposal_lm_dropout),
                    force_lm_dropout_during_inference=bool(
                        getattr(self.config, "force_proposal_lm_dropout", False)
                    ),
                )
            if hasattr(experimental_model, "set_kernel_backend"):
                try:
                    experimental_model.set_kernel_backend(None)
                except Exception as exc:  # pragma: no cover - optional kernel path
                    logger.warning("Could not set ESMFold2 kernel backend fallback: %s", exc)
            self.experimental_model = experimental_model.to(self.device).eval().requires_grad_(False)

        self.optimizations.memory.place("proposal", self.experimental_model)
        self.optimizations.proposal(self.experimental_model)
        if self.config.esmc_optimization != "off":
            internal = getattr(self.experimental_model, "_esmc", None)
            if internal is not None:
                self.optimizations.esmc(internal, "proposal_esmc")

        lambda_esmc = resolve_lambda_esmc(self.config.lambda_esmc, task_type=self.task_type)
        if lambda_esmc != 0.0 and self.esmc_model is None:
            from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM

            model_name = getattr(self.config, "esmc_model_name", "biohub/ESMC-6B")
            logger.info("Loading ESMC sequence-prior model: %s", model_name)
            self.esmc_model = ESMCForMaskedLM.from_pretrained(
                resolve_model_source(model_name),
                torch_dtype=torch.float32,
            ).to(self.device).eval().requires_grad_(False)
        if self.esmc_model is not None:
            if not self.optimizations.memory.active:
                self.optimizations.memory.place("prior", self.esmc_model)
            self.optimizations.esmc(self.esmc_model, "prior_esmc")
            if self.config.memory_prior_residency == "per_step":
                self.optimizations.memory.offload_model("prior", self.esmc_model)
        self._record_memory_ready()

    def _build_structure_input(self, targets: Sequence[str], binder_seq: str) -> Any:
        self._validate_target_assembly(targets)
        if self.config.target_assembly == "homotrimer" and (not binder_seq or set(binder_seq) - set(ESM_AA_ORDER)):
            raise ValueError("homotrimer requires one nonempty canonical binder sequence")
        if len(targets) + 1 > len(CHAIN_IDS):
            raise ValueError("Too many chains for first ESMFold2 backend")
        proteins = []
        for idx, sequence in enumerate(targets):
            proteins.append(self.protein_input_cls(id=CHAIN_IDS[idx], sequence=sequence, msa=None))
        proteins.append(
            self.protein_input_cls(id=CHAIN_IDS[len(targets)], sequence=binder_seq, msa=None)
        )
        return self.structure_prediction_input_cls(sequences=proteins)

    def _prepare_sequence_atom_features(
        self,
        targets: Sequence[str],
        binder_seq: str,
        *,
        seed: Optional[int],
    ) -> dict[str, torch.Tensor]:
        """Build the atom identity layout for a discrete parent sequence."""

        from esm.models.esmfold2 import prepare_esmfold2_input

        input_spec = self._build_structure_input(targets, binder_seq)
        with _seed_context(seed):
            features, _ = prepare_esmfold2_input(input_spec, seed=seed)
        return {
            key: value.unsqueeze(0)
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
        }

    def _summary_from_fold_result(
        self,
        fold_result: Any,
        *,
        n_target_chains: int,
        target_chain_roles: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        roles = _normalize_target_chain_roles(
            target_chain_roles,
            n_target_chains=n_target_chains,
        )
        plddt = _as_float(getattr(fold_result, "plddt", None))
        ptm = _as_float(getattr(fold_result, "ptm", None))
        iptm = _as_float(getattr(fold_result, "iptm", None))
        pair_chain = _mean_target_to_binder_pair_iptm(
            getattr(fold_result, "pair_chains_iptm", None),
            n_target_chains=n_target_chains,
            target_chain_roles=roles,
        )
        scaffold_pair_chain = _mean_target_to_binder_pair_iptm(
            getattr(fold_result, "pair_chains_iptm", None),
            n_target_chains=n_target_chains,
            target_chain_roles=roles,
            selected_role="scaffold",
        )
        if plddt is not None:
            summary["plddt"] = plddt
        if ptm is not None:
            summary["ptm"] = ptm
        if iptm is not None:
            summary["iptm"] = iptm
        if pair_chain is not None:
            summary["pair_chain_iptm"] = pair_chain
        if scaffold_pair_chain is not None:
            summary["scaffold_binder_pair_chain_iptm"] = scaffold_pair_chain
        summary["target_chain_roles"] = list(roles)
        complex_obj = getattr(fold_result, "complex", None)
        if complex_obj is not None:
            if hasattr(complex_obj, "to_mmcif"):
                summary["mmcif"] = complex_obj.to_mmcif()
            elif hasattr(complex_obj, "to_mmcif_string"):
                # Compatibility with older ESM SDK result objects.
                summary["mmcif"] = complex_obj.to_mmcif_string()
        return summary

    def _run_soft_update(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        seed: Optional[int],
        mutable_mask: Optional[Sequence[bool]],
        action_group: Optional[str],
        proposal_logits: Optional[torch.Tensor],
        reference_coordinates: Optional[torch.Tensor] = None,
        reference_features: Optional[Mapping[str, torch.Tensor]] = None,
        hotspot_params: Optional[Dict[str, Any]] = None,
        hotspot_weight_override: Optional[float] = None,
        preserve_proposal_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_before, logits_after, _, _ = self._run_soft_update_impl(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=global_step,
            seed=seed,
            mutable_mask=mutable_mask,
            action_group=action_group,
            proposal_logits=proposal_logits,
            reference_coordinates=reference_coordinates,
            reference_features=reference_features,
            hotspot_params=hotspot_params,
            hotspot_weight_override=hotspot_weight_override,
            track_confidence=False,
            preserve_proposal_logits=preserve_proposal_logits,
        )
        return logits_before, logits_after

    def _run_soft_update_with_tracking(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        seed: Optional[int],
        mutable_mask: Optional[Sequence[bool]],
        action_group: Optional[str],
        proposal_logits: Optional[torch.Tensor],
        reference_coordinates: Optional[torch.Tensor],
        reference_features: Optional[Mapping[str, torch.Tensor]],
        hotspot_params: Optional[Dict[str, Any]],
        hotspot_weight_override: Optional[float] = None,
        track_confidence_override: Optional[bool] = None,
        soft_steps_override: Optional[int] = None,
        temperature_step_offset: int = 0,
        temperature_total_steps: Optional[int] = None,
        preserve_proposal_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]]]:
        track_confidence = bool(getattr(self.config, "track_proposal_confidence", False))
        if track_confidence_override is not None:
            track_confidence = bool(track_confidence_override)
        return self._run_soft_update_impl(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=global_step,
            seed=seed,
            mutable_mask=mutable_mask,
            action_group=action_group,
            proposal_logits=proposal_logits,
            reference_coordinates=reference_coordinates,
            reference_features=reference_features,
            hotspot_params=hotspot_params,
            hotspot_weight_override=hotspot_weight_override,
            track_confidence=track_confidence,
            soft_steps_override=soft_steps_override,
            temperature_step_offset=temperature_step_offset,
            temperature_total_steps=temperature_total_steps,
            preserve_proposal_logits=preserve_proposal_logits,
        )

    @measured("proposal_diffusion_gradient")
    def _one_step_diffusion_gradient(
        self,
        *,
        logits: torch.Tensor,
        fold_result: Mapping[str, Any],
        targets: Sequence[str],
        current_seq: str,
        mutable_mask: Optional[Sequence[bool]],
        target_chain_roles: Sequence[str],
        reference_coordinates: Optional[torch.Tensor],
        reference_features: Optional[Mapping[str, torch.Tensor]],
        diffusion_seed: Optional[int],
        hotspot_params: Optional[Dict[str, Any]],
        effective_hotspot_weight: float,
    ) -> Dict[str, Any]:
        if self.config.memory_prior_residency == "between_diffusion":
            # The prior's own backward has completed. Keep the proposal model
            # and its live graph; move only the independent sequence prior.
            self._prepare_memory_phase("diffusion", targets, len(current_seq))
        one_step_mode = str(self.config.one_step_diffusion_mode).strip().lower()
        template_report: Optional[Mapping[str, Any]] = None
        if one_step_mode == "truncated_trajectory":
            one_step = run_esmfold2_truncated_diffusion(
                model=self.experimental_model,
                captured=fold_result["_captured"],
                features=fold_result["_features"],
                tau=float(self.config.one_step_diffusion_tau),
                seed=diffusion_seed,
                num_sampling_steps=int(
                    self.config.one_step_diffusion_truncated_num_sampling_steps
                ),
                truncated_schedule_mode=str(
                    self.config.one_step_diffusion_truncated_schedule_mode
                ),
                dense_num_sampling_steps=int(
                    self.config.one_step_diffusion_truncated_dense_num_sampling_steps
                ),
            )
            source_coordinates = one_step["source_coordinates"]
        elif one_step_mode == "template_anchored_truncated_trajectory":
            template_clamp = build_template_clamp(
                features=fold_result["_features"],
                template_path=str(self.config.template_structure_path),
                template_chain_ids=self.config.template_chain_ids or (),
                template_chain_sequences=self.config.template_chain_sequences or (),
                binder_structure_regeneration_mask=(
                    self.config.binder_structure_regeneration_mask or ()
                ),
            )
            one_step = run_esmfold2_template_anchored_truncated_diffusion(
                model=self.experimental_model,
                captured=fold_result["_captured"],
                features=fold_result["_features"],
                template_coordinates=template_clamp.coordinates,
                fixed_atom_mask=template_clamp.fixed_atom_mask,
                tau=float(self.config.one_step_diffusion_tau),
                seed=diffusion_seed,
                num_sampling_steps=int(
                    self.config.one_step_diffusion_truncated_num_sampling_steps
                ),
                truncated_schedule_mode=str(
                    self.config.one_step_diffusion_truncated_schedule_mode
                ),
                dense_num_sampling_steps=int(
                    self.config.one_step_diffusion_truncated_dense_num_sampling_steps
                ),
            )
            source_coordinates = one_step["template_coordinates"]
            template_report = template_clamp.report
        elif one_step_mode == "reference_noising":
            if reference_coordinates is None:
                raise ValueError(
                    "reference_noising one-step diffusion requires verifier coordinates"
                )
            source_features = reference_features
            if source_features is None:
                source_features = self._prepare_sequence_atom_features(
                    targets,
                    current_seq,
                    seed=diffusion_seed,
                )
            remapped_reference = remap_reference_coordinates(
                reference_coordinates,
                source_features=source_features,
                destination_features=fold_result["_features"],
            )
            one_step = run_esmfold2_one_step_diffusion(
                model=self.experimental_model,
                captured=fold_result["_captured"],
                features=fold_result["_features"],
                reference_coordinates=remapped_reference,
                tau=float(self.config.one_step_diffusion_tau),
                seed=diffusion_seed,
            )
            source_coordinates = remapped_reference
        else:
            raise ValueError(f"Unsupported one-step diffusion mode: {one_step_mode}")
        diffusion_total, diffusion_losses = compute_esmfold2_common_losses(
            **self._interface_options(),
            coordinates=one_step["coordinates"],
            plddt_logits=one_step["plddt_logits"],
            pae_logits=one_step["pae_logits"],
            distogram_logits=fold_result["distogram_logits"],
            features=fold_result["_features"],
            binder_length=len(current_seq),
            task_type=self.task_type,
            target_chain_roles=target_chain_roles,
            binder_mutable_mask=mutable_mask,
            distogram_losses=None,
            weights={
                "con": 0.0,
                "i_con": 0.0,
                "helix": 0.0,
                "rg": float(self.config.one_step_diffusion_rg_weight),
            },
            hotspot_params=(
                hotspot_params
                if self.config.one_step_diffusion_hotspot_enabled
                else None
            ),
            hotspot_weight=(
                effective_hotspot_weight
                if self.config.one_step_diffusion_hotspot_enabled
                else 0.0
            ),
            hotspot_interface_mode=str(self.config.hotspot_interface_mode),
        )
        diffusion_grad = torch.autograd.grad(diffusion_total, logits)[0]
        diffusion_grad_norm = float(diffusion_grad.detach().norm().item())
        mutable_diffusion_grad_norm = diffusion_grad_norm
        if mutable_mask is not None:
            if diffusion_grad.ndim < 2 or diffusion_grad.shape[-2] != len(
                mutable_mask
            ):
                raise ValueError(
                    "one-step diffusion gradient does not match mutable mask"
                )
            mutable_selector = torch.as_tensor(
                mutable_mask,
                device=diffusion_grad.device,
                dtype=diffusion_grad.dtype,
            ).view(*((1,) * (diffusion_grad.ndim - 2)), -1, 1)
            mutable_diffusion_grad_norm = float(
                (diffusion_grad.detach() * mutable_selector).norm().item()
            )

        denoised_sample = _first_coordinate_sample(one_step["coordinates"])
        reference_sample = _first_coordinate_sample(source_coordinates).to(
            device=denoised_sample.device,
            dtype=denoised_sample.dtype,
        )
        atom_mask = one_step.get(
            "fixed_atom_mask",
            fold_result["_features"]["atom_attention_mask"],
        )
        if atom_mask.ndim > 1:
            atom_mask = atom_mask[0]
        atom_mask = atom_mask.to(device=denoised_sample.device).bool()
        coordinate_rmsd = torch.sqrt(
            torch.square(
                denoised_sample[atom_mask] - reference_sample[atom_mask]
            )
            .sum(dim=-1)
            .mean()
        )
        result = {
            "gradient": diffusion_grad,
            "total": diffusion_total,
            "losses": diffusion_losses,
            "coordinates": one_step["coordinates"],
            "features": fold_result["_features"],
            "reference_rmsd": coordinate_rmsd,
            "mode": one_step_mode,
            "truncated_steps": int(one_step.get("truncated_steps", 0)),
            "truncated_schedule_mode": str(
                self.config.one_step_diffusion_truncated_schedule_mode
            ),
            "gradient_norm": diffusion_grad_norm,
            "mutable_gradient_norm": mutable_diffusion_grad_norm,
        }
        if template_report is not None:
            result["template_report"] = dict(template_report)
            result["template_fixed_atom_count"] = int(atom_mask.sum().item())
            result["template_fixed_rmsd"] = coordinate_rmsd
        return result

    @measured("proposal_update")
    def _run_soft_update_impl(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        seed: Optional[int],
        mutable_mask: Optional[Sequence[bool]],
        action_group: Optional[str],
        proposal_logits: Optional[torch.Tensor],
        reference_coordinates: Optional[torch.Tensor],
        reference_features: Optional[Mapping[str, torch.Tensor]],
        hotspot_params: Optional[Dict[str, Any]],
        hotspot_weight_override: Optional[float],
        track_confidence: bool,
        soft_steps_override: Optional[int] = None,
        temperature_step_offset: int = 0,
        temperature_total_steps: Optional[int] = None,
        preserve_proposal_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.soft_update_fn is not None:
            kwargs = dict(
                current_seq=current_seq,
                target_seq=target_seq,
                global_step=global_step,
                seed=seed,
                action_group=action_group,
                mutable_mask=mutable_mask,
            )
            if _callable_accepts_argument(self.soft_update_fn, "proposal_logits"):
                kwargs["proposal_logits"] = proposal_logits
            if _callable_accepts_argument(self.soft_update_fn, "soft_steps_override"):
                kwargs["soft_steps_override"] = soft_steps_override
            if _callable_accepts_argument(
                self.soft_update_fn, "temperature_step_offset"
            ):
                kwargs["temperature_step_offset"] = temperature_step_offset
            if _callable_accepts_argument(
                self.soft_update_fn, "temperature_total_steps"
            ):
                kwargs["temperature_total_steps"] = temperature_total_steps
            if _callable_accepts_argument(
                self.soft_update_fn, "preserve_proposal_logits"
            ):
                kwargs["preserve_proposal_logits"] = preserve_proposal_logits
            if _callable_accepts_argument(self.soft_update_fn, "hotspot_params"):
                kwargs["hotspot_params"] = hotspot_params
            if _callable_accepts_argument(
                self.soft_update_fn, "hotspot_weight_override"
            ):
                kwargs["hotspot_weight_override"] = hotspot_weight_override
            logits_before, logits_after = self.soft_update_fn(**kwargs)
            return logits_before, logits_after, [], []

        targets = _target_sequences(target_seq)
        explicit_target_chain_roles = self.config.target_chain_roles
        target_chain_roles = _normalize_target_chain_roles(
            explicit_target_chain_roles,
            n_target_chains=len(targets),
        )
        configured_mutable_mask = self.config.binder_mutable_mask
        if configured_mutable_mask is not None:
            configured_mutable_mask = tuple(
                bool(value) for value in configured_mutable_mask
            )
            if len(configured_mutable_mask) != len(current_seq):
                raise ValueError(
                    "configured binder_mutable_mask length does not match current sequence"
                )
            if mutable_mask is None:
                mutable_mask = configured_mutable_mask
            elif tuple(bool(value) for value in mutable_mask) != configured_mutable_mask:
                raise ValueError(
                    "runtime mutable_mask does not match configured binder_mutable_mask"
                )
        self._prepare_memory_phase("proposal", targets, len(current_seq))
        self._ensure_proposal_models_loaded()
        effective_hotspot_weight = (
            float(self.config.hotspot_weight)
            if hotspot_weight_override is None
            else float(hotspot_weight_override)
        )
        if effective_hotspot_weight < 0.0:
            raise ValueError("hotspot_weight_override must be non-negative")
        if (
            str(self.config.hotspot_interface_mode).strip().lower()
            == "localized"
            and effective_hotspot_weight > 1.0
        ):
            raise ValueError(
                "localized hotspot_weight_override must be in [0, 1]"
            )
        effective_distogram_hotspot_weight = (
            effective_hotspot_weight
            if self.config.distogram_hotspot_enabled
            else 0.0
        )

        logits = self._init_current_sequence_logits(
            current_seq,
            mutable_mask=mutable_mask,
            proposal_logits=proposal_logits,
            preserve_proposal_logits=preserve_proposal_logits,
        )
        logits_before = logits.detach().clone()[0]
        gradient_mask = build_gradient_mask(
            current_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            device=self.device,
        )
        optimizer = optim.SGD([logits], lr=float(self.config.learning_rate))
        lambda_esmc = resolve_lambda_esmc(self.config.lambda_esmc, task_type=self.task_type)
        confidence_candidates: List[Dict[str, Any]] = []
        soft_step_metrics: List[Dict[str, Any]] = []
        soft_steps = (
            int(soft_steps_override)
            if soft_steps_override is not None
            else int(self.config.soft_steps)
        )
        if soft_steps <= 0:
            raise ValueError("ESMFold2 soft steps must be positive")
        temperature_step_offset = int(temperature_step_offset)
        if temperature_step_offset < 0:
            raise ValueError("temperature_step_offset must be non-negative")
        resolved_temperature_total_steps = (
            int(temperature_total_steps)
            if temperature_total_steps is not None
            else soft_steps
        )
        if resolved_temperature_total_steps <= 0:
            raise ValueError("temperature_total_steps must be positive")

        one_step_enabled = bool(self.config.one_step_diffusion_enabled)
        one_step_interval = int(self.config.one_step_diffusion_interval)
        scheduled_one_step_steps = set(
            _scheduled_one_step_soft_steps(
                soft_steps=soft_steps,
                interval=one_step_interval,
            )
            if one_step_enabled
            else ()
        )
        one_step_mode = str(self.config.one_step_diffusion_mode).strip().lower()
        if (
            one_step_enabled
            and one_step_mode == "reference_noising"
            and reference_coordinates is None
        ):
            raise ValueError(
                "ESMFold2 one-step diffusion requires coordinates from the "
                "selected verifier node"
            )
        recycle_one_step_coordinates = bool(
            self.config.one_step_diffusion_recycle_coordinates
        )
        recycled_reference_coordinates: Optional[torch.Tensor] = None
        recycled_reference_features: Optional[Mapping[str, torch.Tensor]] = None
        one_step_injection_count = 0

        for local_step in range(soft_steps):
            optimizer.zero_grad()
            one_based_soft_step = local_step + 1
            run_scheduled_one_step = (
                one_based_soft_step in scheduled_one_step_steps
            )
            temperature_step = temperature_step_offset + local_step
            temperature = esm_temperature(
                temperature_step,
                total_steps=resolved_temperature_total_steps,
                temperature_min=float(self.config.temperature_min),
                temperature_max=float(self.config.temperature_max),
            )
            should_track_confidence = self._should_track_proposal_confidence(
                track_confidence=track_confidence,
                local_step=local_step,
                temperature=temperature,
                soft_steps=soft_steps,
            )
            design = F.softmax(logits / temperature, dim=-1)
            fold_result = self._fold_distogram(
                target_seq=targets,
                design=design,
                seed=None if seed is None else int(seed) + int(global_step) + local_step,
                calculate_confidence=should_track_confidence,
                num_loops=(
                    int(self.config.proposal_confidence_num_loops)
                    if should_track_confidence
                    else int(self.config.num_loops)
                ),
                num_sampling_steps=(
                    int(self.config.proposal_confidence_num_sampling_steps)
                    if should_track_confidence
                    else int(self.config.num_sampling_steps)
                ),
                capture_diffusion_state=run_scheduled_one_step,
            )
            if should_track_confidence:
                record = self._proposal_confidence_record(
                    fold_result,
                    logits=logits,
                    current_seq=current_seq,
                    mutable_mask=mutable_mask,
                    local_step=local_step,
                    global_step=global_step,
                    temperature=temperature,
                )
                if record is not None:
                    confidence_candidates.append(record)
            losses = _structure_losses(
                fold_result["distogram_logits"],
                len(current_seq),
                **self._interface_options(),
                target_chain_lengths=(
                    [len(sequence) for sequence in targets]
                    if explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted"
                    else None
                ),
                target_chain_roles=explicit_target_chain_roles,
                binder_mutable_mask=(
                    mutable_mask
                    if explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted"
                    else None
                ),
                hotspot_params=hotspot_params,
                hotspot_weight=effective_distogram_hotspot_weight,
                hotspot_cutoff=float(self.config.hotspot_cutoff),
                hotspot_interface_mode=str(self.config.hotspot_interface_mode),
            )
            structure_loss, _ = _stageb_structure_loss(
                losses,
                action_group,
                group_weight=getattr(self.config, "stageb_group_weight", 1.0),
                hotspot_interface_mode=str(self.config.hotspot_interface_mode),
            )
            structure_grad = torch.autograd.grad(
                structure_loss.mean(),
                logits,
                retain_graph=run_scheduled_one_step,
            )[0]

            esmc_grad = None
            if lambda_esmc != 0.0:
                design_for_esmc = F.softmax(logits / temperature, dim=-1)
                score_mask = gradient_mask.sum(dim=-1) > 0
                with _seed_context(None if seed is None else int(seed) + int(global_step) + local_step):
                    esmc_loss = self._compute_esmc_pseudoperplexity(
                        design_for_esmc,
                        score_mask=score_mask,
                    )
                esmc_grad = torch.autograd.grad(esmc_loss.mean(), logits)[0]
                self._prepare_memory_phase("proposal", targets, len(current_seq))

            combined_grad = combine_normalized_gradients(
                structure_grad,
                esmc_grad,
                gradient_mask=gradient_mask,
                lambda_esmc=lambda_esmc,
            )
            scheduled_one_step: Optional[Dict[str, Any]] = None
            if run_scheduled_one_step:
                diffusion_seed = (
                    None
                    if seed is None
                    else int(seed)
                    + int(global_step)
                    + one_based_soft_step
                    + 100_003
                )
                use_recycled_reference = bool(
                    recycle_one_step_coordinates
                    and recycled_reference_coordinates is not None
                    and recycled_reference_features is not None
                )
                scheduled_one_step = self._one_step_diffusion_gradient(
                    logits=logits,
                    fold_result=fold_result,
                    targets=targets,
                    current_seq=current_seq,
                    mutable_mask=mutable_mask,
                    target_chain_roles=target_chain_roles,
                    reference_coordinates=(
                        recycled_reference_coordinates
                        if use_recycled_reference
                        else reference_coordinates
                    ),
                    reference_features=(
                        recycled_reference_features
                        if use_recycled_reference
                        else reference_features
                    ),
                    diffusion_seed=diffusion_seed,
                    hotspot_params=hotspot_params,
                    effective_hotspot_weight=effective_hotspot_weight,
                )
                combined_grad = combined_grad + float(
                    self.config.one_step_diffusion_weight
                ) * normalized_gradient_tensor(
                    scheduled_one_step["gradient"],
                    gradient_mask,
                )
                if recycle_one_step_coordinates:
                    recycled_reference_coordinates = scheduled_one_step[
                        "coordinates"
                    ].detach()
                    recycled_reference_features = {
                        key: (
                            value.detach()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in scheduled_one_step["features"].items()
                    }
                one_step_injection_count += 1
            logits.grad = combined_grad * gradient_mask
            for group in optimizer.param_groups:
                group["lr"] = float(self.config.learning_rate) * temperature
            optimizer.step()
            structure_total = losses["total_loss"]
            combined_total = structure_total + (esmc_loss if lambda_esmc != 0.0 else 0.0)
            argmax_sequence = sequence_from_logits_argmax(
                logits.detach()[0],
                current_seq=current_seq,
                mutable_mask=mutable_mask,
            )
            metrics = {
                "global_step": int(global_step),
                "local_step": int(local_step),
                "temperature_step": int(temperature_step),
                "temperature_total_steps": int(resolved_temperature_total_steps),
                "temperature": float(temperature),
                "argmax_sequence": argmax_sequence,
                "proposal_loss_source": _stageb_structure_loss(
                    losses,
                    action_group,
                    group_weight=getattr(self.config, "stageb_group_weight", 1.0),
                    hotspot_interface_mode=str(self.config.hotspot_interface_mode),
                )[1],
                "intra_contact_loss": _as_float(losses.get("intra_contact_loss")),
                "inter_contact_loss": _as_float(losses.get("inter_contact_loss")),
                "glob_loss": _as_float(losses.get("glob_loss")),
                "structure_total_loss": _as_float(structure_total),
                "plm_loss": None if lambda_esmc == 0.0 else _as_float(esmc_loss),
                "total_loss": _as_float(combined_total),
                "backprop_loss": _as_float(structure_loss),
                "lambda_esmc": float(lambda_esmc),
                "confidence_tracked": bool(should_track_confidence),
            }
            if "hotspot_contact_loss" in losses:
                metrics.update(
                    {
                        "base_structure_loss": _as_float(losses.get("base_structure_loss")),
                        "hotspot_contact_loss": _as_float(losses.get("hotspot_contact_loss")),
                        "hotspot_weight": effective_distogram_hotspot_weight,
                        "distogram_hotspot_enabled": bool(
                            self.config.distogram_hotspot_enabled
                        ),
                        "distogram_hotspot_weight": float(
                            effective_distogram_hotspot_weight
                        ),
                        "hotspot_weight_configured": float(
                            self.config.hotspot_weight
                        ),
                        "hotspot_attraction_suppressed": bool(
                            effective_distogram_hotspot_weight
                            < float(self.config.hotspot_weight)
                        ),
                        "hotspot_cutoff": float(self.config.hotspot_cutoff),
                        "hotspot_interface_mode": str(
                            self.config.hotspot_interface_mode
                        ),
                        "hotspot_weighted_contribution": _as_float(
                            losses.get("hotspot_weighted_contribution")
                        ),
                        "hotspot_objective_adjustment": _as_float(
                            losses.get("hotspot_objective_adjustment")
                        ),
                        "effective_inter_contact_loss": _as_float(
                            losses.get("effective_inter_contact_loss")
                        ),
                    }
                )
            if scheduled_one_step is not None:
                metrics.update(
                    {
                        "one_step_diffusion": True,
                        "one_step_diffusion_schedule": "interval",
                        "one_step_diffusion_interval": int(one_step_interval),
                        "one_step_diffusion_injection_index": int(
                            one_step_injection_count
                        ),
                        "one_step_diffusion_recycle_coordinates": bool(
                            recycle_one_step_coordinates
                        ),
                        "one_step_diffusion_used_recycled_coordinates": bool(
                            recycle_one_step_coordinates
                            and one_step_injection_count > 1
                        ),
                        "one_step_diffusion_tau": float(
                            self.config.one_step_diffusion_tau
                        ),
                        "one_step_diffusion_mode": str(
                            scheduled_one_step.get("mode", one_step_mode)
                        ),
                        "one_step_diffusion_truncated_steps": int(
                            scheduled_one_step.get("truncated_steps", 0)
                        ),
                        "one_step_diffusion_truncated_schedule_mode": str(
                            scheduled_one_step.get(
                                "truncated_schedule_mode",
                                self.config.one_step_diffusion_truncated_schedule_mode,
                            )
                        ),
                        "one_step_diffusion_weight": float(
                            self.config.one_step_diffusion_weight
                        ),
                        "one_step_diffusion_rg_weight": float(
                            self.config.one_step_diffusion_rg_weight
                        ),
                        "one_step_diffusion_hotspot_enabled": bool(
                            self.config.one_step_diffusion_hotspot_enabled
                            and hotspot_params is not None
                        ),
                        "one_step_diffusion_hotspot_weight": float(
                            effective_hotspot_weight
                            if self.config.one_step_diffusion_hotspot_enabled
                            else 0.0
                        ),
                        "one_step_diffusion_hotspot_interface_mode": str(
                            self.config.hotspot_interface_mode
                        ),
                        "distogram_hotspot_enabled": bool(
                            self.config.distogram_hotspot_enabled
                        ),
                        "distogram_hotspot_weight": float(
                            effective_distogram_hotspot_weight
                        ),
                        "one_step_diffusion_loss": _as_float(
                            scheduled_one_step["total"]
                        ),
                        "one_step_diffusion_reference_rmsd": _as_float(
                            scheduled_one_step["reference_rmsd"]
                        ),
                        "one_step_diffusion_gradient_norm": float(
                            scheduled_one_step["gradient_norm"]
                        ),
                        "one_step_diffusion_mutable_gradient_norm": float(
                            scheduled_one_step["mutable_gradient_norm"]
                        ),
                    }
                )
                if scheduled_one_step.get("template_report") is not None:
                    metrics["one_step_template_report"] = dict(
                        scheduled_one_step["template_report"]
                    )
                    metrics["one_step_template_fixed_atom_count"] = int(
                        scheduled_one_step["template_fixed_atom_count"]
                    )
                    metrics["one_step_template_fixed_rmsd"] = _as_float(
                        scheduled_one_step["template_fixed_rmsd"]
                    )
                for name, value in scheduled_one_step["losses"].items():
                    metrics[f"diffusion_{name}_loss"] = _as_float(value)
                if "rg" in scheduled_one_step["losses"]:
                    metrics["diffusion_rg_weighted_contribution"] = _as_float(
                        scheduled_one_step["losses"]["rg"]
                        * float(self.config.one_step_diffusion_rg_weight)
                    )
            if (
                local_step == 0
                or local_step == soft_steps - 1
                or (local_step + 1) % 10 == 0
            ):
                metrics["logits_snapshot"] = logits.detach().clone()[0].cpu()
            if "_interface_selection" in losses:
                metrics["target_interface_selection"] = dict(losses["_interface_selection"])
            soft_step_metrics.append(metrics)

        if one_step_enabled and one_step_interval == 0:
            optimizer.zero_grad()
            final_temperature = esm_temperature(
                min(
                    temperature_step_offset + soft_steps,
                    resolved_temperature_total_steps - 1,
                ),
                total_steps=resolved_temperature_total_steps,
                temperature_min=float(self.config.temperature_min),
                temperature_max=float(self.config.temperature_max),
            )
            design = F.softmax(logits / final_temperature, dim=-1)
            diffusion_seed = (
                None
                if seed is None
                else int(seed) + int(global_step) + soft_steps + 100_003
            )
            fold_result = self._fold_distogram(
                target_seq=targets,
                design=design,
                seed=diffusion_seed,
                calculate_confidence=False,
                num_loops=int(self.config.num_loops),
                num_sampling_steps=int(self.config.num_sampling_steps),
                capture_diffusion_state=True,
            )
            losses = _structure_losses(
                fold_result["distogram_logits"],
                len(current_seq),
                **self._interface_options(),
                target_chain_lengths=(
                    [len(sequence) for sequence in targets]
                    if explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted"
                    else None
                ),
                target_chain_roles=explicit_target_chain_roles,
                binder_mutable_mask=(
                    mutable_mask if (explicit_target_chain_roles is not None or self.config.target_interface_mode == "contacted") else None
                ),
                hotspot_params=hotspot_params,
                hotspot_weight=effective_distogram_hotspot_weight,
                hotspot_cutoff=float(self.config.hotspot_cutoff),
                hotspot_interface_mode=str(self.config.hotspot_interface_mode),
            )
            structure_loss, proposal_loss_source = _stageb_structure_loss(
                losses,
                action_group,
                group_weight=getattr(self.config, "stageb_group_weight", 1.0),
                hotspot_interface_mode=str(self.config.hotspot_interface_mode),
            )
            structure_grad = torch.autograd.grad(
                structure_loss.mean(),
                logits,
                retain_graph=True,
            )[0]

            esmc_loss = None
            esmc_grad = None
            if lambda_esmc != 0.0:
                score_mask = gradient_mask.sum(dim=-1) > 0
                with _seed_context(diffusion_seed):
                    esmc_loss = self._compute_esmc_pseudoperplexity(
                        F.softmax(logits / final_temperature, dim=-1),
                        score_mask=score_mask,
                    )
                esmc_grad = torch.autograd.grad(esmc_loss.mean(), logits)[0]

            self._prepare_memory_phase("proposal", targets, len(current_seq))
            one_step = self._one_step_diffusion_gradient(
                logits=logits,
                fold_result=fold_result,
                targets=targets,
                current_seq=current_seq,
                mutable_mask=mutable_mask,
                target_chain_roles=target_chain_roles,
                reference_coordinates=reference_coordinates,
                reference_features=reference_features,
                diffusion_seed=diffusion_seed,
                hotspot_params=hotspot_params,
                effective_hotspot_weight=effective_hotspot_weight,
            )
            diffusion_total = one_step["total"]
            diffusion_losses = one_step["losses"]
            diffusion_grad = one_step["gradient"]
            combined_grad = combine_normalized_gradients(
                structure_grad,
                esmc_grad,
                gradient_mask=gradient_mask,
                lambda_esmc=lambda_esmc,
            )
            combined_grad = combined_grad + float(
                self.config.one_step_diffusion_weight
            ) * normalized_gradient_tensor(diffusion_grad, gradient_mask)
            logits.grad = combined_grad * gradient_mask
            for group in optimizer.param_groups:
                group["lr"] = float(self.config.learning_rate) * final_temperature
            optimizer.step()

            one_step_metrics: Dict[str, Any] = {
                "global_step": int(global_step),
                "local_step": int(soft_steps),
                "temperature": float(final_temperature),
                "proposal_loss_source": proposal_loss_source,
                "one_step_diffusion": True,
                "one_step_diffusion_schedule": "legacy_final_only",
                "one_step_diffusion_interval": 0,
                "one_step_diffusion_injection_index": 1,
                "one_step_diffusion_recycle_coordinates": False,
                "one_step_diffusion_used_recycled_coordinates": False,
                "one_step_diffusion_tau": float(
                    self.config.one_step_diffusion_tau
                ),
                "one_step_diffusion_mode": str(one_step["mode"]),
                "one_step_diffusion_truncated_steps": int(
                    one_step["truncated_steps"]
                ),
                "one_step_diffusion_truncated_schedule_mode": str(
                    one_step.get(
                        "truncated_schedule_mode",
                        self.config.one_step_diffusion_truncated_schedule_mode,
                    )
                ),
                "one_step_diffusion_weight": float(
                    self.config.one_step_diffusion_weight
                ),
                "one_step_diffusion_rg_weight": float(
                    self.config.one_step_diffusion_rg_weight
                ),
                "one_step_diffusion_hotspot_enabled": bool(
                    self.config.one_step_diffusion_hotspot_enabled
                    and hotspot_params is not None
                ),
                "one_step_diffusion_hotspot_weight": float(
                    effective_hotspot_weight
                    if self.config.one_step_diffusion_hotspot_enabled
                    else 0.0
                ),
                "one_step_diffusion_hotspot_interface_mode": str(
                    self.config.hotspot_interface_mode
                ),
                "distogram_hotspot_enabled": bool(
                    self.config.distogram_hotspot_enabled
                ),
                "distogram_hotspot_weight": float(
                    effective_distogram_hotspot_weight
                ),
                "one_step_diffusion_loss": _as_float(diffusion_total),
                "one_step_diffusion_reference_rmsd": _as_float(
                    one_step["reference_rmsd"]
                ),
                "one_step_diffusion_gradient_norm": float(
                    one_step["gradient_norm"]
                ),
                "one_step_diffusion_mutable_gradient_norm": float(
                    one_step["mutable_gradient_norm"]
                ),
                "structure_total_loss": _as_float(losses["total_loss"]),
                "backprop_loss": _as_float(structure_loss),
                "plm_loss": None if esmc_loss is None else _as_float(esmc_loss),
                "lambda_esmc": float(lambda_esmc),
                "argmax_sequence": sequence_from_logits_argmax(
                    logits.detach()[0],
                    current_seq=current_seq,
                    mutable_mask=mutable_mask,
                ),
                "logits_snapshot": logits.detach().clone()[0].cpu(),
            }
            if one_step.get("template_report") is not None:
                one_step_metrics["one_step_template_report"] = dict(
                    one_step["template_report"]
                )
                one_step_metrics["one_step_template_fixed_atom_count"] = int(
                    one_step["template_fixed_atom_count"]
                )
                one_step_metrics["one_step_template_fixed_rmsd"] = _as_float(
                    one_step["template_fixed_rmsd"]
                )
            for name, value in diffusion_losses.items():
                one_step_metrics[f"diffusion_{name}_loss"] = _as_float(value)
            if "rg" in diffusion_losses:
                one_step_metrics["diffusion_rg_weighted_contribution"] = _as_float(
                    diffusion_losses["rg"]
                    * float(self.config.one_step_diffusion_rg_weight)
                )
            if "_interface_selection" in losses:
                one_step_metrics["target_interface_selection"] = dict(losses["_interface_selection"])
            soft_step_metrics.append(one_step_metrics)

        return (
            logits_before.detach().cpu(),
            logits.detach().clone()[0].cpu(),
            confidence_candidates,
            soft_step_metrics,
        )

    def _should_track_proposal_confidence(
        self,
        *,
        track_confidence: bool,
        local_step: int,
        temperature: float,
        soft_steps: Optional[int] = None,
    ) -> bool:
        if not track_confidence:
            return False
        total_soft_steps = int(soft_steps if soft_steps is not None else self.config.soft_steps)
        last_step = total_soft_steps - 1
        tail_steps = int(getattr(self.config, "proposal_confidence_tail_steps", 0) or 0)
        if tail_steps > 0:
            first_tail_step = max(0, total_soft_steps - tail_steps)
            if int(local_step) < first_tail_step:
                return False
        elif float(temperature) > float(self.config.proposal_confidence_temperature):
            return False
        interval = max(int(self.config.proposal_confidence_interval), 1)
        return int(local_step) == last_step or int(local_step) % interval == 0

    def _proposal_confidence_record(
        self,
        fold_result: Dict[str, Any],
        *,
        logits: torch.Tensor,
        current_seq: str,
        mutable_mask: Optional[Sequence[bool]],
        local_step: int,
        global_step: int,
        temperature: float,
    ) -> Optional[Dict[str, Any]]:
        iptm = _as_float(fold_result.get("iptm"))
        pair_chain = _as_float(fold_result.get("pair_chain_iptm"))
        score = pair_chain if pair_chain is not None else iptm
        if score is None:
            return None
        sequence = sequence_from_logits_argmax(
            logits.detach()[0],
            current_seq=current_seq,
            mutable_mask=mutable_mask,
        )
        record: Dict[str, Any] = {
            "sequence": sequence,
            "global_step": int(global_step),
            "local_step": int(local_step),
            "temperature": float(temperature),
            "confidence_score": float(score),
            "confidence_source": "pair_chain_iptm" if pair_chain is not None else "iptm",
        }
        if iptm is not None:
            record["iptm"] = float(iptm)
        if pair_chain is not None:
            record["pair_chain_iptm"] = float(pair_chain)
        ptm = _as_float(fold_result.get("ptm"))
        plddt = _as_float(fold_result.get("plddt"))
        if ptm is not None:
            record["ptm"] = float(ptm)
        if plddt is not None:
            record["plddt"] = float(plddt)
        return record

    def _top_confidence_sequences(
        self,
        confidence_candidates: Sequence[Dict[str, Any]],
    ) -> List[str]:
        ordered = sorted(
            [dict(item) for item in confidence_candidates if item.get("sequence")],
            key=lambda item: float(item.get("confidence_score", float("-inf"))),
            reverse=True,
        )
        max_candidates = max(int(self.config.proposal_confidence_max_candidates), 1)
        return list(dict.fromkeys(str(item["sequence"]) for item in ordered))[:max_candidates]

    def _init_current_sequence_logits(
        self,
        current_seq: str,
        *,
        mutable_mask: Optional[Sequence[bool]],
        proposal_logits: Optional[torch.Tensor] = None,
        preserve_proposal_logits: bool = False,
    ) -> torch.Tensor:
        if proposal_logits is not None:
            current_seq = str(current_seq)
            if preserve_proposal_logits or (
                sequence_from_logits_argmax(
                    proposal_logits,
                    current_seq=current_seq,
                    mutable_mask=mutable_mask,
                )
                == current_seq
            ):
                logits = proposal_logits.detach().clone().to(device=self.device)
            else:
                logits = anchor_binder_logits_to_sequence(
                    proposal_logits,
                    current_seq,
                    mutable_mask=mutable_mask,
                ).to(device=self.device)
            return logits.unsqueeze(0).detach().clone().requires_grad_(True)

        logits = init_binder_logits(
            current_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            random_scale=0.0,
            device=self.device,
        )
        with torch.no_grad():
            from proteocraft.hallucination.esmfold2_backend import AA_TO_ESM_IDX, CYS_INDEX

            for pos, aa in enumerate(current_seq):
                if aa in AA_TO_ESM_IDX and AA_TO_ESM_IDX[aa] != CYS_INDEX:
                    logits[:, pos, AA_TO_ESM_IDX[aa]] = 2.0
        return logits

    def _sequence_to_design_one_hot(self, sequence: str) -> torch.Tensor:
        indices = [AA_TO_ESM_IDX[aa] for aa in sequence]
        return F.one_hot(
            torch.tensor(indices, device=self.device),
            num_classes=len(ESM_AA_ORDER),
        ).unsqueeze(0).float()

    @measured("proposal_forward")
    def _fold_distogram(
        self,
        *,
        target_seq: Union[str, Sequence[str]],
        design: torch.Tensor,
        seed: Optional[int],
        calculate_confidence: bool = False,
        num_loops: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        capture_diffusion_state: bool = False,
    ) -> dict[str, Any]:
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, prepare_esmfold2_input

        targets = _target_sequences(target_seq)
        if not targets:
            raise ValueError("ESMFold2 target sequence is empty")
        if len(targets) + 1 > len(CHAIN_IDS):
            raise ValueError("Too many chains for first ESMFold2 backend")
        target_chain_roles = _normalize_target_chain_roles(
            self.config.target_chain_roles,
            n_target_chains=len(targets),
        )
        target_one_hot = torch.cat(
            [self._sequence_to_folding_one_hot(sequence) for sequence in targets],
            dim=1,
        )
        padded_design = F.pad(design, (2, 11), mode="constant", value=0.0)
        token_ids = torch.argmax(padded_design, dim=-1)
        token_to_aa = self._folding_tokens_to_aa()
        designed_seq = "".join(token_to_aa[int(token.item())] for token in token_ids[0])
        input_raw = StructurePredictionInput(
            sequences=[
                *(
                    ProteinInput(id=CHAIN_IDS[idx], sequence=sequence, msa=None)
                    for idx, sequence in enumerate(targets)
                ),
                ProteinInput(id=CHAIN_IDS[len(targets)], sequence=designed_seq, msa=None),
            ]
        )
        with _seed_context(seed):
            features, _ = prepare_esmfold2_input(input_raw, seed=seed)
            inputs = {
                key: value.unsqueeze(0).to(self.device)
                for key, value in features.items()
                if isinstance(value, torch.Tensor)
            }
            inputs["res_type_soft"] = torch.cat((target_one_hot, padded_design), dim=1)

            if capture_diffusion_state:
                with capture_experimental_trunk_state(
                    self.experimental_model
                ) as captured:
                    output = self.experimental_model(
                        **inputs,
                        num_diffusion_samples=1,
                        num_sampling_steps=int(
                            num_sampling_steps or self.config.num_sampling_steps
                        ),
                        num_loops=int(num_loops or self.config.num_loops),
                        calculate_confidence=bool(calculate_confidence),
                        seed=seed,
                    )
            else:
                captured = {}
                output = self.experimental_model(
                    **inputs,
                    num_diffusion_samples=1,
                    num_sampling_steps=int(
                        num_sampling_steps or self.config.num_sampling_steps
                    ),
                    num_loops=int(num_loops or self.config.num_loops),
                    calculate_confidence=bool(calculate_confidence),
                    seed=seed,
                )
        result: dict[str, Any] = {"distogram_logits": output["distogram_logits"]}
        if capture_diffusion_state:
            result["_features"] = inputs
            result["_captured"] = dict(captured)
        for key in ("iptm", "ptm", "plddt", "pair_chain_iptm"):
            if key in output:
                result[key] = output.get(key)
        if "pair_chains_iptm" in output:
            result["pair_chain_iptm"] = _mean_target_to_binder_pair_iptm(
                output.get("pair_chains_iptm"),
                n_target_chains=len(targets),
                target_chain_roles=target_chain_roles,
            )
            scaffold_pair_chain = _mean_target_to_binder_pair_iptm(
                output.get("pair_chains_iptm"),
                n_target_chains=len(targets),
                target_chain_roles=target_chain_roles,
                selected_role="scaffold",
            )
            if scaffold_pair_chain is not None:
                result["scaffold_binder_pair_chain_iptm"] = scaffold_pair_chain
        if self.config.target_interface_mode == "contacted" and calculate_confidence:
            result["pair_chain_iptm_legacy_all_target_mean"] = result.get("pair_chain_iptm")
            result.update(self._contacted_summary_from_raw(output, inputs, targets, design.shape[1]))
        return result

    @lru_cache(maxsize=1)
    def _folding_tokens(self) -> tuple[str, ...]:
        from esm.models.esmfold2.constants import RES_TYPE_TO_CCD

        return ("<pad>", "-") + tuple(RES_TYPE_TO_CCD[i] for i in range(2, 33))

    @lru_cache(maxsize=1)
    def _folding_tokens_to_aa(self) -> dict[int, str]:
        from esm.models.esmfold2.constants import PROTEIN_3TO1

        return {
            idx: PROTEIN_3TO1[token]
            for idx, token in enumerate(self._folding_tokens())
            if token in PROTEIN_3TO1
        }

    def _sequence_to_folding_one_hot(self, sequence: str) -> torch.Tensor:
        from esm.models.esmfold2.constants import PROTEIN_1TO3

        token_to_idx = {token: idx for idx, token in enumerate(self._folding_tokens())}
        indices = [token_to_idx[PROTEIN_1TO3[aa]] for aa in sequence]
        one_hot = F.one_hot(
            torch.tensor(indices, device=self.device),
            num_classes=len(self._folding_tokens()),
        )
        return one_hot.unsqueeze(0).float()

    @measured("esmc_prior_forward")
    def _compute_esmc_pseudoperplexity(
        self,
        binder_design: torch.Tensor,
        *,
        score_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.esmc_model is None:
            raise RuntimeError("ESMC model is required when lambda_esmc is non-zero")
        if self.optimizations.memory.active:
            self.optimizations.memory.prepare(
                "prior", self.optimizations.memory.total_tokens,
                {"verifier": self.verifier_model, "proposal": self.experimental_model,
                 "prior": self.esmc_model})
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

        device = binder_design.device
        lm_vocab_size = self.esmc_model.config.vocab_size
        model_dtype = self.esmc_model.esmc.embed.weight.dtype
        target_esm = binder_design @ self._folding_to_lm_aa_matrix(device)
        discrete = F.one_hot(torch.argmax(target_esm, dim=-1), num_classes=target_esm.size(-1))
        input_esm = target_esm + (discrete.to(target_esm.dtype) - target_esm).detach()

        input_ids = torch.zeros(
            (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
            dtype=model_dtype,
            device=device,
        )
        tokenizer = ESMCTokenizer()
        input_ids[:, 0, tokenizer.cls_token_id] = 1
        input_ids[:, -1, tokenizer.eos_token_id] = 1
        input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

        if score_mask.ndim == 1:
            score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
        score_mask = score_mask.to(device=device, dtype=torch.bool)
        mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
        mask_token[self.esmc_model.config.mask_token_id] = 1

        losses = []
        for batch_idx in range(binder_design.size(0)):
            position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if position_indices.numel() == 0:
                raise ValueError("ESMC score mask selected zero positions")
            n_passes = int(self.config.esmc_n_passes)
            num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * int(position_indices.numel())))
            random_scores = torch.rand((n_passes, position_indices.numel()), device=device)
            masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
            pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
            pass_masks[
                torch.arange(n_passes, device=device)[:, None],
                position_indices[masked_offsets],
            ] = True
            masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
            mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
            masked_sequences[mask_rows, mask_cols + 1] = mask_token
            target_weights = target_esm[batch_idx]
            masked_nlls = []
            for start in range(0, n_passes, int(self.config.esmc_batch_size)):
                stop = min(start + int(self.config.esmc_batch_size), n_passes)
                chunk = masked_sequences[start:stop]
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    embedded = chunk @ self.esmc_model.esmc.embed.weight.to(
                        chunk.dtype
                    )
                    if self.config.esmc_activation_checkpointing:
                        from torch.utils.checkpoint import checkpoint

                        transformer = self.esmc_model.esmc.transformer
                        hidden = embedded
                        for block in transformer.blocks:
                            def block_forward(
                                value: torch.Tensor,
                                current_block: Any = block,
                            ) -> torch.Tensor:
                                return current_block(
                                    value,
                                    None,
                                    output_attentions=False,
                                )[0]

                            hidden = checkpoint(
                                block_forward,
                                hidden,
                                use_reentrant=False,
                            )
                        hidden = transformer.norm(hidden)
                    else:
                        hidden, *_ = self.esmc_model.esmc.transformer(
                            embedded,
                            sequence_id=None,
                            layers_to_collect=[],
                            output_attentions=False,
                        )
                    logits = self.esmc_model.lm_head(hidden)
                log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
                nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
                masked_nlls.append(nlls[pass_masks[start:stop]])
            losses.append(torch.cat(masked_nlls, dim=0).mean())
        return torch.stack(losses, dim=0)

    def _folding_to_lm_aa_matrix(self, device: torch.device) -> torch.Tensor:
        from esm.models.esmfold2.constants import PROTEIN_1TO3
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

        three_to_one = {value: key for key, value in PROTEIN_1TO3.items()}
        folding_aas = [three_to_one[token] for token in self._folding_tokens()[2:22]]
        lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda item: item[1])
        lm_aas = [lm_vocab[i][0] for i in range(4, 24)]
        matrix = torch.zeros(20, 20, device=device)
        for folding_idx, aa in enumerate(folding_aas):
            matrix[folding_idx, lm_aas.index(aa)] = 1.0
        return matrix
