"""Differentiable one-step diffusion support for the optional ESMFold2 backend.

This module is deliberately ESMFold2-specific.  It does not import the ESM or
Transformers packages at module import time, and it does not alter the existing
Protenix one-step diffusion implementation.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from proteocraft.losses.geometry import compute_rg_loss, compute_soft_clash_loss


# Fixed Biohub ESMFold2 atom-feature vocabulary sizes. Keeping these lightweight
# constants local lets unit tests exercise diffusion orchestration without
# importing the optional Transformers fork.
ESMFOLD2_CHAR_VOCAB_SIZE = 64
ESMFOLD2_MAX_ATOMIC_NUMBER = 128


# Common subset of the tuned Protenix verifier weights.  ESMFold2 does not
# expose PDE logits, so pde/i_pde are intentionally absent.
ESMFOLD2_COMMON_LOSS_WEIGHTS: Dict[str, float] = {
    "plddt": 0.1,
    "pae": 0.4,
    "i_pae": 0.2,
    "con": 0.2,
    "i_con": 3.0,
    "iptm": 0.3,
    "rg": 0.3,
    "helix": -0.3,
    "clash": 0.1,
    "i_clash": 0.1,
}

# Preserve the previous Protenix-style verifier score as a diagnostic.  Its raw
# Rg, helix, and contact terms are not scale-compatible with bounded ESMFold2
# confidence metrics, so this score must not drive MCTS selection.
ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS: Dict[str, float] = {
    **ESMFOLD2_COMMON_LOSS_WEIGHTS,
    "i_con": 0.5,
    "iptm": 1.0,
}

# Full ESMFold2 verification uses bounded terms only.  This keeps pair-chain
# iPTM primary while retaining small fold-confidence and compactness guards.
ESMFOLD2_VERIFIER_LOSS_WEIGHTS: Dict[str, float] = {
    "pair_chain_iptm_loss": 1.0,
    "binder_confidence_loss": 0.2,
    "global_confidence_loss": 0.1,
    "binder_rg_excess_loss": 0.1,
}

ESMFOLD2_TRUNCATED_SCHEDULE_MODES = {
    "official_clipped",
    "dense",
}


def compute_esmfold2_bounded_verifier_losses(
    summary_confidence: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[float, Dict[str, float]]:
    """Build a bounded ESMFold2 verifier score from full-prediction metrics."""

    resolved_weights = dict(ESMFOLD2_VERIFIER_LOSS_WEIGHTS)
    if weights is not None:
        resolved_weights.update({str(key): float(value) for key, value in weights.items()})

    def finite_metric(name: str) -> float:
        value = summary_confidence.get(name)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                raise ValueError(f"ESMFold2 verifier metric {name!r} is empty")
            value = value.detach().float().mean().item()
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ESMFold2 verifier metric {name!r} is missing or non-numeric"
            ) from exc
        if not math.isfinite(scalar):
            raise ValueError(f"ESMFold2 verifier metric {name!r} is not finite")
        return scalar

    pair_chain_iptm = min(max(finite_metric("pair_chain_iptm"), 0.0), 1.0)
    binder_plddt = finite_metric("binder_plddt")
    if binder_plddt > 1.0:
        binder_plddt /= 100.0
    binder_plddt = min(max(binder_plddt, 0.0), 1.0)
    ptm = min(max(finite_metric("ptm"), 0.0), 1.0)
    rg_excess = min(max(finite_metric("binder_rg_excess"), 0.0), 1.0)

    losses = {
        "pair_chain_iptm_loss": 1.0 - pair_chain_iptm,
        "binder_confidence_loss": 1.0 - binder_plddt,
        "global_confidence_loss": 1.0 - ptm,
        "binder_rg_excess_loss": rg_excess,
    }
    total = sum(
        float(resolved_weights.get(name, 0.0)) * value
        for name, value in losses.items()
    )
    losses["total"] = float(total)
    return float(total), losses


def _batch0(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    return tensor[0]


def _sample0_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim == 4:
        return coordinates[0, 0]
    if coordinates.ndim == 3:
        return coordinates[0]
    if coordinates.ndim == 2:
        return coordinates
    raise ValueError(
        "coordinates must have shape [A,3], [B,A,3], or [B,S,A,3]"
    )


def _categorical_mean(
    logits: torch.Tensor,
    *,
    start: float,
    end: float,
) -> torch.Tensor:
    bins = torch.linspace(
        float(start),
        float(end),
        logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )
    return (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)


def _normalize_target_roles(
    roles: Optional[Sequence[str]],
    *,
    n_target_chains: int,
) -> tuple[str, ...]:
    if roles is None:
        return ("target",) * n_target_chains
    normalized = tuple(str(role).strip().lower() for role in roles)
    if len(normalized) != n_target_chains:
        raise ValueError(
            "target_chain_roles length must match the number of target chains"
        )
    if set(normalized) - {"target", "scaffold"}:
        raise ValueError("target_chain_roles only supports target/scaffold")
    if "target" not in normalized:
        raise ValueError("target_chain_roles must contain at least one target")
    return normalized


def _is_tcr_task(task_type: Optional[str]) -> bool:
    return str(task_type or "").strip().lower().replace("-", "_") == "tcr"


def _token_masks(
    features: Mapping[str, torch.Tensor],
    *,
    binder_length: int,
    binder_mutable_mask: Optional[Sequence[bool]],
    target_chain_roles: Optional[Sequence[str]],
) -> Dict[str, torch.Tensor]:
    token_mask = _batch0(features["token_attention_mask"]).bool()
    asym_id = _batch0(features["asym_id"]).long()
    valid_asym = asym_id[token_mask]
    if valid_asym.numel() == 0:
        raise ValueError("ESMFold2 features contain no valid tokens")

    binder_chain_id = valid_asym[-1]
    binder_mask = token_mask & (asym_id == binder_chain_id)
    if int(binder_mask.sum().item()) != int(binder_length):
        raise ValueError(
            "ESMFold2 binder token count does not match binder sequence length"
        )

    chain_ids = [
        int(chain_id)
        for chain_id in torch.unique(valid_asym, sorted=True).tolist()
        if int(chain_id) != int(binder_chain_id.item())
    ]
    roles = _normalize_target_roles(
        target_chain_roles,
        n_target_chains=len(chain_ids),
    )
    target_mask = torch.zeros_like(token_mask)
    target_context_mask = torch.zeros_like(token_mask)
    for chain_id, role in zip(chain_ids, roles):
        if role == "target":
            target_mask |= token_mask & (asym_id == chain_id)
        else:
            target_context_mask |= token_mask & (asym_id == chain_id)

    design_mask = binder_mask.clone()
    if binder_mutable_mask is not None:
        mutable = torch.as_tensor(
            binder_mutable_mask,
            dtype=torch.bool,
            device=binder_mask.device,
        )
        if mutable.numel() != binder_length:
            raise ValueError(
                "binder_mutable_mask length must match binder sequence length"
            )
        design_mask.zero_()
        design_mask[binder_mask.nonzero(as_tuple=True)[0]] = mutable
        if not design_mask.any():
            raise ValueError("binder_mutable_mask selects no residues")

    binder_scaffold_mask = binder_mask & ~design_mask
    scaffold_mask = target_context_mask | binder_scaffold_mask
    return {
        "token": token_mask,
        "asym_id": asym_id,
        "binder": binder_mask,
        "design": design_mask,
        "target": target_mask,
        "target_context": target_context_mask,
        "all_target": target_mask | target_context_mask,
        "binder_scaffold": binder_scaffold_mask,
        "scaffold": scaffold_mask,
    }


def _hotspot_token_mask(
    masks: Mapping[str, torch.Tensor],
    hotspot_params: Optional[Mapping[str, Any]],
    *,
    task_type: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Map target-concatenated hotspot indices onto ESMFold2 token indices."""

    if hotspot_params is None:
        return None
    concat_indices = hotspot_params.get("target_hotspot_concat_indices") or []
    if not concat_indices:
        return None

    # TCR hotspot indices refer to the original alpha+beta concatenation, even
    # when only one chain has role=target. Other tasks retain the established
    # target-only indexing contract.
    hotspot_index_mask = (
        masks["all_target"] if _is_tcr_task(task_type) else masks["target"]
    )
    target_indices = hotspot_index_mask.nonzero(as_tuple=True)[0]
    normalized = [int(index) for index in concat_indices]
    invalid = [
        index
        for index in normalized
        if index < 0 or index >= int(target_indices.numel())
    ]
    if invalid:
        raise ValueError(
            "hotspot target concat indices out of range: "
            f"{invalid}; target_len={int(target_indices.numel())}"
        )

    hotspot_mask = torch.zeros_like(masks["target"], dtype=torch.bool)
    hotspot_indices = torch.as_tensor(
        normalized,
        device=target_indices.device,
        dtype=torch.long,
    )
    hotspot_token_indices = target_indices[hotspot_indices]
    invalid_roles = hotspot_token_indices[
        ~masks["target"][hotspot_token_indices]
    ]
    if invalid_roles.numel() > 0:
        raise ValueError(
            "hotspot target concat indices must belong to a chain with "
            "role=target"
        )
    hotspot_mask[hotspot_token_indices] = True
    return hotspot_mask


def _differentiable_pair_chain_iptm(
    pae_logits: torch.Tensor,
    *,
    masks: Mapping[str, torch.Tensor],
    conservative_bidirectional: bool = False,
) -> torch.Tensor:
    """Compute the differentiable PAE-based target-binder iPTM surrogate.

    The historical path retains its all-token normalization and one direction.
    Contacted-interface mode uses each selected chain pair's own normalization
    and the weaker directional mean.  Neither path substitutes for the model's
    native pair-chain iPTM values used by the full verifier.
    """

    logits = pae_logits
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    valid = masks["token"].to(logits.device)
    target = masks["target"].to(logits.device)
    binder = masks["binder"].to(logits.device)
    asym_id = masks["asym_id"].to(logits.device)

    n_bins = logits.shape[-1]
    bin_width = 32.0 / n_bins
    bin_centers = torch.arange(
        0.5 * bin_width,
        32.0,
        bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    if conservative_bidirectional:
        scores = []
        for chain_id in torch.unique(asym_id[target], sorted=True):
            chain_mask = target & (asym_id == chain_id)
            # Normalize by this chain pair and evaluate only its two PAE
            # blocks, avoiding extra full-trimer softmax tensors per chain.
            pair_n_res = (chain_mask.sum() + binder.sum()).clamp(min=19).to(
                logits.dtype
            )
            pair_d0 = 1.24 * (pair_n_res - 15.0).pow(1.0 / 3.0) - 1.8
            pair_tm_per_bin = 1.0 / (
                1.0 + (bin_centers / pair_d0) ** 2
            )
            forward_logits = logits[:, chain_mask, :, :][:, :, binder, :]
            reverse_logits = logits[:, binder, :, :][:, :, chain_mask, :]
            forward = (
                torch.softmax(forward_logits, dim=-1) * pair_tm_per_bin
            ).sum(dim=-1).mean(dim=(-1, -2))
            reverse = (
                torch.softmax(reverse_logits, dim=-1) * pair_tm_per_bin
            ).sum(dim=-1).mean(dim=(-1, -2))
            scores.append(torch.minimum(forward, reverse))
        if not scores:
            raise ValueError("No target-binder chain pair is available for iPTM")
        return torch.stack(scores, dim=0).mean()

    n_res = valid.sum().clamp(min=19).to(logits.dtype)
    d0 = 1.24 * (n_res - 15.0).pow(1.0 / 3.0) - 1.8
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    tm_expected = (torch.softmax(logits, dim=-1) * tm_per_bin).sum(dim=-1)

    scores = []
    target_chain_ids = torch.unique(asym_id[target], sorted=True)
    for chain_id in target_chain_ids:
        chain_mask = target & (asym_id == chain_id)
        pair_mask = chain_mask.unsqueeze(-1) & binder.unsqueeze(-2)
        denom = pair_mask.sum().clamp(min=1).to(tm_expected.dtype)
        scores.append(
            (tm_expected * pair_mask.unsqueeze(0).to(tm_expected.dtype)).sum(
                dim=(-1, -2)
            )
            / denom
        )
    if not scores:
        raise ValueError("No target-binder chain pair is available for iPTM")
    return torch.stack(scores, dim=0).mean()


def _helix_loss(
    distogram_logits: torch.Tensor,
    *,
    design_mask: torch.Tensor,
    residue_index: torch.Tensor,
    target_interface_mode: str = "all",
) -> torch.Tensor:
    logits = distogram_logits[0] if distogram_logits.ndim == 4 else distogram_logits
    if target_interface_mode == "contacted":
        from proteocraft.hallucination.esmfold2_interface import interface_distogram_bins

        bin_centers = interface_distogram_bins(
            logits.shape[-1], device=logits.device, dtype=logits.dtype
        )
    else:
        boundaries = torch.linspace(
            2.0,
            52.0,
            logits.shape[-1] - 1,
            device=logits.device,
            dtype=logits.dtype,
        )
        expanded = torch.cat(
            (
                torch.tensor([1.0], device=logits.device, dtype=logits.dtype),
                boundaries,
                torch.tensor([57.0], device=logits.device, dtype=logits.dtype),
            )
        )
        bin_centers = (expanded[:-1] + expanded[1:]) / 2.0
    contact_probability = (
        torch.softmax(logits, dim=-1)
        * (bin_centers < 6.0).to(logits.dtype)
    ).sum(dim=-1)
    pair_loss = -torch.log(contact_probability + 1e-8)

    indices = design_mask.nonzero(as_tuple=True)[0]
    if indices.numel() < 4:
        return logits.sum() * 0.0
    design_residue_index = residue_index[indices]
    offset = design_residue_index.unsqueeze(1) - design_residue_index.unsqueeze(0)
    pairs = (offset == 3).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return logits.sum() * 0.0
    return pair_loss[indices[pairs[:, 1]], indices[pairs[:, 0]]].mean()


def compute_esmfold2_common_losses(
    *,
    coordinates: torch.Tensor,
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    distogram_logits: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    binder_length: int,
    task_type: Optional[str] = None,
    target_chain_roles: Optional[Sequence[str]] = None,
    target_interface_mode: str = "all",
    target_interface_contact_cutoff: float = 5.0,
    target_interface_min_contacts: int = 5,
    binder_mutable_mask: Optional[Sequence[bool]] = None,
    distogram_losses: Optional[Mapping[str, torch.Tensor]] = None,
    weights: Optional[Mapping[str, float]] = None,
    hotspot_params: Optional[Mapping[str, Any]] = None,
    hotspot_weight: float = 0.0,
    hotspot_interface_mode: str = "localized",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the Protenix-aligned loss subset available from ESMFold2."""

    resolved_weights = dict(ESMFOLD2_COMMON_LOSS_WEIGHTS)
    if weights is not None:
        resolved_weights.update({str(k): float(v) for k, v in weights.items()})

    masks = _token_masks(
        features,
        binder_length=binder_length,
        binder_mutable_mask=binder_mutable_mask,
        target_chain_roles=target_chain_roles,
    )
    device = coordinates.device
    masks = {key: value.to(device) for key, value in masks.items()}
    resolved_interface_mode = str(target_interface_mode).strip().lower()
    if resolved_interface_mode not in {"all", "contacted"}:
        raise ValueError("target_interface_mode must be 'all' or 'contacted'")
    interface_target_mask = masks["target"]
    interface_selection = None
    if resolved_interface_mode == "contacted":
        from proteocraft.hallucination.esmfold2_interface import (
            select_distogram_interface,
        )

        chain_ids = torch.unique(
            masks["asym_id"][masks["target"]], sorted=True
        )
        chain_masks = [
            masks["target"] & (masks["asym_id"] == chain_id)
            for chain_id in chain_ids
        ]
        interface_selection = select_distogram_interface(
            distogram_logits,
            chain_masks=chain_masks,
            binder_mask=masks["binder"],
            cutoff=target_interface_contact_cutoff,
            min_contacts=target_interface_min_contacts,
        )
        interface_target_mask = torch.zeros_like(masks["target"])
        for index in interface_selection["selected_indices"]:
            interface_target_mask |= chain_masks[index]
    # Preserve the original masks for hotspot indexing and clash checks. Only
    # interface-confidence objectives omit non-contacting target protomers.
    hotspot_mask = _hotspot_token_mask(
        masks,
        hotspot_params,
        task_type=task_type,
    )
    resolved_hotspot_weight = float(hotspot_weight)
    resolved_hotspot_mode = str(hotspot_interface_mode).strip().lower()
    if resolved_hotspot_weight < 0.0:
        raise ValueError("hotspot_weight must be non-negative")
    if resolved_hotspot_mode not in {"additive", "localized"}:
        raise ValueError(
            "hotspot_interface_mode must be 'additive' or 'localized'"
        )
    if resolved_hotspot_mode == "localized" and resolved_hotspot_weight > 1.0:
        raise ValueError(
            "localized hotspot_interface_mode requires hotspot_weight in [0, 1]"
        )

    atom_mask = _batch0(features["atom_attention_mask"]).bool().to(device)
    atom_to_token = _batch0(features["atom_to_token"]).long().to(device)
    residue_index = _batch0(features["residue_index"]).long().to(device)
    rep_idx = _batch0(features["distogram_atom_idx"]).long().to(device)
    coords = _sample0_coordinates(coordinates)
    if coords.shape[0] != atom_mask.shape[0]:
        raise ValueError("coordinates and atom_attention_mask have different sizes")

    design_atom_mask = atom_mask & masks["design"][atom_to_token]
    binder_atom_mask = atom_mask & masks["binder"][atom_to_token]
    target_atom_mask = atom_mask & masks["target"][atom_to_token]
    all_target_atom_mask = atom_mask & masks["all_target"][atom_to_token]
    scaffold_atom_mask = atom_mask & masks["scaffold"][atom_to_token]
    local_redesign = bool((masks["design"] != masks["binder"]).any().item())
    has_scaffold = bool(masks["scaffold"].any().item())

    losses: Dict[str, torch.Tensor] = {}
    if interface_selection is not None:
        losses["interface_selected_chain_count"] = coords.new_tensor(
            len(interface_selection["selected_indices"])
        )
        losses["interface_selection_fallback"] = coords.new_tensor(
            float(interface_selection["fallback"])
        )
    plddt = _categorical_mean(plddt_logits, start=0.0, end=1.0)
    plddt0 = _batch0(plddt)
    plddt_indices = rep_idx[masks["design"] if local_redesign else masks["binder"]]
    losses["plddt"] = 1.0 - plddt0[plddt_indices].mean()

    pae = _categorical_mean(pae_logits, start=0.0, end=32.0) / 32.0
    pae = 0.5 * (pae + pae.transpose(-1, -2))
    design = masks["design"]
    binder = masks["binder"]
    # In TCR design, the non-selected alpha/beta chain is target context, not
    # the binder's immutable framework. Keep it in the full structure and clash
    # masks, but do not ask the binder-internal PAE objective to optimize it.
    pae_scaffold_mask = (
        masks["binder_scaffold"]
        if _is_tcr_task(task_type)
        else masks["scaffold"]
    )
    intra_pair_mask = (
        (design.unsqueeze(-1) & design.unsqueeze(-2))
        | (design.unsqueeze(-1) & pae_scaffold_mask.unsqueeze(-2))
    )
    inter_pair_mask = design.unsqueeze(-1) & interface_target_mask.unsqueeze(-2)

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(0).to(value.dtype)
        return (value * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    losses["pae"] = masked_mean(pae, intra_pair_mask)
    global_i_pae = masked_mean(pae, inter_pair_mask)
    if hotspot_mask is None:
        losses["i_pae"] = global_i_pae
    else:
        hotspot_inter_pair_mask = (
            design.unsqueeze(-1) & hotspot_mask.unsqueeze(-2)
        )
        hotspot_i_pae = masked_mean(pae, hotspot_inter_pair_mask)
        losses["i_pae_global"] = global_i_pae
        losses["i_pae_hotspot"] = hotspot_i_pae
        if resolved_hotspot_mode == "localized":
            losses["i_pae"] = (
                (1.0 - resolved_hotspot_weight) * global_i_pae
                + resolved_hotspot_weight * hotspot_i_pae
            )
        else:
            losses["i_pae"] = (
                global_i_pae + resolved_hotspot_weight * hotspot_i_pae
            )
    losses["iptm"] = 1.0 - _differentiable_pair_chain_iptm(
        pae_logits,
        masks={**masks, "target": interface_target_mask},
        conservative_bidirectional=resolved_interface_mode == "contacted",
    )

    if distogram_losses is not None:
        if "intra_contact_loss" in distogram_losses:
            losses["con"] = distogram_losses["intra_contact_loss"].mean()
        if "inter_contact_loss" in distogram_losses:
            losses["i_con"] = distogram_losses["inter_contact_loss"].mean()

    rep_coords = coords[rep_idx]
    rg_mask = design if local_redesign else binder
    losses["rg"] = compute_rg_loss(
        rep_coords[rg_mask],
        int(rg_mask.sum().item()),
    )
    losses["helix"] = _helix_loss(
        distogram_logits,
        design_mask=design if local_redesign else binder,
        residue_index=residue_index,
        target_interface_mode=resolved_interface_mode,
    )

    if _is_tcr_task(task_type):
        if local_redesign:
            binder_scaffold_atom_mask = (
                atom_mask & masks["binder_scaffold"][atom_to_token]
            )
            clash_a = design_atom_mask
            clash_b = design_atom_mask | binder_scaffold_atom_mask
        else:
            clash_a = binder_atom_mask
            clash_b = binder_atom_mask
        interface_clash_target = all_target_atom_mask
    else:
        clash_a = (
            design_atom_mask
            if (local_redesign or has_scaffold)
            else binder_atom_mask
        )
        clash_b = (
            design_atom_mask | scaffold_atom_mask
            if (local_redesign or has_scaffold)
            else binder_atom_mask
        )
        interface_clash_target = target_atom_mask
    losses["clash"] = compute_soft_clash_loss(
        coords,
        binder_atom_mask,
        target_atom_mask,
        atom_to_token,
        residue_index,
        mode="intra",
        atom_mask_a=clash_a,
        atom_mask_b=clash_b,
        exclude_bonded=True,
    )
    losses["i_clash"] = compute_soft_clash_loss(
        coords,
        binder_atom_mask,
        interface_clash_target,
        atom_to_token,
        residue_index,
        mode="inter",
        atom_mask_a=design_atom_mask if (local_redesign or has_scaffold) else binder_atom_mask,
        atom_mask_b=interface_clash_target,
        detach_b=True,
        exclude_bonded=False,
    )

    total = coords.sum() * 0.0
    for name, weight in resolved_weights.items():
        if name in losses and weight != 0.0:
            total = total + float(weight) * losses[name]
    losses["total"] = total
    return total, losses


@contextmanager
def capture_experimental_trunk_state(model: Any) -> Iterator[Dict[str, torch.Tensor]]:
    """Capture differentiable intermediates from an unmodified Experimental model."""

    captured: Dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                captured[name] = tensor

        return hook

    handles = [
        model.inputs_embedder.register_forward_hook(save("s_inputs")),
        model.rel_pos.register_forward_hook(save("relative_position_encoding")),
        model.token_bonds.register_forward_hook(save("token_bonds_encoding")),
        model.folding_trunk.register_forward_hook(save("z_trunk")),
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def scatter_reference_coordinates(
    coordinates: torch.Tensor,
    *,
    atom_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter decoded real-atom coordinates into ESMFold2's padded atom layout."""

    atom_mask = atom_attention_mask.bool()
    if atom_mask.ndim == 1:
        atom_mask = atom_mask.unsqueeze(0)
    source = coordinates
    if source.ndim == 2:
        source = source.unsqueeze(0)
    if source.ndim != 3 or source.shape[0] != atom_mask.shape[0]:
        raise ValueError("reference coordinates must have shape [B,A,3] or [A,3]")

    padded = torch.zeros(
        atom_mask.shape[0],
        atom_mask.shape[1],
        3,
        device=source.device,
        dtype=source.dtype,
    )
    for batch_index in range(atom_mask.shape[0]):
        expected = int(atom_mask[batch_index].sum().item())
        if source.shape[1] == atom_mask.shape[1]:
            padded[batch_index] = source[batch_index]
        elif source.shape[1] == expected:
            padded[batch_index, atom_mask[batch_index]] = source[batch_index]
        else:
            raise ValueError(
                "reference coordinate count does not match padded or valid atom "
                f"count: source={source.shape[1]}, "
                f"proposal_padded={atom_mask.shape[1]}, proposal_valid={expected}"
            )
    return padded


def remap_reference_coordinates(
    coordinates: torch.Tensor,
    *,
    source_features: Mapping[str, torch.Tensor],
    destination_features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Map parent coordinates into a child layout and rebuild new side chains."""

    source_coords = _sample0_coordinates(coordinates)

    def feature0(
        features: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        value = features[name]
        return value[0] if value.ndim > 1 and value.shape[0] == 1 else value

    def atom_keys(
        features: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, tuple[int, ...], int]],
        torch.Tensor,
    ]:
        mask = feature0(features, "atom_attention_mask").bool()
        atom_to_token = feature0(features, "atom_to_token").long()
        atom_names = feature0(features, "ref_atom_name_chars").long()
        elements = feature0(features, "ref_element").long()
        ref_pos = feature0(features, "ref_pos").float()
        if not (
            mask.shape[0]
            == atom_to_token.shape[0]
            == atom_names.shape[0]
            == elements.shape[0]
            == ref_pos.shape[0]
        ):
            raise ValueError("ESMFold2 atom identity features have inconsistent sizes")
        keys = [
            (
                int(atom_to_token[index].item()),
                tuple(int(value) for value in atom_names[index].tolist()),
                int(elements[index].item()),
            )
            for index in mask.nonzero(as_tuple=True)[0].tolist()
        ]
        return mask, keys, ref_pos[mask]

    source_mask, source_keys, _ = atom_keys(source_features)
    destination_mask, destination_keys, destination_ref_pos = atom_keys(
        destination_features
    )
    if source_coords.shape[0] == source_mask.shape[0]:
        source_valid_coords = source_coords[
            source_mask.to(device=source_coords.device)
        ]
    elif source_coords.shape[0] == int(source_mask.sum().item()):
        source_valid_coords = source_coords
    else:
        raise ValueError(
            "source coordinate count does not match verifier atom layout: "
            f"source={source_coords.shape[0]}, padded={source_mask.shape[0]}, "
            f"valid={int(source_mask.sum().item())}"
        )

    coordinate_by_key: dict[tuple[int, tuple[int, ...], int], torch.Tensor] = {}
    for key, coordinate in zip(source_keys, source_valid_coords, strict=True):
        if key in coordinate_by_key:
            raise ValueError(f"duplicate verifier atom identity: {key}")
        coordinate_by_key[key] = coordinate

    def atom_name(key: tuple[int, tuple[int, ...], int]) -> str:
        return "".join(
            chr(value + 32) if value > 0 else " " for value in key[1]
        ).strip()

    source_by_token_and_name = {
        (key[0], atom_name(key)): coordinate
        for key, coordinate in coordinate_by_key.items()
    }
    destination_ref_by_token_and_name = {
        (key[0], atom_name(key)): ref_pos
        for key, ref_pos in zip(
            destination_keys,
            destination_ref_pos,
            strict=True,
        )
    }

    def rigid_transform(
        moving: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moving_center = moving.mean(dim=0)
        target_center = target.mean(dim=0)
        covariance = (moving - moving_center).transpose(0, 1) @ (
            target - target_center
        )
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.eye(3, device=moving.device, dtype=moving.dtype)
        correction[-1, -1] = torch.sign(torch.det(u @ vh))
        rotation = u @ correction @ vh
        translation = target_center - moving_center @ rotation
        return rotation, translation

    rebuilt_by_key: dict[tuple[int, tuple[int, ...], int], torch.Tensor] = {}
    missing_by_token: dict[int, list[tuple[int, tuple[int, ...], int]]] = {}
    for key in destination_keys:
        if key in coordinate_by_key:
            rebuilt_by_key[key] = coordinate_by_key[key]
        else:
            missing_by_token.setdefault(key[0], []).append(key)

    for token_index, missing_keys in missing_by_token.items():
        anchors = []
        for name in ("N", "CA", "C"):
            source_anchor = source_by_token_and_name.get((token_index, name))
            destination_anchor = destination_ref_by_token_and_name.get(
                (token_index, name)
            )
            if source_anchor is not None and destination_anchor is not None:
                anchors.append((destination_anchor, source_anchor))
        if len(anchors) != 3:
            missing_names = ", ".join(atom_name(key) for key in missing_keys[:3])
            raise ValueError(
                "cannot rebuild proposal side-chain coordinates without N/CA/C "
                f"anchors for token={token_index}; missing_atoms=[{missing_names}]"
            )
        moving = torch.stack([item[0] for item in anchors]).to(
            device=source_coords.device,
            dtype=source_coords.dtype,
        )
        target = torch.stack([item[1] for item in anchors]).to(
            device=source_coords.device,
            dtype=source_coords.dtype,
        )
        rotation, translation = rigid_transform(moving, target)
        for key in missing_keys:
            idealized = destination_ref_by_token_and_name[
                (token_index, atom_name(key))
            ].to(device=source_coords.device, dtype=source_coords.dtype)
            rebuilt_by_key[key] = idealized @ rotation + translation

    destination = torch.zeros(
        1,
        destination_mask.shape[0],
        3,
        dtype=source_coords.dtype,
        device=source_coords.device,
    )
    destination[0, destination_mask.to(source_coords.device)] = torch.stack(
        [rebuilt_by_key[key] for key in destination_keys],
        dim=0,
    )
    return destination


def _diffusion_atom_inputs(
    features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_mask = features["atom_attention_mask"].bool()
    atom_mask_f = atom_mask.float()
    ref_element = F.one_hot(
        features["ref_element"].long(),
        num_classes=ESMFOLD2_MAX_ATOMIC_NUMBER,
    ).float()
    ref_element = ref_element * atom_mask_f.unsqueeze(-1)
    ref_atom_name_chars = F.one_hot(
        features["ref_atom_name_chars"].long(),
        num_classes=ESMFOLD2_CHAR_VOCAB_SIZE,
    ).float()
    ref_atom_name_chars = (
        ref_atom_name_chars * atom_mask_f.unsqueeze(-1).unsqueeze(-1)
    )
    atom_to_token = features["atom_to_token"].long() * atom_mask.long()
    return atom_mask, ref_element, ref_atom_name_chars, atom_to_token


def _one_step_confidence(
    *,
    model: Any,
    captured: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    coordinates: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_to_token: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is None:
        raise RuntimeError(
            "ESMFold2 one-step diffusion requires an Experimental model with a confidence head"
        )
    return confidence_head(
        s_inputs=captured["s_inputs"],
        z=captured["z_trunk"].float(),
        x_pred=coordinates,
        distogram_atom_idx=features["distogram_atom_idx"],
        token_attention_mask=features["token_attention_mask"],
        atom_to_token=atom_to_token,
        atom_attention_mask=atom_mask,
        asym_id=features["asym_id"],
        mol_type=features["mol_type"],
        num_diffusion_samples=1,
        relative_position_encoding=captured["relative_position_encoding"],
        token_bonds_encoding=captured.get("token_bonds_encoding"),
    )


def run_esmfold2_one_step_diffusion(
    *,
    model: Any,
    captured: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    reference_coordinates: torch.Tensor,
    tau: float,
    seed: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Run one differentiable ESMFold2 denoiser call at a fixed noise level."""

    if float(tau) <= 0.0:
        raise ValueError("ESMFold2 one-step diffusion tau must be positive")
    required = {"s_inputs", "relative_position_encoding", "z_trunk"}
    missing = sorted(required - set(captured))
    if missing:
        raise RuntimeError(
            f"Failed to capture ESMFold2 proposal intermediates: {missing}"
        )

    atom_mask, ref_element, ref_atom_name_chars, atom_to_token = (
        _diffusion_atom_inputs(features)
    )
    x_clean = scatter_reference_coordinates(
        reference_coordinates.to(
            device=atom_mask.device,
            dtype=torch.float32,
        ),
        atom_attention_mask=atom_mask,
    )
    devices = []
    if x_clean.is_cuda:
        devices = [x_clean.device.index if x_clean.device.index is not None else 0]
    with torch.random.fork_rng(devices=devices, enabled=True):
        if seed is not None:
            torch.manual_seed(int(seed))
            if x_clean.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
        noise = torch.randn_like(x_clean)
    x_noisy = x_clean.detach() + float(tau) * noise

    diffusion_module = model.structure_head.diffusion_module
    diffusion_output = diffusion_module(
        x_noisy=x_noisy,
        t_hat=torch.full(
            (x_clean.shape[0],),
            float(tau),
            device=x_clean.device,
            dtype=torch.float32,
        ),
        ref_pos=features["ref_pos"],
        ref_charge=features["ref_charge"],
        ref_mask=atom_mask,
        ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars,
        ref_space_uid=features["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=captured["s_inputs"],
        s_trunk=None,
        z_trunk=captured["z_trunk"].float(),
        relative_position_encoding=captured["relative_position_encoding"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        entity_id=features["entity_id"],
        token_index=features["token_index"],
        sym_id=features["sym_id"],
        token_attention_mask=features["token_attention_mask"],
        num_diffusion_samples=1,
    )
    x_denoised = diffusion_output["x_denoised"]
    if x_denoised is None:
        raise RuntimeError("ESMFold2 diffusion module returned no denoised coordinates")

    confidence = _one_step_confidence(
        model=model,
        captured=captured,
        features=features,
        coordinates=x_denoised,
        atom_mask=atom_mask,
        atom_to_token=atom_to_token,
    )
    return {
        "coordinates": x_denoised,
        "plddt_logits": confidence["plddt_logits"],
        "pae_logits": confidence["pae_logits"],
    }


def _build_truncated_noise_schedule(
    *,
    structure_head: Any,
    device: torch.device,
    tau: float,
    num_sampling_steps: int,
    max_inference_sigma: float,
    schedule_mode: str,
    dense_num_sampling_steps: int,
) -> torch.Tensor:
    mode = str(schedule_mode).strip().lower()
    if mode not in ESMFOLD2_TRUNCATED_SCHEDULE_MODES:
        raise ValueError(
            "truncated schedule_mode must be 'official_clipped' or 'dense'"
        )

    if mode == "official_clipped":
        schedule = structure_head.inference_noise_schedule(
            int(num_sampling_steps),
            device,
        )
        schedule = schedule[schedule <= float(max_inference_sigma)]
        schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
        tau_tensor = torch.tensor(float(tau), device=device, dtype=torch.float32)
        return torch.cat((schedule[schedule > tau_tensor], tau_tensor[None]))

    dense_steps = int(dense_num_sampling_steps)
    if dense_steps <= 1:
        raise ValueError("dense_num_sampling_steps must be greater than 1")
    power = float(structure_head.inference_p)
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("ESMFold2 inference_p must be finite and positive")
    schedule = torch.linspace(
        float(max_inference_sigma) ** (1.0 / power),
        float(tau) ** (1.0 / power),
        dense_steps,
        device=device,
        dtype=torch.float32,
    ).pow(power)
    schedule[0] = float(max_inference_sigma)
    schedule[-1] = float(tau)
    return schedule


def run_esmfold2_truncated_diffusion(
    *,
    model: Any,
    captured: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    tau: float,
    seed: Optional[int],
    num_sampling_steps: int,
    max_inference_sigma: float = 256.0,
    truncated_schedule_mode: str = "official_clipped",
    dense_num_sampling_steps: int = 41,
) -> Dict[str, torch.Tensor | int | float]:
    """Run a no-grad reverse trajectory to tau, then one differentiable denoise."""

    if float(tau) <= 0.0:
        raise ValueError("ESMFold2 truncated diffusion tau must be positive")
    if int(num_sampling_steps) <= 1:
        raise ValueError("num_sampling_steps must be greater than 1")
    if float(tau) >= float(max_inference_sigma):
        raise ValueError("tau must be smaller than max_inference_sigma")
    required = {"s_inputs", "relative_position_encoding", "z_trunk"}
    missing = sorted(required - set(captured))
    if missing:
        raise RuntimeError(
            f"Failed to capture ESMFold2 proposal intermediates: {missing}"
        )

    atom_mask, ref_element, ref_atom_name_chars, atom_to_token = (
        _diffusion_atom_inputs(features)
    )
    structure_head = model.structure_head
    diffusion_module = structure_head.diffusion_module
    device = captured["s_inputs"].device
    target_batch = int(captured["s_inputs"].shape[0])
    n_atoms = int(atom_to_token.shape[1])

    truncated_schedule = _build_truncated_noise_schedule(
        structure_head=structure_head,
        device=device,
        tau=float(tau),
        num_sampling_steps=int(num_sampling_steps),
        max_inference_sigma=float(max_inference_sigma),
        schedule_mode=truncated_schedule_mode,
        dense_num_sampling_steps=int(dense_num_sampling_steps),
    )
    if truncated_schedule.numel() < 2:
        raise RuntimeError("truncated diffusion schedule contains no reverse step")

    atom_mask_f = atom_mask.float()
    lam = float(structure_head.noise_scale)
    eta = float(structure_head.step_scale)
    gammas = torch.where(
        truncated_schedule > float(structure_head.gamma_min),
        torch.full_like(truncated_schedule, float(structure_head.gamma_0)),
        torch.zeros_like(truncated_schedule),
    )
    step_pairs = list(
        zip(
            truncated_schedule[:-1],
            truncated_schedule[1:],
            gammas[1:],
        )
    )
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        if seed is not None:
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            x = truncated_schedule[0] * torch.randn(
                target_batch,
                n_atoms,
                3,
                device=device,
                dtype=torch.float32,
            )
            inference_cache: dict[str, torch.Tensor] = {}
            for sigma_tm, sigma_t, gamma in step_pairs:
                x, _ = structure_head._center_random_augmentation(x, atom_mask_f)
                sigma_tm_value = float(sigma_tm.item())
                t_hat_value = sigma_tm_value * (1.0 + float(gamma.item()))
                eps_std = lam * max(
                    t_hat_value**2 - sigma_tm_value**2,
                    0.0,
                ) ** 0.5
                x_noisy = x + eps_std * torch.randn_like(x)
                diffusion_output = diffusion_module(
                    x_noisy=x_noisy,
                    t_hat=torch.full(
                        (target_batch,),
                        t_hat_value,
                        device=device,
                        dtype=torch.float32,
                    ),
                    ref_pos=features["ref_pos"],
                    ref_charge=features["ref_charge"],
                    ref_mask=atom_mask,
                    ref_element=ref_element,
                    ref_atom_name_chars=ref_atom_name_chars,
                    ref_space_uid=features["ref_space_uid"],
                    tok_idx=atom_to_token,
                    s_inputs=captured["s_inputs"],
                    s_trunk=None,
                    z_trunk=captured["z_trunk"].float(),
                    relative_position_encoding=captured["relative_position_encoding"],
                    asym_id=features["asym_id"],
                    residue_index=features["residue_index"],
                    entity_id=features["entity_id"],
                    token_index=features["token_index"],
                    sym_id=features["sym_id"],
                    token_attention_mask=features["token_attention_mask"],
                    num_diffusion_samples=1,
                    return_token_repr=True,
                    inference_cache=inference_cache,
                )
                x_denoised = diffusion_output["x_denoised"]
                x_noisy = structure_head._weighted_rigid_align(
                    x_noisy.float(),
                    x_denoised.float(),
                    atom_mask_f,
                    atom_mask_f,
                ).to(dtype=x_denoised.dtype)
                t_hat_safe = max(t_hat_value, torch.finfo(torch.float32).eps)
                derivative = (x_noisy - x_denoised) / t_hat_safe
                x = x_noisy + eta * (float(sigma_t.item()) - t_hat_value) * derivative
            x_tau = x.detach()

    diffusion_output = diffusion_module(
        x_noisy=x_tau,
        t_hat=torch.full(
            (target_batch,),
            float(tau),
            device=device,
            dtype=torch.float32,
        ),
        ref_pos=features["ref_pos"],
        ref_charge=features["ref_charge"],
        ref_mask=atom_mask,
        ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars,
        ref_space_uid=features["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=captured["s_inputs"],
        s_trunk=None,
        z_trunk=captured["z_trunk"].float(),
        relative_position_encoding=captured["relative_position_encoding"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        entity_id=features["entity_id"],
        token_index=features["token_index"],
        sym_id=features["sym_id"],
        token_attention_mask=features["token_attention_mask"],
        num_diffusion_samples=1,
    )
    x_denoised = diffusion_output["x_denoised"]
    if x_denoised is None:
        raise RuntimeError("ESMFold2 diffusion module returned no denoised coordinates")
    confidence = _one_step_confidence(
        model=model,
        captured=captured,
        features=features,
        coordinates=x_denoised,
        atom_mask=atom_mask,
        atom_to_token=atom_to_token,
    )
    return {
        "coordinates": x_denoised,
        "source_coordinates": x_tau,
        "plddt_logits": confidence["plddt_logits"],
        "pae_logits": confidence["pae_logits"],
        "truncated_steps": len(step_pairs),
        "tau": float(tau),
    }


def _weighted_rigid_align_with_transform(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ESMFold2's Kabsch alignment and expose its rigid transform."""

    weights = (mask * w).unsqueeze(-1)
    denominator = weights.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    moving_center = (x * weights).sum(dim=-2, keepdim=True) / denominator
    target_center = (x_gt * weights).sum(dim=-2, keepdim=True) / denominator
    moving_centered = x - moving_center
    target_centered = x_gt - target_center
    covariance = torch.einsum(
        "bni,bnj->bij",
        weights * target_centered,
        moving_centered,
    )
    covariance32 = covariance.float()
    u, _, vh = torch.linalg.svd(
        covariance32,
        driver="gesvd" if covariance32.is_cuda else None,
    )
    determinant = torch.linalg.det(u @ vh)
    ones = torch.ones_like(determinant)
    rotation = (
        u
        @ torch.diag_embed(torch.stack([ones, ones, determinant], dim=-1))
        @ vh
    ).to(covariance.dtype)
    aligned = moving_centered @ rotation.transpose(-1, -2) + target_center
    return aligned, rotation, moving_center, target_center


def _apply_rigid_transform(
    coordinates: torch.Tensor,
    *,
    rotation: torch.Tensor,
    moving_center: torch.Tensor,
    target_center: torch.Tensor,
) -> torch.Tensor:
    return (
        (coordinates - moving_center) @ rotation.transpose(-1, -2)
        + target_center
    )


def run_esmfold2_template_anchored_truncated_diffusion(
    *,
    model: Any,
    captured: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    template_coordinates: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    tau: float,
    seed: Optional[int],
    num_sampling_steps: int,
    max_inference_sigma: float = 256.0,
    truncated_schedule_mode: str = "official_clipped",
    dense_num_sampling_steps: int = 41,
) -> Dict[str, torch.Tensor | int | float]:
    """Run strict truncated diffusion while clamping a noise-consistent template."""

    if float(tau) <= 0.0:
        raise ValueError("ESMFold2 template-anchored tau must be positive")
    if int(num_sampling_steps) <= 1:
        raise ValueError("num_sampling_steps must be greater than 1")
    if float(tau) >= float(max_inference_sigma):
        raise ValueError("tau must be smaller than max_inference_sigma")
    required = {"s_inputs", "relative_position_encoding", "z_trunk"}
    missing = sorted(required - set(captured))
    if missing:
        raise RuntimeError(
            f"Failed to capture ESMFold2 proposal intermediates: {missing}"
        )

    atom_mask, ref_element, ref_atom_name_chars, atom_to_token = (
        _diffusion_atom_inputs(features)
    )
    structure_head = model.structure_head
    diffusion_module = structure_head.diffusion_module
    device = captured["s_inputs"].device
    target_batch = int(captured["s_inputs"].shape[0])
    n_atoms = int(atom_to_token.shape[1])

    template_clean = template_coordinates.to(device=device, dtype=torch.float32)
    template_mask = fixed_atom_mask.to(device=device, dtype=torch.bool)
    if template_clean.ndim == 2:
        template_clean = template_clean.unsqueeze(0)
    if template_mask.ndim == 1:
        template_mask = template_mask.unsqueeze(0)
    if template_clean.shape != (template_clean.shape[0], n_atoms, 3):
        raise ValueError("template coordinates do not match ESMFold2 atom layout")
    if template_mask.shape != (template_mask.shape[0], n_atoms):
        raise ValueError("fixed_atom_mask does not match ESMFold2 atom layout")
    if template_clean.shape[0] == 1 and target_batch > 1:
        template_clean = template_clean.expand(target_batch, -1, -1).clone()
    if template_mask.shape[0] == 1 and target_batch > 1:
        template_mask = template_mask.expand(target_batch, -1).clone()
    if template_clean.shape[0] != target_batch or template_mask.shape[0] != target_batch:
        raise ValueError("template batch size does not match proposal batch size")
    template_mask &= atom_mask
    if not template_mask.any():
        raise ValueError("fixed_atom_mask selects no valid ESMFold2 atoms")

    truncated_schedule = _build_truncated_noise_schedule(
        structure_head=structure_head,
        device=device,
        tau=float(tau),
        num_sampling_steps=int(num_sampling_steps),
        max_inference_sigma=float(max_inference_sigma),
        schedule_mode=truncated_schedule_mode,
        dense_num_sampling_steps=int(dense_num_sampling_steps),
    )
    if truncated_schedule.numel() < 2:
        raise RuntimeError("template-anchored schedule contains no reverse step")

    atom_mask_f = atom_mask.float()
    template_mask_3d = template_mask.unsqueeze(-1)
    lam = float(structure_head.noise_scale)
    eta = float(structure_head.step_scale)
    gammas = torch.where(
        truncated_schedule > float(structure_head.gamma_min),
        torch.full_like(truncated_schedule, float(structure_head.gamma_0)),
        torch.zeros_like(truncated_schedule),
    )
    step_pairs = list(
        zip(
            truncated_schedule[:-1],
            truncated_schedule[1:],
            gammas[1:],
        )
    )
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        if seed is not None:
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            sigma_start = float(truncated_schedule[0].item())
            x = truncated_schedule[0] * torch.randn(
                target_batch,
                n_atoms,
                3,
                device=device,
                dtype=torch.float32,
            )
            template_generator = torch.Generator(device=device)
            base_seed = int(seed) if seed is not None else int(torch.initial_seed())
            template_generator.manual_seed((base_seed + 104729) % (2**63 - 1))
            template_epsilon = torch.randn(
                template_clean.shape,
                generator=template_generator,
                device=device,
                dtype=torch.float32,
            )
            template_noisy = template_clean + sigma_start * template_epsilon
            x = torch.where(template_mask_3d, template_noisy, x)

            inference_cache: dict[str, torch.Tensor] = {}
            for sigma_tm, sigma_t, gamma in step_pairs:
                paired_template = torch.cat((template_clean, template_noisy), dim=1)
                x, paired_template = structure_head._center_random_augmentation(
                    x,
                    atom_mask_f,
                    paired_template,
                )
                if paired_template is None:
                    raise RuntimeError("ESMFold2 discarded template frame coordinates")
                template_clean, template_noisy = paired_template.split(n_atoms, dim=1)

                sigma_tm_value = float(sigma_tm.item())
                template_epsilon = (
                    template_noisy - template_clean
                ) / max(sigma_tm_value, torch.finfo(torch.float32).eps)
                t_hat_value = sigma_tm_value * (1.0 + float(gamma.item()))
                eps_std = lam * max(
                    t_hat_value**2 - sigma_tm_value**2,
                    0.0,
                ) ** 0.5
                x_noisy = x + eps_std * torch.randn_like(x)
                template_t_hat = template_clean + t_hat_value * template_epsilon
                x_noisy = torch.where(template_mask_3d, template_t_hat, x_noisy)
                diffusion_output = diffusion_module(
                    x_noisy=x_noisy,
                    t_hat=torch.full(
                        (target_batch,),
                        t_hat_value,
                        device=device,
                        dtype=torch.float32,
                    ),
                    ref_pos=features["ref_pos"],
                    ref_charge=features["ref_charge"],
                    ref_mask=atom_mask,
                    ref_element=ref_element,
                    ref_atom_name_chars=ref_atom_name_chars,
                    ref_space_uid=features["ref_space_uid"],
                    tok_idx=atom_to_token,
                    s_inputs=captured["s_inputs"],
                    s_trunk=None,
                    z_trunk=captured["z_trunk"].float(),
                    relative_position_encoding=captured["relative_position_encoding"],
                    asym_id=features["asym_id"],
                    residue_index=features["residue_index"],
                    entity_id=features["entity_id"],
                    token_index=features["token_index"],
                    sym_id=features["sym_id"],
                    token_attention_mask=features["token_attention_mask"],
                    num_diffusion_samples=1,
                    return_token_repr=True,
                    inference_cache=inference_cache,
                )
                x_denoised = diffusion_output["x_denoised"]
                x_noisy, rotation, moving_center, target_center = (
                    _weighted_rigid_align_with_transform(
                        x_noisy.float(),
                        x_denoised.float(),
                        atom_mask_f,
                        atom_mask_f,
                    )
                )
                x_noisy = x_noisy.to(dtype=x_denoised.dtype)
                template_clean = _apply_rigid_transform(
                    template_clean,
                    rotation=rotation,
                    moving_center=moving_center,
                    target_center=target_center,
                )
                template_t_hat = _apply_rigid_transform(
                    template_t_hat,
                    rotation=rotation,
                    moving_center=moving_center,
                    target_center=target_center,
                )
                template_epsilon = (
                    template_t_hat - template_clean
                ) / max(t_hat_value, torch.finfo(torch.float32).eps)
                t_hat_safe = max(t_hat_value, torch.finfo(torch.float32).eps)
                derivative = (x_noisy - x_denoised) / t_hat_safe
                x = x_noisy + eta * (float(sigma_t.item()) - t_hat_value) * derivative
                template_noisy = template_clean + float(sigma_t.item()) * template_epsilon
                x = torch.where(template_mask_3d, template_noisy, x)
            x_tau = x.detach()
            clean_template_at_tau_frame = template_clean.detach()

    diffusion_output = diffusion_module(
        x_noisy=x_tau,
        t_hat=torch.full(
            (target_batch,),
            float(tau),
            device=device,
            dtype=torch.float32,
        ),
        ref_pos=features["ref_pos"],
        ref_charge=features["ref_charge"],
        ref_mask=atom_mask,
        ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars,
        ref_space_uid=features["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=captured["s_inputs"],
        s_trunk=None,
        z_trunk=captured["z_trunk"].float(),
        relative_position_encoding=captured["relative_position_encoding"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        entity_id=features["entity_id"],
        token_index=features["token_index"],
        sym_id=features["sym_id"],
        token_attention_mask=features["token_attention_mask"],
        num_diffusion_samples=1,
    )
    x_denoised = diffusion_output["x_denoised"]
    if x_denoised is None:
        raise RuntimeError("ESMFold2 diffusion module returned no denoised coordinates")
    x_denoised = torch.where(
        template_mask_3d,
        clean_template_at_tau_frame.to(dtype=x_denoised.dtype),
        x_denoised,
    )
    confidence = _one_step_confidence(
        model=model,
        captured=captured,
        features=features,
        coordinates=x_denoised,
        atom_mask=atom_mask,
        atom_to_token=atom_to_token,
    )
    return {
        "coordinates": x_denoised,
        "source_coordinates": x_tau,
        "template_coordinates": clean_template_at_tau_frame,
        "fixed_atom_mask": template_mask,
        "plddt_logits": confidence["plddt_logits"],
        "pae_logits": confidence["pae_logits"],
        "truncated_steps": len(step_pairs),
        "tau": float(tau),
    }


@torch.inference_mode()
def run_esmfold2_scaffold_consistency_diffusion(
    *,
    structure_head: Any,
    template_coordinates: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    z_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
    s_trunk: Optional[torch.Tensor],
    relative_position_encoding: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_charge: torch.Tensor,
    ref_mask: torch.Tensor,
    ref_element: torch.Tensor,
    ref_atom_name_chars: torch.Tensor,
    ref_space_uid: torch.Tensor,
    tok_idx: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
    sym_id: torch.Tensor,
    token_attention_mask: Optional[torch.Tensor] = None,
    num_diffusion_samples: int = 1,
    num_sampling_steps: Optional[int] = None,
    max_inference_sigma: float = 256.0,
) -> Dict[str, torch.Tensor | None]:
    """Run standard full diffusion while keeping a local scaffold consistent.

    This mirrors the official ``DiffusionStructureHead.sample`` loop instead
    of post-processing an unconstrained final structure.  At every noise
    level, fixed atoms receive the same noise-consistent template trajectory;
    only atoms outside ``fixed_atom_mask`` are generated freely.  The caller
    remains responsible for passing the resulting coordinates to the standard
    ESMFold2 confidence head.
    """

    if int(num_diffusion_samples) != 1:
        raise ValueError(
            "Scaffold-consistency verification currently supports one "
            "diffusion sample per candidate"
        )
    if float(max_inference_sigma) <= 0.0:
        raise ValueError("max_inference_sigma must be positive")

    device = s_inputs.device
    target_batch = int(s_inputs.shape[0])
    n_atoms = int(tok_idx.shape[1])
    atom_mask = ref_mask.bool()
    if atom_mask.ndim == 1:
        atom_mask = atom_mask.unsqueeze(0)
    if atom_mask.shape != (target_batch, n_atoms):
        raise ValueError("ref_mask shape does not match verifier atom layout")

    template_clean = template_coordinates.to(device=device, dtype=torch.float32)
    template_mask = fixed_atom_mask.to(device=device, dtype=torch.bool)
    if template_clean.ndim == 2:
        template_clean = template_clean.unsqueeze(0)
    if template_mask.ndim == 1:
        template_mask = template_mask.unsqueeze(0)
    if template_clean.shape == (1, n_atoms, 3) and target_batch > 1:
        template_clean = template_clean.expand(target_batch, -1, -1).clone()
    if template_mask.shape == (1, n_atoms) and target_batch > 1:
        template_mask = template_mask.expand(target_batch, -1).clone()
    if template_clean.shape != (target_batch, n_atoms, 3):
        raise ValueError("template coordinates do not match verifier atom layout")
    if template_mask.shape != (target_batch, n_atoms):
        raise ValueError("fixed_atom_mask does not match verifier atom layout")
    template_mask &= atom_mask
    if not template_mask.any():
        raise ValueError("fixed_atom_mask selects no valid verifier atoms")

    steps = (
        int(structure_head.inference_num_steps)
        if num_sampling_steps is None
        else int(num_sampling_steps)
    )
    if steps <= 1:
        raise ValueError("num_sampling_steps must be greater than 1")
    schedule = structure_head.inference_noise_schedule(steps, device)
    schedule = schedule[schedule <= float(max_inference_sigma)]
    schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
    if schedule.numel() < 2:
        raise RuntimeError("scaffold-consistency schedule contains no reverse step")

    atom_mask_f = atom_mask.float()
    template_mask_3d = template_mask.unsqueeze(-1)
    lam = float(structure_head.noise_scale)
    eta = float(structure_head.step_scale)
    gammas = torch.where(
        schedule > float(structure_head.gamma_min),
        torch.full_like(schedule, float(structure_head.gamma_0)),
        torch.zeros_like(schedule),
    )
    step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
    x = schedule[0] * torch.randn(
        target_batch,
        n_atoms,
        3,
        device=device,
        dtype=torch.float32,
    )
    template_epsilon = torch.randn_like(template_clean)
    template_noisy = template_clean + float(schedule[0].item()) * template_epsilon
    x = torch.where(template_mask_3d, template_noisy, x)

    inference_cache: dict[str, torch.Tensor] = {}
    token_repr: Optional[torch.Tensor] = None
    diffusion_module = structure_head.diffusion_module
    for sigma_tm, sigma_t, gamma in step_pairs:
        paired_template = torch.cat((template_clean, template_noisy), dim=1)
        x, paired_template = structure_head._center_random_augmentation(
            x,
            atom_mask_f,
            paired_template,
        )
        if paired_template is None:
            raise RuntimeError("ESMFold2 discarded scaffold template coordinates")
        template_clean, template_noisy = paired_template.split(n_atoms, dim=1)

        sigma_tm_value = float(sigma_tm.item())
        template_epsilon = (template_noisy - template_clean) / max(
            sigma_tm_value,
            torch.finfo(torch.float32).eps,
        )
        t_hat_value = sigma_tm_value * (1.0 + float(gamma.item()))
        eps_std = lam * max(t_hat_value**2 - sigma_tm_value**2, 0.0) ** 0.5
        x_noisy = x + eps_std * torch.randn_like(x)
        template_t_hat = template_clean + t_hat_value * template_epsilon
        x_noisy = torch.where(template_mask_3d, template_t_hat, x_noisy)

        diffusion_output = diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.full(
                (target_batch,),
                t_hat_value,
                device=device,
                dtype=torch.float32,
            ),
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=atom_mask,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            tok_idx=tok_idx,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=1,
            return_token_repr=True,
            inference_cache=inference_cache,
        )
        x_denoised = diffusion_output.get("x_denoised")
        if x_denoised is None:
            raise RuntimeError("ESMFold2 diffusion module returned no denoised coordinates")
        token_repr = diffusion_output.get("token_repr")
        x_noisy, rotation, moving_center, target_center = (
            _weighted_rigid_align_with_transform(
                x_noisy.float(),
                x_denoised.float(),
                atom_mask_f,
                atom_mask_f,
            )
        )
        x_noisy = x_noisy.to(dtype=x_denoised.dtype)
        template_clean = _apply_rigid_transform(
            template_clean,
            rotation=rotation,
            moving_center=moving_center,
            target_center=target_center,
        )
        template_t_hat = _apply_rigid_transform(
            template_t_hat,
            rotation=rotation,
            moving_center=moving_center,
            target_center=target_center,
        )
        template_epsilon = (template_t_hat - template_clean) / max(
            t_hat_value,
            torch.finfo(torch.float32).eps,
        )
        x = x_noisy + eta * (float(sigma_t.item()) - t_hat_value) * (
            (x_noisy - x_denoised) / max(t_hat_value, torch.finfo(torch.float32).eps)
        )
        template_noisy = template_clean + float(sigma_t.item()) * template_epsilon
        x = torch.where(template_mask_3d, template_noisy, x)

    return {
        "sample_atom_coords": x,
        "diff_token_repr": token_repr,
        "scaffold_template_coordinates": template_clean,
        "fixed_atom_mask": template_mask,
    }
