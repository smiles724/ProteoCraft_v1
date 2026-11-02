"""Opt-in homotrimer interface policy, independent of model installations.

Proposals route gradients using expected representative-atom contacts from a
distogram. Full verification gates native confidence using physical heavy-atom
residue-pair contacts. These are deliberately distinct, recorded observations.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

INTERFACE_POLICY_VERSION = "contacted_bidirectional_min_mean_v1"


def interface_distogram_bins(num_bins: int, *, device: Any = None,
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return explicit distance midpoints for opt-in interface routing only.

    The released ESMFold2 trunk distogram head has 64 training bins spanning
    2--22 Angstrom: Biohub's ``Language Modeling Materializes a World Model of
    Protein Biology``, Appendix A.2.8.1 (Distogram loss),
    https://biohub.ai/papers/esm_protein.pdf. The SDK's
    ``compute_distogram_conditioning`` also specifies uniform 64-bin edges
    ``linspace(2, 22, 65)``. We use adjacent-edge midpoints as a documented
    finite-bin approximation; clipped tail-bin probabilities do not reveal
    their exact distances.

    The Experimental design head has 128 bins. Its midpoint convention is
    retained exactly from Algorithm 12: 2--52 boundary points and outer edges
    1 and 57 Angstrom, giving midpoint endpoints 1.5 and 54.5 Angstrom.

    These are trunk-output bins, not the released confidence head's separate
    39-bin coordinate embedding. Unknown head sizes fail rather than guessing.
    The full verifier's physical heavy-atom contact gate uses coordinates and
    is independent of this approximation. Historical runtime bin paths are not
    changed by this helper.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, int):
        raise ValueError("Interface distogram head size must be an integer")
    if num_bins == 64:
        edges = torch.linspace(2.0, 22.0, 65, device=device, dtype=dtype)
    elif num_bins == 128:
        boundaries = torch.linspace(2.0, 52.0, 127, device=device, dtype=dtype)
        edges = torch.cat((boundaries.new_tensor([1.0]), boundaries,
                           boundaries.new_tensor([57.0])))
    else:
        raise ValueError(
            "Unsupported interface distogram head size; calibrated heads are "
            f"64 (2--22 Angstrom) and 128 (Experimental), got {num_bins}"
        )
    return (edges[:-1] + edges[1:]) / 2.0


def _validate_policy(cutoff: float, min_contacts: int) -> None:
    if not math.isfinite(float(cutoff)) or float(cutoff) <= 0:
        raise ValueError("Interface cutoff must be finite and positive")
    if isinstance(min_contacts, bool) or not isinstance(min_contacts, int) or min_contacts < 1:
        raise ValueError("Interface minimum contacts must be a positive integer")


def _validate_masks(chain_masks, binder_mask, length):
    binder = torch.as_tensor(binder_mask, dtype=torch.bool)
    chains = [torch.as_tensor(mask, dtype=torch.bool, device=binder.device) for mask in chain_masks]
    if binder.ndim != 1 or binder.numel() != length or not bool(binder.any()):
        raise ValueError("Interface binder mask has an invalid token layout")
    occupied = binder.clone()
    for mask in chains:
        if mask.shape != binder.shape or not bool(mask.any()) or bool((occupied & mask).any()):
            raise ValueError("Interface chain masks must be nonempty and disjoint")
        occupied |= mask
    if not chains:
        raise ValueError("Interface requires target chains")
    return chains, binder


def select_distogram_interface(distogram_logits: torch.Tensor, *,
        chain_masks: Sequence[torch.Tensor], binder_mask: torch.Tensor,
        cutoff: float = 5.0, min_contacts: int = 5) -> dict[str, Any]:
    """Detached routing; a single best-contact chain keeps early gradients alive.

Expected contacts sum probabilities over unique residue pairs, using explicitly
calibrated trunk-output midpoint bins. Fallback is proposal-only, never a
full-verifier acceptance gate. No PAE/iPTM values influence chain selection.
    """
    _validate_policy(cutoff, min_contacts)
    logits = distogram_logits.detach()
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[0] != logits.shape[1] or logits.shape[-1] < 2:
        raise ValueError("Interface distogram requires one square token matrix")
    chains, binder = _validate_masks(chain_masks, binder_mask, logits.shape[0])
    chains = [mask.to(logits.device) for mask in chains]
    binder = binder.to(logits.device)
    bin_midpoints = interface_distogram_bins(
        logits.shape[-1], device=logits.device, dtype=torch.float32,
    )
    close = bin_midpoints <= float(cutoff)
    counts = []
    with torch.no_grad():
        for mask in chains:
            # Slice before softmax to avoid allocating a second full LxLxB map.
            forward = logits[mask][:, binder, :].float()
            reverse = logits[binder][:, mask, :].transpose(0, 1).float()
            if not bool(torch.isfinite(forward).all() & torch.isfinite(reverse).all()):
                raise ValueError("Nonfinite interface distogram")
            probs = .5 * (forward.softmax(-1) + reverse.softmax(-1))
            counts.append(float(probs[..., close].sum().item()))
    selected = [i for i, count in enumerate(counts) if count >= min_contacts]
    fallback = not selected
    if fallback:
        # Ties preserve all equivalent nearest chains instead of arbitrarily
        # picking an input chain ID; routing remains permutation invariant.
        best = max(counts)
        selected = [i for i, count in enumerate(counts)
                    if math.isclose(count, best, rel_tol=1e-6, abs_tol=1e-8)]
    return {"selected_indices": selected, "expected_contacts": counts,
            "fallback": fallback, "source": "distogram_expected_residue_pairs",
            "distogram_head_num_bins": int(logits.shape[-1]),
            "distogram_bin_policy": (
                "released64_uniform_2_22_finite_interval_midpoint_approximation"
                if logits.shape[-1] == 64 else "experimental128_algorithm12_midpoints"
            ),
            "policy_version": INTERFACE_POLICY_VERSION}


def select_coordinate_interface(coordinates: torch.Tensor, *,
        atom_to_token: torch.Tensor, atom_mask: torch.Tensor,
        chain_masks: Sequence[torch.Tensor], binder_mask: torch.Tensor,
        cutoff: float = 5.0, min_contacts: int = 5) -> dict[str, Any]:
    """Count each target-residue/binder-residue pair once, regardless of atoms.

Coordinates/atom mask must contain heavy atoms only. Padding is removed using
the original feature mask; chain masks retain token identities and chain order.
    """
    _validate_policy(cutoff, min_contacts)
    coords = torch.as_tensor(coordinates).detach().float().cpu()
    if coords.ndim == 4 and tuple(coords.shape[:2]) == (1, 1):
        coords = coords[0, 0]
    elif coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("Interface coordinates require one sample of shape [atoms,3]")
    mapping = torch.as_tensor(atom_to_token).detach().long().cpu().reshape(-1)
    valid = torch.as_tensor(atom_mask).detach().bool().cpu().reshape(-1)
    if mapping.numel() != coords.shape[0] or valid.numel() != coords.shape[0]:
        raise ValueError("Interface coordinates/atom mapping/mask sizes differ")
    chains, binder = _validate_masks(chain_masks, torch.as_tensor(binder_mask).cpu(), binder_mask.numel())
    chains = [mask.cpu() for mask in chains]
    if bool(((mapping[valid] < 0) | (mapping[valid] >= binder.numel())).any()):
        raise ValueError("Interface atom-to-token mapping is out of bounds")
    if not bool(torch.isfinite(coords[valid]).all()):
        raise ValueError("Nonfinite interface coordinates")
    # Invalid/padded mappings need not be valid token IDs; never index them.
    coords, mapping = coords[valid], mapping[valid]
    binder_atoms = binder[mapping]
    if not bool(binder_atoms.any()):
        raise ValueError("Interface binder contains no valid heavy atoms")
    binder_coords, binder_tokens = coords[binder_atoms], mapping[binder_atoms]
    counts = []
    for mask in chains:
        target_atoms = mask[mapping]
        target_coords, target_tokens = coords[target_atoms], mapping[target_atoms]
        pairs = []
        for start in range(0, len(target_coords), 256):
            distance = torch.cdist(target_coords[start:start + 256], binder_coords)
            t, b = (distance <= float(cutoff)).nonzero(as_tuple=True)
            pairs.append(target_tokens[start:start + 256][t] * binder.numel() + binder_tokens[b])
        counts.append(int(torch.unique(torch.cat(pairs)).numel()) if pairs else 0)
    selected = [i for i, count in enumerate(counts) if count >= min_contacts]
    return {"selected_indices": selected, "contact_counts": counts,
            "fallback": False, "source": "coordinate_heavy_atom_residue_pairs",
            "policy_version": INTERFACE_POLICY_VERSION}


def native_interface_summary(pair_chains_iptm: Any, selection: Mapping[str, Any], *,
        n_target_chains: int) -> dict[str, Any]:
    """Equal mean of conservative native directional scores on contacted chains.

Do not average across seeds/samples, mix global iPTM, or maximize confidence to
choose a chain. The physical selection is performed before reading confidence.
    """
    if pair_chains_iptm is None:
        raise ValueError("Contacted interface requires native pair_chains_iptm")
    matrix = torch.as_tensor(pair_chains_iptm).detach().float().cpu()
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.shape != (n_target_chains + 1, n_target_chains + 1):
        raise ValueError("Native pair matrix must match target chains plus one binder")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Nonfinite native pair-chain iPTM")
    if bool(((matrix < 0) | (matrix > 1)).any()):
        raise ValueError("Native pair-chain iPTM must be in [0,1]")
    selected = list(selection["selected_indices"])
    if len(set(selected)) != len(selected) or any(i not in range(n_target_chains) for i in selected):
        raise ValueError("Invalid selected target chain indices")
    values = [min(float(matrix[i, n_target_chains]), float(matrix[n_target_chains, i]))
              for i in range(n_target_chains)]
    return {"pair_chain_iptm": sum(values[i] for i in selected) / len(selected) if selected else 0.,
            "target_interface_mode": "contacted",
            "target_interface_policy_version": INTERFACE_POLICY_VERSION,
            "target_interface_selection": dict(selection),
            "target_interface_eligible": bool(selected) and not selection.get("fallback", False),
            "pair_chains_iptm_native": matrix.tolist(),
            "pair_chain_iptm_bidirectional_min_by_target": values,
            "pair_chain_iptm_all_target_bidirectional_mean": sum(values) / len(values),
            "pair_chain_iptm_axis_order": [f"target_{i}" for i in range(n_target_chains)] + ["binder"]}
