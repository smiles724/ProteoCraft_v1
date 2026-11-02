"""CPU checks for geometry-gated, opt-in homotrimer interface scoring."""

import pytest
import torch

from proteocraft.hallucination.esmfold2_interface import (
    INTERFACE_POLICY_VERSION,
    interface_distogram_bins,
    native_interface_summary,
    select_coordinate_interface,
    select_distogram_interface,
)


def _masks():
    chain_masks = []
    for start in (0, 2, 4):
        mask = torch.zeros(8, dtype=torch.bool)
        mask[start:start + 2] = True
        chain_masks.append(mask)
    binder = torch.zeros(8, dtype=torch.bool)
    binder[6:] = True
    return chain_masks, binder


def _coordinate_case():
    chains, binder = _masks()
    # Duplicate each residue's atom, yielding sixteen atom contacts but only
    # four unique target-residue/binder-residue pairs for protomer 0.
    token_x = torch.tensor([0., 1., 20., 21., 50., 51., .5, 1.5])
    x = token_x.repeat_interleave(2)
    return {
        "coordinates": torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1),
        "atom_to_token": torch.arange(8).repeat_interleave(2),
        "atom_mask": torch.ones(16, dtype=torch.bool),
        "chain_masks": chains,
        "binder_mask": binder,
        "cutoff": 2.,
        "min_contacts": 4,
    }


def _distogram_case():
    chains, binder = _masks()
    logits = torch.full((1, 8, 8, 64), -12.)
    logits[..., -1] = 12.
    for target in range(4):
        for binder_index in (6, 7):
            for row, col in ((target, binder_index), (binder_index, target)):
                logits[0, row, col, :] = -12.
                logits[0, row, col, 0] = 12.
    return {
        "distogram_logits": logits,
        "chain_masks": chains,
        "binder_mask": binder,
        "min_contacts": 3,
    }


def _native_pairs():
    pairs = torch.zeros(4, 4)
    pairs.fill_diagonal_(1.)
    pairs[:3, 3] = torch.tensor([.8, .6, .99])
    pairs[3, :3] = torch.tensor([.7, .4, .95])
    return pairs


def test_coordinate_contacts_count_residue_pairs_not_atom_pairs():
    case = _coordinate_case()
    selected = select_coordinate_interface(**case)
    assert selected["contact_counts"] == [4, 0, 0]
    assert selected["selected_indices"] == [0]
    assert selected["fallback"] is False
    assert selected["policy_version"] == INTERFACE_POLICY_VERSION
    assert select_coordinate_interface(**{**case, "min_contacts": 5})["selected_indices"] == []


def test_contact_deduplication_survives_coordinate_chunk_boundaries():
    # One residue has over 256 atoms, causing it to appear in multiple chunks.
    coords = torch.zeros(301, 3)
    coords[-1, 0] = 1.
    result = select_coordinate_interface(
        coords,
        atom_to_token=torch.tensor([0] * 300 + [3]),
        atom_mask=torch.ones(301, dtype=torch.bool),
        chain_masks=[torch.tensor([True, False, False, False]),
                     torch.tensor([False, True, False, False]),
                     torch.tensor([False, False, True, False])],
        binder_mask=torch.tensor([False, False, False, True]),
        cutoff=2.,
        min_contacts=2,
    )
    assert result["contact_counts"] == [1, 0, 0]
    assert result["selected_indices"] == []


def test_padding_ignores_invalid_mapping_and_nonfinite_coordinates():
    case = _coordinate_case()
    expected = select_coordinate_interface(**case)
    padded = {
        **case,
        "coordinates": torch.cat((case["coordinates"], torch.tensor([[float("nan"), 0., 0.]]))),
        "atom_to_token": torch.cat((case["atom_to_token"], torch.tensor([-999]))),
        "atom_mask": torch.cat((case["atom_mask"], torch.tensor([False]))),
    }
    assert select_coordinate_interface(**padded) == expected


def test_physical_selection_is_invariant_to_target_chain_order():
    case = _coordinate_case()
    original = select_coordinate_interface(**case)
    order = [2, 0, 1]
    permuted = select_coordinate_interface(
        **{**case, "chain_masks": [case["chain_masks"][index] for index in order]}
    )
    assert permuted["contact_counts"] == [original["contact_counts"][index] for index in order]
    assert [order[index] for index in permuted["selected_indices"]] == original["selected_indices"]


def test_expected_selection_uses_contacts_and_excludes_far_protomer():
    selected = select_distogram_interface(**_distogram_case())
    assert selected["selected_indices"] == [0, 1]
    assert selected["expected_contacts"] == pytest.approx([4., 4., 0.], abs=1e-6)
    assert selected["fallback"] is False


def test_known_distogram_heads_use_explicit_head_specific_distance_midpoints():
    release = interface_distogram_bins(64, dtype=torch.float64)
    experimental = interface_distogram_bins(128, dtype=torch.float64)
    assert release.dtype == torch.float64
    assert release[[0, -1]].tolist() == pytest.approx([2.15625, 21.84375])
    assert experimental[[0, -1]].tolist() == pytest.approx([1.5, 54.5])
    assert torch.all(release[1:] > release[:-1])
    assert torch.all(experimental[1:] > experimental[:-1])


@pytest.mark.parametrize("num_bins", [8, 39, True, 64.0])
def test_distogram_bins_reject_unknown_or_noninteger_heads(num_bins):
    with pytest.raises(ValueError, match="head size"):
        interface_distogram_bins(num_bins)


def test_release_contact_selection_uses_release_bins_instead_of_experimental_range():
    case = _distogram_case()
    logits = case["distogram_logits"]
    # The same ordinal bin has different meanings: release bin 9 is within
    # 5 A, while Experimental bin 9 is outside it. Chain routing must use the
    # actual head definition instead of pretending both heads span 2--52 A.
    for target in range(4):
        for binder_index in (6, 7):
            for row, col in ((target, binder_index), (binder_index, target)):
                logits[0, row, col, :] = -12.
                logits[0, row, col, 9] = 12.
    release = select_distogram_interface(**case)
    assert release["selected_indices"] == [0, 1]
    assert release["fallback"] is False
    experimental = torch.full((1, 8, 8, 128), -12.)
    experimental[..., -1] = 12.
    experimental[:, :4, 6:, :] = -12.
    experimental[:, 6:, :4, :] = -12.
    experimental[:, :4, 6:, 9] = 12.
    experimental[:, 6:, :4, 9] = 12.
    changed = select_distogram_interface(**{**case, "distogram_logits": experimental})
    assert changed["fallback"] is True


def test_expected_fallback_preserves_equal_contact_ties_under_permutation():
    case = _distogram_case()
    case["min_contacts"] = 5
    original = select_distogram_interface(**case)
    assert original["fallback"] is True
    assert original["selected_indices"] == [0, 1]
    order = [2, 1, 0]
    permuted = select_distogram_interface(
        **{**case, "chain_masks": [case["chain_masks"][index] for index in order]}
    )
    assert permuted["fallback"] is True
    assert set(order[index] for index in permuted["selected_indices"]) == {0, 1}


def test_native_score_averages_weaker_directions_of_physical_interfaces():
    selection = {"selected_indices": [0, 1], "fallback": False}
    summary = native_interface_summary(_native_pairs(), selection, n_target_chains=3)
    # min(.8,.7) and min(.6,.4), equal mean; exclude high-scoring third chain.
    assert summary["pair_chain_iptm"] == pytest.approx(.55)
    assert summary["pair_chain_iptm_bidirectional_min_by_target"] == pytest.approx([.7, .4, .95])
    assert summary["pair_chain_iptm_all_target_bidirectional_mean"] == pytest.approx((.7 + .4 + .95) / 3)
    assert summary["target_interface_eligible"] is True
    assert summary["pair_chains_iptm_native"] == _native_pairs().tolist()
    assert summary["pair_chain_iptm_axis_order"] == ["target_0", "target_1", "target_2", "binder"]


def test_native_physical_score_is_invariant_to_chain_permutation():
    pairs = _native_pairs()
    original = native_interface_summary(
        pairs, {"selected_indices": [0, 1], "fallback": False}, n_target_chains=3
    )
    order = torch.tensor([2, 0, 1, 3])
    permuted = native_interface_summary(
        pairs[order][:, order],
        {"selected_indices": [1, 2], "fallback": False},
        n_target_chains=3,
    )
    assert permuted["pair_chain_iptm"] == pytest.approx(original["pair_chain_iptm"])


def test_no_physical_interface_fails_closed_even_with_high_native_scores():
    case = _coordinate_case()
    case["coordinates"] = case["coordinates"].clone()
    case["coordinates"][-4:, 0] += 1000.
    selection = select_coordinate_interface(**case)
    assert selection["selected_indices"] == []
    assert selection["fallback"] is False
    summary = native_interface_summary(torch.ones(1, 4, 4), selection, n_target_chains=3)
    assert summary["pair_chain_iptm"] == 0.
    assert summary["target_interface_eligible"] is False


@pytest.mark.parametrize("shape", [(8, 7, 64), (2, 8, 8, 64), (8, 8, 1)])
def test_distogram_rejects_ambiguous_or_invalid_shapes(shape):
    case = _distogram_case()
    with pytest.raises(ValueError, match="one square token matrix"):
        select_distogram_interface(**{**case, "distogram_logits": torch.zeros(shape)})


def test_distogram_rejects_nonfinite_interface_logits():
    case = _distogram_case()
    case["distogram_logits"][0, 0, 6, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite interface distogram"):
        select_distogram_interface(**case)


@pytest.mark.parametrize("mutation, message", [
    ("shape", "one sample"),
    ("length", "sizes differ"),
    ("negative_mapping", "out of bounds"),
    ("large_mapping", "out of bounds"),
    ("nonfinite", "Nonfinite interface coordinates"),
    ("no_binder_atoms", "no valid heavy atoms"),
])
def test_physical_selection_rejects_invalid_inputs(mutation, message):
    case = _coordinate_case()
    if mutation == "shape":
        case["coordinates"] = torch.zeros(2, 16, 3)
    elif mutation == "length":
        case["atom_mask"] = torch.ones(15, dtype=torch.bool)
    elif mutation == "negative_mapping":
        case["atom_to_token"][0] = -1
    elif mutation == "large_mapping":
        case["atom_to_token"][0] = 8
    elif mutation == "nonfinite":
        case["coordinates"][0, 0] = float("inf")
    else:
        case["atom_mask"][-4:] = False
    with pytest.raises(ValueError, match=message):
        select_coordinate_interface(**case)


@pytest.mark.parametrize("mutation, message", [
    ("missing", "requires native"),
    ("shape", "match target chains"),
    ("batch", "match target chains"),
    ("nonfinite", "Nonfinite native"),
    ("low", r"in \[0,1\]"),
    ("high", r"in \[0,1\]"),
    ("bad_selection", "Invalid selected"),
    ("duplicate_selection", "Invalid selected"),
])
def test_native_score_rejects_invalid_matrix_or_selection(mutation, message):
    matrix = _native_pairs()
    selection = {"selected_indices": [0, 1], "fallback": False}
    if mutation == "missing":
        matrix = None
    elif mutation == "shape":
        matrix = torch.zeros(3, 3)
    elif mutation == "batch":
        matrix = matrix.unsqueeze(0).repeat(2, 1, 1)
    elif mutation == "nonfinite":
        matrix[0, 3] = float("nan")
    elif mutation == "low":
        matrix[0, 3] = -.1
    elif mutation == "high":
        matrix[0, 3] = 1.1
    elif mutation == "bad_selection":
        selection["selected_indices"] = [3]
    else:
        selection["selected_indices"] = [0, 0]
    with pytest.raises(ValueError, match=message):
        native_interface_summary(matrix, selection, n_target_chains=3)


@pytest.mark.parametrize("selector", [select_coordinate_interface, select_distogram_interface])
def test_interface_masks_reject_overlap_with_binder(selector):
    case = _coordinate_case() if selector is select_coordinate_interface else _distogram_case()
    case["chain_masks"][0] = case["binder_mask"].clone()
    with pytest.raises(ValueError, match="nonempty and disjoint"):
        selector(**case)
