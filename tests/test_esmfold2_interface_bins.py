"""Meaningful bin calibration checks for the opt-in contacted interface."""

import pytest
import torch

from proteocraft.hallucination.esmfold2_interface import (
    interface_distogram_bins,
    select_distogram_interface,
)


def test_released_64_bin_midpoints_cover_uniform_training_intervals():
    centers = interface_distogram_bins(64, device="cpu", dtype=torch.float64)
    assert centers.shape == (64,)
    assert centers.dtype == torch.float64
    assert centers[0].item() == pytest.approx(2.15625)
    assert centers[-1].item() == pytest.approx(21.84375)
    assert torch.allclose(torch.diff(centers), torch.full((63,), 0.3125,
                                                       dtype=torch.float64))
    # A 5-A contact is represented by ten finite bins, not a 52-A head.
    assert int((centers <= 5.0).sum()) == 10


def test_experimental_128_bin_midpoints_preserve_algorithm12_convention():
    centers = interface_distogram_bins(128, device="cpu", dtype=torch.float32)
    boundaries = torch.linspace(2.0, 52.0, 127)
    old_edges = torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))
    old_centers = (old_edges[:-1] + old_edges[1:]) / 2.0
    assert torch.equal(centers, old_centers)
    assert centers[0].item() == pytest.approx(1.5)
    assert centers[-1].item() == pytest.approx(54.5)


@pytest.mark.parametrize("num_bins", [4, 8, 39, 63, 65, 127, 129, True, 64.0])
def test_unknown_or_invalid_head_sizes_do_not_guess(num_bins):
    with pytest.raises(ValueError):
        interface_distogram_bins(num_bins)


def test_64_bin_selection_uses_released_distance_scale_and_detaches():
    # Bin 8 represents 4.65625 A on the released head, but >8 A on the old
    # 2--52 mapping. Six distinct residue pairs should select this chain.
    logits = torch.full((1, 5, 5, 64), -40.0, requires_grad=True)
    with torch.no_grad():
        logits[..., 8] = 40.0
    target = torch.tensor([True, True, False, False, False])
    binder = ~target
    selection = select_distogram_interface(
        logits, chain_masks=[target], binder_mask=binder,
        cutoff=5.0, min_contacts=5,
    )
    assert selection["expected_contacts"] == pytest.approx([6.0])
    assert selection["selected_indices"] == [0]
    assert selection["fallback"] is False
    assert logits.grad is None


def test_selector_rejects_unknown_head_even_when_chain_masks_are_valid():
    with pytest.raises(ValueError, match="Unsupported interface distogram head"):
        select_distogram_interface(
            torch.zeros(4, 4, 8),
            chain_masks=[torch.tensor([True, True, False, False])],
            binder_mask=torch.tensor([False, False, True, True]),
        )
