"""CPU-only integration checks for the explicit TNF homotrimer runtime."""

from types import SimpleNamespace

import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import CYS_INDEX, ESMFold2ProposalConfig
from proteocraft.hallucination.esmfold2_runtime import ESMFold2RuntimeBackend
import proteocraft.hallucination.esmfold2_runtime as runtime_module


def _backend(**overrides):
    options = {
        "target_assembly": "homotrimer",
        "target_interface_mode": "contacted",
        "verifier_loss_mode": "aligned_common",
        "lambda_esmc": 0.,
        "soft_steps": 1,
        "one_step_diffusion_enabled": False,
        "one_step_diffusion_interval": 0,
    }
    options.update(overrides)
    return ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(**options), device="cpu",
        experimental_model=object(), protein_input_cls=SimpleNamespace,
        structure_prediction_input_cls=SimpleNamespace,
    )


def _raw_case(*, padding=False):
    x = torch.tensor([-1., 0., 2., 3., 100., 101., 0., 1., 2.])
    coords = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    pairs = torch.eye(4)
    pairs[:3, 3] = torch.tensor([.8, .6, .99])
    pairs[3, :3] = torch.tensor([.7, .4, .95])
    distogram = torch.full((1, 9, 9, 64), -12.)
    distogram[..., -1] = 12.
    for target in range(4):
        for binder in range(6, 9):
            for row, col in ((target, binder), (binder, target)):
                distogram[0, row, col, :] = -12.
                distogram[0, row, col, 0] = 12.
    features = {
        "token_attention_mask": torch.ones(1, 9, dtype=torch.bool),
        "asym_id": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 3]]),
        "atom_attention_mask": torch.ones(1, 9, dtype=torch.bool),
        "atom_to_token": torch.arange(9).unsqueeze(0),
        "distogram_atom_idx": torch.arange(9).unsqueeze(0),
        "residue_index": torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 2]]),
        "ref_element": torch.full((1, 9), 6),
    }
    output = {
        "sample_atom_coords": coords.unsqueeze(0),
        "pair_chains_iptm": pairs.unsqueeze(0),
        "plddt_logits": torch.zeros(1, 9, 50),
        "pae_logits": torch.zeros(1, 9, 9, 64),
        "distogram_logits": distogram,
    }
    if padding:
        output["sample_atom_coords"] = torch.cat(
            (output["sample_atom_coords"], torch.full((1, 1, 3), float("nan"))), dim=1
        )
        features["atom_attention_mask"] = torch.cat(
            (features["atom_attention_mask"], torch.tensor([[False]])), dim=1
        )
        features["atom_to_token"] = torch.cat(
            (features["atom_to_token"], torch.tensor([[-999]])), dim=1
        )
        features["ref_element"] = torch.cat(
            (features["ref_element"], torch.tensor([[0]])), dim=1
        )
    return features, output


def test_raw_physical_interface_uses_two_contacted_chains_and_native_directional_min():
    features, output = _raw_case(padding=True)
    summary = _backend()._contacted_summary_from_raw(output, features, ["TT"] * 3, 3)
    assert summary["pair_chain_iptm"] == pytest.approx(.55)
    assert summary["target_interface_selection"]["contact_counts"] == [6, 6, 0]
    assert summary["target_interface_selection"]["selected_indices"] == [0, 1]
    assert summary["target_interface_eligible"] is True
    assert summary["pair_chain_iptm_axis_order"] == ["A", "B", "C", "D"]
    assert summary["target_interface_chain_token_lengths"] == [2, 2, 2, 3]


def test_raw_physical_interface_fails_closed_without_contacts():
    features, output = _raw_case()
    output["sample_atom_coords"][0, 6:, 0] += 1000.
    summary = _backend()._contacted_summary_from_raw(output, features, ["TT"] * 3, 3)
    assert summary["pair_chain_iptm"] == 0.
    assert summary["target_interface_eligible"] is False
    assert summary["target_interface_selection"]["selected_indices"] == []


def test_raw_interface_excludes_hydrogen_without_mutating_input_feature_mask():
    features, output = _raw_case()
    output["sample_atom_coords"][0, 4:6, 0] = 1.
    features["ref_element"][0, 4:6] = 1
    original_atom_mask = features["atom_attention_mask"].clone()
    summary = _backend()._contacted_summary_from_raw(output, features, ["TT"] * 3, 3)
    assert summary["target_interface_selection"]["contact_counts"] == [6, 6, 0]
    assert torch.equal(features["atom_attention_mask"], original_atom_mask)


@pytest.mark.parametrize("mutation, message", [
    ("axis", "contiguous native chain axes"),
    ("repeated_axis", "contiguous native chain axes"),
    ("token_length", "chain token lengths mismatch"),
    ("layout", "token/asym layout mismatch"),
    ("coordinates", "Nonfinite interface coordinates"),
    ("pair_shape", "match target chains"),
    ("pair_nonfinite", "Nonfinite native"),
    ("pair_missing", "requires native"),
])
def test_raw_contacted_summary_rejects_invalid_model_layout(mutation, message):
    features, output = _raw_case()
    if mutation == "axis":
        features["asym_id"][0, :2] = 7
    elif mutation == "repeated_axis":
        features["asym_id"][0, :4] = torch.tensor([0, 1, 0, 1])
    elif mutation == "token_length":
        features["token_attention_mask"][0, 0] = False
    elif mutation == "layout":
        features["token_attention_mask"] = torch.ones(1, 8, dtype=torch.bool)
    elif mutation == "coordinates":
        output["sample_atom_coords"][0, 0, 0] = float("nan")
    elif mutation == "pair_shape":
        output["pair_chains_iptm"] = torch.ones(3, 3)
    elif mutation == "pair_nonfinite":
        output["pair_chains_iptm"][0, 0, 3] = float("inf")
    else:
        del output["pair_chains_iptm"]
    with pytest.raises(ValueError, match=message):
        _backend()._contacted_summary_from_raw(output, features, ["TT"] * 3, 3)


class _MolecularComplex:
    def __init__(self, coordinates):
        self.sequence = ["THR"] * 6 + ["ALA"] * 3
        self.chain_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3])
        self.atom_positions = coordinates[0].clone()
        self.atom_elements = ["C"] * 9
        self.token_to_atoms = torch.tensor([[i, i + 1] for i in range(9)])

    def to_mmcif(self):
        return "data_trimer\n"


def test_full_verifier_bounded_score_uses_physical_effective_interface():
    features, output = _raw_case()
    fold_result = SimpleNamespace(
        complex=_MolecularComplex(output["sample_atom_coords"]),
        plddt=torch.full((9,), .8), ptm=.7, iptm=.6,
        pair_chains_iptm=output["pair_chains_iptm"],
    )
    result = _backend()._build_verifier_result_from_fold(
        binder_seq="AAA", target_seq="TT|TT|TT", targets=["TT"] * 3,
        target_chain_roles=("target",) * 3, hotspot_params=None,
        fold_result=fold_result, raw_features=features, raw_output=output,
    )
    assert result.summary_confidence["pair_chain_iptm"] == pytest.approx(.55)
    assert result.summary_confidence["pair_chain_iptm_legacy_all_target_mean"] == pytest.approx((.8 + .6 + .99) / 3)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(.45)
    assert result.loss_dict["total"] == pytest.approx(.45 + .2 * .2 + .1 * .3)
    assert result.summary_confidence["target_interface_eligible"] is True
    assert result.coordinate.shape == (1, 9, 3)


@pytest.mark.parametrize("route", ["preview", "soft_update"])
def test_proposal_routes_forward_chain_lengths_and_mask_when_roles_implicit(monkeypatch, route):
    mask = (False, True, False)
    backend = _backend(binder_mutable_mask=mask, target_chain_roles=None)
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)
    original = runtime_module._structure_losses
    captured = []
    captured_metrics = []
    original_update = backend._run_soft_update_impl

    def capture_update(**kwargs):
        result = original_update(**kwargs)
        captured_metrics.extend(result[3])
        return result

    def spy(logits, binder_length, **kwargs):
        captured.append(kwargs)
        return original(logits, binder_length, **kwargs)

    def fake_fold(**kwargs):
        design = kwargs["design"]
        bins = torch.arange(64, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum() * .1
        return {"distogram_logits": signal * bins.reshape(1, 1, 1, -1).expand(1, 9, 9, 64)}

    monkeypatch.setattr(runtime_module, "_structure_losses", spy)
    monkeypatch.setattr(backend, "_fold_distogram", fake_fold)
    monkeypatch.setattr(backend, "_run_soft_update_impl", capture_update)
    if route == "preview":
        scores = backend.preview_loss_dict(binder_seq="AAA", target_seq="TT|TT|TT", seed=3)
        assert all(torch.isfinite(torch.tensor(value)) for value in scores.values())
    else:
        before, after = backend._run_soft_update(
            current_seq="AAA", target_seq="TT|TT|TT", global_step=0,
            seed=3, mutable_mask=mask, action_group=None, proposal_logits=None,
        )
        # The existing sampler deliberately suppresses new Cys with -inf.
        allowed = [index for index in range(20) if index != CYS_INDEX]
        assert torch.isfinite(after[:, allowed]).all()
        assert not torch.isnan(after).any()
        assert torch.equal(before[0], after[0])
        assert torch.equal(before[2], after[2])
        assert len(captured_metrics) == 1
        selection = captured_metrics[0]["target_interface_selection"]
        assert selection["source"] == "distogram_expected_residue_pairs"
        assert selection["policy_version"] == "contacted_bidirectional_min_mean_v1"
        assert selection["distogram_head_num_bins"] == 64
        assert len(selection["expected_contacts"]) == 3
        assert selection["selected_indices"]
        assert isinstance(selection["fallback"], bool)
    assert captured
    assert all(call["target_chain_lengths"] == [2, 2, 2] for call in captured)
    assert all(call["binder_mutable_mask"] == mask for call in captured)
    assert all(call["target_interface_mode"] == "contacted" for call in captured)
    assert all(call["target_interface_contact_cutoff"] == 5. for call in captured)
    assert all(call["target_interface_min_contacts"] == 5 for call in captured)


def test_structure_input_builds_three_targets_and_one_binder_only_in_explicit_mode():
    inputs = _backend()._build_structure_input(["TT"] * 3, "AAA")
    assert [protein.id for protein in inputs.sequences] == ["A", "B", "C", "D"]
    assert [protein.sequence for protein in inputs.sequences] == ["TT", "TT", "TT", "AAA"]
    legacy = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(), device="cpu",
        protein_input_cls=SimpleNamespace, structure_prediction_input_cls=SimpleNamespace,
    )
    assert len(legacy._build_structure_input(["TT", "GG"], "AAA").sequences) == 3


@pytest.mark.parametrize("targets", [["TT"], ["TT"] * 2, ["TT"] * 4, ["TT", "TT", "GG"], ["XX"] * 3, [""] * 3, ["tt"] * 3])
def test_structure_input_rejects_wrong_explicit_homotrimer_assembly(targets):
    with pytest.raises(ValueError, match="homotrimer"):
        _backend()._build_structure_input(targets, "AAA")


@pytest.mark.parametrize("binder", ["", "AXA", "aaa", "AA|A", "A A"])
def test_structure_input_rejects_invalid_single_chain_binder(binder):
    with pytest.raises(ValueError, match="nonempty canonical binder"):
        _backend()._build_structure_input(["TT"] * 3, binder)
