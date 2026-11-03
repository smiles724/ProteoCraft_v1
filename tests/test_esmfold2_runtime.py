import sys
import types
from contextlib import contextmanager

import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESMFold2ProposalConfig,
    init_binder_logits,
    sequence_from_logits_argmax,
)
from proteocraft.hallucination.esmfold2_runtime import (
    ESMFold2RuntimeBackend,
    _hotspot_structure_metrics,
    _mean_target_to_binder_pair_iptm,
    _scheduled_one_step_soft_steps,
    _structure_losses,
    _stageb_structure_loss,
    _validate_decoded_coordinate_layout,
)
import proteocraft.hallucination.esmfold2_runtime as esmfold2_runtime_module


def test_scheduled_one_step_soft_steps_include_interval_and_final_step():
    assert _scheduled_one_step_soft_steps(soft_steps=30, interval=0) == ()
    assert _scheduled_one_step_soft_steps(
        soft_steps=30,
        interval=5,
    ) == (5, 10, 15, 20, 25, 30)
    assert _scheduled_one_step_soft_steps(
        soft_steps=30,
        interval=7,
    ) == (7, 14, 21, 28, 30)


class FakeComplex:
    def __init__(self):
        self.atom37_positions = torch.zeros(7, 37, 3)
        self.atom37_mask = torch.zeros(7, 37, dtype=torch.bool)
        for residue_index, x_coord in enumerate([0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 8.0]):
            self.atom37_positions[residue_index, 1] = torch.tensor([x_coord, 0.0, 0.0])
            self.atom37_mask[residue_index, 1] = True

    def to_mmcif_string(self):
        return "data_fake\n"


class FakeModernComplex(FakeComplex):
    def to_mmcif(self):
        return "data_modern\n"


class FakeGeometryComplex(FakeComplex):
    def __init__(self):
        self.chain_boundaries = [(0, 4), (5, 8)]
        self.atom37_positions = torch.zeros(8, 37, 3)
        self.atom37_mask = torch.zeros(8, 37, dtype=torch.bool)
        target_coordinates = [0.0, 10.0, 24.0, 30.0]
        for residue_index, x_coord in enumerate(target_coordinates):
            self.atom37_positions[residue_index, 0] = torch.tensor(
                [x_coord, 0.0, 0.0]
            )
            self.atom37_mask[residue_index, 0] = True
        for residue_index in range(5, 8):
            self.atom37_positions[residue_index, 0] = torch.tensor(
                [4.0, 0.0, 0.0]
            )
            self.atom37_mask[residue_index, 0] = True


class FakeMolecularComplex(FakeComplex):
    def __init__(self):
        self.sequence = ["ALA"] * 7
        self.chain_id = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        self.atom_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [24.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        self.atom_elements = ["C"] * 7
        self.token_to_atoms = torch.tensor(
            [[index, index + 1] for index in range(7)]
        )


class FakeFoldResult:
    def __init__(self):
        self.complex = FakeComplex()
        self.plddt = torch.tensor([0.8875, 0.8875, 0.8875, 0.8875, 0.7, 0.8, 0.9])
        self.ptm = 0.72
        self.iptm = 0.61
        self.pair_chains_iptm = torch.tensor(
            [
                [1.0, 0.83],
                [0.41, 1.0],
            ]
        )


class FakeModernFoldResult(FakeFoldResult):
    def __init__(self):
        super().__init__()
        self.complex = FakeModernComplex()


class FakeGeometryFoldResult(FakeFoldResult):
    def __init__(self):
        super().__init__()
        self.complex = FakeGeometryComplex()


class FakeMolecularFoldResult(FakeFoldResult):
    def __init__(self):
        super().__init__()
        self.complex = FakeMolecularComplex()


class FakeFoldResultBatchedPairIptm(FakeFoldResult):
    def __init__(self):
        super().__init__()
        self.pair_chains_iptm = torch.tensor(
            [
                [
                    [1.0, 0.74],
                    [0.31, 1.0],
                ]
            ]
        )


def test_validate_decoded_coordinate_layout_uses_valid_atoms_not_padding():
    coordinate = torch.zeros(1, 3, 3)
    atom_mask = torch.tensor([[True, True, False, True, False]])

    assert (
        _validate_decoded_coordinate_layout(coordinate, atom_mask)
        is coordinate
    )


def test_validate_decoded_coordinate_layout_rejects_padded_coordinates():
    coordinate = torch.zeros(1, 5, 3)
    atom_mask = torch.tensor([[True, True, False, True, False]])

    with pytest.raises(
        RuntimeError,
        match=r"decoded=5, expected_valid=3",
    ):
        _validate_decoded_coordinate_layout(coordinate, atom_mask)


def test_role_aware_pair_chain_iptm_excludes_scaffold_chain():
    pair_chains_iptm = torch.tensor(
        [
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.1],
            [0.8, 0.1, 1.0],
        ]
    )

    target_score = _mean_target_to_binder_pair_iptm(
        pair_chains_iptm,
        n_target_chains=2,
        target_chain_roles=("target", "scaffold"),
    )
    scaffold_score = _mean_target_to_binder_pair_iptm(
        pair_chains_iptm,
        n_target_chains=2,
        target_chain_roles=("target", "scaffold"),
        selected_role="scaffold",
    )

    assert target_score == pytest.approx(0.8)
    assert scaffold_score == pytest.approx(0.1)


def test_cdr_structure_loss_uses_true_target_and_mutable_cdr_only(monkeypatch):
    captured_masks = []

    def fake_contact_loss(
        distogram_logits,
        bin_distance,
        *,
        chain_mask,
        binder_mask,
        **kwargs,
    ):
        captured_masks.append(
            (
                chain_mask.detach().cpu().tolist(),
                binder_mask.detach().cpu().tolist(),
            )
        )
        return torch.tensor([0.1], dtype=distogram_logits.dtype)

    monkeypatch.setattr(
        esmfold2_runtime_module,
        "_contact_loss",
        fake_contact_loss,
    )
    _structure_losses(
        torch.zeros(1, 7, 7, 128),
        binder_length=3,
        target_chain_lengths=(2, 2),
        target_chain_roles=("target", "scaffold"),
        binder_mutable_mask=(False, True, False),
    )

    inter_chain_mask, inter_binder_mask = captured_masks[1]
    assert inter_chain_mask == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert inter_binder_mask == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_structure_losses_support_full_verifier_64_bin_distogram():
    losses = _structure_losses(
        torch.zeros(1, 7, 7, 64),
        binder_length=3,
    )
    assert torch.isfinite(losses["total_loss"]).all()


class FakeInputBuilder:
    def __init__(self):
        self.calls = []

    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeFoldResult()


class FakeModernInputBuilder(FakeInputBuilder):
    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeModernFoldResult()


class FakeBatchedPairInputBuilder(FakeInputBuilder):
    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeFoldResultBatchedPairIptm()


class FakeGeometryInputBuilder(FakeInputBuilder):
    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeGeometryFoldResult()


class FakeMolecularInputBuilder(FakeInputBuilder):
    def fold(self, model, input_spec, **kwargs):
        self.calls.append((model, input_spec, kwargs))
        return FakeMolecularFoldResult()


class FakeProteinInput:
    def __init__(self, *, id, sequence, msa=None):
        self.id = id
        self.sequence = sequence
        self.msa = msa


class FakeStructurePredictionInput:
    def __init__(self, *, sequences):
        self.sequences = sequences


def test_runtime_verifier_extracts_pair_chain_metrics_and_mmcif():
    builder = FakeInputBuilder()
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=builder,
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)

    assert result.total_loss == pytest.approx(0.17)
    assert result.summary_confidence["pair_chain_iptm"] == pytest.approx(0.83)
    assert result.summary_confidence["iptm"] == pytest.approx(0.61)
    assert result.summary_confidence["ptm"] == pytest.approx(0.72)
    assert result.summary_confidence["plddt"] == pytest.approx(0.85)
    assert result.summary_confidence["binder_plddt"] == pytest.approx(0.8)
    assert result.loss_dict["binder_confidence_loss"] == pytest.approx(0.2)
    assert result.loss_dict["binder_rg_excess_loss"] == pytest.approx(0.0)
    assert result.summary_confidence["mmcif"] == "data_fake\n"
    assert "verifier_structure_mode" not in result.summary_confidence

    _, input_spec, kwargs = builder.calls[0]
    assert [seq.id for seq in input_spec.sequences] == ["A", "B"]
    assert [seq.sequence for seq in input_spec.sequences] == ["AAAA", "YYY"]
    assert kwargs["seed"] == 7
    assert kwargs["num_loops"] == backend.verifier_num_loops
    assert kwargs["num_sampling_steps"] == backend.verifier_num_sampling_steps


def test_scaffold_consistency_verifier_requires_explicit_opt_in():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    with pytest.raises(ValueError, match="verifier_structure_mode"):
        backend.verify_scaffold_consistency(
            binder_seq="YYY",
            target_seq="AAAA",
            seed=7,
        )


def test_scaffold_consistency_sampler_restores_standard_sampler_after_use():
    class FakeStructureHead:
        def sample(self, **kwargs):
            return {"sample_atom_coords": kwargs["ref_pos"]}

    class FakeVerifierModel:
        def __init__(self):
            self.structure_head = FakeStructureHead()

    model = FakeVerifierModel()
    original = model.structure_head.sample
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=model,
    )
    with backend._temporary_scaffold_consistency_sampler(
        template_coordinates=torch.zeros(1, 3, 3),
        fixed_atom_mask=torch.tensor([[True, False, True]]),
    ):
        assert model.structure_head.sample.__func__ is not original.__func__

    assert model.structure_head.sample.__func__ is original.__func__


def test_runtime_verifier_uses_modern_to_mmcif_api():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeModernInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)

    assert result.summary_confidence["mmcif"] == "data_modern\n"


def test_runtime_verifier_extracts_batched_pair_chain_metrics():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeBatchedPairInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)

    assert result.total_loss == pytest.approx(0.26)
    assert result.summary_confidence["pair_chain_iptm"] == pytest.approx(0.74)
    assert result.summary_confidence["esm_verify_source"] == "pair_chain_iptm"


def test_runtime_hotspot_verifier_combines_full_structure_distance_with_global_loss():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            hotspot_verify_weight=0.2,
            hotspot_verify_near_cutoff=5.0,
            hotspot_verify_far_cutoff=20.0,
        ),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeGeometryInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(
        binder_seq="YYY",
        target_seq="AAAA",
        seed=7,
        hotspot_params={"target_hotspot_concat_indices": [0, 2]},
    )

    assert result.loss_dict["esm_verify_global_loss"] == pytest.approx(0.17)
    assert result.loss_dict["hotspot_verify_distance_loss"] == pytest.approx(0.5)
    assert result.total_loss == pytest.approx(0.27)
    assert result.summary_confidence["hotspot_mean_min_heavy_atom_distance"] == pytest.approx(12.0)
    assert result.summary_confidence["hotspot_heavy_atom_coverage"] == pytest.approx(0.5)
    assert result.summary_confidence["hotspot_heavy_atom_covered_count"] == 1
    assert result.summary_confidence["hotspot_heavy_atom_coverage_5a"] == pytest.approx(0.5)
    assert result.summary_confidence["hotspot_heavy_atom_covered_count_5a"] == 1


def test_runtime_tracks_hotspot_geometry_without_adding_hotspot_reward():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            hotspot_verify_weight=0.0,
            hotspot_verify_too_close_weight=0.0,
            hotspot_verify_near_cutoff=5.0,
            hotspot_verify_far_cutoff=20.0,
        ),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeGeometryInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(
        binder_seq="YYY",
        target_seq="AAAA",
        seed=7,
        hotspot_params={"target_hotspot_concat_indices": [0, 2]},
    )

    assert result.total_loss == pytest.approx(0.17)
    assert result.loss_dict["esm_verify_loss"] == pytest.approx(0.17)
    assert result.loss_dict["hotspot_verify_distance_loss"] == pytest.approx(0.5)
    assert result.summary_confidence["hotspot_heavy_atom_coverage"] == pytest.approx(0.5)
    assert result.summary_confidence["hotspot_heavy_atom_covered_count"] == 1


def test_runtime_hotspot_verifier_supports_current_molecular_complex_output():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            hotspot_verify_weight=0.2,
            hotspot_verify_near_cutoff=5.0,
        ),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeMolecularInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    result = backend.verify(
        binder_seq="YYY",
        target_seq="AAAA",
        seed=7,
        hotspot_params={"target_hotspot_concat_indices": [0, 2]},
    )

    assert result.total_loss == pytest.approx(0.27)
    assert result.summary_confidence["hotspot_mean_min_heavy_atom_distance"] == pytest.approx(12.0)
    assert result.summary_confidence["hotspot_heavy_atom_coverage"] == pytest.approx(0.5)


def test_runtime_hotspot_verifier_requires_hotspot_indices_when_enabled():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(hotspot_verify_weight=0.2),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeGeometryInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    with pytest.raises(
        ValueError,
        match="hotspot-aware verifier requires target_hotspot_concat_indices",
    ):
        backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7)


def test_hotspot_structure_metrics_map_concatenated_indices_across_target_chains():
    complex_obj = FakeGeometryComplex()
    complex_obj.chain_boundaries = [(0, 2), (2, 4), (5, 8)]

    metrics = _hotspot_structure_metrics(
        complex_obj,
        target_sequences=["AA", "AA"],
        hotspot_params={"target_hotspot_concat_indices": [2]},
        near_cutoff=5.0,
        far_cutoff=20.0,
    )

    assert metrics["hotspot_min_heavy_atom_distances"] == pytest.approx([20.0])
    assert metrics["hotspot_verify_distance_loss"] == pytest.approx(1.0)


def test_hotspot_structure_metrics_penalize_unsafe_heavy_atom_distance():
    complex_obj = FakeGeometryComplex()
    complex_obj.chain_boundaries = [(0, 4), (5, 8)]
    complex_obj.atom37_positions[5:8, 0] = torch.tensor([1.8, 0.0, 0.0])

    metrics = _hotspot_structure_metrics(
        complex_obj,
        target_sequences=["AAAA"],
        hotspot_params={"target_hotspot_concat_indices": [0]},
        near_cutoff=8.0,
        far_cutoff=20.0,
        min_safe_distance=2.5,
    )

    assert metrics["hotspot_verify_distance_loss"] == pytest.approx(0.0)
    assert metrics["hotspot_too_close_loss"] == pytest.approx(
        (2.5 - 1.8) / 2.5
    )
    assert metrics["hotspot_verify_min_distance"] == pytest.approx(2.5)


@pytest.mark.parametrize("complex_class", [FakeGeometryComplex, FakeMolecularComplex])
def test_hotspot_cdr_mask_rejects_close_non_cdr_decoys(complex_class):
    complex_obj = complex_class()
    # Only the middle binder residue is a CDR: distant, despite close fixed residues.
    if complex_class is FakeGeometryComplex:
        complex_obj.atom37_positions[6, 0, 0] = 20.0
    else:
        complex_obj.atom_positions[5, 0] = 20.0
    kwargs = dict(target_sequences=["AAAA"], hotspot_params={"target_hotspot_concat_indices": [0]},
                  near_cutoff=5.0, far_cutoff=20.0)
    unmasked = _hotspot_structure_metrics(complex_obj, **kwargs)
    masked = _hotspot_structure_metrics(complex_obj, binder_contact_mask=[False, True, False], **kwargs)
    assert unmasked["hotspot_heavy_atom_coverage_5a"] == 1.0
    assert masked["hotspot_heavy_atom_coverage_5a"] == 0.0
    assert masked["hotspot_verify_distance_loss"] == pytest.approx(1.0)
    assert masked["hotspot_min_heavy_atom_distances"] == pytest.approx([20.0])
    assert masked["hotspot_binder_contact_positions_1based"] == [2]


@pytest.mark.parametrize("mask", [[True], [False, False, False], [0, 0.5, 0]])
def test_hotspot_structure_metrics_rejects_invalid_cdr_mask(mask):
    with pytest.raises(ValueError, match="binder_contact_mask"):
        _hotspot_structure_metrics(FakeGeometryComplex(), target_sequences=["AAAA"],
            hotspot_params={"target_hotspot_concat_indices": [0]}, near_cutoff=5.0,
            far_cutoff=20.0, binder_contact_mask=mask)


@pytest.mark.parametrize("builder_class", [FakeGeometryInputBuilder, FakeMolecularInputBuilder])
def test_vhh_runtime_passes_cdr_mask_into_verifier(builder_class):
    class DistantCdrInputBuilder(builder_class):
        def fold(self, *args, **kwargs):
            result = super().fold(*args, **kwargs)
            if isinstance(result.complex, FakeMolecularComplex):
                result.complex.atom_positions[5, 0] = 20.0
            else:
                result.complex.atom37_positions[6, 0, 0] = 20.0
            return result

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(binder_mutable_mask=(False, True, False),
            hotspot_verify_weight=0.2, hotspot_verify_near_cutoff=5.0, hotspot_verify_far_cutoff=20.0),
        device="cpu", task_type="vhh_redesign", verifier_model=object(),
        input_builder=DistantCdrInputBuilder(), protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput)
    result = backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7,
                            hotspot_params={"target_hotspot_concat_indices": [0]})
    assert result.summary_confidence["hotspot_binder_contact_scope"] == "vhh_cdr"
    assert result.summary_confidence["hotspot_heavy_atom_coverage_5a"] == 0.0
    assert result.total_loss == pytest.approx(0.37)


def test_vhh_hotspot_verification_requires_explicit_cdr_mask():
    backend = ESMFold2RuntimeBackend(config=ESMFold2ProposalConfig(), device="cpu",
        task_type="vhh_redesign", verifier_model=object(), input_builder=FakeGeometryInputBuilder(),
        protein_input_cls=FakeProteinInput, structure_prediction_input_cls=FakeStructurePredictionInput)
    with pytest.raises(ValueError, match="CDR mutable mask"):
        backend.verify(binder_seq="YYY", target_seq="AAAA", seed=7,
                       hotspot_params={"target_hotspot_concat_indices": [0]})


def test_runtime_preserves_matching_free_proposal_logits_without_hard_anchor():
    generator = torch.Generator().manual_seed(11)
    proposal_logits = init_binder_logits(
        "AAA",
        mutable_mask=[True, True, True],
        random_scale=0.01,
        requires_grad=False,
        generator=generator,
    )[0]
    current_seq = sequence_from_logits_argmax(
        proposal_logits,
        current_seq="AAA",
        mutable_mask=[True, True, True],
    )

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
    )
    initialized = backend._init_current_sequence_logits(
        current_seq,
        mutable_mask=[True, True, True],
        proposal_logits=proposal_logits,
    )

    assert initialized.shape == (1, 3, len(AA_TO_ESM_IDX))
    for pos, aa in enumerate(current_seq):
        aa_idx = AA_TO_ESM_IDX[aa]
        assert initialized[0, pos, aa_idx].item() == pytest.approx(
            proposal_logits[pos, aa_idx].item()
        )
        assert initialized[0, pos, aa_idx].item() < 0.5


def test_runtime_can_preserve_soft_logits_when_sequence_is_a_local_branch():
    proposal_logits = torch.full((3, len(AA_TO_ESM_IDX)), -1.0)
    proposal_logits[:, AA_TO_ESM_IDX["C"]] = float("-inf")
    proposal_logits[:, AA_TO_ESM_IDX["A"]] = 0.5
    proposal_logits[1, AA_TO_ESM_IDX["Y"]] = 0.4

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
    )
    initialized = backend._init_current_sequence_logits(
        "AYA",
        mutable_mask=[True, True, True],
        proposal_logits=proposal_logits,
        preserve_proposal_logits=True,
    )

    assert torch.equal(initialized[0].detach().cpu(), proposal_logits)


def test_runtime_configures_official_proposal_lm_dropout(monkeypatch):
    from proteocraft.hallucination import esmfold2_runtime as runtime
    monkeypatch.setattr(
        runtime, "resolve_model_source",
        lambda name: "/cache/pinned-proposal" if name == "biohub/ESMFold2-Experimental-Fast" else name,
    )
    class FakeExperimentalModel:
        last_instance = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.last_instance = cls()
            cls.last_instance.from_pretrained_args = args
            cls.last_instance.from_pretrained_kwargs = kwargs
            cls.last_instance.config = types.SimpleNamespace(esmc_id="fake-esmc")
            return cls.last_instance

        def load_esmc(self, esmc_id):
            self.load_esmc_call = esmc_id

        def configure_lm_dropout(self, lm_dropout, *, force_lm_dropout_during_inference):
            self.dropout_call = {
                "lm_dropout": lm_dropout,
                "force_lm_dropout_during_inference": force_lm_dropout_during_inference,
            }

        def set_kernel_backend(self, backend):
            self.kernel_backend = backend

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def requires_grad_(self, requires_grad):
            self.requires_grad = requires_grad
            return self

    module = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2_experimental")
    module.ESMFold2ExperimentalModel = FakeExperimentalModel
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.models", types.ModuleType("transformers.models"))
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2",
        types.ModuleType("transformers.models.esmfold2"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2.modeling_esmfold2_experimental",
        module,
    )

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            experimental_model_name="biohub/ESMFold2-Experimental-Fast",
            lambda_esmc=0.0,
            proposal_lm_dropout=0.5,
            force_proposal_lm_dropout=True,
        ),
        device="cpu",
    )

    backend._ensure_proposal_models_loaded()

    instance = FakeExperimentalModel.last_instance
    assert instance.from_pretrained_args == ("/cache/pinned-proposal",)
    assert instance.from_pretrained_kwargs["load_esmc"] is False
    assert instance.load_esmc_call == "fake-esmc"
    assert instance.dropout_call == {
        "lm_dropout": 0.5,
        "force_lm_dropout_during_inference": True,
    }


def test_runtime_propose_uses_injected_soft_update_and_logit_delta_candidates():
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}

    seen = {}

    def fake_soft_update(
        *,
        current_seq,
        target_seq,
        global_step,
        seed,
        action_group,
        mutable_mask,
        proposal_logits=None,
    ):
        seen["action_group"] = action_group
        seen["mutable_mask"] = mutable_mask
        seen["proposal_logits"] = proposal_logits
        before = torch.zeros(len(current_seq), 20)
        after = before.clone()
        after[1, aa_to_idx["Y"]] = 3.0
        after[2, aa_to_idx["F"]] = 2.0
        return before, after

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
        soft_update_fn=fake_soft_update,
    )

    candidates = backend.propose(
        current_seq="ALD",
        target_seq="AAAA",
        global_step=0,
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
        seed=5,
        action_group="interface_formation",
        mutable_mask=[False, True, True],
    )

    assert "AYD" in candidates
    assert all(seq[0] == "A" for seq in candidates)
    assert seen["action_group"] == "interface_formation"
    assert seen["mutable_mask"] == [False, True, True]
    assert seen["proposal_logits"] is None


def test_runtime_propose_candidate_sampling_is_reproducible_for_fixed_seed():
    def fake_soft_update(**kwargs):
        current_seq = kwargs["current_seq"]
        before = torch.zeros(len(current_seq), 20)
        after = before.clone()
        for position in range(len(current_seq)):
            after[position, (position + 1) % 20] = 3.0 - 0.1 * position
            after[position, (position + 2) % 20] = 2.0 - 0.1 * position
        return before, after

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
        soft_update_fn=fake_soft_update,
    )
    kwargs = {
        "current_seq": "AAAAAA",
        "target_seq": "TTTT",
        "global_step": 3,
        "K_top": 8,
        "B_candidates": 6,
        "T_sample": 1.0,
        "max_mutations_per_seq": 3,
        "seed": 17,
        "action_group": "interface_formation",
    }

    torch.manual_seed(1)
    first = list(backend.propose(**kwargs).sequences)
    torch.manual_seed(999)
    second = list(backend.propose(**kwargs).sequences)

    assert first == second


@pytest.mark.parametrize("strategy", ["positive_score", "positive_score_guarded"])
def test_positive_score_runtime_excludes_unbounded_confidence_candidates(
    monkeypatch, strategy,
):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )
    before = torch.zeros(4, 20)
    after = before.clone()
    after[0, AA_TO_ESM_IDX["Y"]] = 4.0
    after[1, AA_TO_ESM_IDX["F"]] = 3.0
    monkeypatch.setattr(
        backend,
        "_run_soft_update_with_tracking",
        lambda **kwargs: (
            before,
            after,
            [{"sequence": "YYYY", "confidence_score": 1.0}],
            [],
        ),
    )

    batch = backend.propose(
        current_seq="AAAA",
        target_seq="TTTT",
        global_step=0,
        K_top=10,
        B_candidates=8,
        T_sample=1.0,
        max_mutations_per_seq=3,
        mutation_fraction=0.25,
        candidate_strategy=strategy,
        include_argmax_candidate=False,
        seed=17,
    )

    assert "YYYY" not in batch.sequences
    assert all(
        sum(a != b for a, b in zip("AAAA", sequence)) == 1
        for sequence in batch.sequences
    )
    assert batch.proposal_diagnostics["confidence_candidate_count"] == 1
    assert batch.proposal_diagnostics["returned_confidence_candidate_count"] == 0


def test_runtime_propose_can_continue_from_saved_proposal_logits():
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
    saved_logits = torch.zeros(3, 20)
    saved_logits[1, aa_to_idx["F"]] = 4.0
    seen = {}

    def fake_soft_update(
        *,
        current_seq,
        target_seq,
        global_step,
        seed,
        action_group,
        mutable_mask,
        proposal_logits=None,
    ):
        seen["proposal_logits"] = proposal_logits
        before = proposal_logits.detach().clone()
        after = before.clone()
        after[2, aa_to_idx["Y"]] = 5.0
        return before, after

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
        soft_update_fn=fake_soft_update,
    )

    batch = backend.propose(
        current_seq="AAA",
        target_seq="TTTT",
        global_step=0,
        K_top=2,
        B_candidates=1,
        T_sample=0.01,
        max_mutations_per_seq=1,
        proposal_logits=saved_logits,
    )

    assert seen["proposal_logits"] is not None
    assert seen["proposal_logits"][1, aa_to_idx["F"]].item() == pytest.approx(4.0)
    assert batch.sequences == ["AFY"]
    assert batch.logits_after[1, aa_to_idx["F"]].item() == pytest.approx(4.0)
    assert batch.logits_after[2, aa_to_idx["Y"]].item() == pytest.approx(5.0)


def test_runtime_continuous_design_records_soft_step_metrics(monkeypatch):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=2,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
            one_step_diffusion_enabled=False,
            one_step_diffusion_interval=0,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        design = kwargs["design"]
        full_len = len(kwargs["target_seq"]) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        distogram = signal * bins.reshape(1, 1, 1, -1).expand(1, full_len, full_len, 128)
        return {"distogram_logits": distogram}

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq="TTTT",
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
    )

    assert len(batch.soft_step_metrics) == 2
    assert batch.soft_step_metrics[0]["local_step"] == 0
    assert batch.soft_step_metrics[0]["global_step"] == 0
    assert "intra_contact_loss" in batch.soft_step_metrics[0]
    assert "inter_contact_loss" in batch.soft_step_metrics[0]
    assert "glob_loss" in batch.soft_step_metrics[0]
    assert "structure_total_loss" in batch.soft_step_metrics[0]
    assert "backprop_loss" in batch.soft_step_metrics[0]
    assert "temperature" in batch.soft_step_metrics[0]
    assert "argmax_sequence" in batch.soft_step_metrics[0]
    assert "logits_snapshot" in batch.soft_step_metrics[0]


def test_runtime_schedules_and_recycles_interval_one_step_coordinates(monkeypatch):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=6,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
            one_step_diffusion_enabled=True,
            one_step_diffusion_mode="reference_noising",
            one_step_diffusion_interval=2,
            one_step_diffusion_recycle_coordinates=True,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        design = kwargs["design"]
        full_len = len(kwargs["target_seq"]) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        return {
            "distogram_logits": signal
            * bins.reshape(1, 1, 1, -1).expand(
                1,
                full_len,
                full_len,
                128,
            )
        }

    references = []
    reference_feature_markers = []

    def fake_one_step_diffusion_gradient(**kwargs):
        references.append(kwargs["reference_coordinates"].detach().clone())
        reference_feature_markers.append(
            float(kwargs["reference_features"]["marker"].item())
        )
        call_index = len(references)
        return {
            "gradient": torch.ones_like(kwargs["logits"]),
            "total": torch.tensor(float(call_index)),
            "losses": {"rg": torch.tensor(0.5)},
            "coordinates": torch.full(
                (1, 1, 3),
                float(call_index),
            ),
            "features": {"marker": torch.tensor(float(call_index))},
            "reference_rmsd": torch.tensor(0.25),
            "gradient_norm": 1.0,
            "mutable_gradient_norm": 1.0,
        }

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)
    monkeypatch.setattr(
        backend,
        "_one_step_diffusion_gradient",
        fake_one_step_diffusion_gradient,
    )

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq="TTTT",
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
        reference_coordinates=torch.zeros(1, 1, 3),
        reference_features={"marker": torch.tensor(99.0)},
    )

    assert len(batch.soft_step_metrics) == 6
    assert len(references) == 3
    assert torch.equal(references[0], torch.zeros(1, 1, 3))
    assert torch.equal(references[1], torch.ones(1, 1, 3))
    assert torch.equal(references[2], torch.full((1, 1, 3), 2.0))
    assert reference_feature_markers == [99.0, 1.0, 2.0]
    injection_metrics = [
        metrics
        for metrics in batch.soft_step_metrics
        if metrics.get("one_step_diffusion")
    ]
    assert [metrics["local_step"] for metrics in injection_metrics] == [1, 3, 5]
    assert [
        metrics["one_step_diffusion_used_recycled_coordinates"]
        for metrics in injection_metrics
    ] == [False, True, True]


def test_hotspot_structure_loss_is_additive_and_weight_zero_preserves_base_loss():
    torch.manual_seed(3)
    distogram = torch.randn(1, 7, 7, 128, requires_grad=True)
    hotspot_params = {"target_hotspot_concat_indices": [1, 3, 1]}

    baseline = _structure_losses(distogram, binder_length=3)
    diagnostic = _structure_losses(
        distogram,
        binder_length=3,
        hotspot_params=hotspot_params,
        hotspot_weight=0.0,
    )
    weighted = _structure_losses(
        distogram,
        binder_length=3,
        hotspot_params=hotspot_params,
        hotspot_weight=0.4,
    )

    assert torch.allclose(diagnostic["total_loss"], baseline["total_loss"])
    assert torch.allclose(diagnostic["base_structure_loss"], baseline["total_loss"])
    assert torch.allclose(
        weighted["total_loss"],
        baseline["total_loss"] + 0.4 * weighted["hotspot_contact_loss"],
    )
    weighted["total_loss"].mean().backward()
    assert distogram.grad is not None
    assert torch.isfinite(distogram.grad).all()


def test_hotspot_structure_loss_uses_independent_strict_cutoff():
    distogram = torch.zeros(1, 7, 7, 128)
    bin_distances = esmfold2_runtime_module._midpoint_bins(distogram.device)
    twenty_angstrom_bin = int(torch.argmin(torch.abs(bin_distances - 20.0)).item())
    distogram[0, 1, 4, twenty_angstrom_bin] = 12.0

    broad = _structure_losses(
        distogram,
        binder_length=3,
        hotspot_params={"target_hotspot_concat_indices": [1]},
        hotspot_weight=0.4,
        hotspot_cutoff=22.0,
    )
    strict = _structure_losses(
        distogram,
        binder_length=3,
        hotspot_params={"target_hotspot_concat_indices": [1]},
        hotspot_weight=0.4,
        hotspot_cutoff=8.0,
    )

    assert strict["hotspot_contact_loss"] > broad["hotspot_contact_loss"]
    assert torch.allclose(strict["base_structure_loss"], broad["base_structure_loss"])


def test_localized_hotspot_mode_replaces_global_inter_objective_by_interpolation():
    distogram = torch.randn(1, 7, 7, 128)
    losses = _structure_losses(
        distogram,
        binder_length=3,
        hotspot_params={"target_hotspot_concat_indices": [1]},
        hotspot_weight=1.0,
        hotspot_cutoff=8.0,
        hotspot_interface_mode="localized",
    )

    expected = (
        0.5 * losses["intra_contact_loss"]
        + 0.5 * losses["hotspot_contact_loss"]
        + 0.2 * losses["glob_loss"]
    )
    assert torch.allclose(losses["effective_inter_contact_loss"], losses["hotspot_contact_loss"])
    assert torch.allclose(losses["total_loss"], expected)
    assert torch.allclose(
        losses["hotspot_objective_adjustment"],
        losses["total_loss"] - losses["base_structure_loss"],
    )


def test_localized_hotspot_mode_rejects_weight_above_one():
    with pytest.raises(ValueError, match="hotspot_weight in"):
        _structure_losses(
            torch.randn(1, 7, 7, 128),
            binder_length=3,
            hotspot_params={"target_hotspot_concat_indices": [1]},
            hotspot_weight=1.1,
            hotspot_interface_mode="localized",
        )


@pytest.mark.parametrize("indices", [[4], [-1], [1.5], [True]])
def test_hotspot_structure_loss_rejects_invalid_target_indices(indices):
    distogram = torch.zeros(1, 7, 7, 128)

    with pytest.raises(ValueError):
        _structure_losses(
            distogram,
            binder_length=3,
            hotspot_params={"target_hotspot_concat_indices": indices},
            hotspot_weight=0.2,
        )


def test_hotspot_structure_loss_rejects_positive_weight_without_hotspots():
    with pytest.raises(
        ValueError,
        match="positive hotspot_weight requires target_hotspot_concat_indices",
    ):
        _structure_losses(
            torch.zeros(1, 7, 7, 128),
            binder_length=3,
            hotspot_weight=0.2,
        )


def test_runtime_continuous_design_logs_multichain_hotspot_metrics(monkeypatch):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=1,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
            hotspot_weight=0.3,
            hotspot_cutoff=8.0,
            one_step_diffusion_enabled=False,
            one_step_diffusion_interval=0,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        targets = esmfold2_runtime_module._target_sequences(kwargs["target_seq"])
        design = kwargs["design"]
        full_len = sum(len(seq) for seq in targets) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        return {
            "distogram_logits": signal
            * bins.reshape(1, 1, 1, -1).expand(1, full_len, full_len, 128)
        }

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq=["TT", "GGG"],
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
        hotspot_params={"target_hotspot_concat_indices": [3, 4]},
    )

    metrics = batch.soft_step_metrics[0]
    assert metrics["hotspot_weight"] == pytest.approx(0.3)
    assert metrics["hotspot_cutoff"] == pytest.approx(8.0)
    assert metrics["hotspot_contact_loss"] is not None
    assert metrics["hotspot_weighted_contribution"] == pytest.approx(
        0.3 * metrics["hotspot_contact_loss"]
    )


def test_runtime_can_disable_only_distogram_hotspot(monkeypatch):
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=1,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
            hotspot_weight=0.5,
            hotspot_cutoff=8.0,
            one_step_diffusion_enabled=False,
            one_step_diffusion_interval=0,
            hotspot_interface_mode="localized",
            distogram_hotspot_enabled=False,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        targets = esmfold2_runtime_module._target_sequences(kwargs["target_seq"])
        design = kwargs["design"]
        full_len = sum(len(seq) for seq in targets) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        return {
            "distogram_logits": signal
            * bins.reshape(1, 1, 1, -1).expand(1, full_len, full_len, 128)
        }

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq=["TT", "GGG"],
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
        hotspot_params={"target_hotspot_concat_indices": [3, 4]},
    )

    metrics = batch.soft_step_metrics[0]
    assert metrics["hotspot_weight"] == pytest.approx(0.0)
    assert metrics["distogram_hotspot_enabled"] is False
    assert metrics["distogram_hotspot_weight"] == pytest.approx(0.0)
    assert metrics["hotspot_contact_loss"] is not None
    assert metrics["hotspot_objective_adjustment"] == pytest.approx(0.0)


def test_runtime_continuous_design_accepts_multiple_target_chains(monkeypatch):
    captured = {}
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=1,
            lambda_esmc=0.0,
            num_loops=1,
            num_sampling_steps=1,
            one_step_diffusion_enabled=False,
            one_step_diffusion_interval=0,
        ),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        captured["target_seq"] = kwargs["target_seq"]
        targets = esmfold2_runtime_module._target_sequences(kwargs["target_seq"])
        design = kwargs["design"]
        full_len = sum(len(seq) for seq in targets) + design.shape[1]
        bins = torch.arange(128, dtype=design.dtype, device=design.device)
        signal = design[..., 0].sum()
        distogram = signal * bins.reshape(1, 1, 1, -1).expand(1, full_len, full_len, 128)
        return {"distogram_logits": distogram}

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    batch = backend.continuous_design(
        current_seq="AAA",
        target_seq=["TT", "GGG"],
        global_step=0,
        seed=5,
        mutable_mask=[True, True, True],
        proposal_logits=torch.zeros(3, 20),
    )

    assert captured["target_seq"] == ["TT", "GGG"]
    assert batch.sequences
    assert len(batch.soft_step_metrics) == 1


def test_preview_loss_dict_accepts_multiple_target_chains(monkeypatch):
    captured = {}
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        experimental_model=object(),
    )
    monkeypatch.setattr(backend, "_ensure_proposal_models_loaded", lambda: None)

    def fake_fold_distogram(**kwargs):
        captured["target_seq"] = kwargs["target_seq"]
        targets = esmfold2_runtime_module._target_sequences(kwargs["target_seq"])
        full_len = sum(len(seq) for seq in targets) + len("AAA")
        return {"distogram_logits": torch.zeros(1, full_len, full_len, 128)}

    monkeypatch.setattr(backend, "_fold_distogram", fake_fold_distogram)

    losses = backend.preview_loss_dict(
        binder_seq="AAA",
        target_seq=["TT", "GGG"],
        seed=5,
    )

    assert captured["target_seq"] == ["TT", "GGG"]
    assert set(losses) == {"intra_contact_loss", "inter_contact_loss", "glob_loss"}


def test_fold_distogram_uses_seed_context(monkeypatch):
    entered = []

    @contextmanager
    def fake_seed_context(seed):
        entered.append(seed)
        yield

    monkeypatch.setattr(esmfold2_runtime_module, "_seed_context", fake_seed_context)

    class FakeProteinInput:
        def __init__(self, *, id, sequence, msa=None):
            self.id = id
            self.sequence = sequence
            self.msa = msa

    class FakeStructurePredictionInput:
        def __init__(self, *, sequences):
            self.sequences = sequences

    def fake_prepare(input_raw, seed=None):
        return {"dummy": torch.tensor([1.0])}, None

    module = types.ModuleType("esm.models.esmfold2")
    module.ProteinInput = FakeProteinInput
    module.StructurePredictionInput = FakeStructurePredictionInput
    module.prepare_esmfold2_input = fake_prepare
    monkeypatch.setitem(sys.modules, "esm", types.ModuleType("esm"))
    monkeypatch.setitem(sys.modules, "esm.models", types.ModuleType("esm.models"))
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", module)

    class FakeExperimentalModel:
        def __call__(self, **kwargs):
            return {"distogram_logits": torch.zeros(1, 7, 7, 128)}

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        experimental_model=FakeExperimentalModel(),
    )
    monkeypatch.setattr(
        backend,
        "_sequence_to_folding_one_hot",
        lambda sequence: torch.zeros(1, len(sequence), 33),
    )
    monkeypatch.setattr(
        backend,
        "_folding_tokens_to_aa",
        lambda: {idx: "A" for idx in range(33)},
    )

    backend._fold_distogram(
        target_seq="TTTT",
        design=torch.zeros(1, 3, 20),
        seed=123,
    )

    assert entered == [123]


def test_fold_distogram_accepts_multiple_target_chains(monkeypatch):
    captured = {}

    class FakeProteinInput:
        def __init__(self, *, id, sequence, msa=None):
            self.id = id
            self.sequence = sequence
            self.msa = msa

    class FakeStructurePredictionInput:
        def __init__(self, *, sequences):
            self.sequences = sequences

    def fake_prepare(input_raw, seed=None):
        captured["input_raw"] = input_raw
        return {"dummy": torch.tensor([1.0])}, None

    module = types.ModuleType("esm.models.esmfold2")
    module.ProteinInput = FakeProteinInput
    module.StructurePredictionInput = FakeStructurePredictionInput
    module.prepare_esmfold2_input = fake_prepare
    monkeypatch.setitem(sys.modules, "esm", types.ModuleType("esm"))
    monkeypatch.setitem(sys.modules, "esm.models", types.ModuleType("esm.models"))
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", module)

    class FakeExperimentalModel:
        def __call__(self, **kwargs):
            captured["model_kwargs"] = kwargs
            length = kwargs["res_type_soft"].shape[1]
            return {"distogram_logits": torch.zeros(1, length, length, 128)}

    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(),
        device="cpu",
        experimental_model=FakeExperimentalModel(),
    )
    monkeypatch.setattr(
        backend,
        "_sequence_to_folding_one_hot",
        lambda sequence: torch.zeros(1, len(sequence), 33),
    )
    monkeypatch.setattr(
        backend,
        "_folding_tokens_to_aa",
        lambda: {idx: "G" for idx in range(33)},
    )

    result = backend._fold_distogram(
        target_seq=["AA", "TTT"],
        design=torch.zeros(1, 3, 20),
        seed=123,
    )

    input_sequences = captured["input_raw"].sequences
    assert [seq.id for seq in input_sequences] == ["A", "B", "C"]
    assert [seq.sequence for seq in input_sequences] == ["AA", "TTT", "GGG"]
    assert captured["model_kwargs"]["res_type_soft"].shape == (1, 8, 33)
    assert result["distogram_logits"].shape == (1, 8, 8, 128)


def test_stageb_structure_loss_backpropagates_only_selected_group():
    losses = {
        "total_loss": torch.tensor(10.0),
        "inter_contact_loss": torch.tensor(3.0),
    }

    proposal_loss, source = _stageb_structure_loss(
        losses,
        "interface_formation",
        group_weight=1.0,
    )

    assert proposal_loss.item() == pytest.approx(3.0)
    assert source == "inter_contact_loss"


def test_stageb_structure_loss_keeps_hotspot_constraint_for_every_group():
    losses = {
        "total_loss": torch.tensor(10.0),
        "intra_contact_loss": torch.tensor(2.0),
        "glob_loss": torch.tensor(1.0),
        "hotspot_weighted_contribution": torch.tensor(0.4),
    }

    core_loss, core_source = _stageb_structure_loss(losses, "core_stabilization")
    compact_loss, compact_source = _stageb_structure_loss(losses, "compactness_repair")

    assert core_loss.item() == pytest.approx(2.4)
    assert core_source == "intra_contact_loss+hotspot_weighted_contribution"
    assert compact_loss.item() == pytest.approx(1.4)
    assert compact_source == "glob_loss+hotspot_weighted_contribution"


def test_stageb_interface_hotspot_is_counted_once_for_both_modes():
    losses = {
        "total_loss": torch.tensor(10.0),
        "inter_contact_loss": torch.tensor(3.0),
        "effective_inter_contact_loss": torch.tensor(2.2),
        "hotspot_weighted_contribution": torch.tensor(0.4),
    }

    localized, localized_source = _stageb_structure_loss(
        losses,
        "interface_formation",
        hotspot_interface_mode="localized",
    )
    additive, additive_source = _stageb_structure_loss(
        losses,
        "interface_formation",
        hotspot_interface_mode="additive",
    )

    assert localized.item() == pytest.approx(2.2)
    assert localized_source == "effective_inter_contact_loss"
    assert additive.item() == pytest.approx(3.4)
    assert additive_source == "inter_contact_loss+hotspot_weighted_contribution"


def test_proposal_confidence_tail_steps_override_temperature_gate():
    backend = ESMFold2RuntimeBackend(
        config=ESMFold2ProposalConfig(
            soft_steps=30,
            track_proposal_confidence=True,
            proposal_confidence_temperature=0.01,
            proposal_confidence_tail_steps=20,
        ),
        device="cpu",
        verifier_model=object(),
        input_builder=FakeInputBuilder(),
        protein_input_cls=FakeProteinInput,
        structure_prediction_input_cls=FakeStructurePredictionInput,
    )

    tracked_steps = [
        local_step
        for local_step in range(30)
        if backend._should_track_proposal_confidence(
            track_confidence=True,
            local_step=local_step,
            temperature=1.0,
        )
    ]

    assert tracked_steps == list(range(10, 30))
