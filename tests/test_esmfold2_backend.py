import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import (
    CYS_INDEX,
    ESM_AA_ORDER,
    ESMFold2ProposalConfig,
    build_logit_delta_candidate_batch,
    build_esm_verifier_result,
    build_gradient_mask,
    combine_normalized_gradients,
    compute_esmfold2_action_group_priors,
    esm_temperature,
    init_binder_logits,
    logit_delta_mutation_scores,
    minimally_align_binder_logits_to_sequence,
    normalized_gradient_tensor,
    propose_sequences_from_logit_delta,
    resolve_esm_verifier_loss,
    resolve_lambda_esmc,
    select_esmfold2_action_group,
    softly_anchor_mutation_logits,
)
from proteocraft.hallucination.stage_b_tree import StageBTree


def test_soft_mutation_anchor_changes_only_explicit_edit_positions():
    logits = torch.arange(3 * len(ESM_AA_ORDER), dtype=torch.float32).reshape(
        3, len(ESM_AA_ORDER)
    )
    logits[:, CYS_INDEX] = float("-inf")

    anchored = softly_anchor_mutation_logits(
        logits,
        parent_sequence="AAA",
        child_sequence="AYA",
        mutable_mask=[True, True, True],
        anchor_margin=0.1,
    )

    assert torch.equal(anchored[0], logits[0])
    assert torch.equal(anchored[2], logits[2])
    y_idx = ESM_AA_ORDER.index("Y")
    finite_before = logits[1][torch.isfinite(logits[1])]
    assert anchored[1, y_idx] == pytest.approx(float(finite_before.max()) + 0.1)


def test_minimal_sequence_alignment_preserves_matching_soft_logits_below_two():
    logits = torch.full((2, len(ESM_AA_ORDER)), -1.0)
    logits[:, CYS_INDEX] = float("-inf")
    logits[0, ESM_AA_ORDER.index("A")] = 0.2
    logits[1, ESM_AA_ORDER.index("Y")] = 0.3

    aligned = minimally_align_binder_logits_to_sequence(
        logits,
        "AY",
        mutable_mask=[True, True],
    )

    assert torch.equal(aligned, logits)


def test_minimal_sequence_alignment_changes_only_mismatched_rows():
    logits = torch.full((2, len(ESM_AA_ORDER)), -1.0)
    logits[:, CYS_INDEX] = float("-inf")
    a_idx = ESM_AA_ORDER.index("A")
    v_idx = ESM_AA_ORDER.index("V")
    y_idx = ESM_AA_ORDER.index("Y")
    logits[0, a_idx] = 1.2
    logits[0, v_idx] = 3.4
    logits[1, y_idx] = 0.3

    aligned = minimally_align_binder_logits_to_sequence(
        logits,
        "AY",
        mutable_mask=[True, True],
        anchor_margin=0.1,
    )

    assert aligned[0, a_idx] == pytest.approx(3.5)
    assert torch.equal(aligned[0, 1:], logits[0, 1:])
    assert torch.equal(aligned[1], logits[1])


def test_minimal_sequence_alignment_breaks_argmax_ties():
    logits = torch.full((1, len(ESM_AA_ORDER)), -1.0)
    logits[:, CYS_INDEX] = float("-inf")
    a_idx = ESM_AA_ORDER.index("A")
    v_idx = ESM_AA_ORDER.index("V")
    logits[0, a_idx] = 1.0
    logits[0, v_idx] = 1.0

    aligned = minimally_align_binder_logits_to_sequence(logits, "A")

    assert aligned[0, a_idx] == pytest.approx(1.1)
    assert ESM_AA_ORDER[int(aligned[0].argmax().item())] == "A"


def test_minimal_sequence_alignment_keeps_fixed_positions_hard_constrained():
    logits = torch.zeros(2, len(ESM_AA_ORDER))
    logits[:, CYS_INDEX] = float("-inf")
    y_idx = ESM_AA_ORDER.index("Y")

    aligned = minimally_align_binder_logits_to_sequence(
        logits,
        "YA",
        mutable_mask=[False, True],
    )

    assert aligned[0, y_idx] == pytest.approx(10.0)
    assert ESM_AA_ORDER[int(aligned[0].argmax().item())] == "Y"
    assert aligned[1, ESM_AA_ORDER.index("A")] == pytest.approx(0.1)


def test_verifier_loss_requires_pair_chain_iptm_without_global_fallback():
    loss, source = resolve_esm_verifier_loss(
        {"pair_chain_iptm": 0.82, "iptm": 0.10}
    )
    assert loss == pytest.approx(0.18)
    assert source == "pair_chain_iptm"

    with pytest.raises(ValueError, match="pair-chain iPTM"):
        resolve_esm_verifier_loss({"iptm": 0.73})


def test_verifier_result_is_compatible_with_mcts_cache_contract():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={
            "pair_chain_iptm": 0.75,
            "plddt": 0.91,
            "binder_plddt": 0.88,
            "binder_rg_excess": 0.12,
        },
        coordinate=torch.zeros(1, 8, 3),
    )

    assert result.total_loss == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_loss"] == pytest.approx(0.25)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["binder_confidence_loss"] == pytest.approx(0.12)
    assert result.loss_dict["binder_rg_excess_loss"] == pytest.approx(0.12)
    assert "interface_loss" not in result.loss_dict
    assert result.summary_confidence["esm_verify_source"] == "pair_chain_iptm"

    tree = StageBTree()
    cached = tree.store_verifier_result(result)
    assert cached.binder_seq == "ALY"
    assert cached.total_loss == pytest.approx(0.25)
    assert cached.coordinate.shape == (1, 8, 3)


def test_verifier_result_combines_global_and_hotspot_losses_without_overwriting_global_metric():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.6,
        hotspot_verify_weight=0.2,
        extra_loss_dict={"hotspot_mean_min_heavy_atom_distance": 14.0},
    )

    assert result.total_loss == pytest.approx(0.37)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_global_loss"] == pytest.approx(0.25)
    assert result.loss_dict["hotspot_verify_distance_loss"] == pytest.approx(0.6)
    assert result.loss_dict["hotspot_verify_weighted_contribution"] == pytest.approx(0.12)
    assert result.loss_dict["hotspot_mean_min_heavy_atom_distance"] == pytest.approx(14.0)
    assert result.summary_confidence["esm_verify_loss"] == pytest.approx(0.37)


def test_verifier_result_can_use_aligned_common_loss_without_losing_pair_metric():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        aligned_loss_dict={
            "total": 0.42,
            "plddt": 0.12,
            "i_pae": 0.20,
        },
        hotspot_verify_loss=0.5,
        hotspot_verify_weight=0.2,
    )

    assert result.total_loss == pytest.approx(0.52)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["esm_verify_global_loss"] == pytest.approx(0.42)
    assert result.loss_dict["plddt"] == pytest.approx(0.12)
    assert result.summary_confidence["esm_verify_source"] == "esmfold2_aligned_common"


def test_verifier_result_adds_hotspot_too_close_penalty():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.0,
        hotspot_verify_weight=0.2,
        hotspot_too_close_loss=0.4,
        hotspot_too_close_weight=0.25,
    )

    assert result.total_loss == pytest.approx(0.35)
    assert result.loss_dict["pair_chain_iptm_loss"] == pytest.approx(0.25)
    assert result.loss_dict["hotspot_too_close_loss"] == pytest.approx(0.4)
    assert result.loss_dict[
        "hotspot_too_close_weighted_contribution"
    ] == pytest.approx(0.1)


def test_zero_hotspot_verify_weight_preserves_legacy_verifier_result_shape():
    result = build_esm_verifier_result(
        binder_seq="ALY",
        target_seq="TARGET",
        summary_confidence={"pair_chain_iptm": 0.75},
        hotspot_verify_loss=0.6,
        hotspot_verify_weight=0.0,
    )

    assert result.total_loss == pytest.approx(0.25)
    assert set(result.loss_dict) == {
        "total",
        "esm_verify_loss",
        "pair_chain_iptm_loss",
    }
    assert "hotspot_verify_distance_loss" not in result.summary_confidence

    with pytest.raises(
        ValueError,
        match="positive hotspot_verify_weight requires hotspot_verify_loss",
    ):
        build_esm_verifier_result(
            binder_seq="ALY",
            target_seq="TARGET",
            summary_confidence={"pair_chain_iptm": 0.75},
            hotspot_verify_weight=0.2,
        )


def test_init_logits_and_gradient_mask_keep_fixed_positions_out_of_optimization():
    logits = init_binder_logits(
        "ALD",
        mutable_mask=[False, True, True],
        batch_size=2,
        random_scale=0.0,
    )
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}

    assert logits.shape == (2, 3, 20)
    assert logits.requires_grad
    assert logits[0, 0, aa_to_idx["A"]].item() == pytest.approx(10.0)
    assert torch.isneginf(logits[:, 1:, CYS_INDEX]).all()

    mask = build_gradient_mask("ALD", mutable_mask=[False, True, True], batch_size=2)
    assert mask.shape == (2, 3, 20)
    assert mask[:, 0, :].sum().item() == pytest.approx(0.0)
    assert mask[:, 1:, CYS_INDEX].sum().item() == pytest.approx(0.0)
    assert mask[:, 1:, :].sum().item() > 0.0


def test_logit_delta_scores_rank_mutations_from_short_soft_update():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(2, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 2.0
    after[1, aa_to_idx["L"]] = 0.5
    after[1, aa_to_idx["V"]] = 1.1

    scores = logit_delta_mutation_scores(
        before,
        after,
        current_seq="AL",
        mutable_mask=[False, True],
    )

    assert torch.isneginf(scores[0]).all()
    assert torch.isneginf(scores[1, aa_to_idx["L"]])
    assert torch.isneginf(scores[1, CYS_INDEX])
    assert scores[1, aa_to_idx["Y"]].item() == pytest.approx(1.5)
    assert scores[1, aa_to_idx["V"]].item() == pytest.approx(0.6)


def test_logit_delta_scores_do_not_nan_when_current_mutable_residue_is_cys():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(2, 20)
    after = before.clone()
    before[1, CYS_INDEX] = float("-inf")
    after[1, CYS_INDEX] = float("-inf")
    after[1, aa_to_idx["Y"]] = 2.0

    scores = logit_delta_mutation_scores(
        before,
        after,
        current_seq="AC",
        mutable_mask=[False, True],
    )

    assert not torch.isnan(scores).any()
    assert torch.isneginf(scores[1, CYS_INDEX])
    assert scores[1, aa_to_idx["Y"]].item() == pytest.approx(2.0)


def test_combine_normalized_gradients_respects_mask_and_task_lambda():
    mask = torch.zeros(1, 2, 20)
    mask[:, 1, :] = 1.0
    mask[:, 1, CYS_INDEX] = 0.0

    structure_grad = torch.zeros_like(mask)
    structure_grad[0, 1, 0] = 3.0
    esmc_grad = torch.zeros_like(mask)
    esmc_grad[0, 1, 1] = 4.0

    combined = combine_normalized_gradients(
        structure_grad,
        esmc_grad,
        gradient_mask=mask,
        lambda_esmc=resolve_lambda_esmc("auto", task_type="denovo"),
    )

    assert combined[:, 0, :].sum().item() == pytest.approx(0.0)
    assert combined[0, 1, 0].item() > 0.0
    assert combined[0, 1, 1].item() > 0.0
    assert resolve_lambda_esmc("auto", task_type="cdr") < resolve_lambda_esmc(
        "auto", task_type="denovo"
    )
    assert resolve_lambda_esmc("auto", task_type="vhh_redesign") == pytest.approx(
        0.05
    )


def test_normalized_gradient_tensor_matches_official_eff_length_scaling():
    mask = torch.zeros(1, 3, 20)
    mask[0, 0, 0] = 1.0
    mask[0, 2, 1] = 1.0
    grad = torch.zeros_like(mask)
    grad[0, 0, 0] = 3.0
    grad[0, 2, 1] = 4.0

    normalized = normalized_gradient_tensor(grad, mask)

    expected_scale = (2.0 ** 0.5) / 5.0
    assert normalized[0, 0, 0].item() == pytest.approx(3.0 * expected_scale)
    assert normalized[0, 2, 1].item() == pytest.approx(4.0 * expected_scale)
    assert torch.linalg.vector_norm(normalized).item() == pytest.approx(2.0 ** 0.5)
    assert normalized[0, 1, :].sum().item() == pytest.approx(0.0)


def test_temperature_schedule_and_config_match_short_esm_stage_a_contract():
    default_config = ESMFold2ProposalConfig(lambda_esmc="auto")
    assert default_config.soft_steps == 30
    assert default_config.learning_rate == pytest.approx(0.05)
    assert default_config.proposal_lm_dropout == pytest.approx(0.5)
    assert default_config.force_proposal_lm_dropout is True
    assert default_config.experimental_model_name.endswith(
        "ESMFold2-Experimental-Fast"
    )
    assert default_config.hotspot_weight == pytest.approx(0.0)
    assert default_config.hotspot_satisfied_proposal_weight == pytest.approx(0.0)
    assert default_config.hotspot_cutoff == pytest.approx(8.0)
    assert default_config.hotspot_interface_mode == "additive"
    assert default_config.hotspot_verify_weight == pytest.approx(0.0)
    assert default_config.hotspot_verify_near_cutoff == pytest.approx(8.0)
    assert default_config.hotspot_verify_far_cutoff == pytest.approx(20.0)
    assert default_config.hotspot_verify_min_distance == pytest.approx(2.5)
    assert default_config.hotspot_verify_too_close_weight == pytest.approx(0.0)
    assert default_config.one_step_diffusion_enabled is True
    assert default_config.one_step_diffusion_mode == "truncated_trajectory"
    assert default_config.one_step_diffusion_tau == pytest.approx(1.0)
    assert default_config.one_step_diffusion_truncated_num_sampling_steps == 50
    assert (
        default_config.one_step_diffusion_truncated_schedule_mode
        == "official_clipped"
    )
    assert (
        default_config.one_step_diffusion_truncated_dense_num_sampling_steps
        == 41
    )
    assert default_config.one_step_diffusion_weight == pytest.approx(0.25)
    assert default_config.one_step_diffusion_rg_weight == pytest.approx(0.1)
    assert default_config.one_step_diffusion_interval == 5
    assert default_config.one_step_diffusion_recycle_coordinates is False
    assert default_config.one_step_diffusion_hotspot_enabled is False
    assert default_config.distogram_hotspot_enabled is True
    assert default_config.verifier_loss_mode == "pair_chain_iptm"

    config = ESMFold2ProposalConfig(soft_steps=3, lambda_esmc="auto")
    assert config.soft_steps == 3
    first = esm_temperature(0, total_steps=3)
    last = esm_temperature(2, total_steps=3)
    assert first > last
    assert last >= 1e-2


def test_proposal_config_rejects_negative_hotspot_weight():
    with pytest.raises(ValueError, match="hotspot_weight must be non-negative"):
        ESMFold2ProposalConfig(hotspot_weight=-0.1)


def test_proposal_config_rejects_nonpositive_hotspot_cutoff():
    with pytest.raises(ValueError, match="hotspot_cutoff must be positive"):
        ESMFold2ProposalConfig(hotspot_cutoff=0.0)


def test_proposal_config_validates_hotspot_interface_mode():
    with pytest.raises(ValueError, match="hotspot_interface_mode"):
        ESMFold2ProposalConfig(hotspot_interface_mode="unknown")
    with pytest.raises(ValueError, match="hotspot_weight in"):
        ESMFold2ProposalConfig(
            hotspot_interface_mode="localized",
            hotspot_weight=1.1,
        )


def test_proposal_config_validates_hotspot_verifier_parameters():
    with pytest.raises(ValueError, match="hotspot_verify_weight must be non-negative"):
        ESMFold2ProposalConfig(hotspot_verify_weight=-0.1)
    with pytest.raises(
        ValueError,
        match="hotspot_verify_far_cutoff must be greater",
    ):
        ESMFold2ProposalConfig(
            hotspot_verify_near_cutoff=5.0,
            hotspot_verify_far_cutoff=5.0,
        )


def test_proposal_config_validates_one_step_diffusion_parameters():
    assert ESMFold2ProposalConfig().esmc_activation_checkpointing is False
    with pytest.raises(ValueError, match="one_step_diffusion_tau"):
        ESMFold2ProposalConfig(one_step_diffusion_tau=0.0)
    with pytest.raises(ValueError, match="one_step_diffusion_mode"):
        ESMFold2ProposalConfig(one_step_diffusion_mode="unknown")
    with pytest.raises(ValueError, match="truncated_num_sampling_steps"):
        ESMFold2ProposalConfig(
            one_step_diffusion_truncated_num_sampling_steps=1
        )
    with pytest.raises(ValueError, match="truncated_schedule_mode"):
        ESMFold2ProposalConfig(
            one_step_diffusion_truncated_schedule_mode="unknown"
        )
    with pytest.raises(ValueError, match="truncated_dense_num_sampling_steps"):
        ESMFold2ProposalConfig(
            one_step_diffusion_truncated_dense_num_sampling_steps=1
        )
    with pytest.raises(ValueError, match="only applies to truncated"):
        ESMFold2ProposalConfig(
            one_step_diffusion_mode="reference_noising",
            one_step_diffusion_truncated_schedule_mode="dense",
        )
    with pytest.raises(ValueError, match="one_step_diffusion_weight"):
        ESMFold2ProposalConfig(one_step_diffusion_weight=-0.1)
    with pytest.raises(ValueError, match="one_step_diffusion_rg_weight"):
        ESMFold2ProposalConfig(one_step_diffusion_rg_weight=-0.1)
    with pytest.raises(ValueError, match="one_step_diffusion_interval"):
        ESMFold2ProposalConfig(one_step_diffusion_interval=-1)
    with pytest.raises(
        ValueError,
        match="positive one_step_diffusion_interval",
    ):
        ESMFold2ProposalConfig(
            one_step_diffusion_enabled=False,
            one_step_diffusion_interval=5,
        )
    with pytest.raises(
        ValueError,
        match="one_step_diffusion_recycle_coordinates",
    ):
        ESMFold2ProposalConfig(
            one_step_diffusion_recycle_coordinates=True,
            one_step_diffusion_interval=0,
        )
    with pytest.raises(
        ValueError,
        match="only supported by reference_noising",
    ):
        ESMFold2ProposalConfig(
            one_step_diffusion_mode="truncated_trajectory",
            one_step_diffusion_recycle_coordinates=True,
        )
    with pytest.raises(ValueError, match="verifier_loss_mode"):
        ESMFold2ProposalConfig(verifier_loss_mode="unknown")


def test_template_anchored_config_requires_complete_local_redesign_contract(tmp_path):
    with pytest.raises(ValueError, match="template_structure_path"):
        ESMFold2ProposalConfig(
            one_step_diffusion_mode="template_anchored_truncated_trajectory"
        )

    template_path = tmp_path / "template.pdb"
    template_path.write_text("END\n", encoding="utf-8")
    config = ESMFold2ProposalConfig(
        one_step_diffusion_mode="template_anchored_truncated_trajectory",
        template_structure_path=str(template_path),
        template_chain_ids=("A", "B"),
        template_chain_sequences=("AAAA", "AACAA"),
        binder_mutable_mask=(False, False, True, False, False),
        binder_structure_regeneration_mask=(False, False, True, False, False),
    )

    assert config.template_chain_ids == ("A", "B")


def test_scaffold_consistency_config_is_opt_in_and_requires_template_contract(tmp_path):
    assert ESMFold2ProposalConfig().verifier_structure_mode == "sequence_only"
    with pytest.raises(ValueError, match="verifier_structure_mode"):
        ESMFold2ProposalConfig(verifier_structure_mode="unknown")
    with pytest.raises(ValueError, match="template_structure_path"):
        ESMFold2ProposalConfig(verifier_structure_mode="scaffold_consistency")

    template_path = tmp_path / "template.pdb"
    template_path.write_text("END\n", encoding="utf-8")
    config = ESMFold2ProposalConfig(
        verifier_structure_mode="scaffold_consistency",
        template_structure_path=str(template_path),
        template_chain_ids=("A", "B"),
        template_chain_sequences=("AAAA", "AACAA"),
        binder_mutable_mask=(False, False, True, False, False),
        binder_structure_regeneration_mask=(False, False, True, False, False),
    )
    assert config.verifier_structure_mode == "scaffold_consistency"


def test_logit_delta_candidates_reuse_stage_a_topk_sampling_contract():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 3.0
    after[2, aa_to_idx["F"]] = 2.5

    torch.manual_seed(0)
    candidates = propose_sequences_from_logit_delta(
        before,
        after,
        current_seq="ALD",
        mutable_mask=[False, True, True],
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
    )

    assert len(candidates) == 3
    assert "AYD" in candidates
    assert all(seq[0] == "A" for seq in candidates)


def test_logit_delta_candidates_prepend_full_argmax_sequence():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, aa_to_idx["Y"]] = 3.0
    after[2, aa_to_idx["F"]] = 2.5

    torch.manual_seed(0)
    candidates = propose_sequences_from_logit_delta(
        before,
        after,
        current_seq="ALD",
        mutable_mask=[False, True, True],
        K_top=2,
        B_candidates=3,
        T_sample=0.01,
        max_mutations_per_seq=1,
        include_logit_candidates=True,
    )

    assert candidates[0] == "AYF"
    assert len(candidates) == 3
    assert all(seq[0] == "A" for seq in candidates)


def test_logit_delta_fraction_samples_distinct_mutable_positions():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(10, 20)
    after = before.clone()
    for position in range(10):
        after[position, aa_to_idx["Y"]] = float(20 - position)

    torch.manual_seed(4)
    candidates = propose_sequences_from_logit_delta(
        before,
        after,
        current_seq="A" * 10,
        mutable_mask=[False, False] + [True] * 8,
        K_top=2,
        B_candidates=20,
        T_sample=1.0,
        max_mutations_per_seq=1,
        mutation_fraction=0.5,
    )

    mutation_counts = [
        sum(left != right for left, right in zip("A" * 10, sequence))
        for sequence in candidates
    ]
    assert max(mutation_counts) == 4
    assert all(count <= 4 for count in mutation_counts)
    assert all(sequence[:2] == "AA" for sequence in candidates)


def test_positive_score_candidates_use_fixed_count_and_disable_raw_argmax():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(10, 20)
    after = before.clone()
    for position in range(2, 10):
        after[position, aa_to_idx["Y"]] = float(20 - position)
        after[position, aa_to_idx["F"]] = float(10 - position) + 0.5

    torch.manual_seed(7)
    batch = build_logit_delta_candidate_batch(
        before,
        after,
        current_seq="A" * 10,
        mutable_mask=[False, False] + [True] * 8,
        K_top=2,
        B_candidates=8,
        T_sample=1.0,
        mutation_fraction=0.5,
        include_logit_candidates=False,
        candidate_strategy="positive_score",
    )

    assert len(batch.sequences) >= 2
    assert all(
        sum(left != right for left, right in zip("A" * 10, sequence)) == 4
        for sequence in batch.sequences
    )
    assert batch.sequences[0] == "AAYYYYAAAA"
    assert all(sequence[:2] == "AA" for sequence in batch.sequences)
    assert all(sequence != "AA" + "Y" * 8 for sequence in batch.sequences)
    assert batch.candidate_metadata[0]["source"] == "positive_score_topk"
    assert all(
        item["mutation_count"] == 4 for item in batch.candidate_metadata
    )
    assert batch.diagnostics["positive_position_count"] == 8
    assert batch.diagnostics["mutation_cap"] == 4
    assert batch.diagnostics["actual_mutation_count"] == 4


def test_positive_score_candidates_do_not_force_nonpositive_mutations():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(8, 20)
    after = before.clone()
    after[2, aa_to_idx["Y"]] = 3.0
    after[5, aa_to_idx["F"]] = 2.0
    after[6, aa_to_idx["W"]] = -2.0

    batch = build_logit_delta_candidate_batch(
        before,
        after,
        current_seq="A" * 8,
        mutable_mask=[True] * 8,
        B_candidates=8,
        mutation_fraction=0.5,
        include_logit_candidates=False,
        candidate_strategy="positive_score",
    )

    assert batch.sequences == ["AAYAAFAA"]
    assert batch.diagnostics["positive_position_count"] == 2
    assert batch.diagnostics["mutation_cap"] == 4
    assert batch.diagnostics["actual_mutation_count"] == 2


def test_positive_score_argmax_control_is_explicit_and_unrestricted():
    aa_to_idx = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
    before = torch.zeros(6, 20)
    after = before.clone()
    after[:, aa_to_idx["Y"]] = 3.0

    torch.manual_seed(0)
    batch = build_logit_delta_candidate_batch(
        before,
        after,
        current_seq="A" * 6,
        B_candidates=8,
        mutation_fraction=0.5,
        include_logit_candidates=True,
        candidate_strategy="positive_score",
    )

    assert batch.sequences[0] == "Y" * 6
    assert len(batch.sequences) == 8
    assert batch.candidate_metadata[0]["source"] == "unrestricted_argmax"
    assert batch.candidate_metadata[0]["mutation_count"] == 6
    assert all(
        item["source"] in {"positive_score_topk", "positive_score_weighted"}
        and item["mutation_count"] == 3
        for item in batch.candidate_metadata[1:]
    )


def test_positive_score_strategy_requires_mutation_fraction():
    logits = torch.zeros(2, 20)
    with pytest.raises(ValueError, match="requires mutation_fraction"):
        propose_sequences_from_logit_delta(
            logits,
            logits,
            current_seq="AA",
            candidate_strategy="positive_score",
        )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_logit_delta_fraction_rejects_invalid_values(fraction):
    logits = torch.zeros(2, 20)
    with pytest.raises(ValueError, match="mutation_fraction"):
        propose_sequences_from_logit_delta(
            logits,
            logits,
            current_seq="AA",
            mutation_fraction=fraction,
        )


def test_esmfold2_action_group_routing_uses_native_loss_terms():
    interface = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.7,
            "binder_confidence_loss": 0.1,
            "binder_rg_excess_loss": 0.05,
        }
    )
    assert interface["action_group"] == "interface_formation"
    assert interface["group_priors"]["interface_formation"] == pytest.approx(
        max(interface["group_priors"].values())
    )

    compact = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.1,
            "binder_confidence_loss": 0.1,
            "binder_rg_excess_loss": 0.9,
        }
    )
    assert compact["action_group"] == "compactness_repair"

    with pytest.raises(ValueError, match="full-verifier metrics"):
        compute_esmfold2_action_group_priors({}, temperature=1.0)


def test_esmfold2_routing_uses_only_bounded_full_verifier_residuals():
    routing = select_esmfold2_action_group(
        {
            "pair_chain_iptm_loss": 0.95,
            "binder_confidence_loss": 0.2,
            "binder_rg_excess_loss": 1.2,
        }
    )

    assert routing["group_scores"]["interface_formation"] <= 1.0
    assert routing["group_scores"]["core_stabilization"] <= 1.0
    assert routing["group_scores"]["compactness_repair"] <= 1.0
    assert routing["group_terms"]["interface_formation"]["pair_chain_iptm_loss"]["normalized"] == pytest.approx(0.95)
    assert routing["group_terms"]["compactness_repair"]["binder_rg_excess_loss"]["normalized"] == pytest.approx(1.0)
