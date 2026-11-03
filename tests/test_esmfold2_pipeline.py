import hashlib
import json
import random
from types import SimpleNamespace

import pytest
import torch

from proteocraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESM_AA_ORDER,
    ESMFold2ProposalBatch,
    build_esm_verifier_result,
    minimally_align_binder_logits_to_sequence,
)
from proteocraft.hallucination.stage_a_tree import StageATree
from proteocraft.hallucination.esmfold2_pipeline import (
    run_esmfold2_mcts_hallucination,
)
from proteocraft.hallucination.esmfold2_pipeline import (
    _ESMArtifactRecorder,
    _ESMFold2StageBMCTSTree,
    _allocate_group_budgets,
    _deduplicate_actions_by_sequence,
    _stageb_proposal_call_specs,
    _hotspot_proposal_weight_for_verified_state,
    _load_initial_logits_checkpoint_states,
    _make_phase0_seed_states,
    _make_stageb_proposals,
    _minimally_align_proposal_logits,
    _rerank_stageb_candidates_with_preview,
    _select_stageb_actions_by_group_quota,
    _select_stageb_tree_expansions,
    _select_stagea_tree_results,
    _stagea_child_logits,
    _stageb_guarded_rank,
    _stageb_preview_objective,
    _write_proposal_visualization_trace,
)


def test_proposal_visualization_trace_is_opt_in(tmp_path):
    logits_before = torch.zeros(3, len(ESM_AA_ORDER))
    logits_after = logits_before.clone()
    logits_after[2, AA_TO_ESM_IDX["Y"]] = 1.25
    batch = ESMFold2ProposalBatch(
        sequences=["AAY"],
        logits_before=logits_before,
        logits_after=logits_after,
        candidate_metadata=[
            {
                "sequence": "AAY",
                "source": "positive_score_top",
                "mutation_changes": ["A3Y"],
                "mutation_scores": [1.25],
            }
        ],
    )

    disabled = _ESMArtifactRecorder(tmp_path / "disabled")
    _write_proposal_visualization_trace(
        disabled,
        stage="stage_a",
        step=0,
        selected_sequence="AAA",
        batch=batch,
        mutable_mask=[True, True, True],
    )
    assert not (tmp_path / "disabled" / "visualization_trace").exists()

    enabled = _ESMArtifactRecorder(
        tmp_path / "enabled",
        save_visualization_trace=True,
    )
    _write_proposal_visualization_trace(
        enabled,
        stage="stage_a",
        step=0,
        selected_sequence="AAA",
        batch=batch,
        mutable_mask=[True, True, True],
    )

    index_path = (
        tmp_path / "enabled" / "visualization_trace" / "proposal_trace.jsonl"
    )
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["aa_order"] == ESM_AA_ORDER
    assert records[0]["mutation_scores_shape"] == [3, len(ESM_AA_ORDER)]
    payload = torch.load(
        tmp_path / "enabled" / records[0]["trace_file"],
        map_location="cpu",
        weights_only=False,
    )
    assert payload["candidate_metadata"][0]["mutation_changes"] == ["A3Y"]
    assert payload["mutation_scores"][2, AA_TO_ESM_IDX["Y"]] == pytest.approx(1.25)


def test_minimal_proposal_alignment_preserves_existing_soft_logits():
    logits = torch.zeros(3, len(ESM_AA_ORDER))
    logits[0, AA_TO_ESM_IDX["A"]] = 0.4
    logits[1, AA_TO_ESM_IDX["F"]] = 0.8
    logits[1, AA_TO_ESM_IDX["Y"]] = 0.2
    logits[2, AA_TO_ESM_IDX["A"]] = 0.6

    aligned = _minimally_align_proposal_logits(logits, "AYA", None)

    assert aligned[0, AA_TO_ESM_IDX["A"]] == pytest.approx(0.4)
    assert aligned[1, AA_TO_ESM_IDX["Y"]] == pytest.approx(0.9)
    assert aligned[2, AA_TO_ESM_IDX["A"]] == pytest.approx(0.6)
    assert float(aligned[1, AA_TO_ESM_IDX["Y"]]) < 2.0


def test_stagea_soft_state_minimally_aligns_every_mismatched_row():
    logits = torch.zeros(3, len(ESM_AA_ORDER))
    logits[0, AA_TO_ESM_IDX["F"]] = 0.8
    logits[0, AA_TO_ESM_IDX["A"]] = 0.1
    logits[1, AA_TO_ESM_IDX["F"]] = 1.0
    logits[1, AA_TO_ESM_IDX["Y"]] = 0.2
    logits[2, AA_TO_ESM_IDX["A"]] = 0.4

    aligned = _stagea_child_logits(
        logits_after=logits,
        parent_sequence="AAA",
        child_sequence="AYA",
        argmax_sequence=None,
        mutable_mask=None,
        preserve_soft_state=True,
        soft_anchor_margin=0.1,
    )

    assert aligned[0, AA_TO_ESM_IDX["A"]] == pytest.approx(0.9)
    assert aligned[1, AA_TO_ESM_IDX["Y"]] == pytest.approx(1.1)
    assert aligned[2, AA_TO_ESM_IDX["A"]] == pytest.approx(0.4)
    assert "".join(
        ESM_AA_ORDER[int(index)] for index in aligned.argmax(dim=-1)
    ) == "AYA"


class FakeESMFold2Backend:
    def __init__(self):
        self.verify_calls = []
        self.propose_calls = []
        self.continuous_calls = []
        self.preview_calls = []
        self.scores = {
            "AAA": 0.10,
            "AAY": 0.80,
            "AAF": 0.40,
            "AYY": 0.90,
        }

    def verify(self, *, binder_seq, target_seq, seed=None):
        self.verify_calls.append((binder_seq, target_seq, seed))
        score = self.scores.get(binder_seq, 0.05)
        extra = {
        }
        return build_esm_verifier_result(
            binder_seq=binder_seq,
            target_seq=target_seq,
            summary_confidence={
                "pair_chain_iptm": score,
                "iptm": score,
                "binder_plddt": 0.8,
                "binder_rg_excess": 0.1,
                "mmcif": f"data_{binder_seq}\n#\n",
            },
            extra_loss_dict=extra,
            coordinate=torch.zeros(1, len(binder_seq) + len(str(target_seq)), 3),
        )

    def preview_loss_dict(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
        self.preview_calls.append((binder_seq, target_seq, seed, hotspot_params))
        losses = {
            "inter_contact_loss": 0.6,
            "intra_contact_loss": 0.1,
            "glob_loss": 0.05,
        }
        if hotspot_params:
            losses["hotspot_contact_loss"] = 0.4
        return losses

    def propose(
        self,
        *,
        current_seq,
        target_seq,
        global_step,
        K_top,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
        mutation_fraction=None,
        candidate_strategy=None,
        include_argmax_candidate=None,
        seed=None,
        action_group=None,
        mutable_mask=None,
        proposal_logits=None,
        hotspot_params=None,
        track_confidence_override=None,
        soft_steps_override=None,
        temperature_step_offset=0,
        temperature_total_steps=None,
        preserve_proposal_logits=False,
    ):
        self.propose_calls.append(
            {
                "current_seq": current_seq,
                "target_seq": target_seq,
                "global_step": global_step,
                "seed": seed,
                "action_group": action_group,
                "mutable_mask": mutable_mask,
                "max_mutations_per_seq": max_mutations_per_seq,
                "mutation_fraction": mutation_fraction,
                "candidate_strategy": candidate_strategy,
                "include_argmax_candidate": include_argmax_candidate,
                "proposal_logits": proposal_logits.detach().clone() if proposal_logits is not None else None,
                "hotspot_params": hotspot_params,
                "track_confidence_override": track_confidence_override,
                "soft_steps_override": soft_steps_override,
                "temperature_step_offset": temperature_step_offset,
                "temperature_total_steps": temperature_total_steps,
                "preserve_proposal_logits": preserve_proposal_logits,
            }
        )
        if current_seq == "AAA":
            return ["AAY", "AAF"][:B_candidates]
        if current_seq == "AAY":
            return ["AYY"][:B_candidates]
        return []

    def continuous_design(
        self,
        *,
        current_seq,
        target_seq,
        global_step,
        seed=None,
        mutable_mask=None,
        proposal_logits=None,
        hotspot_params=None,
        track_confidence_override=None,
        soft_steps_override=None,
        temperature_step_offset=0,
        temperature_total_steps=None,
        preserve_proposal_logits=False,
    ):
        self.continuous_calls.append(
            {
                "current_seq": current_seq,
                "target_seq": target_seq,
                "global_step": global_step,
                "seed": seed,
                "mutable_mask": mutable_mask,
                "proposal_logits": proposal_logits.detach().clone() if proposal_logits is not None else None,
                "hotspot_params": hotspot_params,
                "track_confidence_override": track_confidence_override,
                "soft_steps_override": soft_steps_override,
                "temperature_step_offset": temperature_step_offset,
                "temperature_total_steps": temperature_total_steps,
                "preserve_proposal_logits": preserve_proposal_logits,
            }
        )
        logits = (
            proposal_logits.detach().clone()
            if proposal_logits is not None
            else torch.zeros(len(current_seq), 20)
        )
        return ESMFold2ProposalBatch(
            sequences=["AYY"],
            logits_after=logits,
        )


class FakeHotspotVerifyBackend(FakeESMFold2Backend):
    def __init__(self):
        super().__init__()
        self.hotspot_verify_calls = []

    def verify(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
        self.hotspot_verify_calls.append(hotspot_params)
        return super().verify(
            binder_seq=binder_seq,
            target_seq=target_seq,
            seed=seed,
        )












def test_stageb_group_budget_keeps_every_repair_group_represented():
    budgets = _allocate_group_budgets(
        {
            "interface_formation": 0.8,
            "core_stabilization": 0.15,
            "compactness_repair": 0.05,
        },
        8,
    )

    assert sum(budgets.values()) == 8
    assert all(budgets[group] >= 1 for group in budgets)
    assert set(budgets) == {
        "interface_formation",
        "core_stabilization",
        "compactness_repair",
    }


def test_stageb_matched_no_group_uses_three_equal_compute_calls():
    specs = _stageb_proposal_call_specs(
        {"unrouted": 20},
        "no_group_matched_compute",
    )

    assert [group for group, _, _ in specs] == ["unrouted"] * 3
    assert [budget for _, budget, _ in specs] == [7, 7, 6]
    assert len({seed_label for _, _, seed_label in specs}) == 3
    assert sum(budget for _, budget, _ in specs) == 20


def test_stageb_matched_no_group_deduplicates_registered_actions():
    actions = [
        SimpleNamespace(action_id=0, sequence="AAA"),
        SimpleNamespace(action_id=0, sequence="AAA"),
        SimpleNamespace(action_id=1, sequence="AYA"),
    ]

    unique = _deduplicate_actions_by_sequence(actions)

    assert [(item.action_id, item.sequence) for item in unique] == [
        (0, "AAA"),
        (1, "AYA"),
    ]


def test_stageb_verifier_selection_applies_group_quotas_before_action_ranking():
    class FakeTree:
        @staticmethod
        def remaining_expansion_slots(node):
            return 8

        @staticmethod
        def has_sequence(sequence):
            return False

        @staticmethod
        def action_puct_score(node, action):
            return float(action.prior)

    action_id = 0

    def action(group, prior):
        nonlocal action_id
        result = SimpleNamespace(
            action_id=action_id,
            sequence=f"candidate_{action_id}",
            action_group=group,
            prior=prior,
            status="unexpanded",
            child_node_id=None,
        )
        action_id += 1
        return result

    actions = [
        action("interface_formation", 0.10),
        action("interface_formation", 0.09),
        action("interface_formation", 0.08),
        action("core_stabilization", 0.30),
        action("core_stabilization", 0.20),
        action("compactness_repair", 0.60),
        action("compactness_repair", 0.50),
    ]
    selected, quotas = _select_stageb_actions_by_group_quota(
        FakeTree(),
        SimpleNamespace(),
        actions,
        group_priors={
            "interface_formation": 0.40,
            "core_stabilization": 0.35,
            "compactness_repair": 0.25,
        },
        max_actions=4,
    )

    assert len(selected) == 4
    assert quotas == {
        "interface_formation": 2,
        "core_stabilization": 1,
        "compactness_repair": 1,
    }


def test_stageb_rerank_selection_keeps_three_exploit_and_one_exploration():
    class FakeTree:
        @staticmethod
        def remaining_expansion_slots(node):
            return 8

        @staticmethod
        def has_sequence(sequence):
            return False

        @staticmethod
        def action_puct_score(node, action):
            return float(action.prior)

    def action(action_id, group, benefit, gradient_prior):
        proposal = SimpleNamespace(
            rerank_benefit=benefit,
            gradient_prior=gradient_prior,
        )
        return SimpleNamespace(
            action_id=action_id,
            sequence=f"candidate_{action_id}",
            action_group=group,
            group_prior=1.0,
            prior=gradient_prior,
            proposal=proposal,
            status="unexpanded",
            child_node_id=None,
        )

    actions = [
        action(0, "interface_formation", 0.30, 0.7),
        action(1, "interface_formation", 0.10, 0.3),
        action(2, "core_stabilization", 0.20, 0.8),
        action(3, "core_stabilization", 0.05, 0.2),
        action(4, "compactness_repair", 0.15, 0.9),
        action(5, "compactness_repair", 0.01, 0.1),
    ]
    selected, quotas = _select_stageb_actions_by_group_quota(
        FakeTree(),
        SimpleNamespace(),
        actions,
        group_priors={
            "interface_formation": 1.0,
            "core_stabilization": 1.0,
            "compactness_repair": 1.0,
        },
        max_actions=4,
        exploration_actions=1,
        rng=random.Random(7),
    )

    assert len(selected) == 4
    assert {action.action_id for action in selected[:3]} == {0, 2, 4}
    assert sum(quotas.values()) == 4
    assert set(quotas) == {
        "interface_formation",
        "core_stabilization",
        "compactness_repair",
    }


def test_stageb_selection_filters_existing_tree_sequences_before_using_budget():
    class FakeTree:
        @staticmethod
        def remaining_expansion_slots(node):
            return 4

        @staticmethod
        def has_sequence(sequence):
            return sequence == "existing"

        @staticmethod
        def action_puct_score(node, action):
            return float(action.prior)

    def action(action_id, sequence, prior):
        return SimpleNamespace(
            action_id=action_id,
            sequence=sequence,
            action_group="interface_formation",
            group_prior=1.0,
            prior=prior,
            proposal=SimpleNamespace(rerank_benefit=prior, gradient_prior=prior),
            status="unexpanded",
            child_node_id=None,
        )

    selected, _ = _select_stageb_actions_by_group_quota(
        FakeTree(),
        SimpleNamespace(),
        [
            action(0, "existing", 1.0),
            action(1, "new_a", 0.8),
            action(2, "new_b", 0.7),
            action(3, "new_c", 0.6),
            action(4, "new_d", 0.5),
        ],
        group_priors={"interface_formation": 1.0},
        max_actions=4,
    )

    assert [item.sequence for item in selected] == ["new_a", "new_b", "new_c", "new_d"]


def test_stageb_preview_objective_matches_group_and_localized_hotspot():
    preview = {
        "inter_contact_loss": 0.7,
        "effective_inter_contact_loss": 0.4,
        "intra_contact_loss": 0.2,
        "glob_loss": 0.3,
        "hotspot_contact_loss": 0.5,
    }

    assert _stageb_preview_objective(
        preview,
        action_group="interface_formation",
        hotspot_weight=0.5,
        hotspot_interface_mode="localized",
    ) == pytest.approx(0.4)
    assert _stageb_preview_objective(
        preview,
        action_group="core_stabilization",
        hotspot_weight=0.5,
        hotspot_interface_mode="localized",
    ) == pytest.approx(0.45)
    assert _stageb_preview_objective(
        preview,
        action_group="compactness_repair",
        hotspot_weight=0.5,
        hotspot_interface_mode="localized",
    ) == pytest.approx(0.55)

    assert _stageb_preview_objective(
        preview,
        action_group="unrouted",
        hotspot_weight=0.5,
        hotspot_interface_mode="localized",
    ) == pytest.approx(0.36)
    assert _stageb_preview_objective(
        preview,
        action_group="unrouted",
        hotspot_weight=0.5,
        hotspot_interface_mode="additive",
    ) == pytest.approx(0.76)


def test_stageb_discrete_rerank_uses_paired_seed_and_parent_delta():
    class PreviewBackend:
        config = SimpleNamespace(
            hotspot_weight=0.5,
            hotspot_interface_mode="localized",
        )

        def __init__(self):
            self.calls = []

        def preview_loss_dict(
            self, *, binder_seq, target_seq, seed=None, hotspot_params=None
        ):
            self.calls.append((binder_seq, seed, hotspot_params))
            value = {"AAA": 0.8, "AYA": 0.5, "AFA": 0.7}[binder_seq]
            return {
                "inter_contact_loss": value,
                "effective_inter_contact_loss": value,
                "hotspot_contact_loss": value,
            }

    backend = PreviewBackend()
    scores = _rerank_stageb_candidates_with_preview(
        backend,
        {},
        current_seq="AAA",
        candidates=["AYA", "AFA"],
        target_seq="TTTT",
        action_group="interface_formation",
        seed=123,
        hotspot_params={"target_hotspot_concat_indices": [1]},
    )

    assert scores["AYA"]["benefit"] == pytest.approx(0.3)
    assert scores["AFA"]["benefit"] == pytest.approx(0.1)
    assert {call[1] for call in backend.calls} == {123}
    assert [call[0] for call in backend.calls] == ["AAA", "AYA", "AFA"]


def test_esmfold2_stageb_reward_uses_continuous_hotspot_coverage_loss():
    tree = _ESMFold2StageBMCTSTree(use_pareto_pruning=False)
    base = {
        "pair_chain_iptm_loss": 0.3,
        "binder_confidence_loss": 0.2,
        "global_confidence_loss": 0.1,
        "binder_rg_excess_loss": 0.2,
        "hotspot_verify_distance_loss": 0.4,
    }
    hit_reward, hit_status = tree.compute_node_reward(
        0.4,
        {**base, "hotspot_heavy_atom_coverage": 0.5},
    )
    miss_reward, miss_status = tree.compute_node_reward(
        0.4,
        {**base, "hotspot_heavy_atom_coverage": 0.0},
    )

    assert hit_status["reward_mode"] == "esmfold2_stageb_fixed"
    assert hit_status["hotspot_hit"] is False
    assert hit_status["hotspot_coverage_loss"] == pytest.approx(0.5)
    assert hit_status["hotspot_miss_penalty"] == pytest.approx(0.0)
    assert miss_status["hotspot_coverage_loss"] == pytest.approx(1.0)
    assert miss_status["hotspot_miss_penalty"] == pytest.approx(0.0)
    assert hit_reward > miss_reward


def test_esmfold2_stageb_reward_guardrail_rejects_low_quality_hotspot_hit():
    tree = _ESMFold2StageBMCTSTree(
        use_pareto_pruning=False,
        hotspot_distance_weight=0.1,
        hotspot_miss_penalty=0.25,
        min_pair_chain_iptm=0.25,
    )
    shared = {
        "binder_confidence_loss": 0.3,
        "global_confidence_loss": 0.2,
        "binder_rg_excess_loss": 0.2,
    }
    low_quality_hit_reward, low_quality_hit_status = tree.compute_node_reward(
        0.9,
        {
            **shared,
            "pair_chain_iptm_loss": 0.9,
            "hotspot_verify_distance_loss": 0.0,
            "hotspot_heavy_atom_coverage": 1.0,
        },
    )
    acceptable_miss_reward, acceptable_miss_status = tree.compute_node_reward(
        0.7,
        {
            **shared,
            "pair_chain_iptm_loss": 0.7,
            "hotspot_verify_distance_loss": 1.0,
            "hotspot_heavy_atom_coverage": 0.0,
        },
    )

    assert low_quality_hit_status["pair_chain_floor_violated"] is True
    assert low_quality_hit_status["pair_chain_floor_penalty"] == pytest.approx(2.0)
    assert acceptable_miss_status["pair_chain_floor_violated"] is False
    assert acceptable_miss_status["hotspot_miss_penalty"] == pytest.approx(0.25)
    assert acceptable_miss_reward > low_quality_hit_reward


def test_stageb_tree_expansion_prefers_quality_guarded_hotspot_coverage():
    tree = _ESMFold2StageBMCTSTree(
        use_pareto_pruning=False,
        hotspot_distance_weight=0.25,
        hotspot_coverage_weight=0.25,
    )

    def result(sequence, pair_chain_iptm, coverage, distance, total_loss):
        floor = 0.60
        return SimpleNamespace(
            binder_seq=sequence,
            total_loss=total_loss,
            loss_dict={
                "pair_chain_iptm_loss": 1.0 - pair_chain_iptm,
                "binder_confidence_loss": 0.2,
                "global_confidence_loss": 0.2,
                "binder_rg_excess_loss": 0.1,
                "hotspot_verify_distance_loss": min(distance / 20.0, 1.0),
                "hotspot_heavy_atom_coverage": coverage,
                "stageb_pair_chain_quality_floor": floor,
            },
            summary_confidence={
                "hotspot_heavy_atom_coverage": coverage,
                "hotspot_mean_min_heavy_atom_distance": distance,
            },
        )

    actions = [
        SimpleNamespace(action_id=0, sequence="low_quality_hit"),
        SimpleNamespace(action_id=1, sequence="partial"),
        SimpleNamespace(action_id=2, sequence="full"),
    ]
    results = [
        result("low_quality_hit", 0.40, 1.0, 3.0, 0.1),
        result("partial", 0.65, 0.5, 7.0, 0.3),
        result("full", 0.62, 1.0, 5.0, 0.4),
    ]

    selected_actions, selected_results = _select_stageb_tree_expansions(
        actions,
        results,
        tree=tree,
        max_children=2,
    )

    assert [action.sequence for action in selected_actions] == ["full", "partial"]
    assert [item.binder_seq for item in selected_results] == ["full", "partial"]
    assert (
        _stageb_guarded_rank(results[2], tree)
        > _stageb_guarded_rank(results[1], tree)
        > _stageb_guarded_rank(results[0], tree)
    )


def test_stageb_guarded_rank_prefers_global_quality_after_full_coverage():
    tree = _ESMFold2StageBMCTSTree(
        use_pareto_pruning=False,
        hotspot_distance_weight=0.10,
        hotspot_coverage_weight=0.25,
    )

    def result(
        sequence,
        *,
        pair_chain_iptm,
        binder_confidence_loss,
        distance,
    ):
        return SimpleNamespace(
            binder_seq=sequence,
            total_loss=1.0 - pair_chain_iptm,
            loss_dict={
                "pair_chain_iptm_loss": 1.0 - pair_chain_iptm,
                "binder_confidence_loss": binder_confidence_loss,
                "global_confidence_loss": binder_confidence_loss,
                "binder_rg_excess_loss": binder_confidence_loss,
                "hotspot_verify_distance_loss": 0.0,
                "hotspot_heavy_atom_coverage": 1.0,
                "stageb_pair_chain_quality_floor": 0.60,
            },
            summary_confidence={
                "hotspot_heavy_atom_coverage": 1.0,
                "hotspot_mean_min_heavy_atom_distance": distance,
            },
        )

    close_but_weaker = result(
        "close_but_weaker",
        pair_chain_iptm=0.62,
        binder_confidence_loss=0.30,
        distance=2.0,
    )
    farther_but_better = result(
        "farther_but_better",
        pair_chain_iptm=0.70,
        binder_confidence_loss=0.10,
        distance=5.0,
    )

    assert (
        _stageb_guarded_rank(farther_but_better, tree)
        > _stageb_guarded_rank(close_but_weaker, tree)
    )


def test_hotspot_proposal_weight_stops_after_full_verified_coverage():
    hit = SimpleNamespace(
        summary_confidence={"hotspot_heavy_atom_coverage": 1.0}
    )
    miss = SimpleNamespace(
        summary_confidence={"hotspot_heavy_atom_coverage": 0.5}
    )

    assert _hotspot_proposal_weight_for_verified_state(
        hit,
        configured_weight=0.5,
        satisfied_weight=0.0,
    ) == pytest.approx(0.0)
    assert _hotspot_proposal_weight_for_verified_state(
        miss,
        configured_weight=0.5,
        satisfied_weight=0.0,
    ) == pytest.approx(0.5)
    assert _hotspot_proposal_weight_for_verified_state(
        None,
        configured_weight=0.5,
        satisfied_weight=0.0,
    ) == pytest.approx(0.5)


def test_stageb_guarded_rank_accepts_a_run_level_quality_floor():
    tree = _ESMFold2StageBMCTSTree(use_pareto_pruning=False)

    def result(sequence, pair_chain_iptm, branch_floor, coverage):
        return SimpleNamespace(
            binder_seq=sequence,
            total_loss=1.0 - pair_chain_iptm,
            loss_dict={
                "pair_chain_iptm_loss": 1.0 - pair_chain_iptm,
                "binder_confidence_loss": 0.10,
                "global_confidence_loss": 0.10,
                "binder_rg_excess_loss": 0.10,
                "hotspot_verify_distance_loss": 0.0,
                "hotspot_heavy_atom_coverage": coverage,
                "stageb_pair_chain_quality_floor": branch_floor,
            },
            summary_confidence={
                "hotspot_heavy_atom_coverage": coverage,
                "hotspot_mean_min_heavy_atom_distance": 4.0,
            },
        )

    weak_branch_candidate = result("weak", 0.56, 0.50, 1.0)
    strong_root_candidate = result("strong", 0.62, 0.60, 0.75)

    assert (
        _stageb_guarded_rank(
            strong_root_candidate,
            tree,
            quality_floor=0.60,
        )
        > _stageb_guarded_rank(
            weak_branch_candidate,
            tree,
            quality_floor=0.60,
        )
    )


def test_stageb_proposals_are_strict_single_mutations_with_gradient_priors():
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, AA_TO_ESM_IDX["Y"]] = 3.0
    after[1, AA_TO_ESM_IDX["F"]] = 1.0

    proposals = _make_stageb_proposals(
        current_seq="AAA",
        candidates=["AYA", "AFA", "AYY"],
        action_group="interface_formation",
        group_prior=0.6,
        logits_before=before,
        logits_after=after,
        mutable_mask=[True, True, True],
    )

    assert [proposal.sequence for proposal in proposals] == ["AYA", "AFA"]
    assert all(len(proposal.edits) == 1 for proposal in proposals)
    assert proposals[0].benefit > proposals[1].benefit
    assert proposals[0].gradient_prior > proposals[1].gradient_prior
    assert sum(proposal.prior for proposal in proposals) == pytest.approx(0.6)

    uniform = _make_stageb_proposals(
        current_seq="AAA",
        candidates=["AYA", "AFA"],
        action_group="interface_formation",
        group_prior=0.6,
        logits_before=before,
        logits_after=after,
        mutable_mask=[True, True, True],
        use_gradient_prior=False,
    )
    assert [proposal.gradient_prior for proposal in uniform] == pytest.approx([0.5, 0.5])


def test_stageb_proposals_use_discrete_rerank_but_keep_logit_delta_metadata():
    before = torch.zeros(3, 20)
    after = before.clone()
    after[1, AA_TO_ESM_IDX["Y"]] = 3.0
    after[1, AA_TO_ESM_IDX["F"]] = 1.0

    proposals = _make_stageb_proposals(
        current_seq="AAA",
        candidates=["AYA", "AFA"],
        action_group="interface_formation",
        group_prior=0.6,
        logits_before=before,
        logits_after=after,
        mutable_mask=[True, True, True],
        rerank_scores={
            "AYA": {"loss": 0.8, "benefit": 0.1},
            "AFA": {"loss": 0.4, "benefit": 0.5},
        },
    )

    assert proposals[0].logit_delta_benefit > proposals[1].logit_delta_benefit
    assert proposals[0].rerank_benefit == pytest.approx(0.1)
    assert proposals[1].rerank_benefit == pytest.approx(0.5)
    assert proposals[1].rerank_prior > proposals[0].rerank_prior
    assert proposals[1].benefit > proposals[0].benefit


def test_stageb_soft_updates_do_not_overwrite_parent_node_logits():
    class StatefulStageBBackend(FakeESMFold2Backend):
        def __init__(self):
            super().__init__()
            self.stageb_inputs = []
            self.scores.update({"FAY": 0.01, "GAY": 0.01, "IAY": 0.01})

        def propose(self, **kwargs):
            if kwargs.get("action_group") is None:
                return super().propose(**kwargs)
            before = kwargs["proposal_logits"].detach().clone()
            self.stageb_inputs.append(
                (
                    kwargs["global_step"],
                    kwargs["action_group"],
                    kwargs["current_seq"],
                    before.clone(),
                )
            )
            group_to_candidate = {
                "interface_formation": ("FAY", "F"),
                "core_stabilization": ("GAY", "G"),
                "compactness_repair": ("IAY", "I"),
            }
            candidate, amino_acid = group_to_candidate[kwargs["action_group"]]
            after = before.clone()
            after[0, AA_TO_ESM_IDX[amino_acid]] += 5.0
            return ESMFold2ProposalBatch(
                sequences=[candidate],
                logits_before=before,
                logits_after=after,
            )

    backend = StatefulStageBBackend()
    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=3,
        T_sample=1.0,
        max_mutations_per_seq=3,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    step_one = [entry[3] for entry in backend.stageb_inputs if entry[0] == 1]
    step_two = [entry[3] for entry in backend.stageb_inputs if entry[0] == 2]
    assert step_one and step_two
    assert all(torch.equal(logits, step_one[0]) for logits in step_one)
    assert all(torch.equal(logits, step_two[0]) for logits in step_two)

    logits_by_sequence = {}
    for _, _, sequence, logits in backend.stageb_inputs:
        if sequence in logits_by_sequence:
            assert torch.equal(logits, logits_by_sequence[sequence])
        else:
            logits_by_sequence[sequence] = logits


def test_esmfold2_stagea_pipeline_uses_backend_for_verify_and_proposal():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAY", "AAF", "AYY"]
    assert [call["current_seq"] for call in backend.propose_calls] == ["AAA", "AAY"]
    assert [call["action_group"] for call in backend.propose_calls] == [None, None]


def test_stagea_legacy_tree_selects_lower_average_loss_path():
    tree = StageATree(C=0.0)
    low_loss = tree.add_root_node(
        sequence="AAA",
        logits=torch.zeros(3, 20),
        L_real=0.2,
    )
    tree.add_root_node(
        sequence="BBB",
        logits=torch.zeros(3, 20),
        L_real=0.8,
    )

    tree.backpropagate(low_loss, [0.1])

    assert tree.select().sequence == "AAA"


def test_stagea_tree_frontier_keeps_argmax_and_best_remaining_candidate():
    results = [
        SimpleNamespace(binder_seq="AAF", total_loss=0.20),
        SimpleNamespace(binder_seq="AAY", total_loss=0.70),
        SimpleNamespace(binder_seq="AYF", total_loss=0.40),
    ]

    selected = _select_stagea_tree_results(
        results,
        argmax_sequence="AAY",
        max_children=2,
    )

    assert [result.binder_seq for result in selected] == ["AAY", "AAF"]


def test_stagea_argmax_child_preserves_raw_logits_and_only_two_children_enter_tree(
    tmp_path,
):
    class RawLogitsStageABackend(FakeESMFold2Backend):
        def __init__(self):
            super().__init__()
            self.scores.update(
                {
                    "AAA": 0.10,
                    "AAY": 0.90,
                    "AAF": 0.80,
                    "AYF": 0.70,
                }
            )
            self.raw_logits = torch.full((3, len(ESM_AA_ORDER)), -1.0)
            self.raw_logits[:, AA_TO_ESM_IDX["C"]] = float("-inf")
            self.raw_logits[0, AA_TO_ESM_IDX["A"]] = 0.3
            self.raw_logits[1, AA_TO_ESM_IDX["A"]] = 0.4
            self.raw_logits[2, AA_TO_ESM_IDX["Y"]] = 0.5

        def propose(self, **kwargs):
            self.propose_calls.append(dict(kwargs))
            return ESMFold2ProposalBatch(
                sequences=["AAF", "AAY", "AYF"],
                logits_after=self.raw_logits.clone(),
            )

    backend = RawLogitsStageABackend()
    artifact_dir = tmp_path / "artifacts"

    best_seq, final_seq, _ = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=3,
        T_sample=1.0,
        max_mutations_per_seq=3,
        seed=7,
        mcts_strategy="two_stage",
        mcts_handoff_top_k=2,
        esmfold2_stagea_tree_children=2,
        esmfold2_stagea_root_selection_mode="global_ucb",
        esmfold2_stagea_min_root_expansions=0,
        esmfold2_stagea_handoff_checkpoint_interval=0,
        artifact_dir=artifact_dir,
    )

    assert best_seq == final_seq == "AAY"
    assert [call[0] for call in backend.verify_calls] == [
        "AAA",
        "AAF",
        "AAY",
        "AYF",
    ]

    trajectory = [
        json.loads(line)
        for line in (artifact_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    stagea_event = next(event for event in trajectory if event["stage"] == "stage_a")
    assert stagea_event["candidate_count"] == 3
    assert stagea_event["expanded_child_count"] == 2
    assert stagea_event["argmax_sequence"] == "AAY"
    assert stagea_event["argmax_child_in_tree"] is True
    assert stagea_event["root_selection_mode"] == "global_ucb"
    assert stagea_event["root_selection_phase"] == "global_ucb"
    assert stagea_event["selected_root_expansions"] == 1

    nodes = [
        json.loads(line)
        for line in (artifact_dir / "mcts_nodes.jsonl").read_text().splitlines()
    ]
    expanded_sequences = [
        event["sequence"]
        for event in nodes
        if event["stage"] == "stage_a" and event.get("parent_node_id") is not None
    ]
    assert expanded_sequences == ["AAY", "AAF"]

    summary = json.loads((artifact_dir / "run_summary.json").read_text())
    assert summary["stagea_root_selection"] == {
        "mode": "global_ucb",
        "min_root_expansions": 0,
        "root_expansion_counts": {"0": 1},
    }

    checkpoint = torch.load(
        artifact_dir / "handoff_checkpoints" / "stagea_handoff_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    logits_by_sequence = {
        entry["result"]["binder_seq"]: entry["proposal_logits"]
        for entry in checkpoint["handoff_entries"]
    }
    assert torch.equal(logits_by_sequence["AAY"], backend.raw_logits)
    assert not torch.equal(logits_by_sequence["AAF"], backend.raw_logits)
    assert torch.equal(logits_by_sequence["AAF"][0], backend.raw_logits[0])
    assert torch.equal(logits_by_sequence["AAF"][1], backend.raw_logits[1])
    assert logits_by_sequence["AAF"][2, AA_TO_ESM_IDX["F"]] == pytest.approx(0.6)


def test_stagea_uses_one_run_level_verifier_seed_for_all_candidates():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        esmfold2_stagea_paired_verifier_seed=True,
    )

    assert len(backend.verify_calls) == 3
    assert len({call[2] for call in backend.verify_calls}) == 1


def test_stagea_cumulative_annealing_follows_selected_node_soft_depth(tmp_path):
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_C=0.0,
        mcts_strategy="two_stage",
        esmfold2_stagea_soft_steps=30,
        esmfold2_stagea_cumulative_annealing=True,
        esmfold2_stagea_annealing_horizon=150,
        esmfold2_stagea_preserve_soft_state=True,
        esmfold2_stagea_handoff_checkpoint_interval=0,
        artifact_dir=tmp_path / "artifacts",
    )

    assert [call["temperature_step_offset"] for call in backend.propose_calls] == [
        0,
        30,
    ]
    assert [call["temperature_total_steps"] for call in backend.propose_calls] == [
        150,
        150,
    ]
    assert all(call["preserve_proposal_logits"] for call in backend.propose_calls)

    trajectory = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "trajectory.jsonl").read_text().splitlines()
    ]
    stagea_events = [event for event in trajectory if event["stage"] == "stage_a"]
    assert [event["soft_step_start"] for event in stagea_events] == [0, 30]
    assert [event["soft_step_end"] for event in stagea_events] == [30, 60]


def test_stagea_defaults_to_independent_outer_step_annealing():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_C=0.0,
        mcts_strategy="two_stage",
        esmfold2_stagea_soft_steps=30,
        esmfold2_stagea_preserve_soft_state=True,
        esmfold2_stagea_handoff_checkpoint_interval=0,
    )

    assert [call["temperature_step_offset"] for call in backend.propose_calls] == [
        0,
        0,
    ]
    assert [call["temperature_total_steps"] for call in backend.propose_calls] == [
        None,
        None,
    ]














def test_esmfold2_pipeline_propagates_hotspot_to_proposal_and_preview_rerank():
    backend = FakeESMFold2Backend()
    hotspot_params = {"target_hotspot_concat_indices": [1, 2]}

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq=["TT", "GG"],
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        hotspot_params=hotspot_params,
    )

    assert backend.propose_calls
    assert all(call["hotspot_params"] == hotspot_params for call in backend.propose_calls)
    assert backend.preview_calls
    assert all(call[3] == hotspot_params for call in backend.preview_calls)


def test_esmfold2_pipeline_propagates_hotspots_to_verify_when_supported():
    backend = FakeHotspotVerifyBackend()
    hotspot_params = {"target_hotspot_concat_indices": [1, 2]}

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        hotspot_params=hotspot_params,
    )

    assert backend.hotspot_verify_calls
    assert all(call == hotspot_params for call in backend.hotspot_verify_calls)




def test_esmfold2_pipeline_rejects_positive_backend_weight_without_hotspots():
    backend = FakeESMFold2Backend()
    backend.config = type("Config", (), {"hotspot_weight": 0.5})()

    with pytest.raises(
        ValueError,
        match="positive ESMFold2 hotspot_weight requires target hotspot indices",
    ):
        run_esmfold2_mcts_hallucination(
            backend=backend,
            target_seq="TTTT",
            L_binder=3,
            N_steps=1,
            num_beams=1,
            init_binder_seq="AAA",
        )


def test_esmfold2_pipeline_rejects_positive_hotspot_verify_weight_without_hotspots():
    backend = FakeESMFold2Backend()
    backend.config = type(
        "Config",
        (),
        {"hotspot_weight": 0.0, "hotspot_verify_weight": 0.2},
    )()

    with pytest.raises(
        ValueError,
        match="positive ESMFold2 hotspot_verify_weight requires target hotspot indices",
    ):
        run_esmfold2_mcts_hallucination(
            backend=backend,
            target_seq="TTTT",
            L_binder=3,
            N_steps=1,
            num_beams=1,
            init_binder_seq="AAA",
        )


def test_esmfold2_official_free_roots_keep_unanchored_mutable_logits():
    [(current_seq, proposal_logits)] = _make_phase0_seed_states(
        None, num_beams=1, L_binder=3, seed=7,
        mutable_mask=[True, True, True], logit_init_mode="official_free")
    for pos, aa in enumerate(current_seq):
        aa_idx = AA_TO_ESM_IDX[aa]
        assert proposal_logits[pos, aa_idx].item() < 0.5
        assert torch.argmax(proposal_logits[pos]).item() == aa_idx


def test_esmfold2_official_exact_roots_use_direct_torch_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = torch.randn((1, 3, len(ESM_AA_ORDER))) * 0.01
        expected[:, :, AA_TO_ESM_IDX["C"]] = float("-inf")
    expected_logits = expected[0]
    expected_seq = "".join(
        ESM_AA_ORDER[int(torch.argmax(expected_logits[pos]).item())]
        for pos in range(3)
    )

    states = _make_phase0_seed_states(
        None,
        num_beams=1,
        L_binder=3,
        seed=7,
        mutable_mask=[True, True, True],
        logit_init_mode="official_exact",
    )

    seq, logits = states[0]
    assert seq == expected_seq
    assert torch.allclose(logits, expected_logits, equal_nan=False)


def test_official_exact_phase0_handoff_preserves_free_logits(tmp_path):
    expected_sequence, expected_logits = _make_phase0_seed_states(
        None,
        num_beams=1,
        L_binder=3,
        seed=42,
        mutable_mask=[True, True, True],
        logit_init_mode="official_exact",
    )[0]
    artifact_dir = tmp_path / "artifacts"

    run_esmfold2_mcts_hallucination(
        backend=FakeESMFold2Backend(),
        target_seq="TTTT",
        L_binder=3,
        N_steps=0,
        num_beams=1,
        seed=42,
        mcts_strategy="two_stage",
        esmfold2_logit_init_mode="official_exact",
        artifact_dir=artifact_dir,
    )

    payload = torch.load(
        artifact_dir / "handoff_checkpoints" / "stagea_handoff_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    entry = payload["handoff_entries"][0]
    assert entry["result"]["binder_seq"] == expected_sequence
    assert torch.equal(entry["proposal_logits"], expected_logits)
    finite_logits = entry["proposal_logits"][torch.isfinite(entry["proposal_logits"])]
    assert finite_logits.abs().max().item() < 0.1


def test_initial_logits_checkpoint_directory_restores_exact_phase0_roots(tmp_path):
    best_of_n_dir = tmp_path / "best_of_n"
    expected_states = []
    for trajectory_index, sequence in enumerate(("AYY", "FYA")):
        trajectory_dir = best_of_n_dir / f"trajectory_{trajectory_index:03d}"
        trajectory_dir.mkdir(parents=True)
        logits = torch.full((3, len(ESM_AA_ORDER)), -1.0)
        for position, amino_acid in enumerate(sequence):
            logits[position, AA_TO_ESM_IDX[amino_acid]] = 1.0
        torch.save(
            {
                "trajectory_index": trajectory_index,
                "label": "initial_logits",
                "seed": trajectory_index + 10,
                "sequence": sequence,
                "sequence_hash": hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16],
                "logits": logits,
            },
            trajectory_dir / "initial_logits.pt",
        )
        expected_states.append((sequence, logits))

    states, paths = _load_initial_logits_checkpoint_states(
        tmp_path,
        L_binder=3,
        num_beams=2,
        init_binder_seq=None,
        mutable_mask=[True, True, True],
    )

    assert [sequence for sequence, _ in states] == ["AYY", "FYA"]
    assert len(paths) == 2
    for (_, actual_logits), (_, expected_logits) in zip(states, expected_states):
        assert torch.equal(actual_logits, expected_logits)


def test_initial_logits_checkpoint_requires_exact_beam_count(tmp_path):
    logits = torch.zeros((3, len(ESM_AA_ORDER)))
    torch.save({"sequence": "AAA", "logits": logits}, tmp_path / "initial_logits.pt")

    with pytest.raises(ValueError, match="does not match mcts_num_beams"):
        _load_initial_logits_checkpoint_states(
            tmp_path / "initial_logits.pt",
            L_binder=3,
            num_beams=2,
            init_binder_seq=None,
            mutable_mask=[True, True, True],
        )


def test_initial_logits_checkpoint_rejects_fixed_scaffold_mismatch(tmp_path):
    logits = torch.full((3, len(ESM_AA_ORDER)), -1.0)
    for position, amino_acid in enumerate("YAA"):
        logits[position, AA_TO_ESM_IDX[amino_acid]] = 1.0
    checkpoint = tmp_path / "initial_logits.pt"
    torch.save({"logits": logits}, checkpoint)

    with pytest.raises(ValueError, match="fixed binder"):
        _load_initial_logits_checkpoint_states(
            checkpoint,
            L_binder=3,
            num_beams=1,
            init_binder_seq="AAA",
            mutable_mask=[False, True, True],
        )


def test_mcts_uses_loaded_initial_logits_for_first_proposal(tmp_path):
    backend = FakeESMFold2Backend()
    logits = torch.full((3, len(ESM_AA_ORDER)), -1.0)
    for position, amino_acid in enumerate("AAA"):
        logits[position, AA_TO_ESM_IDX[amino_acid]] = 1.0 + position
    checkpoint = tmp_path / "initial_logits.pt"
    torch.save(
        {
            "sequence": "AAA",
            "sequence_hash": hashlib.sha256(b"AAA").hexdigest()[:16],
            "logits": logits,
        },
        checkpoint,
    )

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        esmfold2_initial_logits_checkpoint=checkpoint,
    )

    assert len(backend.propose_calls) == 1
    assert torch.equal(backend.propose_calls[0]["proposal_logits"], logits)


def test_esmfold2_official_free_scaffold_roots_anchor_only_fixed_positions():
    [(sequence, logits)] = _make_phase0_seed_states(
        "AAA", num_beams=1, L_binder=3, seed=7,
        mutable_mask=[False, True, False], logit_init_mode="official_free")
    assert sequence[0] == sequence[2] == "A"
    for pos in (0, 2):
        assert logits[pos, AA_TO_ESM_IDX["A"]].item() == pytest.approx(10.0)
    assert logits[1, torch.argmax(logits[1])].item() < 0.5




def test_esmfold2_stagea_ignores_tracked_confidence_candidate_during_normal_search():
    class ConfidenceProposalBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            super().propose(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=["AAF"],
                confidence_candidates=[
                    {
                        "sequence": "AAY",
                        "local_step": 4,
                        "confidence_score": 0.80,
                        "confidence_source": "iptm",
                    }
                ],
            )

    backend = ConfidenceProposalBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
    )

    assert best_seq == "AAF"
    assert final_seq == "AAF"
    assert losses == [pytest.approx(0.60)]
    assert [call[0] for call in backend.verify_calls] == ["AAA", "AAF"]
    assert backend.propose_calls[0]["track_confidence_override"] is False


def test_esmfold2_stagea_does_not_use_tracked_confidence_candidates_without_final_polish():
    class ManyConfidenceProposalBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            super().propose(**kwargs)
            return ESMFold2ProposalBatch(
                sequences=[],
                confidence_candidates=[
                    {
                        "sequence": "AAY",
                        "local_step": 10,
                        "confidence_score": 0.80,
                        "confidence_source": "iptm",
                    },
                    {
                        "sequence": "AAF",
                        "local_step": 11,
                        "confidence_score": 0.70,
                        "confidence_source": "iptm",
                    },
                    {
                        "sequence": "AYY",
                        "local_step": 12,
                        "confidence_score": 0.60,
                        "confidence_source": "iptm",
                    },
                ],
            )

    backend = ManyConfidenceProposalBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
    )

    assert best_seq == "AAA"
    assert final_seq == "AAA"
    assert losses == [pytest.approx(0.90)]
    assert [call[0] for call in backend.verify_calls] == ["AAA"]
    assert backend.propose_calls[0]["track_confidence_override"] is False








def test_budget_ending_in_stage_a_never_enters_stageb():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=2,
        mcts_bootstrap_max_steps=2,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert [call["action_group"] for call in backend.propose_calls] == [None, None]
    assert backend.preview_calls == []


def test_esmfold2_pipeline_passes_mutable_mask_and_clips_fixed_positions():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        mutable_mask=[False, True, True],
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
    )

    assert best_seq == "AAY"
    assert final_seq == "AAY"
    assert losses[-1] == pytest.approx(0.20)
    assert backend.propose_calls[0]["mutable_mask"] == [False, True, True]
    assert all(seq[0] == "A" for seq, *_ in backend.verify_calls)


def test_esmfold2_two_stage_pipeline_routes_stageb_proposals_by_group():
    backend = FakeESMFold2Backend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert best_seq == "AYY"
    assert final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)
    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.preview_calls


def test_esmfold2_two_stage_pipeline_passes_mutable_mask_to_stageb():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        mutable_mask=[False, True, True],
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert backend.propose_calls[0]["mutable_mask"] == [False, True, True]
    assert backend.propose_calls[1]["mutable_mask"] == [False, True, True]


def test_esmfold2_two_stage_uses_separate_stagea_and_stageb_mutation_budgets():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=3,
        stagea_max_mutations_per_seq=6,
        esmfold2_stagea_mutation_fraction=None,
        esmfold2_stagea_candidate_strategy="legacy",
        stageb_max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[0]["max_mutations_per_seq"] == 6
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.propose_calls[1]["max_mutations_per_seq"] == 1


def test_esmfold2_two_stage_accepts_explicit_stageb_verifier_seed():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        esmfold2_stagea_verifier_seed=111,
        esmfold2_stageb_verifier_seed=222,
    )

    verify_seeds = [call[2] for call in backend.verify_calls]
    assert 111 in verify_seeds
    assert 222 in verify_seeds


def test_esmfold2_stagea_defaults_to_positive_score_half_fraction_and_argmax():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=4,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAAA",
        K_top=4,
        B_candidates=2,
        seed=7,
        mcts_strategy="two_stage",
    )

    proposal = backend.propose_calls[0]
    assert proposal["mutation_fraction"] == pytest.approx(0.5)
    assert proposal["candidate_strategy"] == "positive_score"
    assert proposal["include_argmax_candidate"] is True
    assert proposal["max_mutations_per_seq"] == 2


def test_esmfold2_positive_score_still_requires_a_mutation_fraction():
    with pytest.raises(ValueError, match="requires esmfold2_stagea_mutation_fraction"):
        run_esmfold2_mcts_hallucination(
            backend=FakeESMFold2Backend(),
            target_seq="TTTT",
            L_binder=4,
            N_steps=1,
            num_beams=1,
            init_binder_seq="AAAA",
            esmfold2_stagea_mutation_fraction=None,
            esmfold2_stagea_candidate_strategy="positive_score",
            mcts_strategy="two_stage",
        )


def test_esmfold2_stagea_fraction_uses_mutable_length_and_leaves_stageb_unchanged():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=10,
        N_steps=3,
        num_beams=1,
        init_binder_seq="A" * 10,
        mutable_mask=[False, False] + [True] * 8,
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=3,
        esmfold2_stagea_mutation_fraction=0.5,
        esmfold2_stagea_candidate_strategy="positive_score",
        esmfold2_stagea_include_argmax_candidate=False,
        stageb_max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
    )

    assert backend.propose_calls[0]["max_mutations_per_seq"] == 4
    assert backend.propose_calls[0]["mutation_fraction"] == pytest.approx(0.5)
    assert backend.propose_calls[0]["candidate_strategy"] == "positive_score"
    assert backend.propose_calls[0]["include_argmax_candidate"] is False
    assert backend.propose_calls[1]["max_mutations_per_seq"] == 1
    assert backend.propose_calls[1]["mutation_fraction"] is None
    assert backend.propose_calls[1]["candidate_strategy"] is None
    assert backend.propose_calls[1]["include_argmax_candidate"] is None


def test_esmfold2_two_stage_uses_separate_stagea_and_stageb_soft_steps():
    backend = FakeESMFold2Backend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=3,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        esmfold2_stagea_soft_steps=30,
        esmfold2_stageb_soft_steps=5,
    )

    assert backend.propose_calls[0]["action_group"] is None
    assert backend.propose_calls[0]["soft_steps_override"] == 30
    assert backend.propose_calls[1]["action_group"] == "interface_formation"
    assert backend.propose_calls[1]["soft_steps_override"] == 5


def test_esmfold2_pipeline_persists_node_local_logits_between_stagea_visits():
    class StatefulLogitBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            proposal_logits = kwargs.get("proposal_logits")
            self.propose_calls.append(
                {
                    "current_seq": kwargs["current_seq"],
                    "proposal_logits": (
                        proposal_logits.detach().clone()
                        if proposal_logits is not None
                        else None
                    ),
                }
            )
            if kwargs["current_seq"] == "AAA":
                assert proposal_logits is not None
                updated = proposal_logits.detach().clone()
                updated[1, AA_TO_ESM_IDX["F"]] = 4.0
                updated[2, AA_TO_ESM_IDX["Y"]] = 3.0
                return ESMFold2ProposalBatch(
                    sequences=["AAY"],
                    logits_after=updated,
                )
            return ESMFold2ProposalBatch(
                sequences=[],
                logits_after=proposal_logits,
            )

    backend = StatefulLogitBackend()

    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
    )

    assert [call["current_seq"] for call in backend.propose_calls] == ["AAA", "AAY"]
    second_logits = backend.propose_calls[1]["proposal_logits"]
    assert second_logits is not None
    assert second_logits[1, AA_TO_ESM_IDX["F"]].item() == pytest.approx(4.0)
    assert second_logits[2, AA_TO_ESM_IDX["Y"]].item() >= 3.0


def test_esmfold2_pipeline_remains_compatible_with_list_only_proposal_backend():
    class ListOnlyBackend(FakeESMFold2Backend):
        def propose(
            self,
            *,
            current_seq,
            target_seq,
            global_step,
            K_top,
            B_candidates,
            T_sample,
            max_mutations_per_seq,
            seed=None,
            action_group=None,
            mutable_mask=None,
        ):
            self.propose_calls.append(
                {
                    "current_seq": current_seq,
                    "action_group": action_group,
                }
            )
            return ["AAY"][:B_candidates] if current_seq == "AAA" else []

    backend = ListOnlyBackend()

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=1,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        hotspot_params={"target_hotspot_concat_indices": [1]},
    )

    assert best_seq == final_seq == "AAY"
    assert losses[-1] == pytest.approx(0.20)


def test_esmfold2_pipeline_writes_structured_step_artifacts(tmp_path):
    backend = FakeESMFold2Backend()
    artifact_dir = tmp_path / "esmfold2_artifacts"

    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=3,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        T_sample=1.0,
        max_mutations_per_seq=1,
        seed=7,
        mcts_strategy="two_stage",
        mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1,
        mcts_expand_actions_per_step=1,
        artifact_dir=artifact_dir,
        save_verified_candidate_cif=True,
    )

    assert best_seq == final_seq == "AYY"
    assert losses[-1] == pytest.approx(0.10)

    trajectory_path = artifact_dir / "trajectory.jsonl"
    candidates_path = artifact_dir / "candidates.jsonl"
    nodes_path = artifact_dir / "mcts_nodes.jsonl"
    edges_path = artifact_dir / "mcts_edges.jsonl"
    actions_path = artifact_dir / "mcts_actions.jsonl"
    soft_steps_path = artifact_dir / "soft_steps.jsonl"
    handoff_path = artifact_dir / "mcts_handoff.jsonl"
    summary_path = artifact_dir / "run_summary.json"
    assert trajectory_path.exists()
    assert candidates_path.exists()
    assert nodes_path.exists()
    assert edges_path.exists()
    assert actions_path.exists()
    assert soft_steps_path.exists()
    assert handoff_path.exists()
    assert summary_path.exists()

    trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    candidates = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    nodes = [json.loads(line) for line in nodes_path.read_text().splitlines()]
    edges = [json.loads(line) for line in edges_path.read_text().splitlines()]
    actions = [json.loads(line) for line in actions_path.read_text().splitlines()]
    soft_steps = [json.loads(line) for line in soft_steps_path.read_text().splitlines()]
    handoff = [json.loads(line) for line in handoff_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert [event["stage"] for event in trajectory] == ["phase0", "stage_a", "stage_b"]
    # Both Stage B budget iterations count even when one has no available action.
    assert len(losses) == summary["n_steps"] == 3
    assert trajectory[1]["candidate_count"] == 2
    assert trajectory[2]["primary_action_group"] == "interface_formation"
    assert trajectory[2]["selected_group_counts"] == {"interface_formation": 1}
    assert soft_steps == []
    assert any(event["stage"] == "stage_b" and event["sequence"] == "AYY" for event in candidates)
    assert candidates[-1]["loss_dict"]["total"] == pytest.approx(0.10)
    assert all(event["structure_status"] == "ok" for event in candidates)
    assert all(
        (artifact_dir / event["structure_cif"]).is_file()
        for event in candidates
    )
    assert not list(
        (artifact_dir / "verified_candidate_cif").glob("*.pdb")
    )
    assert any(node["stage"] == "stage_a" and node["parent_node_id"] is None for node in nodes)
    assert any(node["stage"] == "stage_b" and node["sequence"] == "AYY" for node in nodes)
    assert any(edge["stage"] == "stage_a" and edge["edits"] for edge in edges)
    assert any(action["stage"] == "stage_b" and action["selected"] for action in actions)
    assert any(event["source_stage"] == "stage_a" and event["target_stage"] == "stage_b" for event in handoff)

    stage_b_ckpt_dir = artifact_dir / "checkpoints" / "stage_b" / "step_0001"
    stage_b_meta = json.loads((stage_b_ckpt_dir / "step_0001_round_best_meta.json").read_text())
    assert (stage_b_ckpt_dir / "step_0001_round_best.cif").exists()
    assert stage_b_meta["binder_seq"] == "AYY"
    assert stage_b_meta["total_loss"] == pytest.approx(0.10)
    assert stage_b_meta["structure_format"] == "mmcif"
    assert "mmcif" not in stage_b_meta["summary_confidence"]
    assert summary["best_sequence"] == "AYY"
    assert summary["best_loss"] == pytest.approx(0.10)


@pytest.mark.parametrize("n_steps", [1, 2])
def test_final_checkpoint_matches_selected_verifier_result_without_improvement(tmp_path, n_steps):
    class NoMutationBackend(FakeESMFold2Backend):
        def propose(self, **kwargs):
            return ESMFold2ProposalBatch(sequences=[])

        def verify(self, *, binder_seq, target_seq, seed=None, hotspot_params=None):
            self.scores[binder_seq] = 0.9 - 0.1 * len(self.verify_calls)
            result = super().verify(binder_seq=binder_seq, target_seq=target_seq, seed=seed)
            result.summary_confidence["mmcif"] = f"data_verify_{len(self.verify_calls)}\n#\n"
            self.last_result = result
            return result

    backend = NoMutationBackend()
    artifact_dir = tmp_path / "artifacts"
    best, final, losses = run_esmfold2_mcts_hallucination(
        backend=backend, target_seq="ACDE", L_binder=3, N_steps=n_steps,
        num_beams=1, init_binder_seq="AAA", mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1, artifact_dir=artifact_dir,
    )
    summary = json.loads((artifact_dir / "run_summary.json").read_text())
    step_name = f"step_{n_steps:04d}"
    checkpoint_dir = artifact_dir / "checkpoints" / "final" / step_name
    metadata = json.loads((checkpoint_dir / f"{step_name}_best_meta.json").read_text())
    assert best == final == metadata["binder_seq"] == "AAA"
    assert metadata["total_loss"] == summary["best_loss"] == losses[-1]
    assert metadata["summary_confidence"]["pair_chain_iptm"] == backend.last_result.summary_confidence["pair_chain_iptm"]
    assert (checkpoint_dir / metadata["structure_file"]).read_text() == backend.last_result.summary_confidence["mmcif"]
    assert len(backend.verify_calls) == n_steps
    assert len(losses) == n_steps


def test_stagea_handoff_checkpoint_is_saved_and_resumes_directly_into_stageb(tmp_path):
    stagea_backend = FakeESMFold2Backend()
    source_artifacts = tmp_path / "stagea_source"

    run_esmfold2_mcts_hallucination(
        backend=stagea_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        init_binder_seq="AAA",
        K_top=4,
        B_candidates=2,
        seed=7,
        mcts_strategy="two_stage",
        artifact_dir=source_artifacts,
        esmfold2_stagea_handoff_checkpoint_interval=1,
    )

    checkpoint_dir = source_artifacts / "handoff_checkpoints"
    checkpoint_path = checkpoint_dir / "stagea_handoff_final.pt"
    assert (checkpoint_dir / "stagea_handoff_step_0001.pt").exists()
    assert (checkpoint_dir / "stagea_handoff_step_0002.pt").exists()
    assert checkpoint_path.exists()

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["completed_stagea_steps"] == 2
    assert payload["losses_history"] == pytest.approx([0.2, 0.1])
    assert payload["handoff_entries"][0]["result"]["binder_seq"] == "AYY"
    saved_logits = payload["handoff_entries"][0]["proposal_logits"].clone()

    resumed_backend = FakeESMFold2Backend()
    best_seq, final_seq, losses = run_esmfold2_mcts_hallucination(
        backend=resumed_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=4,
        num_beams=1,
        K_top=4,
        B_candidates=2,
        seed=999,
        mcts_strategy="two_stage",
        mcts_expand_actions_per_step=1,
        artifact_dir=tmp_path / "stageb_resume",
        esmfold2_resume_handoff_checkpoint=checkpoint_path,
        esmfold2_stageb_paired_verifier_seed=False,
    )

    assert best_seq == final_seq == "AYY"
    assert losses[:2] == pytest.approx([0.2, 0.1])
    assert len(losses) == 4
    assert resumed_backend.verify_calls == []
    assert resumed_backend.propose_calls
    assert all(call["action_group"] is not None for call in resumed_backend.propose_calls)
    expected_logits = minimally_align_binder_logits_to_sequence(saved_logits, "AYY")
    assert torch.equal(
        resumed_backend.propose_calls[0]["proposal_logits"],
        expected_logits,
    )

    summary = json.loads((tmp_path / "stageb_resume" / "run_summary.json").read_text())
    assert summary["resumed_from_stagea_handoff"] == str(checkpoint_path.resolve())




def test_stagea_handoff_checkpoint_rejects_mismatched_target(tmp_path):
    backend = FakeESMFold2Backend()
    artifact_dir = tmp_path / "source"
    run_esmfold2_mcts_hallucination(
        backend=backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        B_candidates=1,
        mcts_strategy="two_stage",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="target does not match"):
        run_esmfold2_mcts_hallucination(
            backend=FakeESMFold2Backend(),
            target_seq="VVVV",
            L_binder=3,
            N_steps=2,
            num_beams=1,
            B_candidates=1,
            mcts_strategy="two_stage",
            artifact_dir=tmp_path / "resume",
            esmfold2_resume_handoff_checkpoint=(
                artifact_dir / "handoff_checkpoints" / "stagea_handoff_final.pt"
            ),
        )


def test_handoff_resume_checks_only_fixed_positions_against_starting_binder(tmp_path):
    shared = dict(
        target_seq="TTTT", L_binder=3, num_beams=1, B_candidates=2,
        mutable_mask=[False, True, True],
    )
    source = tmp_path / "source"
    run_esmfold2_mcts_hallucination(
        backend=FakeESMFold2Backend(), N_steps=1, init_binder_seq="AAA",
        artifact_dir=source, **shared,
    )
    checkpoint = source / "handoff_checkpoints" / "stagea_handoff_final.pt"
    rejected = FakeESMFold2Backend()
    with pytest.raises(ValueError, match="fixed binder position 1"):
        run_esmfold2_mcts_hallucination(
            backend=rejected, N_steps=2, init_binder_seq="FAA",
            artifact_dir=tmp_path / "rejected",
            esmfold2_resume_handoff_checkpoint=checkpoint, **shared,
        )
    assert rejected.verify_calls == rejected.propose_calls == []

    best, _, _ = run_esmfold2_mcts_hallucination(
        backend=FakeESMFold2Backend(), N_steps=2, init_binder_seq="AYY",
        artifact_dir=tmp_path / "accepted",
        esmfold2_resume_handoff_checkpoint=checkpoint, **shared,
    )
    assert best[0] == "A"


def test_stagea_handoff_can_reverify_roots_after_verifier_config_change(tmp_path):
    hotspot_params = {
        "target_hotspot_residues_1based": [2],
        "target_hotspot_concat_indices": [1],
    }
    stagea_backend = FakeESMFold2Backend()
    stagea_backend.config = SimpleNamespace(
        verifier_model_name="biohub/ESMFold2",
        verifier_num_loops=4,
        verifier_num_sampling_steps=50,
        hotspot_verify_weight=0.10,
        hotspot_verify_near_cutoff=5.0,
        hotspot_verify_far_cutoff=20.0,
    )
    source_artifacts = tmp_path / "source"
    run_esmfold2_mcts_hallucination(
        backend=stagea_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=1,
        num_beams=1,
        init_binder_seq="AAA",
        B_candidates=1,
        mcts_strategy="two_stage",
        hotspot_params=hotspot_params,
        artifact_dir=source_artifacts,
    )
    checkpoint_path = source_artifacts / "handoff_checkpoints" / "stagea_handoff_final.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    handoff_sequences = [
        str(entry["result"]["binder_seq"])
        for entry in payload["handoff_entries"]
    ]

    resumed_backend = FakeESMFold2Backend()
    resumed_backend.config = SimpleNamespace(
        verifier_model_name="biohub/ESMFold2",
        verifier_num_loops=20,
        verifier_num_sampling_steps=200,
        hotspot_verify_weight=0.25,
        hotspot_verify_near_cutoff=5.0,
        hotspot_verify_far_cutoff=20.0,
    )
    with pytest.raises(ValueError, match="verifier configuration does not match"):
        run_esmfold2_mcts_hallucination(
            backend=resumed_backend,
            target_seq="TTTT",
            L_binder=3,
            N_steps=2,
            num_beams=1,
            B_candidates=1,
            mcts_strategy="two_stage",
            hotspot_params=hotspot_params,
            artifact_dir=tmp_path / "rejected",
            esmfold2_resume_handoff_checkpoint=checkpoint_path,
        )

    resumed_backend.verify_calls.clear()
    run_esmfold2_mcts_hallucination(
        backend=resumed_backend,
        target_seq="TTTT",
        L_binder=3,
        N_steps=2,
        num_beams=1,
        B_candidates=1,
        mcts_strategy="two_stage",
        hotspot_params=hotspot_params,
        artifact_dir=tmp_path / "reverified",
        esmfold2_resume_handoff_checkpoint=checkpoint_path,
        esmfold2_resume_reverify_roots=True,
    )

    assert [call[0] for call in resumed_backend.verify_calls[: len(handoff_sequences)]] == (
        handoff_sequences
    )
    summary = json.loads((tmp_path / "reverified" / "run_summary.json").read_text())
    assert summary["resumed_handoff_roots_reverified"] is True
    assert summary["resumed_handoff_verifier_contract_matches"] is False


def test_reverify_roots_requires_handoff_checkpoint():
    with pytest.raises(ValueError, match="requires a StageA handoff checkpoint"):
        run_esmfold2_mcts_hallucination(
            backend=FakeESMFold2Backend(),
            target_seq="TTTT",
            L_binder=3,
            N_steps=1,
            num_beams=1,
            B_candidates=1,
            mcts_strategy="two_stage",
            esmfold2_resume_reverify_roots=True,
        )
