from pathlib import Path

import pytest
import torch

from proteocraft.hallucination import esmfold2_backend as backend
from proteocraft.hallucination.esmfold2_pipeline import _call_backend_propose
from proteocraft.cli import parse_args_with_profile, resolve_search_defaults


def logits():
    before = torch.zeros(6, 20)
    before[:, backend.AA_TO_ESM_IDX["A"]] = 2.0
    after = before.clone()
    after[:, backend.AA_TO_ESM_IDX["G"]] = 0.6
    after[:, backend.AA_TO_ESM_IDX["S"]] = 0.4
    # Only the first two positions have a stable dominant current residue.
    before[2:, backend.AA_TO_ESM_IDX["A"]] = 0.1
    after[2:, backend.AA_TO_ESM_IDX["A"]] = 0.1
    return before, after


def build(strategy, **kwargs):
    before, after = logits()
    return backend.build_logit_delta_candidate_batch(
        before, after, current_seq="AAAAAA", mutation_fraction=0.5,
        candidate_strategy=strategy, include_logit_candidates=True, **kwargs,
    )


@pytest.mark.parametrize("seed", range(10))
def test_two_guarded_slots_preserve_sampler_rng_other_slots_and_metadata(seed):
    torch.manual_seed(seed)
    old = build("positive_score")
    old_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    new = build("positive_score_guarded")
    assert torch.equal(old_rng, torch.get_rng_state())
    slots = [i for i, item in enumerate(old.candidate_metadata)
             if item["source"] == "positive_score_weighted"][-2:]
    assert new.diagnostics["guard_selected_slots"] == slots
    assert new.diagnostics["protected_positions"] == [0, 1]
    assert len(new.sequences) == len(old.sequences) == len(set(new.sequences))
    assert "AAAAAA" not in new.sequences
    for i, (original, sequence, record) in enumerate(zip(
        old.sequences, new.sequences, new.candidate_metadata,
    )):
        if i not in slots:
            assert sequence == original
            assert record == old.candidate_metadata[i]
        else:
            guard = record["dominance_guard"]
            if guard["fallback_reason"]:
                assert sequence == original
                assert guard["restored_positions"] == []
            else:
                assert sequence[:2] == "AA"
        assert record["sequence"] == sequence
        expected_positions = [j for j, aa in enumerate(sequence) if aa != "A"]
        assert record["mutation_positions"] == expected_positions
        assert record["mutation_count"] == len(expected_positions)


@pytest.mark.parametrize("count", [1, 2, 20])
def test_parent_fallback_and_short_pools(count):
    before, after = logits()
    before[:, backend.AA_TO_ESM_IDX["A"]] = 2
    after[:, backend.AA_TO_ESM_IDX["A"]] = 2
    batch = backend.build_logit_delta_candidate_batch(
        before, after, current_seq="AAAAAA", mutation_fraction=0.5,
        candidate_strategy="positive_score_guarded", protected_mutation_candidates=count,
    )
    assert batch.diagnostics["guard_changed_candidate_count"] == 0
    assert batch.diagnostics["guard_fallback_count"] == len(batch.diagnostics["guard_selected_slots"])
    assert all(m["dominance_guard"]["fallback_reason"] == "parent_sequence"
               for m in batch.candidate_metadata if "dominance_guard" in m)


def test_no_weighted_slot_is_noop():
    torch.manual_seed(1)
    old = build("positive_score", B_candidates=1)
    torch.manual_seed(1)
    new = build("positive_score_guarded", B_candidates=1)
    assert new.sequences == old.sequences
    assert new.candidate_metadata == old.candidate_metadata
    assert new.diagnostics["guard_selected_slots"] == []


def test_duplicate_restoration_retains_original_slot():
    before, after = logits()
    sequences = ["AAGAAA", "GAGAAA"]
    scores = backend.logit_delta_mutation_scores(before, after, current_seq="AAAAAA")
    batch = backend.ESMFold2DiscreteCandidateBatch(
        sequences=sequences,
        candidate_metadata=[backend._candidate_metadata_from_sequence(
            current_seq="AAAAAA", sequence=seq, scores=scores, source=source,
        ) for seq, source in zip(sequences, ["positive_score_topk", "positive_score_weighted"])],
    )
    result = backend._guard_weighted_candidates(
        batch, logits_before=before, logits_after=after, current_seq="AAAAAA",
        mutable_mask=None, dominance_gap=0.5, protected_mutation_candidates=2,
    )
    assert result.sequences == sequences
    assert result.candidate_metadata[1]["dominance_guard"]["fallback_reason"] == "duplicate_candidate"
    assert result.diagnostics["guard_fallback_count"] == 1


def test_guard_does_not_modify_input_tensor_mask_or_logits():
    before, after = logits()
    mask = torch.ones(6, dtype=torch.bool)
    saved_before, saved_after = before.clone(), after.clone()
    backend.dominant_current_residue_mask(before, after, current_seq="AAAAAA", mutable_mask=mask)
    assert mask.all()
    assert torch.equal(before, saved_before) and torch.equal(after, saved_after)


def test_empty_pool_when_no_positive_mutations():
    before = torch.zeros(3, 20)
    batch = backend.build_logit_delta_candidate_batch(
        before, before, current_seq="AAA", candidate_strategy="positive_score_guarded",
        mutation_fraction=0.5,
    )
    assert batch.sequences == []
    assert batch.diagnostics["guard_selected_slots"] == []


def test_gap_boundary_masks_cys_and_nonfinite_values():
    before = torch.zeros(7, 20)
    before[:, backend.AA_TO_ESM_IDX["A"]] = 0.5
    after = before.clone()
    after[1, backend.AA_TO_ESM_IDX["G"]] = 0.01
    before[2, backend.AA_TO_ESM_IDX["G"]] = 1.0
    after[3, backend.AA_TO_ESM_IDX["G"]] = float("nan")
    after[4, backend.CYS_INDEX] = 100.0
    after[5, backend.AA_TO_ESM_IDX["G"]] = float("inf")
    result = backend.dominant_current_residue_mask(
        before.unsqueeze(0), after.unsqueeze(0), current_seq="AAAAAAA",
        mutable_mask=[True] * 6 + [False],
    )
    assert result.tolist() == [True, False, False, False, True, False, False]


@pytest.mark.parametrize("gap,count", [(0, 2), (-1, 2), (float("nan"), 2),
                                      (float("inf"), 2), (0.5, 0), (0.5, 1.5), (0.5, True)])
def test_invalid_guard_config(gap, count):
    with pytest.raises(ValueError):
        build("positive_score_guarded", dominance_gap=gap, protected_mutation_candidates=count)


def test_cli_defaults_and_profile_override(tmp_path):
    args, parser = parse_args_with_profile(["--target", "AAAA"])
    assert args.esmfold2_stagea_candidate_strategy == "positive_score_guarded"
    from proteocraft.hallucination.profile_loader import write_case_profile
    profile = write_case_profile("esmfold2_d", {"esmfold2_stagea_dominance_gap": 0.7}, tmp_path / "profile.yaml")
    args, parser = parse_args_with_profile(["--target", "AAAA", "--profile", str(profile)])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_stagea_dominance_gap == 0.7
    assert args.esmfold2_stagea_protected_mutation_candidates == 2


@pytest.mark.parametrize("strategy", ["best_of_n", "raw_lineage_stage_a", "continuous_stage_a"])
def test_unsupported_modes_fail_loudly(strategy):
    with pytest.raises(SystemExit):
        parse_args_with_profile(["--target", "AAAA", "--mcts_strategy", strategy])


def test_old_backend_cannot_silently_drop_guard_parameters():
    class OldBackend:
        def propose(self, current_seq):
            raise AssertionError("must reject before calling backend")
    with pytest.raises(ValueError, match="requires backend.propose"):
        _call_backend_propose(OldBackend(), proposal_logits=None, current_seq="AAAAAA",
                              candidate_strategy="positive_score_guarded")


def test_backend_wrapper_forwards_guard_only_when_requested():
    class RecordingBackend:
        def propose(self, **kwargs):
            self.kwargs = kwargs
            return []
    instance = RecordingBackend()
    _call_backend_propose(instance, proposal_logits=None, candidate_strategy="positive_score")
    assert "dominance_gap" not in instance.kwargs
    _call_backend_propose(instance, proposal_logits=None, candidate_strategy="positive_score_guarded",
                          dominance_gap=0.8, protected_mutation_candidates=3)
    assert instance.kwargs["dominance_gap"] == 0.8
    assert instance.kwargs["protected_mutation_candidates"] == 3


@pytest.mark.parametrize("kind", ["cdr", "vhh"])
def test_task_launchers_forward_guard_config(kind, tmp_path):
    config = {"launch_profile": "esmfold2_d",
              "esmfold2_memory_saved_tensor_pin_memory": False,
              "esmfold2_stagea_candidate_strategy": "positive_score_guarded",
              "esmfold2_stagea_dominance_gap": 0.8,
              "esmfold2_stagea_protected_mutation_candidates": 3}
    if kind == "vhh":
        from tests.test_esmfold2_vhh import _vhh_case
        from proteocraft.hallucination.esmfold2_vhh import prepare_esmfold2_vhh_case
        from benchmark.script.run_esmfold2_vhh_redesign import build_hallucination_command
        case = prepare_esmfold2_vhh_case(_vhh_case(), input_dir=tmp_path)
    else:
        import yaml
        from proteocraft.hallucination.esmfold2_cdr import prepare_esmfold2_cdr_case
        from benchmark.script.run_esmfold2_cdr_redesign import build_hallucination_command
        root = Path(__file__).resolve().parents[1]
        original = yaml.safe_load((root / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml").read_text())
        case = prepare_esmfold2_cdr_case(original["cases"][0], input_dir=root / original["input_dir"])
    command = build_hallucination_command(case, config, case_dir=tmp_path / "out")
    args, parser = parse_args_with_profile(command[2:])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.esmfold2_stagea_candidate_strategy == "positive_score_guarded"
    assert args.esmfold2_stagea_dominance_gap == 0.8
    assert args.esmfold2_stagea_protected_mutation_candidates == 3
    assert args.esmfold2_memory_saved_tensor_pin_memory is False


@pytest.mark.parametrize("strategy", ["two_stage"])
def test_guarded_search_records_config_and_keeps_stageb_unmodified(strategy, tmp_path):
    from tests.test_esmfold2_pipeline import FakeESMFold2Backend
    from proteocraft.hallucination.esmfold2_pipeline import run_esmfold2_mcts_hallucination

    class GuardedBackend(FakeESMFold2Backend):
        def __init__(self):
            super().__init__()
            self.guard_calls = []

        def propose(self, **kwargs):
            self.guard_calls.append(dict(kwargs))
            kwargs.pop("dominance_gap", None)
            kwargs.pop("protected_mutation_candidates", None)
            # The real candidate builder is tested above; this checks orchestration.
            return super().propose(**kwargs)

    instance = GuardedBackend()
    run_esmfold2_mcts_hallucination(
        backend=instance, target_seq="TTT", L_binder=3, init_binder_seq="AAA",
        N_steps=3, mcts_strategy=strategy, mcts_bootstrap_steps=1,
        mcts_bootstrap_max_steps=1, esmfold2_stagea_candidate_strategy="positive_score_guarded",
        esmfold2_stagea_dominance_gap=0.8, artifact_dir=tmp_path, seed=43,
    )
    guarded = [call for call in instance.guard_calls if call.get("candidate_strategy") == "positive_score_guarded"]
    assert guarded and all(call["dominance_gap"] == 0.8 for call in guarded)
    assert all(call["protected_mutation_candidates"] == 2 for call in guarded)
    for call in instance.guard_calls:
        if call not in guarded:
            assert "dominance_gap" not in call
    checkpoint = torch.load(tmp_path / "handoff_checkpoints/stagea_handoff_final.pt", weights_only=False)
    assert checkpoint["stagea_config"]["stagea_dominance_gap"] == 0.8
