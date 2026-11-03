"""ESMFold2 sequence search, two-stage MCTS and candidate artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from proteocraft.hallucination.esmfold2_backend import (
    ESM_AA_ORDER,
    ESMFold2ProposalBatch,
    ESMFold2VerifierResult,
    anchor_binder_logits_to_sequence,
    init_binder_logits,
    logit_delta_mutation_scores,
    minimally_align_binder_logits_to_sequence,
    select_esmfold2_action_group,
    sequence_from_logits_argmax,
)
from proteocraft.hallucination.stage_a_tree import (
    LEGACY_SELECTION_MODES,
    StageATree,
)
from proteocraft.hallucination.stage_b_tree import StageBTree


logger = logging.getLogger(__name__)
VALID_ESM_BINDER_AAS = ESM_AA_ORDER.replace("C", "")
ESMFOLD2_LOGIT_INIT_MODES = {"legacy_anchor", "official_free", "official_exact"}
ESMFOLD2_STAGEA_HANDOFF_SCHEMA_VERSION = 1
ESMFOLD2_STAGEB_REWARD_KEYS = (
    "pair_chain_iptm_loss",
    "binder_confidence_loss",
    "global_confidence_loss",
    "binder_rg_excess_loss",
)
ESMFOLD2_STAGEB_HOTSPOT_DISTANCE_WEIGHT = 0.25
ESMFOLD2_STAGEB_HOTSPOT_COVERAGE_WEIGHT = 0.25
ESMFOLD2_STAGEB_HOTSPOT_MISS_PENALTY = 0.0
ESMFOLD2_STAGEB_PAIR_CHAIN_FLOOR_PENALTY = 2.0
ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY = "stageb_pair_chain_quality_floor"


@dataclass(frozen=True)
class ESMFold2MutationProposal:
    sequence: str
    edits: List[Tuple[int, str, str]]
    prior: float
    benefit: float = 0.0
    action_group: str = "unrouted"
    group_prior: float = 1.0
    gradient_prior: float = 1.0
    logit_delta_benefit: float = 0.0
    rerank_benefit: Optional[float] = None
    rerank_loss: Optional[float] = None
    rerank_prior: Optional[float] = None
    proposal_logits: Optional[torch.Tensor] = None


class _NoopArtifactRecorder:
    def write_step(self, **event: Any) -> None:
        return None

    def write_candidate(self, **event: Any) -> None:
        return None

    def write_mcts_node(self, **event: Any) -> None:
        return None

    def write_mcts_edge(self, **event: Any) -> None:
        return None

    def write_mcts_action(self, **event: Any) -> None:
        return None

    def write_soft_step(self, **event: Any) -> None:
        return None

    def write_mcts_handoff(self, **event: Any) -> None:
        return None

    def write_checkpoint(self, **event: Any) -> None:
        return None


    def write_proposal_trace(self, **event: Any) -> None:
        return None

    def write_summary(self, **summary: Any) -> None:
        return None




class _ESMArtifactRecorder(_NoopArtifactRecorder):
    """Append-only JSONL artifact writer for ESMFold2 MCTS diagnostics."""

    def __init__(
        self,
        artifact_dir: Union[str, os.PathLike[str]],
        *,
        append: bool = False,
        save_verified_candidate_cif: bool = False,
        save_visualization_trace: bool = False,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.save_verified_candidate_cif = bool(save_verified_candidate_cif)
        self.save_visualization_trace = bool(save_visualization_trace)
        self.trajectory_path = self.artifact_dir / "trajectory.jsonl"
        self.candidates_path = self.artifact_dir / "candidates.jsonl"
        self.nodes_path = self.artifact_dir / "mcts_nodes.jsonl"
        self.edges_path = self.artifact_dir / "mcts_edges.jsonl"
        self.actions_path = self.artifact_dir / "mcts_actions.jsonl"
        self.soft_steps_path = self.artifact_dir / "soft_steps.jsonl"
        self.handoff_path = self.artifact_dir / "mcts_handoff.jsonl"
        self.summary_path = self.artifact_dir / "run_summary.json"
        self.visualization_trace_dir = self.artifact_dir / "visualization_trace"
        self.visualization_trace_index_path = (
            self.visualization_trace_dir / "proposal_trace.jsonl"
        )
        jsonl_paths = (
            self.trajectory_path,
            self.candidates_path,
            self.nodes_path,
            self.edges_path,
            self.actions_path,
            self.soft_steps_path,
            self.handoff_path,
        )
        for path in jsonl_paths:
            if append:
                path.touch(exist_ok=True)
            else:
                path.write_text("", encoding="utf-8")
        self._visualization_trace_counter = 0
        if self.save_visualization_trace:
            self.visualization_trace_dir.mkdir(parents=True, exist_ok=True)
            if append:
                self.visualization_trace_index_path.touch(exist_ok=True)
                self._visualization_trace_counter = sum(
                    1
                    for line in self.visualization_trace_index_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
            else:
                self.visualization_trace_index_path.write_text("", encoding="utf-8")

    def write_step(self, **event: Any) -> None:
        self._append_jsonl(self.trajectory_path, event)

    def write_candidate(self, **event: Any) -> None:
        payload = dict(event)
        mmcif = payload.pop("_mmcif", None)
        if self.save_verified_candidate_cif:
            structure_status = "missing_mmcif"
            if isinstance(mmcif, str) and mmcif.strip():
                sequence = str(payload.get("sequence", ""))
                sequence_hash = hashlib.sha256(
                    sequence.encode("utf-8")
                ).hexdigest()[:16]
                structure_hash = hashlib.sha256(
                    mmcif.encode("utf-8")
                ).hexdigest()
                structure_dir = (
                    self.artifact_dir / "verified_candidate_cif"
                )
                structure_dir.mkdir(parents=True, exist_ok=True)
                structure_name = (
                    f"{sequence_hash}_{structure_hash[:16]}.cif"
                )
                structure_path = structure_dir / structure_name
                if not structure_path.exists():
                    temporary_path = structure_path.with_suffix(".cif.tmp")
                    temporary_path.write_text(mmcif, encoding="utf-8")
                    temporary_path.replace(structure_path)
                payload["structure_cif"] = str(
                    structure_path.relative_to(self.artifact_dir)
                )
                payload["structure_sha256"] = structure_hash
                structure_status = "ok"
            payload["structure_status"] = structure_status
        self._append_jsonl(self.candidates_path, payload)

    def write_mcts_node(self, **event: Any) -> None:
        self._append_jsonl(self.nodes_path, event)

    def write_mcts_edge(self, **event: Any) -> None:
        self._append_jsonl(self.edges_path, event)

    def write_mcts_action(self, **event: Any) -> None:
        self._append_jsonl(self.actions_path, event)

    def write_soft_step(self, **event: Any) -> None:
        self._append_jsonl(self.soft_steps_path, event)

    def write_mcts_handoff(self, **event: Any) -> None:
        self._append_jsonl(self.handoff_path, event)

    def write_checkpoint(self, **event: Any) -> None:
        stage = _safe_path_token(str(event.get("stage", "unknown")))
        step = event.get("step")
        label = _safe_path_token(str(event.get("label", "round_best")))
        step_index = 0 if step is None else int(step)
        step_name = f"step_{step_index:04d}"
        step_dir = self.artifact_dir / "checkpoints" / stage / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        result = event.get("result")
        if result is None:
            return
        sequence = str(getattr(result, "binder_seq", ""))
        summary = dict(getattr(result, "summary_confidence", {}) or {})
        mmcif = summary.get("mmcif")
        structure_file = None
        structure_status = "missing_mmcif"
        if isinstance(mmcif, str) and mmcif.strip():
            structure_file = f"{step_name}_{label}.cif"
            (step_dir / structure_file).write_text(mmcif, encoding="utf-8")
            structure_status = "ok"

        metadata = {
            "stage": event.get("stage"),
            "step": step,
            "label": label,
            "binder_seq": sequence,
            "binder_seq_hash": _sequence_hash(sequence),
            "target_seq": getattr(result, "target_seq", None),
            "total_loss": float(getattr(result, "total_loss")),
            "loss_dict": _loss_dict_for_artifact(result),
            "summary_confidence": _summary_without_structure(summary),
            "structure_file": structure_file,
            "structure_format": "mmcif" if structure_file else None,
            "structure_status": structure_status,
            "selected_sequence": event.get("selected_sequence"),
            "parent_sequence": event.get("parent_sequence"),
            "is_global_best": bool(event.get("is_global_best", False)),
        }
        meta_file = f"{step_name}_{label}_meta.json"
        (step_dir / meta_file).write_text(
            json.dumps(_json_ready(metadata), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )


    def write_proposal_trace(self, **event: Any) -> None:
        if not self.save_visualization_trace:
            return
        logits_before = event.pop("logits_before", None)
        logits_after = event.pop("logits_after", None)
        mutation_scores = event.pop("mutation_scores", None)
        if logits_before is None or logits_after is None or mutation_scores is None:
            return

        trace_index = self._visualization_trace_counter
        self._visualization_trace_counter += 1
        stage = _safe_path_token(str(event.get("stage", "unknown")))
        step = event.get("step")
        step_token = "none" if step is None else f"{int(step):04d}"
        action_group = _safe_path_token(str(event.get("action_group") or "unrouted"))
        trace_name = (
            f"{trace_index:05d}_{stage}_step_{step_token}_{action_group}.pt"
        )
        trace_path = self.visualization_trace_dir / trace_name
        tensor_payload = {
            **dict(event),
            "trace_index": trace_index,
            "aa_order": ESM_AA_ORDER,
            "logits_before": torch.as_tensor(logits_before).detach().cpu().clone(),
            "logits_after": torch.as_tensor(logits_after).detach().cpu().clone(),
            "mutation_scores": torch.as_tensor(mutation_scores).detach().cpu().clone(),
        }
        temporary_path = trace_path.with_suffix(trace_path.suffix + ".tmp")
        torch.save(tensor_payload, temporary_path)
        temporary_path.replace(trace_path)
        self._append_jsonl(
            self.visualization_trace_index_path,
            {
                **dict(event),
                "trace_index": trace_index,
                "aa_order": ESM_AA_ORDER,
                "trace_file": str(trace_path.relative_to(self.artifact_dir)),
                "logits_shape": list(torch.as_tensor(logits_before).shape),
                "mutation_scores_shape": list(torch.as_tensor(mutation_scores).shape),
            },
        )

    def write_summary(self, **summary: Any) -> None:
        self._write_json_atomic(self.summary_path, summary)



    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(_json_ready(dict(payload)), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)


def _make_artifact_recorder(
    artifact_dir: Optional[Union[str, os.PathLike[str]]],
    *,
    append: bool = False,
    save_verified_candidate_cif: bool = False,
    save_visualization_trace: bool = False,
) -> _NoopArtifactRecorder:
    if artifact_dir is None:
        return _NoopArtifactRecorder()
    return _ESMArtifactRecorder(
        artifact_dir,
        append=append,
        save_verified_candidate_cif=save_verified_candidate_cif,
        save_visualization_trace=save_visualization_trace,
    )


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_ready(float(value.detach().cpu().item()))
        return {
            "tensor_shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    scalar = _finite_float(value)
    if scalar is not None:
        return scalar
    return str(value)


def _safe_path_token(text: str) -> str:
    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def _sequence_hash(sequence: Optional[str]) -> Optional[str]:
    if sequence is None:
        return None
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()[:16]


def _summary_without_structure(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(summary or {}).items()
        if str(key) not in {"mmcif"}
    }


def _loss_dict_for_artifact(result: Any) -> Dict[str, float]:
    return {
        str(key): float(scalar)
        for key, value in dict(getattr(result, "loss_dict", {}) or {}).items()
        if (scalar := _finite_float(value)) is not None
    }


def _summary_for_artifact(result: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in dict(getattr(result, "summary_confidence", {}) or {}).items():
        if str(key) == "mmcif":
            continue
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            continue
        summary[str(key)] = _json_ready(value)
    return summary


def _write_candidate_artifact(
    recorder: _NoopArtifactRecorder,
    *,
    stage: str,
    step: Optional[int],
    parent_sequence: Optional[str],
    result: Any,
    action_group: Optional[str] = None,
    proposal_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    sequence = str(result.binder_seq)
    edits = _sequence_edits(parent_sequence, sequence) if parent_sequence else []
    recorder.write_candidate(
        stage=stage,
        step=step,
        parent_sequence=parent_sequence,
        sequence=sequence,
        edits=edits,
        mutation_count=len(edits),
        action_group=action_group,
        proposal_metadata=dict(proposal_metadata or {}),
        total_loss=float(result.total_loss),
        loss_dict=_loss_dict_for_artifact(result),
        summary_confidence=_summary_for_artifact(result),
        _mmcif=dict(
            getattr(result, "summary_confidence", {}) or {}
        ).get("mmcif"),
    )


def _write_soft_step_artifacts(
    recorder: _NoopArtifactRecorder,
    *,
    stage: str,
    step: Optional[int],
    selected_sequence: str,
    batch: ESMFold2ProposalBatch,
    action_group: Optional[str] = None,
) -> None:
    for metric in list(getattr(batch, "soft_step_metrics", []) or []):
        metric = dict(metric)
        metric.pop("logits_snapshot", None)
        recorder.write_soft_step(
            stage=stage,
            step=step,
            selected_sequence=str(selected_sequence),
            action_group=action_group,
            **metric,
        )


def _write_proposal_visualization_trace(
    recorder: _NoopArtifactRecorder,
    *,
    stage: str,
    step: Optional[int],
    selected_sequence: str,
    batch: ESMFold2ProposalBatch,
    mutable_mask: Optional[Sequence[bool]],
    action_group: Optional[str] = None,
    trace_label: Optional[str] = None,
) -> None:
    if not bool(getattr(recorder, "save_visualization_trace", False)):
        return
    logits_before = getattr(batch, "logits_before", None)
    logits_after = getattr(batch, "logits_after", None)
    if logits_before is None or logits_after is None:
        return
    mutation_scores = logit_delta_mutation_scores(
        logits_before,
        logits_after,
        current_seq=str(selected_sequence),
        mutable_mask=mutable_mask,
    )
    recorder.write_proposal_trace(
        stage=stage,
        step=step,
        selected_sequence=str(selected_sequence),
        selected_sequence_hash=_sequence_hash(selected_sequence),
        action_group=action_group,
        trace_label=trace_label,
        mutable_mask=(
            [bool(value) for value in mutable_mask]
            if mutable_mask is not None
            else [True] * len(selected_sequence)
        ),
        candidate_sequences=[str(sequence) for sequence in batch.sequences],
        candidate_metadata=[dict(item) for item in batch.candidate_metadata],
        proposal_diagnostics=dict(batch.proposal_diagnostics),
        logits_before=logits_before,
        logits_after=logits_after,
        mutation_scores=mutation_scores,
    )


def _mean_legacy_loss(node: Any) -> Optional[float]:
    if not hasattr(node, "W") or not hasattr(node, "visit_count"):
        return None
    visits = max(int(getattr(node, "visit_count", 0)), 1)
    return float(getattr(node, "W")) / visits


def _mcts_node_artifact(
    *,
    stage: str,
    node: Any,
    step_created: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sequence = str(getattr(node, "sequence", ""))
    parent = getattr(node, "parent", None)
    payload: Dict[str, Any] = {
        "stage": stage,
        "step_created": step_created,
        "node_id": int(getattr(node, "node_id")),
        "node_uid": f"{stage}:{int(getattr(node, 'node_id'))}",
        "parent_node_id": None if parent is None else int(getattr(parent, "node_id")),
        "parent_node_uid": None if parent is None else f"{stage}:{int(getattr(parent, 'node_id'))}",
        "depth": int(getattr(node, "depth", 0)),
        "sequence": sequence,
        "sequence_hash": _sequence_hash(sequence),
        "parent_sequence_hash": _sequence_hash(getattr(parent, "sequence", None)) if parent is not None else None,
        "edits": list(getattr(node, "edit_from_parent", []) or []),
        "total_loss": float(getattr(node, "L_real")),
        "visit_count": int(getattr(node, "visit_count", 0)),
        "soft_step": int(getattr(node, "soft_step", 0)),
    }
    if parent is not None and not payload["edits"]:
        payload["edits"] = _sequence_edits(str(getattr(parent, "sequence", "")), sequence)
    if hasattr(node, "W"):
        payload["W"] = float(getattr(node, "W"))
        payload["mean_loss"] = _mean_legacy_loss(node)
    if hasattr(node, "tree_reward"):
        payload["tree_reward"] = float(getattr(node, "tree_reward"))
        payload["value_sum"] = float(getattr(node, "value_sum", 0.0))
        payload["q_value"] = float(getattr(node, "q_value", 0.0))
        payload["prior"] = float(getattr(node, "prior", 1.0))
        payload["group_id"] = str(getattr(node, "group_id", "ungrouped"))
        payload["loss_vector"] = dict(getattr(node, "loss_vector", {}) or {})
        payload["status_flags"] = dict(getattr(node, "status_flags", {}) or {})
        payload["structural_signature"] = dict(getattr(node, "structural_signature", {}) or {})
    if extra:
        payload.update(extra)
    return payload


def _mcts_edge_artifact(
    *,
    stage: str,
    parent: Any,
    child: Any,
    step_created: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parent_seq = str(getattr(parent, "sequence", ""))
    child_seq = str(getattr(child, "sequence", ""))
    payload: Dict[str, Any] = {
        "stage": stage,
        "step_created": step_created,
        "parent_node_id": int(getattr(parent, "node_id")),
        "child_node_id": int(getattr(child, "node_id")),
        "parent_node_uid": f"{stage}:{int(getattr(parent, 'node_id'))}",
        "child_node_uid": f"{stage}:{int(getattr(child, 'node_id'))}",
        "parent_sequence_hash": _sequence_hash(parent_seq),
        "child_sequence_hash": _sequence_hash(child_seq),
        "edits": _sequence_edits(parent_seq, child_seq),
        "num_mutations": sum(1 for a, b in zip(parent_seq, child_seq) if a != b),
        "child_total_loss": float(getattr(child, "L_real")),
    }
    if extra:
        payload.update(extra)
    return payload


def _mcts_action_artifact(
    *,
    stage: str,
    step: Optional[int],
    action: Any,
    selected: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sequence = str(getattr(action, "sequence", ""))
    payload: Dict[str, Any] = {
        "stage": stage,
        "step": step,
        "action_id": int(getattr(action, "action_id")),
        "parent_node_id": int(getattr(action, "parent_node_id")),
        "parent_node_uid": f"{stage}:{int(getattr(action, 'parent_node_id'))}",
        "child_node_id": getattr(action, "child_node_id", None),
        "child_node_uid": (
            None
            if getattr(action, "child_node_id", None) is None
            else f"{stage}:{int(getattr(action, 'child_node_id'))}"
        ),
        "sequence": sequence,
        "sequence_hash": _sequence_hash(sequence),
        "edits": list(getattr(action, "edits", []) or []),
        "prior": float(getattr(action, "prior", 0.0)),
        "benefit": float(getattr(action, "benefit", 0.0)),
        "action_group": str(getattr(action, "action_group", "unrouted")),
        "group_prior": float(getattr(action, "group_prior", 1.0)),
        "gradient_prior": float(
            getattr(getattr(action, "proposal", None), "gradient_prior", 1.0)
        ),
        "logit_delta_benefit": float(
            getattr(getattr(action, "proposal", None), "logit_delta_benefit", 0.0)
        ),
        "rerank_benefit": getattr(
            getattr(action, "proposal", None), "rerank_benefit", None
        ),
        "rerank_loss": getattr(
            getattr(action, "proposal", None), "rerank_loss", None
        ),
        "rerank_prior": getattr(
            getattr(action, "proposal", None), "rerank_prior", None
        ),
        "visit_count": int(getattr(action, "visit_count", 0)),
        "value_sum": float(getattr(action, "value_sum", 0.0)),
        "q_value": float(getattr(action, "q_value", 0.0)),
        "status": str(getattr(action, "status", "unknown")),
        "selected": bool(selected),
        "status_flags": dict(getattr(action, "status_flags", {}) or {}),
    }
    if extra:
        payload.update(extra)
    return payload


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _normalize_target_key(target_seq: Union[str, Sequence[str]]) -> str:
    if isinstance(target_seq, str):
        return target_seq
    return "|".join(str(seq) for seq in target_seq)




def _initial_proposal_logits(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]],
) -> torch.Tensor:
    logits = init_binder_logits(
        sequence,
        mutable_mask=mutable_mask,
        batch_size=1,
        random_scale=0.0,
        requires_grad=False,
    )[0]
    with torch.no_grad():
        for pos, aa in enumerate(str(sequence)):
            aa_index = ESM_AA_ORDER.index(aa)
            if torch.isfinite(logits[pos, aa_index]):
                logits[pos, aa_index] = max(float(logits[pos, aa_index].item()), 2.0)
    return anchor_binder_logits_to_sequence(
        logits,
        str(sequence),
        mutable_mask=mutable_mask,
    )


def _anchor_proposal_logits(
    logits: Optional[torch.Tensor],
    sequence: str,
    mutable_mask: Optional[Sequence[bool]],
) -> torch.Tensor:
    if logits is None:
        return _initial_proposal_logits(sequence, mutable_mask)
    return anchor_binder_logits_to_sequence(
        logits,
        str(sequence),
        mutable_mask=mutable_mask,
    )


def _minimally_align_proposal_logits(
    logits: Optional[torch.Tensor],
    sequence: str,
    mutable_mask: Optional[Sequence[bool]],
) -> torch.Tensor:
    if logits is None:
        return _initial_proposal_logits(sequence, mutable_mask)
    return minimally_align_binder_logits_to_sequence(
        logits,
        str(sequence),
        mutable_mask=mutable_mask,
    )


def _select_stagea_tree_results(
    round_results: Sequence[Any],
    *,
    argmax_sequence: Optional[str],
    max_children: int,
) -> List[Any]:
    """Select a narrow Stage-A tree frontier from fully verified candidates.

    The direct argmax sequence represents the proposal model's continuous
    soft-logit trajectory, so it receives the first tree slot when available.
    Remaining slots use verifier loss. All candidates can still be recorded and
    considered for the run-level best result without widening the MCTS tree.
    """

    limit = max(int(max_children), 1)
    selected: List[Any] = []
    selected_sequences = set()

    if argmax_sequence:
        for result in round_results:
            sequence = str(result.binder_seq)
            if sequence == str(argmax_sequence):
                selected.append(result)
                selected_sequences.add(sequence)
                break

    for result in sorted(round_results, key=lambda item: float(item.total_loss)):
        sequence = str(result.binder_seq)
        if sequence in selected_sequences:
            continue
        selected.append(result)
        selected_sequences.add(sequence)
        if len(selected) >= limit:
            break

    return selected[:limit]


def _stagea_child_logits(
    *,
    logits_after: Optional[torch.Tensor],
    parent_sequence: str,
    child_sequence: str,
    argmax_sequence: Optional[str],
    mutable_mask: Optional[Sequence[bool]],
    preserve_soft_state: bool,
    soft_anchor_margin: float,
) -> torch.Tensor:
    if logits_after is None:
        return _initial_proposal_logits(child_sequence, mutable_mask)
    if argmax_sequence is not None and str(child_sequence) == str(argmax_sequence):
        return logits_after.detach().cpu().clone()
    if preserve_soft_state:
        return _minimally_align_proposal_logits(
            logits_after,
            str(child_sequence),
            mutable_mask,
        )
    return _anchor_proposal_logits(logits_after, child_sequence, mutable_mask)


def _normalize_proposal_batch(raw: Any) -> ESMFold2ProposalBatch:
    if isinstance(raw, ESMFold2ProposalBatch):
        return raw
    return ESMFold2ProposalBatch(sequences=[str(sequence) for sequence in raw])


def _proposal_confidence_records(batch: ESMFold2ProposalBatch) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in list(getattr(batch, "confidence_candidates", []) or []):
        if not isinstance(item, dict):
            continue
        sequence = str(item.get("sequence", "")).strip().upper()
        if not sequence:
            continue
        record = dict(item)
        record["sequence"] = sequence
        records.append(record)
    records.sort(key=_proposal_confidence_score, reverse=True)
    return records


def _proposal_confidence_score(record: Dict[str, Any]) -> float:
    for key in ("confidence_score", "pair_chain_iptm", "iptm"):
        scalar = _finite_float(record.get(key))
        if scalar is not None:
            return float(scalar)
    return float("-inf")


def _proposal_sequences_from_batch(
    batch: ESMFold2ProposalBatch,
    *,
    include_confidence: bool = True,
) -> List[str]:
    confidence_sequences = []
    if include_confidence:
        confidence_sequences = [
            str(record["sequence"])
            for record in _proposal_confidence_records(batch)
        ]
    ordinary_sequences = [str(sequence) for sequence in list(batch.sequences or [])]
    return list(dict.fromkeys([*confidence_sequences, *ordinary_sequences]))


def _proposal_confidence_summary(batch: ESMFold2ProposalBatch) -> Dict[str, Any]:
    records = _proposal_confidence_records(batch)
    if not records:
        return {
            "proposal_confidence_candidate_count": 0,
            "proposal_confidence_best": None,
        }
    return {
        "proposal_confidence_candidate_count": len(records),
        "proposal_confidence_best": records[0],
        "proposal_confidence_candidates": records,
    }




def _call_backend_propose(
    backend: Any,
    *,
    proposal_logits: Optional[torch.Tensor],
    reference_coordinates: Optional[torch.Tensor] = None,
    reference_features: Optional[Mapping[str, torch.Tensor]] = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    hotspot_weight_override: Optional[float] = None,
    track_confidence_override: Optional[bool] = None,
    soft_steps_override: Optional[int] = None,
    temperature_step_offset: int = 0,
    temperature_total_steps: Optional[int] = None,
    preserve_proposal_logits: bool = False,
    mutation_fraction: Optional[float] = None,
    candidate_strategy: Optional[str] = None,
    dominance_gap: float = 0.5,
    protected_mutation_candidates: int = 2,
    include_argmax_candidate: Optional[bool] = None,
    **kwargs: Any,
) -> ESMFold2ProposalBatch:
    propose_kwargs = dict(kwargs)
    if candidate_strategy == "positive_score_guarded":
        for name, value in (
            ("candidate_strategy", candidate_strategy),
            ("dominance_gap", dominance_gap),
            ("protected_mutation_candidates", protected_mutation_candidates),
        ):
            if not _callable_accepts_argument(backend.propose, name):
                raise ValueError(f"guarded candidate mode requires backend.propose argument {name}")
            propose_kwargs[name] = value
    if (
        mutation_fraction is not None
        and _callable_accepts_argument(backend.propose, "mutation_fraction")
    ):
        propose_kwargs["mutation_fraction"] = mutation_fraction
    if (
        candidate_strategy is not None
        and _callable_accepts_argument(backend.propose, "candidate_strategy")
    ):
        propose_kwargs["candidate_strategy"] = str(candidate_strategy)
    if (
        include_argmax_candidate is not None
        and _callable_accepts_argument(
            backend.propose,
            "include_argmax_candidate",
        )
    ):
        propose_kwargs["include_argmax_candidate"] = bool(
            include_argmax_candidate
        )
    if _callable_accepts_argument(backend.propose, "proposal_logits"):
        propose_kwargs["proposal_logits"] = (
            proposal_logits.detach().clone() if proposal_logits is not None else None
        )
    one_step_enabled = bool(
        getattr(
            getattr(backend, "config", None),
            "one_step_diffusion_enabled",
            False,
        )
    )
    if one_step_enabled and _callable_accepts_argument(
        backend.propose,
        "reference_coordinates",
    ):
        propose_kwargs["reference_coordinates"] = (
            reference_coordinates.detach().clone()
            if reference_coordinates is not None
            else None
        )
    if (
        one_step_enabled
        and reference_features is not None
        and _callable_accepts_argument(
            backend.propose,
            "reference_features",
        )
    ):
        propose_kwargs["reference_features"] = {
            str(key): value.detach().clone()
            for key, value in reference_features.items()
        }
    if _callable_accepts_argument(backend.propose, "track_confidence_override"):
        propose_kwargs["track_confidence_override"] = track_confidence_override
    if _callable_accepts_argument(backend.propose, "soft_steps_override"):
        propose_kwargs["soft_steps_override"] = soft_steps_override
    if (
        (int(temperature_step_offset) != 0 or temperature_total_steps is not None)
        and _callable_accepts_argument(backend.propose, "temperature_step_offset")
    ):
        propose_kwargs["temperature_step_offset"] = int(temperature_step_offset)
    if (
        temperature_total_steps is not None
        and _callable_accepts_argument(backend.propose, "temperature_total_steps")
    ):
        propose_kwargs["temperature_total_steps"] = temperature_total_steps
    if _callable_accepts_argument(backend.propose, "preserve_proposal_logits"):
        propose_kwargs["preserve_proposal_logits"] = bool(preserve_proposal_logits)
    if _callable_accepts_argument(backend.propose, "hotspot_params"):
        propose_kwargs["hotspot_params"] = (
            dict(hotspot_params) if hotspot_params is not None else None
        )
    if (
        hotspot_params is not None
        and hotspot_weight_override is not None
        and _callable_accepts_argument(backend.propose, "hotspot_weight_override")
    ):
        propose_kwargs["hotspot_weight_override"] = hotspot_weight_override
    raw = backend.propose(**propose_kwargs)
    return _normalize_proposal_batch(raw)




def _callable_accepts_argument(func: Any, name: str) -> bool:
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


def _sequence_edits(parent_seq: str, child_seq: str) -> List[Tuple[int, str, str]]:
    return [
        (idx, old, new)
        for idx, (old, new) in enumerate(zip(parent_seq, child_seq))
        if old != new
    ]


def _allocate_group_budgets(group_priors: Dict[str, float], total_budget: int) -> Dict[str, int]:
    if total_budget <= 0:
        return {}
    valid = [
        (str(group), max(float(prior), 0.0))
        for group, prior in (group_priors or {}).items()
        if float(prior) > 0.0
    ]
    if not valid:
        return {}
    total_prior = sum(prior for _, prior in valid)
    if total_prior <= 0.0:
        return {}
    valid.sort(key=lambda item: item[1], reverse=True)
    if total_budget < len(valid):
        return {group: 1 for group, _ in valid[:total_budget]}

    budgets = {group: 1 for group, _ in valid}
    remaining = total_budget - len(valid)
    scaled = [(group, remaining * prior / total_prior) for group, prior in valid]
    for group, raw in scaled:
        budgets[group] += int(math.floor(raw))
    used = sum(budgets.values())
    remainders = sorted(
        ((raw - math.floor(raw), group) for group, raw in scaled),
        reverse=True,
    )
    for _, group in remainders:
        if used >= total_budget:
            break
        budgets[group] += 1
        used += 1
    return {group: count for group, count in budgets.items() if count > 0}


def _select_stageb_actions_by_group_quota(
    tree: StageBTree,
    node: Any,
    actions: Sequence[Any],
    *,
    group_priors: Dict[str, float],
    max_actions: int,
    exploration_actions: int = 0,
    rng: Optional[random.Random] = None,
    limit_to_tree_slots: bool = True,
) -> Tuple[List[Any], Dict[str, int]]:
    """Select group-balanced exploit actions plus a small exploration budget."""

    limit = max(int(max_actions), 0)
    if limit_to_tree_slots:
        limit = min(limit, int(tree.remaining_expansion_slots(node)))
    if limit <= 0:
        return [], {}

    available = [
        action
        for action in actions
        if action.status == "unexpanded" and action.child_node_id is None
        and not tree.has_sequence(action.sequence)
    ]
    if not available:
        return [], {}

    actions_by_group: Dict[str, List[Any]] = {}
    for action in available:
        actions_by_group.setdefault(str(action.action_group), []).append(action)

    available_priors = {
        group: max(float(group_priors.get(group, 0.0)), 0.0)
        for group in actions_by_group
    }
    if not any(prior > 0.0 for prior in available_priors.values()):
        available_priors = {group: 1.0 for group in actions_by_group}
    explore_limit = min(max(int(exploration_actions), 0), max(limit - 1, 0))
    exploit_limit = limit - explore_limit
    quotas = _allocate_group_budgets(available_priors, exploit_limit)

    def exploit_score(action: Any) -> Tuple[int, float]:
        rerank_benefit = getattr(getattr(action, "proposal", None), "rerank_benefit", None)
        if rerank_benefit is not None and math.isfinite(float(rerank_benefit)):
            return 1, float(rerank_benefit)
        return 0, float(tree.action_puct_score(node, action))

    selected: List[Any] = []
    selected_ids = set()
    for group, quota in quotas.items():
        ranked = sorted(
            actions_by_group.get(group, []),
            key=exploit_score,
            reverse=True,
        )
        for action in ranked[: int(quota)]:
            selected.append(action)
            selected_ids.add(int(action.action_id))

    if len(selected) < exploit_limit:
        remaining = sorted(
            [action for action in available if int(action.action_id) not in selected_ids],
            key=exploit_score,
            reverse=True,
        )
        for action in remaining[: exploit_limit - len(selected)]:
            selected.append(action)
            selected_ids.add(int(action.action_id))

    rng = rng or random.Random(0)
    while len(selected) < limit:
        remaining = [
            action for action in available if int(action.action_id) not in selected_ids
        ]
        if not remaining:
            break
        weights = []
        for action in remaining:
            proposal = getattr(action, "proposal", None)
            group_prior = max(float(getattr(action, "group_prior", 1.0)), 1.0e-12)
            gradient_prior = max(float(getattr(proposal, "gradient_prior", 1.0)), 1.0e-12)
            weights.append(group_prior * gradient_prior)
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(chosen)
        selected_ids.add(int(chosen.action_id))

    actual_quotas = _count_items([str(action.action_group) for action in selected])
    return selected, actual_quotas


class _ESMFold2StageBMCTSTree(StageBTree):
    """MCTS tree with a fixed, ESMFold2-native Stage-B reward."""

    def __init__(
        self,
        *args: Any,
        hotspot_distance_weight: float = ESMFOLD2_STAGEB_HOTSPOT_DISTANCE_WEIGHT,
        hotspot_coverage_weight: float = ESMFOLD2_STAGEB_HOTSPOT_COVERAGE_WEIGHT,
        hotspot_miss_penalty: float = ESMFOLD2_STAGEB_HOTSPOT_MISS_PENALTY,
        min_pair_chain_iptm: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hotspot_distance_weight = max(float(hotspot_distance_weight), 0.0)
        self.hotspot_coverage_weight = max(float(hotspot_coverage_weight), 0.0)
        self.hotspot_miss_penalty = max(float(hotspot_miss_penalty), 0.0)
        self.min_pair_chain_iptm = min(max(float(min_pair_chain_iptm), 0.0), 1.0)

    def _compute_base_node_reward(
        self,
        total_loss: float,
        loss_vector: Dict[str, float],
        base_reward: Optional[float] = None,
        observe: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        del total_loss, base_reward, observe
        losses = {str(key): float(value) for key, value in loss_vector.items()}
        required = (
            "pair_chain_iptm_loss",
            "binder_confidence_loss",
            "binder_rg_excess_loss",
        )
        missing = [key for key in required if key not in losses]
        if missing:
            raise ValueError(
                "ESMFold2 Stage-B reward requires full-verifier metrics: "
                + ", ".join(missing)
            )

        components = {
            key: min(max(float(losses[key]), 0.0), 1.0)
            for key in ESMFOLD2_STAGEB_REWARD_KEYS
            if key in losses
        }
        base_loss = sum(components.values()) / len(components)
        hotspot_distance_loss = min(
            max(float(losses.get("hotspot_verify_distance_loss", 0.0)), 0.0),
            1.0,
        )
        hotspot_coverage = losses.get("hotspot_heavy_atom_coverage")
        hotspot_coverage_loss = 0.0
        hotspot_miss_penalty = 0.0
        hotspot_hit = None
        if hotspot_coverage is not None:
            hotspot_coverage = min(max(float(hotspot_coverage), 0.0), 1.0)
            hotspot_coverage_loss = 1.0 - hotspot_coverage
            hotspot_hit = hotspot_coverage >= 1.0 - 1.0e-8
            if hotspot_coverage <= 0.0:
                hotspot_miss_penalty = self.hotspot_miss_penalty

        pair_chain_iptm = 1.0 - float(losses["pair_chain_iptm_loss"])
        pair_chain_floor = max(
            self.min_pair_chain_iptm,
            min(
                max(float(losses.get(ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY, 0.0)), 0.0),
                1.0,
            ),
        )
        pair_chain_floor_violated = pair_chain_iptm < pair_chain_floor
        pair_chain_floor_penalty = (
            ESMFOLD2_STAGEB_PAIR_CHAIN_FLOOR_PENALTY
            if pair_chain_floor_violated
            else 0.0
        )

        reward_loss = (
            base_loss
            + self.hotspot_distance_weight * hotspot_distance_loss
            + self.hotspot_coverage_weight * hotspot_coverage_loss
            + hotspot_miss_penalty
            + pair_chain_floor_penalty
        )
        return -float(reward_loss), {
            "valid": True,
            "hard_invalid": False,
            "hard_penalty": 0.0,
            "hard_violations": {},
            "reward_mode": "esmfold2_stageb_fixed",
            "normalized_loss": float(reward_loss),
            "normalized_components": components,
            "hotspot_hit": hotspot_hit,
            "hotspot_coverage": hotspot_coverage,
            "hotspot_coverage_loss": hotspot_coverage_loss,
            "hotspot_coverage_weight": self.hotspot_coverage_weight,
            "hotspot_distance_loss": hotspot_distance_loss,
            "hotspot_distance_weight": self.hotspot_distance_weight,
            "hotspot_miss_penalty": hotspot_miss_penalty,
            "pair_chain_iptm": pair_chain_iptm,
            "min_pair_chain_iptm": pair_chain_floor,
            "pair_chain_floor_violated": pair_chain_floor_violated,
            "pair_chain_floor_penalty": pair_chain_floor_penalty,
        }


def _normalize_prior_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "full").strip().lower()
    if normalized not in {
        "full",
        "uniform",
        "no_gradient",
        "no_group",
        "no_group_matched_compute",
        "no_sequence",
    }:
        return "full"
    return normalized


def _apply_stageb_group_prior_mode(
    prior_mode: str,
    action_group: str,
    group_priors: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    prior_mode = _normalize_prior_mode(prior_mode)
    if prior_mode in {"uniform", "no_group", "no_group_matched_compute"}:
        return "unrouted", {"unrouted": 1.0}
    return action_group, group_priors or {action_group: 1.0}


def _stageb_proposal_call_specs(
    group_budgets: Mapping[str, int],
    prior_mode: str,
) -> List[Tuple[str, int, str]]:
    """Return proposal calls while preserving the legacy modes exactly."""

    calls = [
        (str(group), int(budget), str(group))
        for group, budget in group_budgets.items()
        if int(budget) > 0
    ]
    if _normalize_prior_mode(prior_mode) != "no_group_matched_compute":
        return calls

    total_budget = sum(budget for _, budget, _ in calls)
    if total_budget <= 0:
        return []
    replica_count = min(3, total_budget)
    base_budget, remainder = divmod(total_budget, replica_count)
    return [
        (
            "unrouted",
            base_budget + (1 if replica_index < remainder else 0),
            f"unrouted_replica_{replica_index}",
        )
        for replica_index in range(replica_count)
    ]


def _deduplicate_actions_by_sequence(actions: Sequence[Any]) -> List[Any]:
    unique = []
    seen = set()
    for action in actions:
        sequence = str(getattr(action, "sequence", ""))
        if not sequence or sequence in seen:
            continue
        seen.add(sequence)
        unique.append(action)
    return unique


def _stageb_result_pair_chain_iptm(result: Any) -> float:
    loss_dict = dict(getattr(result, "loss_dict", {}) or {})
    value = _finite_float(loss_dict.get("pair_chain_iptm_loss"))
    if value is None:
        raise ValueError("Stage-B result is missing pair_chain_iptm_loss")
    return 1.0 - min(max(float(value), 0.0), 1.0)


def _stageb_result_quality_floor(result: Any) -> float:
    loss_dict = dict(getattr(result, "loss_dict", {}) or {})
    return min(
        max(float(loss_dict.get(ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY, 0.0)), 0.0),
        1.0,
    )


def _stageb_result_quality_ok(result: Any) -> bool:
    return _stageb_result_pair_chain_iptm(result) >= _stageb_result_quality_floor(result)


def _stageb_result_hotspot_coverage(result: Any) -> Optional[float]:
    summary = dict(getattr(result, "summary_confidence", {}) or {})
    value = _finite_float(summary.get("hotspot_heavy_atom_coverage"))
    if value is None:
        value = _finite_float(
            dict(getattr(result, "loss_dict", {}) or {}).get(
                "hotspot_heavy_atom_coverage"
            )
        )
    if value is None:
        return None
    return min(max(float(value), 0.0), 1.0)


def _stageb_result_hotspot_distance(result: Any) -> Optional[float]:
    summary = dict(getattr(result, "summary_confidence", {}) or {})
    value = _finite_float(summary.get("hotspot_mean_min_heavy_atom_distance"))
    return None if value is None else float(value)


def _annotate_stageb_quality_floor(result: Any, quality_floor: float) -> None:
    floor = min(max(float(quality_floor), 0.0), 1.0)
    result.loss_dict[ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY] = floor
    result.summary_confidence[ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY] = floor
    result.summary_confidence["stageb_pair_chain_quality_ok"] = (
        _stageb_result_pair_chain_iptm(result) >= floor
    )


def _stageb_guarded_rank(
    result: Any,
    tree: "_ESMFold2StageBMCTSTree",
    *,
    quality_floor: Optional[float] = None,
) -> Tuple[float, ...]:
    pair_chain_iptm = _stageb_result_pair_chain_iptm(result)
    effective_quality_floor = (
        _stageb_result_quality_floor(result)
        if quality_floor is None
        else min(max(float(quality_floor), 0.0), 1.0)
    )
    quality_ok = 1.0 if pair_chain_iptm >= effective_quality_floor else 0.0
    coverage = _stageb_result_hotspot_coverage(result)
    distance = _stageb_result_hotspot_distance(result)
    reward = float(tree.result_reward(result))
    if coverage is None:
        return quality_ok, reward, pair_chain_iptm, -float(result.total_loss)
    fully_covered = 1.0 if coverage >= 1.0 - 1.0e-8 else 0.0
    finite_distance = distance if distance is not None else float("inf")
    if fully_covered:
        return (
            quality_ok,
            fully_covered,
            float(coverage),
            reward,
            pair_chain_iptm,
            -float(finite_distance),
            -float(result.total_loss),
        )
    return (
        quality_ok,
        fully_covered,
        float(coverage),
        -float(finite_distance),
        reward,
        pair_chain_iptm,
        -float(result.total_loss),
    )


def _select_stageb_tree_expansions(
    actions: Sequence[Any],
    results: Sequence[Any],
    *,
    tree: "_ESMFold2StageBMCTSTree",
    max_children: int,
    allow_one_quality_exploration: bool = True,
) -> Tuple[List[Any], List[Any]]:
    """Select a small, quality-guarded subset of verified candidates for the tree."""

    pairs = list(zip(actions, results))
    if not pairs or max_children <= 0:
        return [], []
    eligible = [pair for pair in pairs if _stageb_result_quality_ok(pair[1])]
    eligible.sort(key=lambda pair: _stageb_guarded_rank(pair[1], tree), reverse=True)
    selected = eligible[: int(max_children)]
    selected_ids = {id(action) for action, _ in selected}
    if allow_one_quality_exploration and len(selected) < int(max_children):
        ineligible = [
            pair for pair in pairs if id(pair[0]) not in selected_ids
        ]
        ineligible.sort(
            key=lambda pair: (
                _stageb_result_pair_chain_iptm(pair[1])
                - _stageb_result_quality_floor(pair[1]),
                float(tree.result_reward(pair[1])),
            ),
            reverse=True,
        )
        if ineligible:
            selected.append(ineligible[0])
    return (
        [action for action, _ in selected],
        [result for _, result in selected],
    )


def _stageb_selection_surface(
    result: Any,
    tree: "_ESMFold2StageBMCTSTree",
    *,
    quality_floor: Optional[float] = None,
) -> Dict[str, Any]:
    summary = dict(getattr(result, "summary_confidence", {}) or {})
    pair_chain_iptm = _stageb_result_pair_chain_iptm(result)
    effective_quality_floor = (
        _stageb_result_quality_floor(result)
        if quality_floor is None
        else min(max(float(quality_floor), 0.0), 1.0)
    )
    return {
        "sequence": str(result.binder_seq),
        "total_loss": float(result.total_loss),
        "tree_reward": float(tree.result_reward(result)),
        "pair_chain_iptm": pair_chain_iptm,
        "quality_floor": effective_quality_floor,
        "quality_ok": pair_chain_iptm >= effective_quality_floor,
        "hotspot_coverage": _stageb_result_hotspot_coverage(result),
        "hotspot_coverage_5a": _finite_float(
            summary.get("hotspot_heavy_atom_coverage_5a")
        ),
        "hotspot_mean_distance": _stageb_result_hotspot_distance(result),
    }


def _make_phase0_seed_sequences(
    init_binder_seq: Optional[str],
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Optional[Sequence[bool]] = None,
) -> List[str]:
    num_beams = max(int(num_beams), 1)
    rng = random.Random(seed)
    mask = _normalize_mutable_mask(mutable_mask, L_binder)

    if init_binder_seq is not None:
        init = str(init_binder_seq).strip().upper()
        if len(init) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(init)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(init) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
        mutable_positions = [idx for idx, is_mutable in enumerate(mask) if is_mutable]
        seeds = [init]
        while len(seeds) < num_beams and mutable_positions:
            seq = list(init)
            pos = rng.choice(mutable_positions)
            aa_choices = [aa for aa in VALID_ESM_BINDER_AAS if aa != seq[pos]]
            seq[pos] = rng.choice(aa_choices)
            candidate = "".join(seq)
            if candidate not in seeds:
                seeds.append(candidate)
        return seeds

    seeds: List[str] = []
    while len(seeds) < num_beams:
        candidate = "".join(rng.choice(VALID_ESM_BINDER_AAS) for _ in range(L_binder))
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


def _normalize_logit_init_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "legacy_anchor").strip().lower()
    if normalized not in ESMFOLD2_LOGIT_INIT_MODES:
        raise ValueError(
            "esmfold2_logit_init_mode must be one of: "
            + ", ".join(sorted(ESMFOLD2_LOGIT_INIT_MODES))
        )
    return normalized


@contextmanager
def _torch_seed_context(seed: Optional[int], device: Optional[Union[str, torch.device]] = None):
    devices: List[int] = []
    if device is not None:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        yield


def _make_official_free_phase0_seed_states(
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    base_sequence: Optional[str] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Create official-style free mutable logits and matching argmax roots.

    Fixed scaffold positions come from ``base_sequence`` and are hard anchored
    by ``init_binder_logits``.  Mutable positions stay near-uniform/free and
    the initial discrete residue is selected by row-wise argmax, matching the
    ESMFold2 binder-design style more closely than anchoring mutable residues to
    an arbitrary starting sequence.
    """

    if base_sequence is not None:
        base_sequence = str(base_sequence).strip().upper()
        if len(base_sequence) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(base_sequence)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(base_sequence) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
    elif not all(bool(item) for item in mutable_mask):
        raise ValueError(
            "official_free ESMFold2 root initialization without init_binder_seq "
            "requires all binder positions to be mutable"
        )

    base_rng = random.Random(seed)
    states: List[Tuple[str, torch.Tensor]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(int(num_beams), 1) * 100
    while len(states) < max(int(num_beams), 1) and attempts < max_attempts:
        attempts += 1
        base_seq = base_sequence or "".join(
            base_rng.choice(VALID_ESM_BINDER_AAS) for _ in range(L_binder)
        )
        generator = torch.Generator()
        generator.manual_seed(_stable_seed(seed, "esm_phase0_free_logits", attempts))
        logits = init_binder_logits(
            base_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            random_scale=0.01,
            requires_grad=False,
            generator=generator,
        )[0].detach().cpu()
        seq = sequence_from_logits_argmax(
            logits,
            current_seq=base_seq,
            mutable_mask=mutable_mask,
        )
        if seq in seen:
            continue
        seen.add(seq)
        states.append((seq, logits))
    if len(states) < max(int(num_beams), 1):
        raise RuntimeError("failed to generate unique official_free ESMFold2 phase0 roots")
    return states


def _make_official_exact_phase0_seed_states(
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    base_sequence: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Create official-seed logits for exact official-binder-design diagnosis."""

    if base_sequence is not None:
        base_sequence = str(base_sequence).strip().upper()
        if len(base_sequence) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(base_sequence)} does not match L_binder {L_binder}"
            )
        invalid = sorted(set(base_sequence) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
    elif not all(bool(item) for item in mutable_mask):
        raise ValueError(
            "official_exact ESMFold2 root initialization without init_binder_seq "
            "requires all binder positions to be mutable"
        )

    template_seq = base_sequence or ("A" * int(L_binder))
    batch_size = max(int(num_beams), 1)
    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    with _torch_seed_context(seed, torch_device):
        logits_batch = init_binder_logits(
            template_seq,
            mutable_mask=mutable_mask,
            batch_size=batch_size,
            random_scale=0.01,
            requires_grad=False,
            device=torch_device,
        ).detach().cpu()
    states: List[Tuple[str, torch.Tensor]] = []
    for idx in range(batch_size):
        logits = logits_batch[idx]
        seq = sequence_from_logits_argmax(
            logits,
            current_seq=template_seq,
            mutable_mask=mutable_mask,
        )
        states.append((seq, logits))
    return states


def _make_phase0_seed_states(
    init_binder_seq: Optional[str],
    *,
    num_beams: int,
    L_binder: int,
    seed: Optional[int],
    mutable_mask: Sequence[bool],
    logit_init_mode: str,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Tuple[str, Optional[torch.Tensor]]]:
    mode = _normalize_logit_init_mode(logit_init_mode)
    if mode == "official_free":
        return _make_official_free_phase0_seed_states(
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
            base_sequence=init_binder_seq,
        )
    if mode == "official_exact":
        return _make_official_exact_phase0_seed_states(
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
            base_sequence=init_binder_seq,
            device=device,
        )
    return [
        (seq, None)
        for seq in _make_phase0_seed_sequences(
            init_binder_seq,
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
        )
    ]


def _resolve_initial_logits_checkpoint_paths(
    source: Union[str, os.PathLike[str]],
) -> List[Path]:
    """Resolve one checkpoint file or a Best-of-N artifact directory."""

    path = Path(source).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.exists():
        raise FileNotFoundError(f"initial logits checkpoint does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"initial logits checkpoint must be a file or directory: {path}")

    direct = path / "initial_logits.pt"
    if direct.is_file():
        return [direct.resolve()]

    checkpoint_paths = sorted(path.glob("trajectory_*/initial_logits.pt"))
    if not checkpoint_paths:
        checkpoint_paths = sorted(
            path.glob("best_of_n/trajectory_*/initial_logits.pt")
        )
    if not checkpoint_paths:
        raise FileNotFoundError(
            "no initial_logits.pt checkpoints found under "
            f"{path}; expected trajectory_*/initial_logits.pt or "
            "best_of_n/trajectory_*/initial_logits.pt"
        )
    return [checkpoint.resolve() for checkpoint in checkpoint_paths]


def _load_initial_logits_checkpoint_states(
    source: Union[str, os.PathLike[str]],
    *,
    L_binder: int,
    num_beams: int,
    init_binder_seq: Optional[str],
    mutable_mask: Sequence[bool],
    allow_extra_checkpoints: bool = False,
) -> Tuple[List[Tuple[str, torch.Tensor]], List[str]]:
    """Load exact MCTS Phase-0 roots from saved Best-of-N logits."""

    checkpoint_paths = _resolve_initial_logits_checkpoint_paths(source)
    expected_count = max(int(num_beams), 1)
    if len(checkpoint_paths) < expected_count or (
        not allow_extra_checkpoints and len(checkpoint_paths) != expected_count
    ):
        raise ValueError(
            f"initial logits checkpoint count {len(checkpoint_paths)} does not "
            f"match mcts_num_beams {expected_count}"
        )
    checkpoint_paths = checkpoint_paths[:expected_count]

    reference_sequence = None
    if init_binder_seq is not None:
        reference_sequence = str(init_binder_seq).strip().upper()
        if len(reference_sequence) != int(L_binder):
            raise ValueError(
                f"init_binder_seq length {len(reference_sequence)} does not "
                f"match L_binder {L_binder}"
            )
        invalid = sorted(set(reference_sequence) - set(ESM_AA_ORDER))
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in init_binder_seq: {invalid}")
    elif not all(bool(item) for item in mutable_mask):
        raise ValueError(
            "loading initial logits with fixed binder positions requires "
            "init_binder_seq"
        )

    states: List[Tuple[str, torch.Tensor]] = []
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(payload, Mapping):
            if "logits" not in payload:
                raise ValueError(
                    f"checkpoint has no 'logits' tensor: {checkpoint_path}"
                )
            logits_value = payload["logits"]
            stored_sequence_value = payload.get("sequence")
            stored_hash = payload.get("sequence_hash")
        else:
            logits_value = payload
            stored_sequence_value = None
            stored_hash = None

        if not isinstance(logits_value, torch.Tensor):
            raise ValueError(
                f"checkpoint logits must be a torch.Tensor: {checkpoint_path}"
            )
        logits = logits_value.detach().cpu()
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        expected_shape = (int(L_binder), len(ESM_AA_ORDER))
        if logits.ndim != 2 or tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"checkpoint logits shape {tuple(logits.shape)} does not match "
                f"{expected_shape}: {checkpoint_path}"
            )
        if not torch.is_floating_point(logits):
            raise ValueError(
                f"checkpoint logits must use a floating dtype: {checkpoint_path}"
            )
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError(
                f"checkpoint logits contain NaN or +inf: {checkpoint_path}"
            )
        if torch.isneginf(logits).all(dim=-1).any():
            raise ValueError(
                f"checkpoint logits contain a position with no valid amino acid: "
                f"{checkpoint_path}"
            )

        stored_sequence = None
        if stored_sequence_value is not None:
            stored_sequence = str(stored_sequence_value).strip().upper()
            if len(stored_sequence) != int(L_binder):
                raise ValueError(
                    f"checkpoint sequence length {len(stored_sequence)} does not "
                    f"match L_binder {L_binder}: {checkpoint_path}"
                )
            invalid = sorted(set(stored_sequence) - set(ESM_AA_ORDER))
            if invalid:
                raise ValueError(
                    f"checkpoint sequence contains invalid amino acid(s) "
                    f"{invalid}: {checkpoint_path}"
                )
            if stored_hash is not None and str(stored_hash) != _sequence_hash(
                stored_sequence
            ):
                raise ValueError(
                    f"checkpoint sequence hash does not match sequence: "
                    f"{checkpoint_path}"
                )

        base_sequence = reference_sequence or stored_sequence or ("A" * int(L_binder))
        derived_sequence = sequence_from_logits_argmax(
            logits,
            current_seq=base_sequence,
            mutable_mask=mutable_mask,
        )
        if stored_sequence is not None and stored_sequence != derived_sequence:
            raise ValueError(
                f"checkpoint sequence does not match logits argmax: {checkpoint_path}"
            )
        if reference_sequence is not None:
            for index, is_mutable in enumerate(mutable_mask):
                checkpoint_amino_acid = ESM_AA_ORDER[
                    int(torch.argmax(logits[index]).item())
                ]
                if (
                    not is_mutable
                    and checkpoint_amino_acid != reference_sequence[index]
                ):
                    raise ValueError(
                        f"checkpoint changes fixed binder position {index + 1}: "
                        f"{checkpoint_path}"
                    )
        states.append((derived_sequence, logits.clone()))

    return states, [str(path) for path in checkpoint_paths]








def _normalize_mutable_mask(
    mutable_mask: Optional[Sequence[bool]],
    length: int,
) -> List[bool]:
    if mutable_mask is None:
        return [True] * int(length)
    mask = [bool(item) for item in mutable_mask]
    if len(mask) != int(length):
        raise ValueError(f"mutable_mask length {len(mask)} does not match binder length {length}")
    return mask


def _clip_sequence_by_mutable_mask(
    sequence: str,
    mutable_mask: Sequence[bool],
    reference_sequence: str,
) -> str:
    if len(sequence) != len(reference_sequence):
        raise ValueError("candidate sequence length must match reference sequence length")
    if len(mutable_mask) != len(reference_sequence):
        raise ValueError("mutable_mask length must match reference sequence length")
    chars = list(str(sequence))
    for idx, is_mutable in enumerate(mutable_mask):
        if not is_mutable:
            chars[idx] = reference_sequence[idx]
    return "".join(chars)


def _verify_with_cache(
    backend: Any,
    verifier_cache: Dict[str, Any],
    *,
    binder_seq: str,
    target_seq: Union[str, Sequence[str]],
    seed: Optional[int],
    hotspot_params: Optional[Dict[str, Any]] = None,
) -> Any:
    if binder_seq in verifier_cache:
        return verifier_cache[binder_seq]
    verify_kwargs = dict(
        binder_seq=binder_seq,
        target_seq=target_seq,
        seed=seed,
    )
    if _callable_accepts_argument(backend.verify, "hotspot_params"):
        verify_kwargs["hotspot_params"] = (
            dict(hotspot_params) if hotspot_params is not None else None
        )
    result = backend.verify(**verify_kwargs)
    verifier_cache[str(result.binder_seq)] = result
    return result


def _finite_float(value: Any) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _hotspot_proposal_weight_for_verified_state(
    state: Any,
    *,
    configured_weight: float,
    satisfied_weight: float,
) -> float:
    """Reduce hotspot attraction after every requested hotspot is covered."""

    configured = max(float(configured_weight), 0.0)
    satisfied = max(float(satisfied_weight), 0.0)
    if state is None or configured <= 0.0:
        return configured

    mappings: List[Mapping[str, Any]] = []
    if isinstance(state, Mapping):
        mappings.append(state)
    for attr in ("summary_confidence", "loss_dict", "loss_vector"):
        value = getattr(state, attr, None)
        if isinstance(value, Mapping):
            mappings.append(value)

    for values in mappings:
        coverage = _finite_float(values.get("hotspot_heavy_atom_coverage"))
        if coverage is not None:
            return satisfied if coverage >= 1.0 - 1.0e-8 else configured
    return configured




def _stageb_preview_objective(
    preview: Mapping[str, Any],
    *,
    action_group: str,
    hotspot_weight: float,
    hotspot_interface_mode: str,
) -> float:
    """Return the cheap discrete objective matching one Stage-B repair group."""

    normalized_group = str(action_group)
    localized = str(hotspot_interface_mode).strip().lower() == "localized"
    if normalized_group == "interface_formation":
        key = (
            "effective_inter_contact_loss"
            if localized and "effective_inter_contact_loss" in preview
            else "inter_contact_loss"
        )
    elif normalized_group == "core_stabilization":
        key = "intra_contact_loss"
    elif normalized_group == "compactness_repair":
        key = "glob_loss"
    elif normalized_group == "unrouted":
        intra = _finite_float(preview.get("intra_contact_loss"))
        inter_key = (
            "effective_inter_contact_loss"
            if localized and "effective_inter_contact_loss" in preview
            else "inter_contact_loss"
        )
        inter = _finite_float(preview.get(inter_key))
        glob = _finite_float(preview.get("glob_loss"))
        if intra is None or inter is None or glob is None:
            raise ValueError(
                "Preview output is missing unified-loss components for 'unrouted'"
            )
        value = 0.5 * intra + 0.5 * inter + 0.2 * glob
        if not localized:
            hotspot_contact = _finite_float(preview.get("hotspot_contact_loss"))
            if hotspot_contact is not None:
                value += max(float(hotspot_weight), 0.0) * hotspot_contact
        return float(value)
    else:
        raise ValueError(f"Unsupported ESMFold2 Stage-B action group: {action_group!r}")

    value = _finite_float(preview.get(key))
    if value is None:
        raise ValueError(f"Preview output is missing {key!r} for {action_group!r}")

    hotspot_contact = _finite_float(preview.get("hotspot_contact_loss"))
    hotspot_already_localized = normalized_group == "interface_formation" and localized
    if hotspot_contact is not None and not hotspot_already_localized:
        value += max(float(hotspot_weight), 0.0) * hotspot_contact
    return float(value)


def _rerank_stageb_candidates_with_preview(
    backend: Any,
    preview_cache: Dict[str, Dict[str, float]],
    *,
    current_seq: str,
    candidates: Sequence[str],
    target_seq: Union[str, Sequence[str]],
    action_group: str,
    seed: Optional[int],
    hotspot_params: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Score discrete single mutants with the cheap proposal-model preview.

    The same seed is reused for the parent and every candidate so proposal LM
    dropout is paired.  Any preview failure disables reranking for the whole
    group instead of mixing incomparable score sources.
    """

    if not hasattr(backend, "preview_loss_dict"):
        return {}

    config = getattr(backend, "config", None)
    hotspot_weight = float(getattr(config, "hotspot_weight", 0.0))
    hotspot_interface_mode = str(
        getattr(config, "hotspot_interface_mode", "additive")
    )

    def preview(sequence: str) -> Dict[str, float]:
        if sequence not in preview_cache:
            kwargs = {
                "binder_seq": sequence,
                "target_seq": target_seq,
                "seed": seed,
            }
            if _callable_accepts_argument(backend.preview_loss_dict, "hotspot_params"):
                kwargs["hotspot_params"] = dict(hotspot_params) if hotspot_params else None
            raw = backend.preview_loss_dict(**kwargs)
            preview_cache[sequence] = {
                str(key): float(scalar)
                for key, value in dict(raw or {}).items()
                if (scalar := _finite_float(value)) is not None
            }
        return preview_cache[sequence]

    try:
        parent_loss = _stageb_preview_objective(
            preview(current_seq),
            action_group=action_group,
            hotspot_weight=hotspot_weight,
            hotspot_interface_mode=hotspot_interface_mode,
        )
        scores: Dict[str, Dict[str, float]] = {}
        for sequence in dict.fromkeys(str(item) for item in candidates):
            candidate_loss = _stageb_preview_objective(
                preview(sequence),
                action_group=action_group,
                hotspot_weight=hotspot_weight,
                hotspot_interface_mode=hotspot_interface_mode,
            )
            scores[sequence] = {
                "parent_loss": float(parent_loss),
                "loss": float(candidate_loss),
                "benefit": float(parent_loss - candidate_loss),
            }
        return scores
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        logger.warning(
            "ESM StageB discrete preview rerank failed for group %s: %s",
            action_group,
            exc,
        )
        return {}


def _format_float_map(values: Dict[str, float], *, precision: int = 3) -> str:
    if not values:
        return "-"
    return ",".join(
        f"{key}:{float(values[key]):.{precision}f}"
        for key in sorted(values)
    )


def _format_count_map(values: Dict[str, int]) -> str:
    if not values:
        return "-"
    return ",".join(f"{key}:{int(values[key])}" for key in sorted(values))


def _count_items(items: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_handoff_results(
    verifier_cache: Dict[str, Any],
    *,
    top_k: int,
    min_distance: int = 1,
) -> List[Any]:
    ordered = sorted(verifier_cache.values(), key=lambda result: float(result.total_loss))
    selected: List[Any] = []
    for result in ordered:
        seq = str(result.binder_seq)
        if any(
            sum(a != b for a, b in zip(seq, str(existing.binder_seq))) < min_distance
            for existing in selected
        ):
            continue
        selected.append(result)
        if len(selected) >= max(int(top_k), 1):
            break
    if not selected and ordered:
        selected.append(ordered[0])
    return selected


def _checkpoint_cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _checkpoint_cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_cpu_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_cpu_value(item) for item in value)
    return value


def _serialize_handoff_result(result: Any) -> Dict[str, Any]:
    return {
        "binder_seq": str(result.binder_seq),
        "target_seq": _checkpoint_cpu_value(result.target_seq),
        "total_loss": float(result.total_loss),
        "loss_dict": {
            str(key): float(value)
            for key, value in dict(result.loss_dict).items()
        },
        "summary_confidence": _checkpoint_cpu_value(
            _summary_without_structure(dict(result.summary_confidence or {}))
        ),
        "coordinate": _checkpoint_cpu_value(result.coordinate),
        "plddt_logits": _checkpoint_cpu_value(result.plddt_logits),
    }


def _deserialize_handoff_result(payload: Dict[str, Any]) -> ESMFold2VerifierResult:
    return ESMFold2VerifierResult(
        binder_seq=str(payload["binder_seq"]),
        target_seq=payload["target_seq"],
        total_loss=float(payload["total_loss"]),
        loss_dict={
            str(key): float(value)
            for key, value in dict(payload["loss_dict"]).items()
        },
        summary_confidence=dict(payload.get("summary_confidence", {})),
        coordinate=payload.get("coordinate"),
        plddt_logits=payload.get("plddt_logits"),
    )


def _stagea_handoff_verifier_contract(backend_config: Any) -> Dict[str, Any]:
    fields = (
        "verifier_model_name",
        "verifier_num_loops",
        "verifier_num_sampling_steps",
        "hotspot_verify_weight",
        "hotspot_verify_near_cutoff",
        "hotspot_verify_far_cutoff",
        "hotspot_verify_min_distance",
        "hotspot_verify_too_close_weight",
        "target_chain_roles",
        "binder_mutable_mask",
    )
    return {
        field: _checkpoint_cpu_value(getattr(backend_config, field, None))
        for field in fields
    }


def _save_stagea_handoff_checkpoint(
    *,
    artifact_dir: Optional[Union[str, os.PathLike[str]]],
    completed_stagea_steps: int,
    final: bool,
    target_seq: Union[str, Sequence[str]],
    L_binder: int,
    mutable_mask: Sequence[bool],
    hotspot_params: Optional[Dict[str, Any]],
    seed: Optional[int],
    handoff_top_k: int,
    verifier_cache: Dict[str, Any],
    proposal_logits_by_sequence: Dict[str, torch.Tensor],
    losses_history: Sequence[float],
    stagea_config: Dict[str, Any],
    verifier_contract: Dict[str, Any],
) -> Optional[Path]:
    if artifact_dir is None:
        return None
    handoff_results = _top_handoff_results(
        verifier_cache,
        top_k=handoff_top_k,
        min_distance=1,
    )
    if not handoff_results:
        return None

    entries = []
    for result in handoff_results:
        sequence = str(result.binder_seq)
        logits = proposal_logits_by_sequence.get(sequence)
        if logits is None:
            logits = _initial_proposal_logits(sequence, mutable_mask)
        entries.append(
            {
                "result": _serialize_handoff_result(result),
                "proposal_logits": logits.detach().cpu().clone(),
            }
        )

    checkpoint_dir = Path(artifact_dir) / "handoff_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "stagea_handoff_final.pt"
        if final
        else f"stagea_handoff_step_{int(completed_stagea_steps):04d}.pt"
    )
    path = checkpoint_dir / filename
    payload = {
        "schema_version": ESMFOLD2_STAGEA_HANDOFF_SCHEMA_VERSION,
        "checkpoint_type": "esmfold2_stagea_handoff",
        "completed_stagea_steps": int(completed_stagea_steps),
        "target_seq": _checkpoint_cpu_value(target_seq),
        "target_key": _normalize_target_key(target_seq),
        "binder_length": int(L_binder),
        "mutable_mask": [bool(value) for value in mutable_mask],
        "hotspot_params": _checkpoint_cpu_value(hotspot_params),
        "seed": seed,
        "handoff_top_k": int(handoff_top_k),
        "losses_history": [float(value) for value in losses_history],
        "stagea_config": _checkpoint_cpu_value(stagea_config),
        "verifier_contract": _checkpoint_cpu_value(verifier_contract),
        "handoff_entries": entries,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    logger.info(
        "Saved ESMFold2 StageA handoff checkpoint: %s (step=%d roots=%d)",
        path,
        completed_stagea_steps,
        len(entries),
    )
    return path


def _load_stagea_handoff_checkpoint(
    checkpoint_path: Union[str, os.PathLike[str]],
    *,
    target_seq: Union[str, Sequence[str]],
    L_binder: int,
    mutable_mask: Sequence[bool],
    hotspot_params: Optional[Dict[str, Any]],
    verifier_contract: Dict[str, Any],
    allow_verifier_mismatch: bool = False,
    init_binder_seq: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ESMFold2 StageA handoff checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ESMFold2 StageA handoff checkpoint: {path}")
    if int(payload.get("schema_version", -1)) != ESMFOLD2_STAGEA_HANDOFF_SCHEMA_VERSION:
        raise ValueError(
            "unsupported ESMFold2 StageA handoff checkpoint schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("checkpoint_type") != "esmfold2_stagea_handoff":
        raise ValueError(f"not an ESMFold2 StageA handoff checkpoint: {path}")
    if str(payload.get("target_key")) != _normalize_target_key(target_seq):
        raise ValueError("StageA handoff checkpoint target does not match current target")
    if int(payload.get("binder_length", -1)) != int(L_binder):
        raise ValueError("StageA handoff checkpoint binder length does not match")
    if list(payload.get("mutable_mask", [])) != [bool(value) for value in mutable_mask]:
        raise ValueError("StageA handoff checkpoint mutable mask does not match")
    expected_hotspots = json.dumps(_json_ready(hotspot_params), sort_keys=True)
    saved_hotspots = json.dumps(_json_ready(payload.get("hotspot_params")), sort_keys=True)
    if saved_hotspots != expected_hotspots:
        raise ValueError("StageA handoff checkpoint hotspot parameters do not match")
    expected_verifier = json.dumps(_json_ready(verifier_contract), sort_keys=True)
    saved_verifier = json.dumps(
        _json_ready(payload.get("verifier_contract")),
        sort_keys=True,
    )
    verifier_contract_matches = saved_verifier == expected_verifier
    if not verifier_contract_matches and not allow_verifier_mismatch:
        raise ValueError("StageA handoff checkpoint verifier configuration does not match")
    entries = list(payload.get("handoff_entries", []))
    if not entries:
        raise ValueError("StageA handoff checkpoint contains no handoff roots")
    completed_steps = int(payload.get("completed_stagea_steps", -1))
    if completed_steps < 0 or len(payload.get("losses_history", [])) != completed_steps:
        raise ValueError("StageA handoff checkpoint has an inconsistent loss history")
    reference_sequence = None
    if init_binder_seq is not None:
        reference_sequence = str(init_binder_seq).strip().upper()
        if len(reference_sequence) != int(L_binder):
            raise ValueError("init_binder_seq length does not match L_binder")
        if set(reference_sequence) - set(ESM_AA_ORDER):
            raise ValueError("init_binder_seq contains invalid amino acids")
    for entry in entries:
        result_payload = dict(entry.get("result", {}))
        sequence = str(result_payload.get("binder_seq", ""))
        logits = entry.get("proposal_logits")
        if len(sequence) != int(L_binder):
            raise ValueError("StageA handoff checkpoint contains an invalid binder sequence")
        if reference_sequence is not None:
            for index, is_mutable in enumerate(mutable_mask):
                if not is_mutable and sequence[index] != reference_sequence[index]:
                    raise ValueError(
                        f"StageA handoff checkpoint changes fixed binder position {index + 1}"
                    )
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
            int(L_binder),
            len(ESM_AA_ORDER),
        ):
            raise ValueError("StageA handoff checkpoint contains invalid proposal logits")
    payload["checkpoint_path"] = str(path)
    payload["verifier_contract_matches"] = verifier_contract_matches
    return payload


def _make_stageb_proposals(
    *,
    current_seq: str,
    candidates: Sequence[str],
    action_group: str,
    group_prior: float,
    logits_before: Optional[torch.Tensor] = None,
    logits_after: Optional[torch.Tensor] = None,
    mutable_mask: Optional[Sequence[bool]] = None,
    use_gradient_prior: bool = True,
    rerank_scores: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> List[ESMFold2MutationProposal]:
    unique = []
    edits_by_sequence: Dict[str, List[Tuple[int, str, str]]] = {}
    for item in candidates:
        sequence = str(item)
        edits = _sequence_edits(current_seq, sequence)
        if sequence == current_seq or len(edits) != 1 or sequence in edits_by_sequence:
            continue
        unique.append(sequence)
        edits_by_sequence[sequence] = edits
    if not unique:
        return []

    logit_benefits = torch.zeros(len(unique), dtype=torch.float32)
    if use_gradient_prior and logits_before is not None and logits_after is not None:
        mutation_scores = logit_delta_mutation_scores(
            logits_before,
            logits_after,
            current_seq=current_seq,
            mutable_mask=mutable_mask,
        ).detach().cpu()
        for index, sequence in enumerate(unique):
            pos, _, new_aa = edits_by_sequence[sequence][0]
            score = mutation_scores[int(pos), ESM_AA_ORDER.index(str(new_aa))]
            logit_benefits[index] = score if torch.isfinite(score) else -1.0e9
    gradient_priors = torch.softmax(logit_benefits, dim=0)

    rerank_scores = dict(rerank_scores or {})
    rerank_enabled = bool(rerank_scores) and all(seq in rerank_scores for seq in unique)
    if rerank_enabled:
        ranking_benefits = torch.tensor(
            [float(rerank_scores[seq]["benefit"]) for seq in unique],
            dtype=torch.float32,
        )
        ranking_priors = torch.softmax(ranking_benefits, dim=0)
    else:
        ranking_benefits = logit_benefits
        ranking_priors = gradient_priors

    proposals: List[ESMFold2MutationProposal] = []
    normalized_group_prior = max(float(group_prior), 1.0e-12)
    for index, seq in enumerate(unique):
        gradient_prior = float(gradient_priors[index].item())
        ranking_prior = float(ranking_priors[index].item())
        rerank_record = rerank_scores.get(seq) if rerank_enabled else None
        proposals.append(
            ESMFold2MutationProposal(
                sequence=seq,
                edits=edits_by_sequence[seq],
                prior=normalized_group_prior * ranking_prior,
                benefit=float(ranking_benefits[index].item()),
                action_group=action_group,
                group_prior=normalized_group_prior,
                gradient_prior=gradient_prior,
                logit_delta_benefit=float(logit_benefits[index].item()),
                rerank_benefit=(
                    None if rerank_record is None else float(rerank_record["benefit"])
                ),
                rerank_loss=(
                    None if rerank_record is None else float(rerank_record["loss"])
                ),
                rerank_prior=ranking_prior if rerank_record is not None else None,
                proposal_logits=logits_after.detach().clone() if logits_after is not None else None,
            )
        )
    return proposals


def run_esmfold2_mcts_hallucination(
    *,
    backend: Any,
    target_seq: Union[str, Sequence[str]],
    L_binder: int = 40,
    N_steps: int = 100,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    stagea_max_mutations_per_seq: Optional[int] = None,
    esmfold2_stagea_mutation_fraction: Optional[float] = 0.5,
    esmfold2_stagea_candidate_strategy: str = "positive_score",
    esmfold2_stagea_dominance_gap: float = 0.5,
    esmfold2_stagea_protected_mutation_candidates: int = 2,
    esmfold2_stagea_include_argmax_candidate: bool = True,
    stageb_max_mutations_per_seq: Optional[int] = None,
    num_beams: int = 8,
    init_binder_seq: Optional[str] = None,
    esmfold2_initial_logits_checkpoint: Optional[
        Union[str, os.PathLike[str]]
    ] = None,
    mcts_C: float = 1.0,
    mcts_max_nodes: int = 5000,
    mcts_strategy: str = "two_stage",
    mcts_bootstrap_steps: int = 50,
    mcts_bootstrap_patience: int = 10,
    mcts_bootstrap_min_delta: float = 0.02,
    mcts_bootstrap_max_steps: int = 80,
    mcts_handoff_top_k: Optional[int] = None,
    mcts_expand_actions_per_step: int = 2,
    mcts_prior_mode: str = "full",
    seed: Optional[int] = 42,
    mutable_mask: Optional[Sequence[bool]] = None,
    hotspot_params: Optional[Dict[str, Any]] = None,
    artifact_dir: Optional[Union[str, os.PathLike[str]]] = None,
    save_verified_candidate_cif: bool = False,
    save_visualization_trace: bool = False,
    esmfold2_logit_init_mode: str = "legacy_anchor",
    esmfold2_stagea_soft_steps: Optional[int] = None,
    esmfold2_stagea_tree_children: int = 2,
    esmfold2_stagea_root_selection_mode: str = "hierarchical_ucb",
    esmfold2_stagea_min_root_expansions: int = 2,
    esmfold2_stagea_preserve_soft_state: bool = True,
    esmfold2_stagea_cumulative_annealing: bool = False,
    esmfold2_stagea_annealing_horizon: int = 150,
    esmfold2_stagea_soft_anchor_margin: float = 0.1,
    esmfold2_stagea_paired_verifier_seed: bool = True,
    esmfold2_stagea_verifier_seed: Optional[int] = None,
    esmfold2_stageb_soft_steps: Optional[int] = None,
    esmfold2_stageb_discrete_rerank: bool = True,
    esmfold2_stageb_rerank_pool_size: int = 20,
    esmfold2_stageb_exploration_actions: int = 1,
    esmfold2_stageb_verifier_candidates: int = 8,
    esmfold2_stageb_hotspot_coverage_weight: float = ESMFOLD2_STAGEB_HOTSPOT_COVERAGE_WEIGHT,
    esmfold2_stageb_hotspot_distance_weight: float = ESMFOLD2_STAGEB_HOTSPOT_DISTANCE_WEIGHT,
    esmfold2_stageb_hotspot_miss_penalty: float = ESMFOLD2_STAGEB_HOTSPOT_MISS_PENALTY,
    esmfold2_stageb_min_pair_chain_iptm: float = 0.0,
    esmfold2_stageb_handoff_quality_tolerance: float = 0.05,
    esmfold2_stageb_paired_verifier_seed: bool = True,
    esmfold2_stageb_verifier_seed: Optional[int] = None,
    esmfold2_stagea_handoff_checkpoint_interval: int = 10,
    esmfold2_resume_handoff_checkpoint: Optional[Union[str, os.PathLike[str]]] = None,
    esmfold2_resume_reverify_roots: bool = False,
) -> Tuple[str, str, List[float]]:
    """
    Run the first ESMFold2 closed-loop path.

    The backend supplies two operations:
      - verify(binder_seq, target_seq, seed) -> MCTS-compatible result
      - propose(current_seq, target_seq, global_step, K/B/T args, seed) -> sequences

    Stage A explores soft sequence proposals; Stage B refines verified
    candidates with action-level MCTS. A budget ending during Stage A can
    produce a handoff checkpoint without entering Stage B.
    """

    if backend is None:
        raise ValueError("backend is required for ESMFold2 MCTS")
    L_binder = int(L_binder)
    N_steps = max(int(N_steps), 0)
    K_top = max(int(K_top), 1)
    B_candidates = max(int(B_candidates), 1)
    max_mutations_per_seq = max(int(max_mutations_per_seq), 1)
    stagea_max_mutations = (
        max(int(stagea_max_mutations_per_seq), 1)
        if stagea_max_mutations_per_seq is not None
        else max_mutations_per_seq
    )
    stagea_mutation_fraction = (
        float(esmfold2_stagea_mutation_fraction)
        if esmfold2_stagea_mutation_fraction is not None
        else None
    )
    if stagea_mutation_fraction is not None and (
        not math.isfinite(stagea_mutation_fraction)
        or not 0.0 < stagea_mutation_fraction <= 1.0
    ):
        raise ValueError("esmfold2_stagea_mutation_fraction must be in (0, 1]")
    stagea_candidate_strategy = str(
        esmfold2_stagea_candidate_strategy or "legacy"
    ).strip().lower()
    if stagea_candidate_strategy == "positive_score_guarded":
        from proteocraft.hallucination.esmfold2_backend import validate_dominance_guard
        validate_dominance_guard(
            esmfold2_stagea_dominance_gap, esmfold2_stagea_protected_mutation_candidates,
        )
    if stagea_candidate_strategy not in {"legacy", "positive_score", "positive_score_guarded"}:
        raise ValueError(
            "esmfold2_stagea_candidate_strategy must be 'legacy' or "
            "'positive_score' or 'positive_score_guarded'"
        )
    if (
        stagea_candidate_strategy in {"positive_score", "positive_score_guarded"}
        and stagea_mutation_fraction is None
    ):
        raise ValueError(
            "positive_score StageA candidate strategy requires "
            "esmfold2_stagea_mutation_fraction"
        )
    stagea_include_argmax_candidate = bool(
        esmfold2_stagea_include_argmax_candidate
    )
    stageb_max_mutations = (
        max(int(stageb_max_mutations_per_seq), 1)
        if stageb_max_mutations_per_seq is not None
        else 1
    )
    strategy = str(mcts_strategy).strip().lower()
    if strategy != "two_stage":
        raise ValueError("ProteoCraft supports only mcts_strategy='two_stage'")
    target_key = _normalize_target_key(target_seq)
    stagea_paired_verifier_seed = bool(esmfold2_stagea_paired_verifier_seed)
    stagea_shared_verify_seed = (
        (
            int(esmfold2_stagea_verifier_seed)
            if esmfold2_stagea_verifier_seed is not None
            else _stable_seed(seed, "esm_stagea_paired_verify", target_key)
        )
        if stagea_paired_verifier_seed
        else None
    )
    stageb_shared_verify_seed = (
        (
            int(esmfold2_stageb_verifier_seed)
            if esmfold2_stageb_verifier_seed is not None
            else _stable_seed(seed, "esm_stageb_paired_verify", target_key)
        )
        if bool(esmfold2_stageb_paired_verifier_seed)
        else None
    )
    mutable_mask = _normalize_mutable_mask(mutable_mask, L_binder)
    stagea_mutable_residue_count = sum(bool(value) for value in mutable_mask)
    if stagea_mutation_fraction is not None:
        stagea_max_mutations = (
            max(
                1,
                math.ceil(
                    stagea_mutation_fraction * stagea_mutable_residue_count
                ),
            )
            if stagea_mutable_residue_count > 0
            else 0
        )
    hotspot_params = dict(hotspot_params) if hotspot_params else None
    hotspot_indices = [
        int(index)
        for index in (hotspot_params or {}).get("target_hotspot_concat_indices", [])
    ]
    backend_config = getattr(backend, "config", None)
    hotspot_weight = float(getattr(backend_config, "hotspot_weight", 0.0))
    hotspot_satisfied_proposal_weight = float(
        getattr(backend_config, "hotspot_satisfied_proposal_weight", 0.0)
    )
    hotspot_cutoff = float(getattr(backend_config, "hotspot_cutoff", 8.0))
    hotspot_interface_mode = str(
        getattr(backend_config, "hotspot_interface_mode", "additive")
    )
    hotspot_verify_weight = float(
        getattr(backend_config, "hotspot_verify_weight", 0.0)
    )
    hotspot_verify_near_cutoff = float(
        getattr(backend_config, "hotspot_verify_near_cutoff", 8.0)
    )
    hotspot_verify_far_cutoff = float(
        getattr(backend_config, "hotspot_verify_far_cutoff", 20.0)
    )
    hotspot_verify_min_distance = float(
        getattr(backend_config, "hotspot_verify_min_distance", 0.0)
    )
    hotspot_verify_too_close_weight = float(
        getattr(backend_config, "hotspot_verify_too_close_weight", 0.0)
    )
    if hotspot_weight > 0.0 and not hotspot_indices:
        raise ValueError(
            "positive ESMFold2 hotspot_weight requires target hotspot indices"
        )
    if hotspot_verify_weight > 0.0 and not hotspot_indices:
        raise ValueError(
            "positive ESMFold2 hotspot_verify_weight requires target hotspot indices"
        )
    if hotspot_verify_too_close_weight > 0.0 and not hotspot_indices:
        raise ValueError(
            "positive ESMFold2 hotspot_verify_too_close_weight requires "
            "target hotspot indices"
        )
    hotspot_summary = {
        "hotspot_indices_0based": hotspot_indices,
        "hotspot_weight": hotspot_weight,
        "hotspot_satisfied_proposal_weight": hotspot_satisfied_proposal_weight,
        "hotspot_cutoff": hotspot_cutoff,
        "hotspot_interface_mode": hotspot_interface_mode,
        "hotspot_enabled": bool(hotspot_indices and hotspot_weight > 0.0),
        "hotspot_verify_weight": hotspot_verify_weight,
        "hotspot_verify_near_cutoff": hotspot_verify_near_cutoff,
        "hotspot_verify_far_cutoff": hotspot_verify_far_cutoff,
        "hotspot_verify_min_distance": hotspot_verify_min_distance,
        "hotspot_verify_too_close_weight": hotspot_verify_too_close_weight,
        "hotspot_verify_enabled": bool(
            hotspot_indices
            and (
                hotspot_verify_weight > 0.0
                or hotspot_verify_too_close_weight > 0.0
            )
        ),
    }
    logit_init_mode = _normalize_logit_init_mode(esmfold2_logit_init_mode)
    stagea_soft_steps = (
        int(esmfold2_stagea_soft_steps)
        if esmfold2_stagea_soft_steps is not None
        else None
    )
    stageb_soft_steps = (
        int(esmfold2_stageb_soft_steps)
        if esmfold2_stageb_soft_steps is not None
        else stagea_soft_steps
    )
    if stagea_soft_steps is not None and stagea_soft_steps <= 0:
        raise ValueError("esmfold2_stagea_soft_steps must be positive")
    stagea_soft_step_count = int(
        stagea_soft_steps
        if stagea_soft_steps is not None
        else getattr(backend_config, "soft_steps", 30)
    )
    if stagea_soft_step_count <= 0:
        raise ValueError("resolved ESMFold2 Stage A soft steps must be positive")
    stagea_tree_children = max(int(esmfold2_stagea_tree_children), 1)
    stagea_root_selection_mode = str(
        esmfold2_stagea_root_selection_mode
    ).strip().lower()
    if stagea_root_selection_mode not in LEGACY_SELECTION_MODES:
        raise ValueError(
            "esmfold2_stagea_root_selection_mode must be one of: "
            + ", ".join(sorted(LEGACY_SELECTION_MODES))
        )
    stagea_min_root_expansions = max(
        int(esmfold2_stagea_min_root_expansions),
        0,
    )
    stagea_preserve_soft_state = bool(esmfold2_stagea_preserve_soft_state)
    stagea_cumulative_annealing = bool(esmfold2_stagea_cumulative_annealing)
    stagea_annealing_horizon = int(esmfold2_stagea_annealing_horizon)
    if stagea_annealing_horizon <= 0:
        raise ValueError("esmfold2_stagea_annealing_horizon must be positive")
    stagea_soft_anchor_margin = float(esmfold2_stagea_soft_anchor_margin)
    if stagea_soft_anchor_margin <= 0.0:
        raise ValueError("esmfold2_stagea_soft_anchor_margin must be positive")
    if stageb_soft_steps is not None and stageb_soft_steps <= 0:
        raise ValueError("esmfold2_stageb_soft_steps must be positive")
    stageb_rerank_pool_size = max(int(esmfold2_stageb_rerank_pool_size), 1)
    stageb_exploration_actions = max(int(esmfold2_stageb_exploration_actions), 0)
    stageb_verifier_candidates = max(int(esmfold2_stageb_verifier_candidates), 1)
    stageb_discrete_rerank = bool(esmfold2_stageb_discrete_rerank)
    stageb_hotspot_coverage_weight = float(
        esmfold2_stageb_hotspot_coverage_weight
    )
    stageb_hotspot_distance_weight = float(esmfold2_stageb_hotspot_distance_weight)
    stageb_hotspot_miss_penalty = float(esmfold2_stageb_hotspot_miss_penalty)
    stageb_min_pair_chain_iptm = float(esmfold2_stageb_min_pair_chain_iptm)
    stageb_handoff_quality_tolerance = float(
        esmfold2_stageb_handoff_quality_tolerance
    )
    stageb_paired_verifier_seed = bool(esmfold2_stageb_paired_verifier_seed)
    if stageb_hotspot_coverage_weight < 0.0:
        raise ValueError("esmfold2_stageb_hotspot_coverage_weight must be non-negative")
    if stageb_hotspot_distance_weight < 0.0:
        raise ValueError("esmfold2_stageb_hotspot_distance_weight must be non-negative")
    if stageb_hotspot_miss_penalty < 0.0:
        raise ValueError("esmfold2_stageb_hotspot_miss_penalty must be non-negative")
    if not 0.0 <= stageb_min_pair_chain_iptm <= 1.0:
        raise ValueError("esmfold2_stageb_min_pair_chain_iptm must be within [0, 1]")
    if not 0.0 <= stageb_handoff_quality_tolerance <= 1.0:
        raise ValueError(
            "esmfold2_stageb_handoff_quality_tolerance must be within [0, 1]"
        )
    hotspot_summary.update(
        {
            "stageb_hotspot_distance_weight": stageb_hotspot_distance_weight,
            "stageb_hotspot_coverage_weight": stageb_hotspot_coverage_weight,
            "stageb_hotspot_miss_penalty": stageb_hotspot_miss_penalty,
            "stageb_min_pair_chain_iptm": stageb_min_pair_chain_iptm,
            "stageb_handoff_quality_tolerance": stageb_handoff_quality_tolerance,
            "stageb_paired_verifier_seed": stageb_paired_verifier_seed,
            "stageb_verifier_candidates": stageb_verifier_candidates,
            "stageb_tree_children_per_step": max(
                int(mcts_expand_actions_per_step), 1
            ),
        }
    )
    resolved_handoff_top_k = max(
        int(
            mcts_handoff_top_k
            if mcts_handoff_top_k is not None
            else min(int(num_beams), 2)
        ),
        1,
    )
    checkpoint_interval = int(esmfold2_stagea_handoff_checkpoint_interval)
    if checkpoint_interval < 0:
        raise ValueError("esmfold2_stagea_handoff_checkpoint_interval must be non-negative")
    if (
        esmfold2_initial_logits_checkpoint is not None
        and esmfold2_resume_handoff_checkpoint is not None
    ):
        raise ValueError(
            "esmfold2_initial_logits_checkpoint and "
            "esmfold2_resume_handoff_checkpoint are mutually exclusive"
        )
    if esmfold2_resume_reverify_roots and esmfold2_resume_handoff_checkpoint is None:
        raise ValueError(
            "esmfold2_resume_reverify_roots requires a StageA handoff checkpoint"
        )
    artifact_recorder = _make_artifact_recorder(
        artifact_dir,
        save_verified_candidate_cif=save_verified_candidate_cif,
        save_visualization_trace=save_visualization_trace,
    )

    tree = StageATree(
        C=mcts_C,
        max_nodes=mcts_max_nodes,
        selection_mode=stagea_root_selection_mode,
        min_root_expansions=stagea_min_root_expansions,
    )
    verifier_cache: Dict[str, Any] = {}
    resume_state = None
    if esmfold2_resume_handoff_checkpoint is not None:
        resume_state = _load_stagea_handoff_checkpoint(
            esmfold2_resume_handoff_checkpoint,
            target_seq=target_seq,
            L_binder=L_binder,
            mutable_mask=mutable_mask,
            hotspot_params=hotspot_params,
            verifier_contract=_stagea_handoff_verifier_contract(backend_config),
            allow_verifier_mismatch=bool(esmfold2_resume_reverify_roots),
            init_binder_seq=init_binder_seq,
        )
        completed_steps = int(resume_state["completed_stagea_steps"])
        if completed_steps >= N_steps:
            raise ValueError(
                "N_steps must exceed the checkpoint's completed StageA steps "
                f"({completed_steps}) so the resumed search has at least one step"
            )
        if not esmfold2_resume_reverify_roots:
            for entry in resume_state["handoff_entries"]:
                result = _deserialize_handoff_result(dict(entry["result"]))
                verifier_cache[str(result.binder_seq)] = result

    logger.info("=" * 70)
    logger.info("ESMFold2 MCTS closed loop")
    logger.info("Target length key=%s | binder length=%d | steps=%d", target_key, L_binder, N_steps)
    logger.info(
        "K_top=%d B=%d T_sample=%.3f max_mut=%d stagea_max_mut=%d stageb_max_mut=%d",
        K_top,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
        stagea_max_mutations,
        stageb_max_mutations,
    )
    logger.info(
        "strategy=%s bootstrap_steps=%d max=%d logit_init=%s",
        strategy,
        mcts_bootstrap_steps,
        mcts_bootstrap_max_steps,
        logit_init_mode,
    )
    logger.info(
        "ESMFold2 soft_steps: stage_a=%s stage_b=%s",
        str(stagea_soft_steps or "backend_default"),
        str(stageb_soft_steps or "backend_default"),
    )
    logger.info(
        "ESMFold2 StageA soft state: preserve=%s cumulative_annealing=%s "
        "horizon=%d soft_anchor_margin=%.3f children=%d",
        stagea_preserve_soft_state,
        stagea_cumulative_annealing,
        stagea_annealing_horizon,
        stagea_soft_anchor_margin,
        stagea_tree_children,
    )
    logger.info(
        "ESMFold2 StageA root selection: mode=%s min_root_expansions=%d",
        stagea_root_selection_mode,
        stagea_min_root_expansions,
    )
    logger.info(
        "ESMFold2 StageA verifier seed: %s",
        str(
            stagea_shared_verify_seed
            if stagea_paired_verifier_seed
            else "sequence_specific"
        ),
    )
    if resume_state is not None:
        logger.info(
            "Resuming %s from StageA handoff checkpoint: %s "
            "(completed_stagea_steps=%d reverify_roots=%s verifier_contract_matches=%s)",
            "StageB",
            resume_state["checkpoint_path"],
            int(resume_state["completed_stagea_steps"]),
            bool(esmfold2_resume_reverify_roots),
            bool(resume_state["verifier_contract_matches"]),
        )
    elif esmfold2_initial_logits_checkpoint is not None:
        logger.info(
            "Loading exact Phase0 logits from checkpoint source: %s",
            esmfold2_initial_logits_checkpoint,
        )
    logger.info(
        "ESMFold2 hotspot: indices_0based=%s weight=%.4f "
        "satisfied_weight=%.4f cutoff=%.2fA interface_mode=%s enabled=%s",
        hotspot_indices or "none",
        hotspot_weight,
        hotspot_satisfied_proposal_weight,
        hotspot_cutoff,
        hotspot_interface_mode,
        hotspot_summary["hotspot_enabled"],
    )
    logger.info(
        "ESMFold2 hotspot verifier: attraction_weight=%.4f near=%.2fA "
        "far=%.2fA min_safe=%.2fA too_close_weight=%.4f enabled=%s",
        hotspot_verify_weight,
        hotspot_verify_near_cutoff,
        hotspot_verify_far_cutoff,
        hotspot_verify_min_distance,
        hotspot_verify_too_close_weight,
        hotspot_summary["hotspot_verify_enabled"],
    )
    logger.info(
        "ESMFold2 StageB V5: verify_candidates=%d tree_children=%d "
        "handoff_roots=%d paired_verify_seed=%s quality_tolerance=%.3f "
        "coverage_weight=%.3f",
        stageb_verifier_candidates,
        max(int(mcts_expand_actions_per_step), 1),
        resolved_handoff_top_k,
        str(
            stageb_shared_verify_seed
            if stageb_paired_verifier_seed
            else "sequence_specific"
        ),
        stageb_handoff_quality_tolerance,
        stageb_hotspot_coverage_weight,
    )
    logger.info("=" * 70)

    initial_logits_checkpoint_paths: List[str] = []
    if resume_state is None and esmfold2_initial_logits_checkpoint is not None:
        phase0_states, initial_logits_checkpoint_paths = (
            _load_initial_logits_checkpoint_states(
                esmfold2_initial_logits_checkpoint,
                L_binder=L_binder,
                num_beams=num_beams,
                init_binder_seq=init_binder_seq,
                mutable_mask=mutable_mask,
            )
        )
    elif resume_state is None:
        phase0_states = _make_phase0_seed_states(
            init_binder_seq,
            num_beams=num_beams,
            L_binder=L_binder,
            seed=seed,
            mutable_mask=mutable_mask,
            logit_init_mode=logit_init_mode,
            device=getattr(backend, "device", None) if logit_init_mode == "official_exact" else None,
        )
    else:
        phase0_states = [
            (
                str(entry["result"]["binder_seq"]),
                _minimally_align_proposal_logits(
                    entry["proposal_logits"].detach().cpu().clone(),
                    str(entry["result"]["binder_seq"]),
                    mutable_mask,
                ),
            )
            for entry in resume_state["handoff_entries"]
        ]
    phase0_logits_by_sequence = {
        str(seq): logits.detach().cpu().clone()
        for seq, logits in phase0_states
        if logits is not None
    }
    phase0_results = []
    phase0_artifact_stage = "handoff_resume" if resume_state is not None else "phase0"
    for idx, (binder_seq, _) in enumerate(phase0_states):
        phase0_verify_seed = (
            stageb_shared_verify_seed
            if resume_state is not None and stageb_paired_verifier_seed
            else (
                stagea_shared_verify_seed
                if stagea_paired_verifier_seed
                else _stable_seed(seed, "esm_phase0", target_key, binder_seq)
            )
        )
        result = _verify_with_cache(
            backend,
            verifier_cache,
            binder_seq=binder_seq,
            target_seq=target_seq,
            seed=phase0_verify_seed,
            hotspot_params=hotspot_params,
        )
        phase0_results.append(result)
        _write_candidate_artifact(
            artifact_recorder,
            stage=phase0_artifact_stage,
            step=None,
            parent_sequence=None,
            result=result,
        )
        logger.info(
            "%s %d/%d | seq=%s L=%.4f",
            "ESM Handoff root" if resume_state is not None else "ESM Phase0",
            idx + 1,
            len(phase0_states),
            binder_seq[:20],
            result.total_loss,
        )

    phase0_results.sort(key=lambda result: result.total_loss)
    proposal_logits_by_sequence: Dict[str, torch.Tensor] = {}
    stagea_node_by_sequence: Dict[str, Any] = {}
    for result in phase0_results[: max(int(num_beams), 1)]:
        root_logits = phase0_logits_by_sequence.get(str(result.binder_seq))
        if root_logits is None:
            root_logits = _initial_proposal_logits(result.binder_seq, mutable_mask)
        root_node = tree.add_root_node(
            sequence=result.binder_seq,
            logits=root_logits,
            L_real=result.total_loss,
            coordinate=result.coordinate,
        )
        stagea_node_by_sequence[str(root_node.sequence)] = root_node
        proposal_logits_by_sequence[str(result.binder_seq)] = root_logits
        artifact_recorder.write_mcts_node(
            **_mcts_node_artifact(
                stage="stage_a",
                node=root_node,
                step_created=None,
                extra={
                    "source": (
                        "handoff_checkpoint_root" if resume_state is not None else "phase0_root"
                    )
                },
            )
        )

    best_result = phase0_results[0]
    best_seq = str(best_result.binder_seq)
    best_loss = float(best_result.total_loss)
    # Phase0 sequences are already row-wise argmaxes of their own logits.
    # Preserve official_free/exact uncertainty verbatim so handoff, resume,
    # do not silently re-anchor mutable positions to 2.0.
    best_logits = proposal_logits_by_sequence[best_seq].detach().cpu().clone()
    proposal_logits_by_sequence[best_seq] = best_logits
    losses_history: List[float] = (
        [float(value) for value in resume_state["losses_history"]]
        if resume_state is not None
        else []
    )
    latest_handoff_checkpoint: Optional[Path] = None

    def _save_handoff_state(*, final: bool = False) -> Optional[Path]:
        nonlocal latest_handoff_checkpoint
        if resume_state is not None:
            return None
        completed_steps = len(losses_history)
        if not final and (
            checkpoint_interval == 0
            or completed_steps == 0
            or completed_steps % checkpoint_interval != 0
        ):
            return None
        latest_handoff_checkpoint = _save_stagea_handoff_checkpoint(
            artifact_dir=artifact_dir,
            completed_stagea_steps=completed_steps,
            final=final,
            target_seq=target_seq,
            L_binder=L_binder,
            mutable_mask=mutable_mask,
            hotspot_params=hotspot_params,
            seed=seed,
            handoff_top_k=resolved_handoff_top_k,
            verifier_cache=verifier_cache,
            proposal_logits_by_sequence=proposal_logits_by_sequence,
            losses_history=losses_history,
            stagea_config={
                "K_top": K_top,
                "B_candidates": B_candidates,
                "T_sample": T_sample,
                "stagea_max_mutations_per_seq": stagea_max_mutations,
                "stagea_mutation_fraction": stagea_mutation_fraction,
                "stagea_candidate_strategy": stagea_candidate_strategy,
                **({
                    "stagea_dominance_gap": float(esmfold2_stagea_dominance_gap),
                    "stagea_protected_mutation_candidates": esmfold2_stagea_protected_mutation_candidates,
                } if stagea_candidate_strategy == "positive_score_guarded" else {}),
                "stagea_include_argmax_candidate": (
                    stagea_include_argmax_candidate
                ),
                "stagea_mutable_residue_count": stagea_mutable_residue_count,
                "num_beams": num_beams,
                "mcts_C": mcts_C,
                "mcts_max_nodes": mcts_max_nodes,
                "mcts_bootstrap_steps": mcts_bootstrap_steps,
                "mcts_bootstrap_patience": mcts_bootstrap_patience,
                "mcts_bootstrap_min_delta": mcts_bootstrap_min_delta,
                "mcts_bootstrap_max_steps": mcts_bootstrap_max_steps,
                "esmfold2_logit_init_mode": logit_init_mode,
                "esmfold2_stagea_soft_steps": stagea_soft_steps,
                "esmfold2_stagea_tree_children": stagea_tree_children,
                "esmfold2_stagea_root_selection_mode": (
                    stagea_root_selection_mode
                ),
                "esmfold2_stagea_min_root_expansions": (
                    stagea_min_root_expansions
                ),
                "esmfold2_stagea_preserve_soft_state": stagea_preserve_soft_state,
                "esmfold2_stagea_cumulative_annealing": stagea_cumulative_annealing,
                "esmfold2_stagea_annealing_horizon": stagea_annealing_horizon,
                "esmfold2_stagea_soft_anchor_margin": stagea_soft_anchor_margin,
                "esmfold2_stagea_paired_verifier_seed": (
                    stagea_paired_verifier_seed
                ),
                "esmfold2_stagea_verifier_seed": stagea_shared_verify_seed,
                "hotspot_weight": hotspot_weight,
                "hotspot_satisfied_proposal_weight": (
                    hotspot_satisfied_proposal_weight
                ),
                "hotspot_cutoff": hotspot_cutoff,
                "hotspot_interface_mode": hotspot_interface_mode,
            },
            verifier_contract=_stagea_handoff_verifier_contract(backend_config),
        )
        return latest_handoff_checkpoint

    def _remember_global_best(
        result: Any,
        logits: Optional[torch.Tensor] = None,
        *,
        anchor_logits: bool = True,
    ) -> None:
        nonlocal best_result, best_seq, best_loss, best_logits
        seq = str(result.binder_seq)
        source_logits = logits
        if source_logits is None:
            source_logits = proposal_logits_by_sequence.get(seq)
        best_result = result
        best_seq = seq
        best_loss = float(result.total_loss)
        if source_logits is not None and not anchor_logits:
            best_logits = source_logits.detach().cpu().clone()
        else:
            best_logits = _anchor_proposal_logits(
                source_logits,
                seq,
                mutable_mask,
            ).detach().cpu().clone()
        proposal_logits_by_sequence[seq] = best_logits

    artifact_recorder.write_step(
        stage=phase0_artifact_stage,
        step=None,
        candidate_count=len(phase0_results),
        round_best_loss=float(best_result.total_loss),
        global_best_loss=best_loss,
        best_sequence=best_seq,
        selected_sequence=None,
        logit_init_mode=logit_init_mode,
    )
    artifact_recorder.write_checkpoint(
        stage=phase0_artifact_stage,
        step=0,
        label="round_best",
        result=best_result,
        is_global_best=True,
    )




    if resume_state is not None:
        stagea_start_step = 0
        stagea_steps = 0
    else:
        stagea_start_step = 0
        min_bootstrap = max(int(mcts_bootstrap_steps), 0)
        max_bootstrap = max(int(mcts_bootstrap_max_steps), min_bootstrap)
        stagea_steps = min(N_steps, max_bootstrap)
        if min_bootstrap > 0:
            stagea_steps = min(N_steps, max(stagea_steps, min_bootstrap))

    stagea_best = best_loss
    stagea_stale = 0
    for step in range(stagea_start_step, stagea_steps):
        selected_node = tree.select()
        root_selection = tree.last_selection
        current_seq = selected_node.sequence
        proposal_hotspot_weight = _hotspot_proposal_weight_for_verified_state(
            verifier_cache.get(str(current_seq)),
            configured_weight=hotspot_weight,
            satisfied_weight=hotspot_satisfied_proposal_weight,
        )
        temperature_step_offset = (
            int(getattr(selected_node, "soft_step", 0))
            if stagea_cumulative_annealing
            else 0
        )
        temperature_total_steps = (
            stagea_annealing_horizon if stagea_cumulative_annealing else None
        )
        proposal_seed = _stable_seed(seed, "esm_propose", target_key, current_seq, step)
        proposal_batch = _call_backend_propose(
            backend,
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=step,
            K_top=K_top,
            B_candidates=B_candidates,
            T_sample=T_sample,
            max_mutations_per_seq=stagea_max_mutations,
            mutation_fraction=stagea_mutation_fraction,
            candidate_strategy=stagea_candidate_strategy,
            dominance_gap=esmfold2_stagea_dominance_gap,
            protected_mutation_candidates=esmfold2_stagea_protected_mutation_candidates,
            include_argmax_candidate=stagea_include_argmax_candidate,
            seed=proposal_seed,
            action_group=None,
            mutable_mask=mutable_mask,
            proposal_logits=selected_node.logits,
            reference_coordinates=selected_node.coordinate,
            hotspot_params=hotspot_params,
            hotspot_weight_override=proposal_hotspot_weight,
            track_confidence_override=False,
            soft_steps_override=stagea_soft_steps,
            temperature_step_offset=temperature_step_offset,
            temperature_total_steps=temperature_total_steps,
            preserve_proposal_logits=stagea_preserve_soft_state,
        )
        _write_soft_step_artifacts(
            artifact_recorder,
            stage="stage_a",
            step=step,
            selected_sequence=current_seq,
            batch=proposal_batch,
        )
        _write_proposal_visualization_trace(
            artifact_recorder,
            stage="stage_a",
            step=step,
            selected_sequence=current_seq,
            batch=proposal_batch,
            mutable_mask=mutable_mask,
        )
        if proposal_batch.logits_after is not None:
            argmax_sequence = sequence_from_logits_argmax(
                proposal_batch.logits_after,
                current_seq=current_seq,
                mutable_mask=mutable_mask,
            )
        else:
            argmax_sequence = None
        candidates = _proposal_sequences_from_batch(proposal_batch, include_confidence=False)
        candidates = [
            _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
            for seq in candidates
        ]
        candidates = [seq for seq in candidates if seq != current_seq]
        candidate_limit = max(
            int(B_candidates),
            0,
        )
        candidates = list(dict.fromkeys(candidates))[:candidate_limit]
        proposal_metadata_by_sequence = {
            str(item.get("sequence")): dict(item)
            for item in list(
                getattr(proposal_batch, "candidate_metadata", []) or []
            )
            if item.get("sequence")
        }
        if not candidates:
            logger.warning("ESM StageA step %d: no candidates from backend", step)
            losses_history.append(best_loss)
            _save_handoff_state()
            continue

        round_results = []
        for cand_seq in candidates:
            result = _verify_with_cache(
                backend,
                verifier_cache,
                binder_seq=cand_seq,
                target_seq=target_seq,
                seed=(
                    stagea_shared_verify_seed
                    if stagea_paired_verifier_seed
                    else _stable_seed(seed, "esm_verify", target_key, cand_seq)
                ),
                hotspot_params=hotspot_params,
            )
            round_results.append(result)
            _write_candidate_artifact(
                artifact_recorder,
                stage="stage_a",
                step=step,
                parent_sequence=current_seq,
                result=result,
                proposal_metadata=proposal_metadata_by_sequence.get(cand_seq),
            )

        tree_results = _select_stagea_tree_results(
            round_results,
            argmax_sequence=argmax_sequence,
            max_children=stagea_tree_children,
        )
        new_nodes = tree.expand(selected_node, tree_results)
        for node in new_nodes:
            node.logits = _stagea_child_logits(
                logits_after=proposal_batch.logits_after,
                parent_sequence=str(current_seq),
                child_sequence=str(node.sequence),
                argmax_sequence=argmax_sequence,
                mutable_mask=mutable_mask,
                preserve_soft_state=stagea_preserve_soft_state,
                soft_anchor_margin=stagea_soft_anchor_margin,
            )
            node.soft_step = (
                int(getattr(selected_node, "soft_step", 0))
                + stagea_soft_step_count
            )
            proposal_logits_by_sequence[str(node.sequence)] = node.logits
            stagea_node_by_sequence[str(node.sequence)] = node
        tree.backpropagate(
            selected_node,
            [result.total_loss for result in tree_results],
        )
        for node in new_nodes:
            artifact_recorder.write_mcts_node(
                **_mcts_node_artifact(
                    stage="stage_a",
                    node=node,
                    step_created=step,
                    extra={
                        "source": "stage_a_expand",
                        "soft_state_kind": (
                            "raw_argmax"
                            if argmax_sequence is not None
                            and str(node.sequence) == str(argmax_sequence)
                            else (
                                "minimal_sequence_alignment"
                                if stagea_preserve_soft_state
                                else "full_anchor"
                            )
                        ),
                    },
                )
            )
            artifact_recorder.write_mcts_edge(
                **_mcts_edge_artifact(
                    stage="stage_a",
                    parent=selected_node,
                    child=node,
                    step_created=step,
                )
            )
        tree.prune()

        round_best = min(round_results, key=lambda result: result.total_loss)
        if float(round_best.total_loss) < best_loss:
            round_best_logits = _stagea_child_logits(
                logits_after=proposal_batch.logits_after,
                parent_sequence=str(current_seq),
                child_sequence=str(round_best.binder_seq),
                argmax_sequence=argmax_sequence,
                mutable_mask=mutable_mask,
                preserve_soft_state=stagea_preserve_soft_state,
                soft_anchor_margin=stagea_soft_anchor_margin,
            )
            _remember_global_best(
                round_best,
                logits=round_best_logits,
                anchor_logits=(
                    not stagea_preserve_soft_state
                    and (
                        argmax_sequence is None
                        or str(round_best.binder_seq) != str(argmax_sequence)
                    )
                ),
            )
        if best_loss < stagea_best - float(mcts_bootstrap_min_delta):
            stagea_best = best_loss
            stagea_stale = 0
        else:
            stagea_stale += 1
        losses_history.append(best_loss)
        _save_handoff_state()
        artifact_recorder.write_step(
            stage="stage_a",
            step=step,
            selected_node_id=int(selected_node.node_id),
            selected_root_id=root_selection.get("selected_root_id"),
            root_selection_mode=root_selection.get("selection_mode"),
            root_selection_phase=root_selection.get("selection_phase"),
            selected_root_expansions=root_selection.get(
                "selected_root_expansions"
            ),
            root_score=root_selection.get("root_score"),
            root_expansion_counts=root_selection.get(
                "root_expansion_counts"
            ),
            selected_sequence=current_seq,
            candidate_count=len(round_results),
            expanded_child_count=len(new_nodes),
            argmax_sequence=argmax_sequence,
            argmax_child_in_tree=any(
                str(node.sequence) == str(argmax_sequence)
                for node in new_nodes
            ) if argmax_sequence is not None else False,
            proposal_hotspot_weight=proposal_hotspot_weight,
            soft_step_start=temperature_step_offset,
            soft_step_end=temperature_step_offset + stagea_soft_step_count,
            temperature_total_steps=temperature_total_steps,
            preserve_soft_state=stagea_preserve_soft_state,
            stagea_mutation_fraction=stagea_mutation_fraction,
            stagea_candidate_strategy=stagea_candidate_strategy,
            stagea_include_argmax_candidate=stagea_include_argmax_candidate,
            stagea_mutable_residue_count=stagea_mutable_residue_count,
            stagea_sampled_mutation_cap=stagea_max_mutations,
            proposal_diagnostics=dict(
                getattr(proposal_batch, "proposal_diagnostics", {}) or {}
            ),
            proposal_candidate_metadata=list(
                getattr(proposal_batch, "candidate_metadata", []) or []
            ),
            round_best_loss=float(round_best.total_loss),
            global_best_loss=best_loss,
            best_sequence=best_seq,
            **_proposal_confidence_summary(proposal_batch),
        )
        artifact_recorder.write_checkpoint(
            stage="stage_a",
            step=step,
            label="round_best",
            result=round_best,
            selected_sequence=current_seq,
            parent_sequence=current_seq,
            is_global_best=(str(round_best.binder_seq) == best_seq),
        )

        logger.info(
            "ESM StageA step %3d/%d | selected=%s verified=%d expanded=%d "
            "round_best=%.4f global_best=%.4f",
            step,
            N_steps,
            current_seq[:20],
            len(round_results),
            len(new_nodes),
            round_best.total_loss,
            best_loss,
        )
        if (
            strategy == "two_stage"
            and step + 1 >= max(int(mcts_bootstrap_steps), 0)
            and stagea_stale >= max(int(mcts_bootstrap_patience), 1)
        ):
            logger.info("ESM StageA bootstrap stopped by patience at step %d", step)
            break

    _save_handoff_state(final=True)

    if len(losses_history) >= N_steps:
        final_seq = best_seq
        artifact_recorder.write_checkpoint(
            stage="final",
            step=len(losses_history),
            label="best",
            result=best_result,
            is_global_best=True,
        )
        artifact_recorder.write_summary(
            best_sequence=best_seq,
            final_sequence=final_seq,
            best_loss=best_loss,
            n_steps=len(losses_history),
            strategy=strategy,
            artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
            stagea_handoff_checkpoint=(
                str(latest_handoff_checkpoint) if latest_handoff_checkpoint is not None else None
            ),
            initial_logits_checkpoint_source=(
                str(Path(esmfold2_initial_logits_checkpoint).expanduser())
                if esmfold2_initial_logits_checkpoint is not None
                else None
            ),
            initial_logits_checkpoint_paths=initial_logits_checkpoint_paths,
            stagea_soft_state={
                "preserve_soft_state": stagea_preserve_soft_state,
                "cumulative_annealing": stagea_cumulative_annealing,
                "annealing_horizon": stagea_annealing_horizon,
                "soft_steps_per_expansion": stagea_soft_step_count,
                "soft_anchor_margin": stagea_soft_anchor_margin,
                "tree_children": stagea_tree_children,
            },
            stagea_root_selection={
                "mode": stagea_root_selection_mode,
                "min_root_expansions": stagea_min_root_expansions,
                "root_expansion_counts": tree.root_expansion_counts,
            },
            **hotspot_summary,
        )
        return best_seq, final_seq, losses_history

    handoff_k = resolved_handoff_top_k
    handoff_results = _top_handoff_results(
        verifier_cache,
        top_k=handoff_k,
        min_distance=1,
    )
    stageb_verifier_cache: Dict[str, Any] = (
        {} if stageb_paired_verifier_seed else verifier_cache
    )
    if stageb_paired_verifier_seed:
        if resume_state is not None and esmfold2_resume_reverify_roots:
            stageb_verifier_cache.update(
                {
                    str(result.binder_seq): result
                    for result in handoff_results
                }
            )
        else:
            reverified_handoff_results = []
            for source_result in handoff_results:
                reverified = _verify_with_cache(
                    backend,
                    stageb_verifier_cache,
                    binder_seq=str(source_result.binder_seq),
                    target_seq=target_seq,
                    seed=stageb_shared_verify_seed,
                    hotspot_params=hotspot_params,
                )
                reverified_handoff_results.append(reverified)
                _write_candidate_artifact(
                    artifact_recorder,
                    stage="stage_b_handoff_reverify",
                    step=len(losses_history),
                    parent_sequence=None,
                    result=reverified,
                )
            handoff_results = reverified_handoff_results

    strict_tree = _ESMFold2StageBMCTSTree(
        C=mcts_C,
        max_nodes=mcts_max_nodes,
        max_children_per_node=max(int(B_candidates), 1),
        expand_actions_per_step=max(int(mcts_expand_actions_per_step), 1),
        use_pareto_pruning=False,
        use_diversity_bonus=False,
        use_structural_grouping=False,
        hotspot_distance_weight=stageb_hotspot_distance_weight,
        hotspot_coverage_weight=stageb_hotspot_coverage_weight,
        hotspot_miss_penalty=stageb_hotspot_miss_penalty,
        min_pair_chain_iptm=stageb_min_pair_chain_iptm,
    )
    for idx, result in enumerate(handoff_results):
        branch_quality_floor = max(
            stageb_min_pair_chain_iptm,
            _stageb_result_pair_chain_iptm(result)
            - stageb_handoff_quality_tolerance,
        )
        _annotate_stageb_quality_floor(result, branch_quality_floor)
        strict_tree.store_verifier_result(result)
        handoff_logits = proposal_logits_by_sequence.get(str(result.binder_seq))
        if handoff_logits is None:
            stageb_root_logits = _initial_proposal_logits(
                str(result.binder_seq),
                mutable_mask,
            )
            alignment_changed_positions = None
        else:
            stageb_root_logits = minimally_align_binder_logits_to_sequence(
                handoff_logits,
                str(result.binder_seq),
                mutable_mask,
            )
            source_handoff_logits = (
                handoff_logits[0]
                if handoff_logits.ndim == 3
                else handoff_logits
            ).detach().cpu()
            alignment_changed_positions = sum(
                not torch.equal(source_handoff_logits[pos], stageb_root_logits[pos])
                for pos in range(int(L_binder))
            )
        logger.info(
            "ESM StageB handoff root rank=%d minimal_logit_alignment_changed_positions=%s",
            idx + 1,
            str(alignment_changed_positions),
        )
        proposal_logits_by_sequence[str(result.binder_seq)] = stageb_root_logits
        root_node = strict_tree.add_root_node(
            sequence=result.binder_seq,
            logits=stageb_root_logits,
            L_real=result.total_loss,
            coordinate=result.coordinate,
            loss_vector=result.loss_dict,
            prior=1.0 / max(len(handoff_results), 1),
            status_flags={
                "esm_stage": "handoff",
                "handoff_rank": idx + 1,
                "minimal_logit_alignment_changed_positions": (
                    alignment_changed_positions
                ),
                ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY: branch_quality_floor,
            },
        )
        source_node = stagea_node_by_sequence.get(str(result.binder_seq))
        artifact_recorder.write_mcts_node(
            **_mcts_node_artifact(
                stage="stage_b",
                node=root_node,
                step_created=len(losses_history),
                extra={
                    "source": "handoff_root",
                    "handoff_rank": idx + 1,
                    "source_stage": "stage_a",
                    "source_node_id": None if source_node is None else int(source_node.node_id),
                    "source_sequence_hash": _sequence_hash(str(result.binder_seq)),
                },
            )
        )
        artifact_recorder.write_mcts_handoff(
            source_stage="stage_a",
            target_stage="stage_b",
            handoff_rank=idx + 1,
            source_node_id=None if source_node is None else int(source_node.node_id),
            target_node_id=int(root_node.node_id),
            source_node_uid=None if source_node is None else f"stage_a:{int(source_node.node_id)}",
            target_node_uid=f"stage_b:{int(root_node.node_id)}",
            sequence=str(result.binder_seq),
            sequence_hash=_sequence_hash(str(result.binder_seq)),
            total_loss=float(result.total_loss),
            prior=1.0 / max(len(handoff_results), 1),
        )

    best_by_total_loss = min(
        handoff_results,
        key=lambda result: float(result.total_loss),
    )
    best_by_composite = max(
        handoff_results,
        key=strict_tree.result_reward,
    )
    final_quality_floor = max(
        stageb_min_pair_chain_iptm,
        max(_stageb_result_pair_chain_iptm(result) for result in handoff_results)
        - stageb_handoff_quality_tolerance,
    )
    best_by_guarded = max(
        handoff_results,
        key=lambda result: _stageb_guarded_rank(
            result,
            strict_tree,
            quality_floor=final_quality_floor,
        ),
    )
    _remember_global_best(
        best_by_guarded,
        logits=proposal_logits_by_sequence.get(str(best_by_guarded.binder_seq)),
        anchor_logits=False,
    )

    for step in range(len(losses_history), N_steps):
        selection = strict_tree.select_decision()
        selected_node = selection.node
        current_seq = selected_node.sequence
        proposal_hotspot_weight = _hotspot_proposal_weight_for_verified_state(
            selected_node.loss_vector,
            configured_weight=hotspot_weight,
            satisfied_weight=hotspot_satisfied_proposal_weight,
        )

        if selected_node.loss_vector is None:
            raise ValueError("ESMFold2 Stage-B node is missing full-verifier loss metrics")
        routing = select_esmfold2_action_group(selected_node.loss_vector, temperature=1.0)
        action_group = str(routing["action_group"])
        group_priors = {
            str(group): float(prior)
            for group, prior in dict(routing.get("group_priors", {})).items()
        }
        action_group, group_priors = _apply_stageb_group_prior_mode(
            mcts_prior_mode,
            action_group,
            group_priors,
        )
        use_gradient_prior = _normalize_prior_mode(mcts_prior_mode) not in {
            "uniform",
            "no_gradient",
        }
        group_prior = float(
            group_priors.get(
                action_group,
                max(group_priors.values()) if group_priors else 1.0,
            )
        )

        tree_child_budget = min(
            max(int(mcts_expand_actions_per_step), 1),
            strict_tree.remaining_expansion_slots(selected_node),
        )
        if tree_child_budget <= 0:
            losses_history.append(best_loss)
            continue

        proposal_pool_budget = max(
            B_candidates,
            stageb_verifier_candidates,
            stageb_rerank_pool_size if stageb_discrete_rerank else 0,
        )
        group_budgets = _allocate_group_budgets(group_priors, proposal_pool_budget)
        proposal_call_specs = _stageb_proposal_call_specs(
            group_budgets,
            mcts_prior_mode,
        )
        proposal_pool: List[ESMFold2MutationProposal] = []
        rerank_preview_cache: Dict[str, Dict[str, float]] = {}
        rerank_seed = _stable_seed(
            seed,
            "esm_stageb_discrete_rerank",
            target_key,
            current_seq,
            step,
        )
        for group_name, group_budget, proposal_seed_label in proposal_call_specs:
            proposal_batch = _call_backend_propose(
                backend,
                current_seq=current_seq,
                target_seq=target_seq,
                global_step=step,
                K_top=K_top,
                B_candidates=group_budget,
                T_sample=T_sample,
                max_mutations_per_seq=stageb_max_mutations,
                seed=_stable_seed(
                    seed,
                    "esm_stageb_propose",
                    target_key,
                    current_seq,
                    step,
                    proposal_seed_label,
                ),
                action_group=group_name,
                mutable_mask=mutable_mask,
                proposal_logits=selected_node.logits,
                reference_coordinates=selected_node.coordinate,
                hotspot_params=hotspot_params,
                hotspot_weight_override=proposal_hotspot_weight,
                track_confidence_override=False,
                soft_steps_override=stageb_soft_steps,
            )
            _write_soft_step_artifacts(
                artifact_recorder,
                stage="stage_b",
                step=step,
                selected_sequence=current_seq,
                batch=proposal_batch,
                action_group=group_name,
            )
            _write_proposal_visualization_trace(
                artifact_recorder,
                stage="stage_b",
                step=step,
                selected_sequence=current_seq,
                batch=proposal_batch,
                mutable_mask=mutable_mask,
                action_group=group_name,
                trace_label=proposal_seed_label,
            )
            candidates = _proposal_sequences_from_batch(proposal_batch, include_confidence=False)
            candidates = [
                _clip_sequence_by_mutable_mask(str(seq), mutable_mask, current_seq)
                for seq in candidates
            ]
            rerank_scores = (
                _rerank_stageb_candidates_with_preview(
                    backend,
                    rerank_preview_cache,
                    current_seq=current_seq,
                    candidates=candidates,
                    target_seq=target_seq,
                    action_group=group_name,
                    seed=rerank_seed,
                    hotspot_params=hotspot_params,
                )
                if stageb_discrete_rerank
                else {}
            )
            proposal_pool.extend(
                _make_stageb_proposals(
                    current_seq=current_seq,
                    candidates=candidates,
                    action_group=group_name,
                    group_prior=group_priors.get(group_name, group_prior),
                    logits_before=proposal_batch.logits_before,
                    logits_after=proposal_batch.logits_after,
                    mutable_mask=mutable_mask,
                    use_gradient_prior=use_gradient_prior,
                    rerank_scores=rerank_scores,
                )
            )

        generated_group_counts = _count_items(
            [str(proposal.action_group) for proposal in proposal_pool]
        )

        if not proposal_pool:
            logger.warning("ESM StageB step %d: no candidates from backend", step)
            losses_history.append(best_loss)
            continue

        registered_actions = strict_tree.register_action_proposals(selected_node, proposal_pool)
        if _normalize_prior_mode(mcts_prior_mode) == "no_group_matched_compute":
            registered_actions = _deduplicate_actions_by_sequence(registered_actions)
        verifier_actions, verifier_group_quotas = _select_stageb_actions_by_group_quota(
            strict_tree,
            selected_node,
            registered_actions,
            group_priors=group_priors,
            max_actions=stageb_verifier_candidates,
            exploration_actions=(
                stageb_exploration_actions if stageb_discrete_rerank else 0
            ),
            rng=random.Random(
                _stable_seed(
                    seed,
                    "esm_stageb_exploration",
                    target_key,
                    current_seq,
                    step,
                )
            ),
            limit_to_tree_slots=False,
        )
        if not verifier_actions:
            losses_history.append(best_loss)
            continue

        verified_actions: List[Any] = []
        verified_results: List[Any] = []
        verified_action_groups: List[str] = []
        branch_quality_floor = float(
            selected_node.loss_vector.get(
                ESMFOLD2_STAGEB_QUALITY_FLOOR_KEY,
                max(
                    stageb_min_pair_chain_iptm,
                    1.0
                    - float(selected_node.loss_vector["pair_chain_iptm_loss"])
                    - stageb_handoff_quality_tolerance,
                ),
            )
        )
        for action in verifier_actions:
            cached = strict_tree.get_cached_verifier_result(action.sequence)
            if cached is None:
                cached = _verify_with_cache(
                    backend,
                    stageb_verifier_cache,
                    binder_seq=action.sequence,
                    target_seq=target_seq,
                    seed=(
                        stageb_shared_verify_seed
                        if stageb_paired_verifier_seed
                        else _stable_seed(
                            seed,
                            "esm_stageb_verify",
                            target_key,
                            action.sequence,
                        )
                    ),
                    hotspot_params=hotspot_params,
                )
            _annotate_stageb_quality_floor(cached, branch_quality_floor)
            strict_tree.store_verifier_result(cached)
            _write_candidate_artifact(
                artifact_recorder,
                stage="stage_b",
                step=step,
                parent_sequence=current_seq,
                result=cached,
                action_group=str(action.action_group),
            )
            verified_actions.append(action)
            verified_results.append(cached)
            verified_action_groups.append(str(action.action_group))

        tree_actions, expand_results = _select_stageb_tree_expansions(
            verified_actions,
            verified_results,
            tree=strict_tree,
            max_children=tree_child_budget,
            allow_one_quality_exploration=True,
        )
        tree_action_ids = {int(action.action_id) for action in tree_actions}
        for action in verified_actions:
            if int(action.action_id) not in tree_action_ids:
                action.status = "evaluated_pruned"

        expand_proposals = [action.proposal for action in tree_actions]
        action_by_sequence = {
            str(action.sequence): action
            for action in tree_actions
        }
        reward_by_sequence: Dict[str, float] = {}
        new_nodes = strict_tree.expand(selected_node, expand_proposals, expand_results)
        proposal_by_sequence = {
            str(proposal.sequence): proposal
            for proposal in expand_proposals
        }
        for node in new_nodes:
            proposal = proposal_by_sequence.get(str(node.sequence))
            node.logits = _minimally_align_proposal_logits(
                getattr(proposal, "proposal_logits", None),
                node.sequence,
                mutable_mask,
            )
            proposal_logits_by_sequence[str(node.sequence)] = node.logits
            reward_by_sequence[node.sequence] = node.tree_reward
            action = action_by_sequence.get(str(node.sequence))
            artifact_recorder.write_mcts_node(
                **_mcts_node_artifact(
                    stage="stage_b",
                    node=node,
                    step_created=step,
                    extra={
                        "source": "stage_b_expand",
                        "soft_state_kind": "minimal_sequence_alignment",
                        "action_id": None if action is None else int(action.action_id),
                        "action_group": None if action is None else str(action.action_group),
                    },
                )
            )
            artifact_recorder.write_mcts_edge(
                **_mcts_edge_artifact(
                    stage="stage_b",
                    parent=selected_node,
                    child=node,
                    step_created=step,
                    extra={
                        "action_id": None if action is None else int(action.action_id),
                        "action_group": None if action is None else str(action.action_group),
                    },
                )
            )

        for result in expand_results:
            reward_by_sequence.setdefault(
                result.binder_seq,
                strict_tree.result_reward(result),
            )

        for action in verified_actions:
            expanded_into_tree = int(action.action_id) in tree_action_ids
            if expanded_into_tree and action.sequence in reward_by_sequence:
                strict_tree.backpropagate_action(
                    selection,
                    action,
                    reward_by_sequence[action.sequence],
                )
            artifact_recorder.write_mcts_action(
                **_mcts_action_artifact(
                    stage="stage_b",
                    step=step,
                    action=action,
                    selected=expanded_into_tree,
                    extra={
                        "full_verified": True,
                        "expanded_into_tree": expanded_into_tree,
                        "selection_decision_type": selection.decision_type,
                        "selection_best_action_score": selection.best_action_score,
                        "selection_frontier_score": selection.frontier_score,
                        "parent_sequence_hash": _sequence_hash(current_seq),
                    },
                )
            )

        strict_tree.prune()
        round_results = verified_results
        if round_results:
            round_best = max(
                round_results,
                key=lambda result: _stageb_guarded_rank(
                    result,
                    strict_tree,
                    quality_floor=final_quality_floor,
                ),
            )

            round_total_best = min(
                round_results,
                key=lambda result: float(result.total_loss),
            )
            if float(round_total_best.total_loss) < float(best_by_total_loss.total_loss):
                best_by_total_loss = round_total_best

            round_composite_best = max(
                round_results,
                key=strict_tree.result_reward,
            )
            if (
                strict_tree.result_reward(round_composite_best)
                > strict_tree.result_reward(best_by_composite)
            ):
                best_by_composite = round_composite_best

            if (
                _stageb_guarded_rank(
                    round_best,
                    strict_tree,
                    quality_floor=final_quality_floor,
                )
                > _stageb_guarded_rank(
                    best_by_guarded,
                    strict_tree,
                    quality_floor=final_quality_floor,
                )
            ):
                best_by_guarded = round_best
                guarded_action = next(
                    (
                        action
                        for action, result in zip(verified_actions, verified_results)
                        if str(result.binder_seq) == str(round_best.binder_seq)
                    ),
                    None,
                )
                guarded_logits = (
                    None
                    if guarded_action is None
                    else getattr(guarded_action.proposal, "proposal_logits", None)
                )
                guarded_logits = _minimally_align_proposal_logits(
                    guarded_logits,
                    str(best_by_guarded.binder_seq),
                    mutable_mask,
                )
                _remember_global_best(
                    best_by_guarded,
                    logits=guarded_logits,
                    anchor_logits=False,
                )
        losses_history.append(best_loss)
        parent_pair_chain = 1.0 - float(
            selected_node.loss_vector["pair_chain_iptm_loss"]
        )
        best_reward_gain = None
        best_pair_chain_gain = None
        if verified_results:
            best_reward_gain = max(
                strict_tree.result_reward(result) - float(selected_node.tree_reward)
                for result in verified_results
            )
            child_pair_chain_scores = [
                1.0 - float(result.loss_dict["pair_chain_iptm_loss"])
                for result in verified_results
            ]
            best_pair_chain_gain = max(child_pair_chain_scores) - parent_pair_chain
        artifact_recorder.write_step(
            stage="stage_b",
            step=step,
            selected_node_id=int(selected_node.node_id),
            selected_action_ids=[int(action.action_id) for action in tree_actions],
            full_verified_action_ids=[
                int(action.action_id) for action in verified_actions
            ],
            tree_expanded_action_ids=[
                int(action.action_id) for action in tree_actions
            ],
            selected_sequence=current_seq,
            candidate_count=len(round_results),
            full_verified_candidate_count=len(verified_results),
            tree_child_count=len(tree_actions),
            round_best_loss=float(round_best.total_loss) if round_results else None,
            global_best_loss=best_loss,
            best_sequence=best_seq,
            primary_action_group=action_group,
            group_prior=group_prior,
            group_scores=dict(routing.get("group_scores", {})),
            group_priors=group_priors,
            group_budgets=group_budgets,
            proposal_call_budgets={
                seed_label: budget
                for _, budget, seed_label in proposal_call_specs
            },
            verifier_group_quotas=verifier_group_quotas,
            generated_group_counts=generated_group_counts,
            generated_unique_candidate_count=len(
                {str(proposal.sequence) for proposal in proposal_pool}
            ),
            verified_group_counts=_count_items(verified_action_groups),
            selected_group_counts=_count_items(
                [str(action.action_group) for action in tree_actions]
            ),
            selected_action_count=len(tree_actions),
            discrete_rerank_enabled=stageb_discrete_rerank,
            discrete_rerank_pool_size=proposal_pool_budget,
            exploration_action_count=(
                min(stageb_exploration_actions, max(len(verifier_actions) - 1, 0))
                if stageb_discrete_rerank
                else 0
            ),
            best_reward_gain=best_reward_gain,
            best_pair_chain_iptm_gain=best_pair_chain_gain,
            proposal_hotspot_weight=proposal_hotspot_weight,
            loss_vector_keys=sorted(str(key) for key in selected_node.loss_vector.keys()),
        )
        if round_results:
            artifact_recorder.write_checkpoint(
                stage="stage_b",
                step=step,
                label="round_best",
                result=round_best,
                selected_sequence=current_seq,
                parent_sequence=current_seq,
                is_global_best=(str(round_best.binder_seq) == best_seq),
            )

        logger.info(
            "ESM StageB step %3d/%d | selected=%s primary_group=%s Pg=%.3f scores=%s priors=%s budgets=%s generated=%s verified=%s selected_groups=%s reward_gain=%s pair_iptm_gain=%s actions=%d global_best=%.4f",
            step,
            N_steps,
            current_seq[:20],
            action_group,
            group_prior,
            _format_float_map(dict(routing.get("group_scores", {}))),
            _format_float_map(group_priors),
            _format_count_map(group_budgets),
            _format_count_map(generated_group_counts),
            _format_count_map(_count_items(verified_action_groups)),
            _format_count_map(
                _count_items([str(action.action_group) for action in tree_actions])
            ),
            "n/a" if best_reward_gain is None else f"{best_reward_gain:.4f}",
            "n/a" if best_pair_chain_gain is None else f"{best_pair_chain_gain:.4f}",
            len(tree_actions),
            best_loss,
        )

    final_seq = best_seq
    artifact_recorder.write_checkpoint(
        stage="final",
        step=len(losses_history),
        label="best",
        result=best_result,
        is_global_best=True,
    )
    artifact_recorder.write_summary(
        best_sequence=best_seq,
        final_sequence=final_seq,
        best_loss=best_loss,
        n_steps=len(losses_history),
        strategy=strategy,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
        stagea_handoff_checkpoint=(
            str(latest_handoff_checkpoint) if latest_handoff_checkpoint is not None else None
        ),
        resumed_from_stagea_handoff=(
            None if resume_state is None else str(resume_state["checkpoint_path"])
        ),
        resumed_handoff_roots_reverified=(
            bool(esmfold2_resume_reverify_roots) if resume_state is not None else False
        ),
        resumed_handoff_verifier_contract_matches=(
            None if resume_state is None else bool(resume_state["verifier_contract_matches"])
        ),
        initial_logits_checkpoint_source=(
            str(Path(esmfold2_initial_logits_checkpoint).expanduser())
            if esmfold2_initial_logits_checkpoint is not None
            else None
        ),
        initial_logits_checkpoint_paths=initial_logits_checkpoint_paths,
        stagea_soft_state={
            "preserve_soft_state": stagea_preserve_soft_state,
            "cumulative_annealing": stagea_cumulative_annealing,
            "annealing_horizon": stagea_annealing_horizon,
            "soft_steps_per_expansion": stagea_soft_step_count,
            "soft_anchor_margin": stagea_soft_anchor_margin,
            "tree_children": stagea_tree_children,
        },
        stagea_root_selection={
            "mode": stagea_root_selection_mode,
            "min_root_expansions": stagea_min_root_expansions,
            "root_expansion_counts": tree.root_expansion_counts,
        },
        stageb_shared_verifier_seed=(
            stageb_shared_verify_seed if stageb_paired_verifier_seed else None
        ),
        stageb_selection_surfaces={
            "best_by_total_loss": _stageb_selection_surface(
                best_by_total_loss,
                strict_tree,
                quality_floor=final_quality_floor,
            ),
            "best_by_composite_reward": _stageb_selection_surface(
                best_by_composite,
                strict_tree,
                quality_floor=final_quality_floor,
            ),
            "best_by_hotspot_guarded_quality": _stageb_selection_surface(
                best_by_guarded,
                strict_tree,
                quality_floor=final_quality_floor,
            ),
        },
        **hotspot_summary,
    )
    return best_seq, final_seq, losses_history
