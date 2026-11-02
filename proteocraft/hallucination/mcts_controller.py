"""
BindCraft v2 - MCTS 控制器

Stage 1 MCTS:
- full AF3 verifier loss vector -> normalized tree_reward
- gradient proposal -> child prior
- selection -> PUCT over the existing tree
- exact sequence deduplication via a transposition table

The tree stores only lightweight CPU state. GPU tensors and autograd graphs must
not be retained by nodes.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union

import torch

logger = logging.getLogger(__name__)


LOSS_GROUP_KEYS = (
    "total",
    "plddt",
    "pae",
    "con",
    "i_pae",
    "i_pde",
    "i_con",
    "rg",
    "clash",
    "i_clash",
)

HARD_CONSTRAINT_RULES = {
    "clash": {"epsilon": 5.0, "scale": 1.0},
    "i_clash": {"epsilon": 5.0, "scale": 1.0},
    "bond_violation": {"epsilon": 0.5, "scale": 0.5},
    "chainbreak": {"epsilon": 0.5, "scale": 0.5},
    "endpoint_loss": {"epsilon": 1.0, "scale": 1.0},
}
HARD_CONSTRAINT_INVALID_THRESHOLD = 1.0
HARD_CONSTRAINT_PENALTY_WEIGHT = 10.0
HARD_CONSTRAINT_INVALID_REWARD = -1.0e6
DIVERSITY_BONUS_WEIGHT = 0.03
DIVERSITY_BONUS_MAX = 1.0
DIVERSITY_REFERENCE_LIMIT = 64
DIVERSITY_SEQUENCE_WEIGHT = 0.65
DIVERSITY_STRUCTURAL_WEIGHT = 0.25
DIVERSITY_GROUP_WEIGHT = 0.10
REWARD_LOSS_WEIGHTS = {
    "plddt": 1.0,
    "pae": 1.0,
    "con": 1.0,
    "i_pae": 1.0,
    "i_pde": 1.0,
    "i_con": 1.0,
    "pde": 1.0,
    "rg": 0.5,
    "clash": 1.0,
    "i_clash": 1.0,
}
DEFAULT_REWARD_LOSS_SCALES = {
    "plddt": 1.0,
    "pae": 1.0,
    "con": 1.0,
    "i_pae": 1.0,
    "i_pde": 1.0,
    "i_con": 1.0,
    "pde": 1.0,
    "rg": 1.0,
    "clash": 1.0,
    "i_clash": 1.0,
    "total": 1.0,
}
PARETO_EPS = 1e-8
PARETO_LOSS_DIMENSIONS = {
    "fold": ("plddt", "pae", "pde", "con"),
    "interface": ("i_pae", "i_pde", "i_con"),
    "physical": ("clash", "i_clash", "bond_violation", "chainbreak"),
    "specificity": ("background_contact", "decoy_margin", "specificity"),
    "sequence": ("sequence_prior", "exposed_hydrophobicity", "solubility"),
    "total": ("total",),
}
STRUCTURAL_GROUP_KEYS = (
    "hotspot_contact_bitset",
    "interface_patch_id",
    "secondary_structure_pattern",
    "contact_fingerprint",
    "length_bin",
)
ROUTING_SIGNATURE_KEYS = (
    "interface_residue_indices",
    "target_facing_residue_indices",
    "hotspot_facing_residue_indices",
    "antigen_facing_residue_indices",
    "background_contact_residue_indices",
    "non_hotspot_contact_residue_indices",
    "wrong_contact_residue_indices",
    "clash_residue_indices",
    "clash_near_residue_indices",
    "steric_conflict_residue_indices",
    "surface_residue_indices",
    "non_interface_residue_indices",
    "core_residue_indices",
    "buried_residue_indices",
    "helix_bundle_residue_indices",
    "exposed_hydrophobic_residue_indices",
    "solubility_residue_indices",
)


def _cpu_detached(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def _copy_target_sequence(target_seq: Any) -> Union[str, List[str]]:
    """Preserve multi-chain target sequences while normalizing scalar values."""
    if target_seq is None:
        return ""
    if isinstance(target_seq, list):
        return [str(seq) for seq in target_seq]
    if isinstance(target_seq, tuple):
        return [str(seq) for seq in target_seq]
    return str(target_seq)


def compute_hard_constraint_status(loss_vector: Dict[str, float]) -> Dict[str, Any]:
    violations: Dict[str, float] = {}
    penalty = 0.0
    for key, rule in HARD_CONSTRAINT_RULES.items():
        if key not in loss_vector:
            continue
        value = max(float(loss_vector[key]), 0.0)
        excess = max(value - float(rule["epsilon"]), 0.0)
        if excess <= 0.0:
            continue
        scaled = excess / max(float(rule["scale"]), 1e-8)
        violations[key] = scaled
        penalty += scaled * scaled

    invalid = penalty > HARD_CONSTRAINT_INVALID_THRESHOLD
    return {
        "valid": not invalid,
        "hard_invalid": invalid,
        "hard_penalty": penalty,
        "hard_violations": violations,
    }


def compute_tree_reward(
    total_loss: float,
    loss_vector: Dict[str, float],
    base_reward: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    hard_status = compute_hard_constraint_status(loss_vector)
    if hard_status["hard_invalid"]:
        return HARD_CONSTRAINT_INVALID_REWARD, hard_status

    reward = float(-total_loss if base_reward is None else base_reward)
    reward -= HARD_CONSTRAINT_PENALTY_WEIGHT * float(hard_status["hard_penalty"])
    return reward, hard_status


class RunningLossNormalizer:
    """
    Online z-score normalizer for verifier loss components.

    Early in a run, before each component has enough observations for a stable
    variance estimate, components fall back to fixed domain scales. This keeps
    Phase 0/root rewards finite and deterministic while still moving toward the
    document's normalized loss-vector reward once the tree has samples.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        default_scales: Optional[Dict[str, float]] = None,
        min_count: int = 4,
        min_std: float = 1e-6,
    ):
        self.weights = dict(weights or REWARD_LOSS_WEIGHTS)
        self.default_scales = dict(default_scales or DEFAULT_REWARD_LOSS_SCALES)
        self.min_count = int(min_count)
        self.min_std = float(min_std)
        self._stats: Dict[str, Dict[str, float]] = {}
        self.observations: int = 0

    def observe(self, loss_vector: Dict[str, float]) -> None:
        self.observations += 1
        for key in self.weights:
            if key not in loss_vector:
                continue
            value = float(loss_vector[key])
            stat = self._stats.setdefault(key, {"count": 0.0, "mean": 0.0, "M2": 0.0})
            stat["count"] += 1.0
            delta = value - stat["mean"]
            stat["mean"] += delta / stat["count"]
            delta2 = value - stat["mean"]
            stat["M2"] += delta * delta2

    def component_stats(self, key: str) -> Dict[str, float]:
        return dict(self._stats.get(key, {"count": 0.0, "mean": 0.0, "M2": 0.0}))

    def _std(self, key: str) -> float:
        stat = self._stats.get(key)
        if stat is None or stat["count"] < 2:
            return 0.0
        variance = stat["M2"] / max(stat["count"] - 1.0, 1.0)
        return math.sqrt(max(variance, 0.0))

    def normalized_components(self, loss_vector: Dict[str, float]) -> Dict[str, float]:
        components: Dict[str, float] = {}
        for key in self.weights:
            if key not in loss_vector:
                continue
            value = float(loss_vector[key])
            stat = self._stats.get(key)
            std = self._std(key)
            if stat is not None and stat["count"] >= self.min_count and std >= self.min_std:
                components[key] = (value - stat["mean"]) / std
            else:
                components[key] = value / max(float(self.default_scales.get(key, 1.0)), 1e-8)
        return components

    def normalized_loss(self, loss_vector: Dict[str, float], total_loss: float) -> Tuple[float, Dict[str, float], str]:
        components = self.normalized_components(loss_vector)
        if components:
            weight_sum = sum(abs(float(self.weights[key])) for key in components)
            normalized = sum(float(self.weights[key]) * value for key, value in components.items())
            return normalized / max(weight_sum, 1e-8), components, "normalized_loss_vector"

        fallback = float(total_loss) / max(float(self.default_scales.get("total", 10.0)), 1e-8)
        return fallback, {"total": fallback}, "normalized_total_fallback"


@dataclass
class CachedVerifierResult:
    """CPU-only verifier result cached by exact binder sequence."""

    binder_seq: str
    target_seq: Union[str, List[str]]
    total_loss: float
    loss_dict: Dict[str, float]
    summary_confidence: Dict[str, Any]
    coordinate: Optional[torch.Tensor] = field(default=None, repr=False)
    plddt_logits: Optional[torch.Tensor] = field(default=None, repr=False)
    from_verifier_cache: bool = True


@dataclass
class EliteArchiveEntry:
    """CPU-only archive entry for a high-value candidate in one search basin."""

    node_id: int
    sequence: str
    group_id: str
    tree_reward: float
    L_real: float
    loss_vector: Dict[str, float]
    status_flags: Dict[str, Any]
    structural_signature: Dict[str, Any]
    depth: int


@dataclass
class MCTSAction:
    """One mutation action under a parent node."""

    action_id: int
    parent_node_id: int
    sequence: str
    edits: List[Tuple[int, str, str]]
    prior: float
    benefit: float = 0.0
    action_group: str = "unrouted"
    group_prior: float = 1.0
    proposal: Any = field(default=None, repr=False)
    visit_count: int = 0
    value_sum: float = 0.0
    child_node_id: Optional[int] = None
    status: str = "unexpanded"
    status_flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    @property
    def is_expanded(self) -> bool:
        return self.child_node_id is not None and self.status == "expanded"


@dataclass
class MCTSSelection:
    """Action-level selection result."""

    node: "MCTSNode"
    action: Optional[MCTSAction]
    node_path: List["MCTSNode"]
    action_path: List[MCTSAction]
    decision_type: str
    best_action_score: Optional[float] = None
    frontier_score: Optional[float] = None


@dataclass
class MCTSNode:
    """
    MCTS 树节点。每个节点对应一条经过 full AF3 verifier 的离散序列。
    """

    node_id: int
    sequence: str
    logits: torch.Tensor
    coordinate: Optional[torch.Tensor]
    L_real: float
    loss_vector: Dict[str, float] = field(default_factory=dict)
    tree_reward: float = 0.0
    value_sum: float = 0.0
    visit_count: int = 0
    prior: float = 1.0
    edit_from_parent: List[Tuple[int, str, str]] = field(default_factory=list)
    status_flags: Dict[str, Any] = field(default_factory=dict)
    structural_signature: Dict[str, Any] = field(default_factory=dict)
    group_id: str = "ungrouped"
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    children: List["MCTSNode"] = field(default_factory=list, repr=False)
    actions: List[MCTSAction] = field(default_factory=list, repr=False)
    action_sequence_index: Dict[str, int] = field(default_factory=dict, repr=False)
    depth: int = field(default=0)

    @property
    def q_value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_path_to_root(self) -> List["MCTSNode"]:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path


class MCTSTree:
    """
    MCTS 树管理器。

    Stage 1 使用 child-level PUCT:
        Q(child) + C * P(child) * sqrt(N(parent)) / (1 + N(child))
    """

    def __init__(
        self,
        C: float = 1.0,
        max_nodes: int = 5000,
        mcts_tau: float = 4.0,
        progressive_widening_k: float = 4.0,
        progressive_widening_alpha: float = 0.5,
        max_children_per_node: int = 16,
        use_loss_grouping: bool = True,
        representatives_per_group: int = 5,
        loss_group_bin_width: float = 1.0,
        use_reward_normalization: bool = True,
        use_pareto_pruning: bool = True,
        use_elite_archive: bool = True,
        elite_archive_size_per_group: int = 5,
        elite_min_sequence_distance: int = 1,
        use_structural_grouping: bool = True,
        length_group_bin_size: int = 10,
        use_diversity_bonus: bool = False,
        diversity_bonus_weight: float = DIVERSITY_BONUS_WEIGHT,
        diversity_bonus_max: float = DIVERSITY_BONUS_MAX,
        min_children_before_descend: int = 4,
        descend_score_margin: float = 0.0,
        expand_actions_per_step: int = 1,
    ):
        self.C = C
        self.max_nodes = max_nodes
        self.mcts_tau = mcts_tau
        self.progressive_widening_k = progressive_widening_k
        self.progressive_widening_alpha = progressive_widening_alpha
        self.max_children_per_node = max_children_per_node
        self.use_loss_grouping = use_loss_grouping
        self.representatives_per_group = representatives_per_group
        self.loss_group_bin_width = max(float(loss_group_bin_width), 1e-8)
        self.use_reward_normalization = bool(use_reward_normalization)
        self.use_pareto_pruning = bool(use_pareto_pruning)
        self.use_elite_archive = bool(use_elite_archive)
        self.elite_archive_size_per_group = int(elite_archive_size_per_group)
        self.elite_min_sequence_distance = int(elite_min_sequence_distance)
        self.use_structural_grouping = bool(use_structural_grouping)
        self.length_group_bin_size = max(int(length_group_bin_size), 1)
        self.use_diversity_bonus = bool(use_diversity_bonus)
        self.diversity_bonus_weight = max(float(diversity_bonus_weight), 0.0)
        self.diversity_bonus_max = max(float(diversity_bonus_max), 0.0)
        self.min_children_before_descend = max(int(min_children_before_descend), 1)
        self.descend_score_margin = float(descend_score_margin)
        self.expand_actions_per_step = max(int(expand_actions_per_step), 1)
        self.reward_normalizer = RunningLossNormalizer()

        self._nodes: Dict[int, MCTSNode] = {}
        self._sequence_to_node_id: Dict[str, int] = {}
        self._group_to_node_ids: Dict[str, List[int]] = {}
        self._verifier_cache: Dict[str, CachedVerifierResult] = {}
        self._elite_archive: Dict[str, List[EliteArchiveEntry]] = {}
        self._next_id: int = 0
        self._next_action_id: int = 0
        self._roots: List[MCTSNode] = []
        self.N_total: int = 0
        self.group_rejections: int = 0
        self.group_pruned: int = 0
        self.verifier_cache_hits: int = 0
        self.verifier_cache_stores: int = 0
        self.duplicate_merges: int = 0
        self.pareto_rejections: int = 0
        self.pareto_pruned: int = 0
        self.elite_archive_updates: int = 0
        self.elite_archive_rejections: int = 0

    def _compute_base_node_reward(
        self,
        total_loss: float,
        loss_vector: Dict[str, float],
        base_reward: Optional[float] = None,
        observe: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        loss_data = {k: float(v) for k, v in (loss_vector or {}).items()}

        if not self.use_reward_normalization:
            reward, status = compute_tree_reward(
                float(total_loss),
                loss_data,
                base_reward=base_reward,
            )
            status = dict(status)
            status["reward_mode"] = "raw_total_loss"
            status["normalized_loss"] = None
            status["reward_stats_count"] = self.reward_normalizer.observations
            return reward, status

        if observe:
            self.reward_normalizer.observe(loss_data)

        hard_status = compute_hard_constraint_status(loss_data)
        status = dict(hard_status)
        status["reward_stats_count"] = self.reward_normalizer.observations
        if hard_status["hard_invalid"]:
            normalized_loss, components, mode = self.reward_normalizer.normalized_loss(
                loss_data,
                float(total_loss),
            )
            status["reward_mode"] = mode
            status["normalized_loss"] = normalized_loss
            status["normalized_components"] = components
            return HARD_CONSTRAINT_INVALID_REWARD, status

        normalized_loss, components, mode = self.reward_normalizer.normalized_loss(
            loss_data,
            float(total_loss),
        )
        reward = -float(normalized_loss)
        reward -= HARD_CONSTRAINT_PENALTY_WEIGHT * float(hard_status["hard_penalty"])
        status["reward_mode"] = mode
        status["normalized_loss"] = normalized_loss
        status["normalized_components"] = components
        return reward, status

    def _new_action_id(self) -> int:
        action_id = self._next_action_id
        self._next_action_id += 1
        return action_id

    def compute_node_reward(
        self,
        total_loss: float,
        loss_vector: Dict[str, float],
        base_reward: Optional[float] = None,
        observe: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        return self._compute_base_node_reward(
            total_loss=total_loss,
            loss_vector=loss_vector,
            base_reward=base_reward,
            observe=observe,
        )

    def _diversity_reference_sequences(self, group_id: Optional[str]) -> List[str]:
        references: List[str] = []
        seen = set()

        def add(seq: str) -> None:
            if seq and seq not in seen:
                seen.add(seq)
                references.append(seq)

        if group_id:
            for entry in self._elite_archive.get(group_id, []):
                add(entry.sequence)
            for node in self._active_group_nodes(group_id):
                add(node.sequence)

        if not references:
            for root in self._roots:
                add(root.sequence)
            if self._nodes:
                best = max(self._nodes.values(), key=lambda node: node.tree_reward)
                add(best.sequence)

        if len(references) < DIVERSITY_REFERENCE_LIMIT:
            elite_entries = [
                entry
                for entries in self._elite_archive.values()
                for entry in entries
            ]
            elite_entries.sort(key=lambda entry: entry.tree_reward, reverse=True)
            for entry in elite_entries:
                add(entry.sequence)
                if len(references) >= DIVERSITY_REFERENCE_LIMIT:
                    break

        return references[:DIVERSITY_REFERENCE_LIMIT]

    def _normalized_sequence_novelty(
        self,
        sequence: Optional[str],
        references: List[str],
    ) -> float:
        if not sequence or not references:
            return 0.0
        distances = [
            self._sequence_distance(sequence, ref) / max(len(sequence), len(ref), 1)
            for ref in references
            if ref != sequence
        ]
        if not distances:
            return 0.0
        return min(max(min(distances), 0.0), 1.0)

    def _structural_key_from_signature(
        self,
        structural_signature: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[Tuple[str, str], ...]]:
        if not structural_signature:
            return None

        parts = []
        has_informative_value = False
        for key in STRUCTURAL_GROUP_KEYS:
            if key not in structural_signature:
                continue
            value = self._normalize_group_value(structural_signature.get(key))
            parts.append((key, value))
            if key != "length_bin" and value not in {"unknown", "none"}:
                has_informative_value = True

        if not parts or not has_informative_value:
            return None
        return tuple(parts)

    def _normalized_structural_novelty(
        self,
        structural_signature: Optional[Dict[str, Any]],
    ) -> float:
        candidate_key = self._structural_key_from_signature(structural_signature)
        if candidate_key is None:
            return 0.0

        known_keys = {
            key
            for key in (
                self._structural_key_from_signature(node.structural_signature)
                for node in self._nodes.values()
            )
            if key is not None
        }
        if not known_keys:
            return 0.0
        return 0.0 if candidate_key in known_keys else 1.0

    def compute_diversity_bonus(
        self,
        sequence: Optional[str],
        group_id: Optional[str] = None,
        structural_signature: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Lightweight diversity bonus used by the tree reward.

        The score is deliberately small and bounded. It rewards candidates that
        are not close duplicates of the current basin's elite/archive entries
        and gives a small bonus to genuinely new structural/loss groups. It is
        not a verifier term and must not override hard constraints.
        """
        status = {
            "diversity_score": 0.0,
            "diversity_bonus": 0.0,
            "diversity_sequence_score": 0.0,
            "diversity_structural_score": 0.0,
            "diversity_group_score": 0.0,
            "diversity_weight": self.diversity_bonus_weight,
        }
        if (
            not self.use_diversity_bonus
            or self.diversity_bonus_weight <= 0.0
            or self.diversity_bonus_max <= 0.0
            or not sequence
        ):
            return 0.0, status

        references = self._diversity_reference_sequences(group_id)
        sequence_score = self._normalized_sequence_novelty(sequence, references)
        structural_score = self._normalized_structural_novelty(structural_signature)
        group_score = (
            1.0
            if group_id
            and bool(self._nodes)
            and group_id not in self._group_to_node_ids
            else 0.0
        )
        score = (
            DIVERSITY_SEQUENCE_WEIGHT * sequence_score
            + DIVERSITY_STRUCTURAL_WEIGHT * structural_score
            + DIVERSITY_GROUP_WEIGHT * group_score
        )
        score = min(max(score, 0.0), self.diversity_bonus_max)
        bonus = self.diversity_bonus_weight * score

        status.update(
            {
                "diversity_score": score,
                "diversity_bonus": bonus,
                "diversity_sequence_score": sequence_score,
                "diversity_structural_score": structural_score,
                "diversity_group_score": group_score,
            }
        )
        return bonus, status

    def _apply_diversity_bonus(
        self,
        reward: float,
        status: Dict[str, Any],
        sequence: Optional[str],
        group_id: Optional[str],
        structural_signature: Optional[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        bonus, diversity_status = self.compute_diversity_bonus(
            sequence=sequence,
            group_id=group_id,
            structural_signature=structural_signature,
        )
        updated_status = dict(status)
        updated_status.update(diversity_status)
        if updated_status.get("hard_invalid", False):
            return reward, updated_status
        return float(reward) + float(bonus), updated_status

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _register_node(self, node: MCTSNode) -> None:
        self._nodes[node.node_id] = node
        self._sequence_to_node_id[node.sequence] = node.node_id
        self._group_to_node_ids.setdefault(node.group_id, []).append(node.node_id)

    def _unregister_node(self, node: MCTSNode) -> None:
        self._sequence_to_node_id.pop(node.sequence, None)
        group_ids = self._group_to_node_ids.get(node.group_id)
        if group_ids is not None:
            self._group_to_node_ids[node.group_id] = [
                node_id for node_id in group_ids if node_id != node.node_id
            ]
            if not self._group_to_node_ids[node.group_id]:
                del self._group_to_node_ids[node.group_id]
        del self._nodes[node.node_id]

    def make_loss_group_id(
        self,
        loss_vector: Optional[Dict[str, Any]],
        status_flags: Optional[Dict[str, Any]] = None,
        structural_signature: Optional[Dict[str, Any]] = None,
        sequence: Optional[str] = None,
    ) -> str:
        if not self.use_loss_grouping:
            return "ungrouped"

        losses = loss_vector or {}
        flags = status_flags or {}
        structural = self.extract_structural_signature(
            sequence=sequence,
            loss_vector=losses,
            structural_signature=structural_signature,
        )
        structural_parts = [
            f"{key}:{structural.get(key, 'unknown')}"
            for key in STRUCTURAL_GROUP_KEYS
            if key in structural
        ]
        bins = []
        for key in LOSS_GROUP_KEYS:
            if key not in losses:
                continue
            value = float(losses[key])
            bin_id = int(math.floor(max(value, 0.0) / self.loss_group_bin_width))
            bins.append(f"{key}:{bin_id}")

        if not bins:
            bins.append("lossless")

        action_group = flags.get("action_group", "unrouted")
        return "|".join([f"ag={action_group}"] + structural_parts + bins)

    def _active_group_nodes(self, group_id: str) -> List[MCTSNode]:
        return [
            self._nodes[node_id]
            for node_id in self._group_to_node_ids.get(group_id, [])
            if node_id in self._nodes
        ]

    def _should_admit_group(self, group_id: str, reward: float) -> bool:
        if not self.use_loss_grouping or self.representatives_per_group <= 0:
            return True

        representatives = self._active_group_nodes(group_id)
        if len(representatives) < self.representatives_per_group:
            return True

        worst = min(representatives, key=lambda node: node.tree_reward)
        if reward <= worst.tree_reward:
            self.group_rejections += 1
            return False
        return True

    def _enforce_group_representatives(self, group_id: str) -> int:
        if not self.use_loss_grouping or self.representatives_per_group <= 0:
            return 0

        pruned = 0
        while len(self._active_group_nodes(group_id)) > self.representatives_per_group:
            candidates = [
                node for node in self._active_group_nodes(group_id)
                if node.is_leaf() and not node.is_root()
            ]
            if not candidates:
                break
            worst = min(candidates, key=lambda node: node.tree_reward)
            self._remove_node(worst)
            pruned += 1

        self.group_pruned += pruned
        return pruned

    def make_pareto_vector(self, loss_vector: Optional[Dict[str, Any]]) -> Dict[str, float]:
        losses = loss_vector or {}
        vector: Dict[str, float] = {}
        for dimension, keys in PARETO_LOSS_DIMENSIONS.items():
            present_values = [
                float(losses[key])
                for key in keys
                if key in losses and math.isfinite(float(losses[key]))
            ]
            if not present_values:
                continue
            vector[dimension] = sum(present_values)
        return vector

    def _pareto_dominates(self, lhs: Dict[str, float], rhs: Dict[str, float]) -> bool:
        """
        Loss-space Pareto dominance. Lower is better.

        Sparse vectors are handled conservatively: lhs can dominate rhs only when
        lhs contains every dimension available in rhs. This avoids letting a
        partial verifier result reject a richer one.
        """
        if not rhs:
            return False
        for dimension in rhs:
            if dimension not in lhs:
                return False
            if lhs[dimension] > rhs[dimension] + PARETO_EPS:
                return False
        return any(lhs[dimension] < rhs[dimension] - PARETO_EPS for dimension in rhs)

    def _is_pareto_dominated(self, group_id: str, loss_vector: Dict[str, float]) -> bool:
        if not self.use_pareto_pruning:
            return False

        candidate_vector = self.make_pareto_vector(loss_vector)
        if not candidate_vector:
            return False

        for node in self._active_group_nodes(group_id):
            node_vector = self.make_pareto_vector(node.loss_vector)
            if self._pareto_dominates(node_vector, candidate_vector):
                self.pareto_rejections += 1
                logger.debug(
                    "[MCTS] Pareto reject: group=%s candidate=%s dominated_by=%d vector=%s",
                    group_id, candidate_vector, node.node_id, node_vector,
                )
                return True
        return False

    def _prune_pareto_dominated_leaves(self, group_id: str, new_node: MCTSNode) -> int:
        if not self.use_pareto_pruning:
            return 0

        new_vector = self.make_pareto_vector(new_node.loss_vector)
        if not new_vector:
            return 0

        pruned = 0
        for node in list(self._active_group_nodes(group_id)):
            if node is new_node or node.is_root() or not node.is_leaf():
                continue
            node_vector = self.make_pareto_vector(node.loss_vector)
            if self._pareto_dominates(new_vector, node_vector):
                self._remove_node(node)
                pruned += 1

        self.pareto_pruned += pruned
        if pruned:
            logger.debug(
                "[MCTS] Pareto pruned %d dominated leaves in group=%s by node=%d",
                pruned, group_id, new_node.node_id,
            )
        return pruned

    @staticmethod
    def _sequence_distance(lhs: str, rhs: str) -> int:
        overlap = min(len(lhs), len(rhs))
        mismatches = sum(1 for i in range(overlap) if lhs[i] != rhs[i])
        return mismatches + abs(len(lhs) - len(rhs))

    def _node_to_elite_entry(self, node: MCTSNode) -> EliteArchiveEntry:
        return EliteArchiveEntry(
            node_id=node.node_id,
            sequence=node.sequence,
            group_id=node.group_id,
            tree_reward=float(node.tree_reward),
            L_real=float(node.L_real),
            loss_vector=dict(node.loss_vector),
            status_flags=dict(node.status_flags),
            structural_signature=dict(node.structural_signature),
            depth=int(node.depth),
        )

    def update_elite_archive(self, node: MCTSNode) -> bool:
        """
        Keep a small, CPU-only elite set per search basin.

        The current basin key is group_id. Once structural grouping is added,
        the same archive code will automatically become structure-aware because
        group_id will include contact/patch/fingerprint information.
        """
        if (
            not self.use_elite_archive
            or self.elite_archive_size_per_group <= 0
            or node.group_id not in self._group_to_node_ids
        ):
            return False

        entries = self._elite_archive.setdefault(node.group_id, [])
        new_entry = self._node_to_elite_entry(node)

        for idx, entry in enumerate(entries):
            if entry.sequence == node.sequence:
                if node.tree_reward > entry.tree_reward:
                    entries[idx] = new_entry
                    self.elite_archive_updates += 1
                    return True
                self.elite_archive_rejections += 1
                return False

        close_indices = [
            idx for idx, entry in enumerate(entries)
            if self._sequence_distance(entry.sequence, node.sequence) <= self.elite_min_sequence_distance
        ]
        if close_indices:
            best_close_idx = max(close_indices, key=lambda idx: entries[idx].tree_reward)
            if node.tree_reward <= entries[best_close_idx].tree_reward:
                self.elite_archive_rejections += 1
                return False
            for idx in sorted(close_indices, reverse=True):
                del entries[idx]

        entries.append(new_entry)
        entries.sort(key=lambda entry: entry.tree_reward, reverse=True)
        del entries[self.elite_archive_size_per_group:]
        self.elite_archive_updates += 1
        return True

    def get_elite_archive(self, group_id: Optional[str] = None) -> Dict[str, List[EliteArchiveEntry]]:
        if group_id is not None:
            return {group_id: list(self._elite_archive.get(group_id, []))}
        return {
            archive_group_id: list(entries)
            for archive_group_id, entries in self._elite_archive.items()
        }

    def _normalize_group_value(self, value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, dict):
            parts = [
                f"{key}:{self._normalize_group_value(value[key])}"
                for key in sorted(value)
            ]
            return ",".join(parts) if parts else "none"
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if all(isinstance(v, (bool, int, float)) for v in values):
                return "".join("1" if float(v) > 0.0 else "0" for v in values) or "none"
            return ",".join(self._normalize_group_value(v) for v in values) or "none"
        text = str(value).strip()
        return text if text else "unknown"

    def _copy_routing_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {
                str(key): self._copy_routing_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                self._copy_routing_value(item)
                for item in list(value)
            ]
        return value

    def extract_structural_signature(
        self,
        result: Optional[Any] = None,
        sequence: Optional[str] = None,
        summary_confidence: Optional[Dict[str, Any]] = None,
        loss_vector: Optional[Dict[str, Any]] = None,
        structural_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the structural part of the group key described in the MCTS docs.

        Verifier-provided contact and structure metadata is used when present.
        Missing metadata is represented explicitly as "unknown" rather than
        guessed from unrelated tensors.
        """
        if not self.use_structural_grouping:
            return {}

        provided = dict(structural_signature or {})
        summary = dict(summary_confidence or {})
        losses = dict(loss_vector or {})
        if result is not None:
            summary.update(dict(getattr(result, "summary_confidence", {}) or {}))
            losses.update(dict(getattr(result, "loss_dict", {}) or {}))
            if sequence is None:
                sequence = str(getattr(result, "binder_seq", "") or "")

        def first_available(*keys: str) -> Any:
            for key in keys:
                if key in provided:
                    return provided[key]
                if key in summary:
                    return summary[key]
                if key in losses:
                    return losses[key]
                if result is not None and hasattr(result, key):
                    return getattr(result, key)
            return None

        seq_len = len(sequence or "")
        length_bin_start = (seq_len // self.length_group_bin_size) * self.length_group_bin_size
        length_bin = f"{length_bin_start}-{length_bin_start + self.length_group_bin_size - 1}"

        signature = {
            "hotspot_contact_bitset": self._normalize_group_value(
                first_available(
                    "hotspot_contact_bitset",
                    "target_hotspot_contact_bitset",
                    "hotspot_contacts",
                    "hotspot_contact",
                )
            ),
            "interface_patch_id": self._normalize_group_value(
                first_available("interface_patch_id", "target_patch_id", "patch_id")
            ),
            "secondary_structure_pattern": self._normalize_group_value(
                first_available("secondary_structure_pattern", "secondary_structure", "ss_pattern")
            ),
            "contact_fingerprint": self._normalize_group_value(
                first_available("contact_fingerprint", "interface_contact_fingerprint", "contact_bitset")
            ),
            "length_bin": length_bin,
        }
        for key in ROUTING_SIGNATURE_KEYS:
            value = first_available(key)
            if value is not None:
                signature[key] = self._copy_routing_value(value)
        return signature

    def has_sequence(self, sequence: str) -> bool:
        return sequence in self._sequence_to_node_id

    def get_node_by_sequence(self, sequence: str) -> Optional[MCTSNode]:
        node_id = self._sequence_to_node_id.get(sequence)
        if node_id is None:
            return None
        return self._nodes.get(node_id)

    def store_verifier_result(self, result: Any) -> CachedVerifierResult:
        cached = CachedVerifierResult(
            binder_seq=str(result.binder_seq),
            target_seq=_copy_target_sequence(getattr(result, "target_seq", "")),
            total_loss=float(result.total_loss),
            loss_dict={k: float(v) for k, v in result.loss_dict.items()},
            summary_confidence=dict(getattr(result, "summary_confidence", {}) or {}),
            coordinate=_cpu_detached(getattr(result, "coordinate", None)),
            plddt_logits=_cpu_detached(getattr(result, "plddt_logits", None)),
            from_verifier_cache=True,
        )
        if cached.binder_seq not in self._verifier_cache:
            self.verifier_cache_stores += 1
        self._verifier_cache[cached.binder_seq] = cached
        return cached

    def get_cached_verifier_result(self, sequence: str) -> Optional[CachedVerifierResult]:
        cached = self._verifier_cache.get(sequence)
        if cached is not None:
            self.verifier_cache_hits += 1
        return cached

    def node_to_cached_result(self, node: MCTSNode) -> CachedVerifierResult:
        return CachedVerifierResult(
            binder_seq=node.sequence,
            target_seq="",
            total_loss=float(node.L_real),
            loss_dict=dict(node.loss_vector),
            summary_confidence={},
            coordinate=_cpu_detached(node.coordinate),
            plddt_logits=None,
            from_verifier_cache=True,
        )

    def merge_duplicate(
        self,
        parent: MCTSNode,
        proposal: Optional[Any],
        result: Optional[Any] = None,
    ) -> float:
        sequence = (
            str(getattr(proposal, "sequence"))
            if proposal is not None and hasattr(proposal, "sequence")
            else str(getattr(result, "binder_seq"))
        )
        existing = self.get_node_by_sequence(sequence)
        if existing is None:
            raise KeyError(f"[MCTS] Cannot merge duplicate, sequence is absent: {sequence[:20]}")

        reward = existing.tree_reward
        existing.value_sum += float(reward)
        existing.visit_count += 1
        existing.status_flags["duplicate_hits"] = int(existing.status_flags.get("duplicate_hits", 0)) + 1
        existing.status_flags["last_duplicate_parent"] = parent.node_id
        if proposal is not None:
            existing.status_flags["last_duplicate_prior"] = float(getattr(proposal, "prior", 0.0))
            existing.status_flags["last_duplicate_benefit"] = float(getattr(proposal, "benefit", 0.0))

        self.duplicate_merges += 1
        logger.debug(
            "[MCTS] Duplicate merge: parent=%d existing=%d seq=%s R=%.4f hits=%d",
            parent.node_id,
            existing.node_id,
            sequence[:20],
            reward,
            existing.status_flags["duplicate_hits"],
        )
        return float(reward)

    @property
    def roots(self) -> List[MCTSNode]:
        return self._roots

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    def add_root_node(
        self,
        sequence: str,
        logits: torch.Tensor,
        L_real: float,
        coordinate: Optional[torch.Tensor] = None,
        loss_vector: Optional[Dict[str, Any]] = None,
        tree_reward: Optional[float] = None,
        prior: float = 1.0,
        status_flags: Optional[Dict[str, Any]] = None,
        structural_signature: Optional[Dict[str, Any]] = None,
    ) -> MCTSNode:
        """
        添加 Phase 0 根节点。若序列已存在，返回已有节点。
        """
        existing = self.get_node_by_sequence(sequence)
        if existing is not None:
            logger.debug("[MCTS] Root dedup hit: seq=%s node=%d", sequence[:20], existing.node_id)
            return existing

        loss_data = {k: float(v) for k, v in (loss_vector or {}).items()}
        flags = dict(status_flags or {})
        reward, hard_status = self.compute_node_reward(
            float(L_real),
            loss_data,
            base_reward=tree_reward,
            observe=True,
        )
        flags = {**flags, **hard_status}
        structural = self.extract_structural_signature(
            sequence=sequence,
            loss_vector=loss_data,
            structural_signature=structural_signature,
        )
        group_id = self.make_loss_group_id(
            loss_data,
            flags,
            structural_signature=structural,
            sequence=sequence,
        )
        reward, flags = self._apply_diversity_bonus(
            reward,
            flags,
            sequence=sequence,
            group_id=group_id,
            structural_signature=structural,
        )
        node = MCTSNode(
            node_id=self._new_id(),
            sequence=sequence,
            logits=_cpu_detached(logits),
            coordinate=_cpu_detached(coordinate),
            L_real=float(L_real),
            loss_vector=loss_data,
            tree_reward=reward,
            value_sum=reward,
            visit_count=1,
            prior=float(prior),
            status_flags=flags,
            structural_signature=structural,
            group_id=group_id,
            parent=None,
            children=[],
            depth=0,
        )
        self._register_node(node)
        self._roots.append(node)
        self.update_elite_archive(node)
        self.N_total += 1
        logger.debug(
            "[MCTS] Root node added: id=%d seq=%s L=%.4f R=%.4f N_total=%d",
            node.node_id, sequence[:20], node.L_real, node.tree_reward, self.N_total,
        )
        return node

    def puct_score(self, parent: Optional[MCTSNode], child: MCTSNode) -> float:
        parent_visits = self.N_total if parent is None else parent.visit_count
        exploration = (
            self.C
            * child.prior
            * math.sqrt(max(parent_visits, 1))
            / (1 + child.visit_count)
        )
        return child.q_value + exploration

    def action_puct_score(self, parent: MCTSNode, action: MCTSAction) -> float:
        parent_visits = max(parent.visit_count, 1)
        q_value = action.q_value if action.visit_count > 0 else parent.q_value
        exploration = (
            self.C
            * max(float(action.prior), 1e-12)
            * math.sqrt(parent_visits)
            / (1 + action.visit_count)
        )
        diversity_bonus = 0.0
        if self.use_diversity_bonus:
            diversity_bonus, diversity_status = self.compute_diversity_bonus(
                sequence=action.sequence,
                group_id=parent.group_id,
                structural_signature=parent.structural_signature,
            )
            action.status_flags.update(
                {
                    "action_diversity_bonus": diversity_bonus,
                    "action_diversity_score": diversity_status.get("diversity_score", 0.0),
                }
            )
        return q_value + exploration + diversity_bonus

    def progressive_widening_limit(self, node: MCTSNode) -> int:
        visits = max(node.visit_count, 1)
        raw_limit = self.progressive_widening_k * (visits ** self.progressive_widening_alpha)
        limit = int(math.ceil(raw_limit))
        return max(1, min(self.max_children_per_node, limit))

    def remaining_expansion_slots(self, node: MCTSNode) -> int:
        if self.total_nodes >= self.max_nodes:
            return 0
        node_slots = self.progressive_widening_limit(node) - len(node.children)
        tree_slots = self.max_nodes - self.total_nodes
        return max(0, min(node_slots, tree_slots))

    def can_expand(self, node: MCTSNode) -> bool:
        return self.remaining_expansion_slots(node) > 0

    def _action_from_proposal(self, node: MCTSNode, proposal: Any) -> MCTSAction:
        return MCTSAction(
            action_id=self._new_action_id(),
            parent_node_id=node.node_id,
            sequence=str(getattr(proposal, "sequence")),
            edits=list(getattr(proposal, "edits", [])),
            prior=max(float(getattr(proposal, "prior", 1.0)), 1e-12),
            benefit=float(getattr(proposal, "benefit", 0.0)),
            action_group=str(getattr(proposal, "action_group", "unrouted")),
            group_prior=float(getattr(proposal, "group_prior", 1.0)),
            proposal=proposal,
            status="unexpanded",
            status_flags={
                "proposal_prior": float(getattr(proposal, "prior", 1.0)),
                "proposal_benefit": float(getattr(proposal, "benefit", 0.0)),
            },
        )

    def register_action_proposals(self, node: MCTSNode, proposals: List[Any]) -> List[MCTSAction]:
        """
        Add gradient-prior proposals to a node-local action table.

        The table is keyed by exact resulting sequence, so repeated visits to
        the same node reuse action statistics instead of treating every proposal
        batch as a fresh set of siblings.
        """
        registered: List[MCTSAction] = []
        for proposal in proposals:
            sequence = str(getattr(proposal, "sequence", ""))
            if not sequence:
                continue
            existing_idx = node.action_sequence_index.get(sequence)
            if existing_idx is not None:
                action = node.actions[existing_idx]
                action.proposal = action.proposal or proposal
                action.prior = max(float(action.prior), float(getattr(proposal, "prior", action.prior)))
                action.benefit = max(float(action.benefit), float(getattr(proposal, "benefit", action.benefit)))
                registered.append(action)
                continue

            action = self._action_from_proposal(node, proposal)
            node.action_sequence_index[sequence] = len(node.actions)
            node.actions.append(action)
            registered.append(action)
        return registered

    def _expanded_actions(self, node: MCTSNode) -> List[MCTSAction]:
        return [
            action for action in node.actions
            if action.is_expanded and action.child_node_id in self._nodes
        ]

    def _unexpanded_actions(self, node: MCTSNode) -> List[MCTSAction]:
        return [
            action for action in node.actions
            if action.status == "unexpanded" and action.child_node_id is None
        ]

    def _best_selectable_action(self, node: MCTSNode) -> Optional[MCTSAction]:
        candidates: List[MCTSAction] = list(self._expanded_actions(node))
        if self.can_expand(node):
            candidates.extend(self._unexpanded_actions(node))
        if not candidates:
            return None
        return max(candidates, key=lambda action: self.action_puct_score(node, action))

    def select_unexpanded_actions(
        self,
        node: MCTSNode,
        max_actions: Optional[int] = None,
        candidates: Optional[List[MCTSAction]] = None,
    ) -> List[MCTSAction]:
        slots = self.remaining_expansion_slots(node)
        if slots <= 0:
            return []
        limit = min(slots, max_actions or self.expand_actions_per_step)
        action_pool = candidates if candidates is not None else self._unexpanded_actions(node)
        action_pool = [
            action for action in action_pool
            if action.status == "unexpanded" and action.child_node_id is None
        ]
        candidates_sorted = sorted(
            action_pool,
            key=lambda action: self.action_puct_score(node, action),
            reverse=True,
        )
        return candidates_sorted[:limit]

    def select_decision(self) -> MCTSSelection:
        """
        Document-style action-level PUCT selection with progressive widening.

        A node with no verified children is always a frontier. Once a node has
        any verified child, the best expanded action competes directly with the
        current-node frontier score. Progressive widening only caps how many
        children a node may add; it is not a hard "fill width before descent"
        gate. This lets high-value mutation basins deepen as soon as PUCT says
        they are more promising than another local expansion.
        """
        if not self._roots:
            raise RuntimeError("[MCTS] Tree is empty, cannot select")

        node = max(self._roots, key=lambda n: self.puct_score(None, n))
        node_path = [node]
        action_path: List[MCTSAction] = []
        last_best_action_score: Optional[float] = None
        last_frontier_score: Optional[float] = None

        while True:
            expanded_actions = self._expanded_actions(node)
            if not expanded_actions:
                return MCTSSelection(
                    node=node,
                    action=None,
                    node_path=node_path,
                    action_path=action_path,
                    decision_type=(
                        "descend_expand_frontier" if action_path and self.can_expand(node)
                        else "expand_frontier" if self.can_expand(node)
                        else "descend_terminal" if action_path
                        else "terminal"
                    ),
                    best_action_score=last_best_action_score,
                    frontier_score=(
                        node.q_value if not action_path and self.can_expand(node)
                        else last_frontier_score
                    ),
                )

            if self.can_expand(node):
                best_expanded = max(
                    expanded_actions,
                    key=lambda item: self.action_puct_score(node, item),
                )
                best_score = self.action_puct_score(node, best_expanded)
                frontier_score = node.q_value + self.descend_score_margin
                if best_score < frontier_score:
                    return MCTSSelection(
                        node=node,
                        action=None,
                        node_path=node_path,
                        action_path=action_path,
                        decision_type="descend_expand_frontier" if action_path else "expand_frontier",
                        best_action_score=best_score,
                        frontier_score=frontier_score,
                    )
                action = best_expanded
                decision_type = "descend_puct"
                last_best_action_score = best_score
                last_frontier_score = frontier_score
            else:
                action = max(expanded_actions, key=lambda item: self.action_puct_score(node, item))
                decision_type = "descend_full"
                best_score = self.action_puct_score(node, action)
                last_best_action_score = best_score
                last_frontier_score = None

            child = self._nodes.get(action.child_node_id)
            if child is None:
                action.status = "invalid"
                continue
            action_path.append(action)
            node = child
            node_path.append(node)

    def should_descend(self, node: MCTSNode) -> bool:
        """
        Progressive-widening descend gate used by compatibility callers.

        The full selection path is implemented in select_decision(); this helper
        mirrors the same PUCT-vs-frontier rule for older call sites.
        """
        if not node.children:
            return False
        if not self.can_expand(node):
            return True
        best_action = max(
            self._expanded_actions(node),
            key=lambda item: self.action_puct_score(node, item),
            default=None,
        )
        if best_action is None:
            return False
        return self.action_puct_score(node, best_action) >= node.q_value + self.descend_score_margin

    def select(self) -> MCTSNode:
        """
        从 roots 出发，按严格 progressive widening 选择 frontier node。

        默认 pipeline 应优先调用 select_decision()，以便 action/edge
        backup 获得完整路径。此方法保留为兼容旧测试和旧调用。
        """
        selection = self.select_decision()
        node = selection.node

        logger.debug(
            "[MCTS] Selected leaf: id=%d depth=%d seq=%s L=%.4f R=%.4f Q=%.4f "
            "P=%.4f V=%d tree_size=%d N_total=%d children=%d/%d decision=%s",
            node.node_id, node.depth, node.sequence[:20], node.L_real,
            node.tree_reward, node.q_value, node.prior, node.visit_count,
            self.total_nodes, self.N_total, len(node.children),
            self.progressive_widening_limit(node), selection.decision_type,
        )
        return node

    def expand(
        self,
        parent: MCTSNode,
        proposals: List[Any],
        results: Optional[List[Any]] = None,
    ) -> List[MCTSNode]:
        """
        将 proposals/results 作为 parent 的 child 插入。

        New interface:
            expand(parent, proposals, zeroth_results)
        Backward-compatible interface:
            expand(parent, zeroth_results)
        """
        if results is None:
            results = proposals
            proposals = [None] * len(results)

        if not results:
            return []

        new_nodes: List[MCTSNode] = []
        fallback_prior = 1.0 / max(len(results), 1)
        slots = self.remaining_expansion_slots(parent)
        if slots <= 0:
            logger.debug("[MCTS] Expand skipped: parent=%d has no widening slots", parent.node_id)
            return []

        for proposal, result in zip(proposals, results):
            if len(new_nodes) >= slots:
                break
            seq = result.binder_seq
            action: Optional[MCTSAction] = None
            if proposal is not None:
                action = self.register_action_proposals(parent, [proposal])[0]
            else:
                existing_idx = parent.action_sequence_index.get(seq)
                if existing_idx is not None:
                    action = parent.actions[existing_idx]
                else:
                    action = MCTSAction(
                        action_id=self._new_action_id(),
                        parent_node_id=parent.node_id,
                        sequence=seq,
                        edits=[],
                        prior=fallback_prior,
                        proposal=None,
                    )
                    parent.action_sequence_index[seq] = len(parent.actions)
                    parent.actions.append(action)

            if self.has_sequence(seq):
                if action is not None:
                    action.status = "duplicate"
                logger.debug("[MCTS] Expand dedup skip: parent=%d seq=%s", parent.node_id, seq[:20])
                continue

            prior = float(getattr(proposal, "prior", fallback_prior) if proposal is not None else fallback_prior)
            edits = list(getattr(proposal, "edits", []) if proposal is not None else [])
            proposal_status = {}
            if proposal is not None:
                proposal_status = {
                    "action_group": getattr(proposal, "action_group", "unrouted"),
                    "group_prior": float(getattr(proposal, "group_prior", 1.0)),
                    "proposal_prior": prior,
                    "proposal_benefit": float(getattr(proposal, "benefit", 0.0)),
                }
            loss_data = {k: float(v) for k, v in result.loss_dict.items()}
            observe_result = not bool(getattr(result, "from_verifier_cache", False))
            reward, hard_status = self.compute_node_reward(
                float(result.total_loss),
                loss_data,
                observe=observe_result,
            )
            status_flags = {**proposal_status, **hard_status}
            structural = self.extract_structural_signature(
                result=result,
                sequence=seq,
                loss_vector=loss_data,
            )
            group_id = self.make_loss_group_id(
                loss_data,
                status_flags,
                structural_signature=structural,
                sequence=seq,
            )
            reward, status_flags = self._apply_diversity_bonus(
                reward,
                status_flags,
                sequence=seq,
                group_id=group_id,
                structural_signature=structural,
            )
            if self._is_pareto_dominated(group_id, loss_data):
                if action is not None:
                    action.status = "pareto_rejected"
                continue
            if not self._should_admit_group(group_id, reward):
                if action is not None:
                    action.status = "group_rejected"
                logger.debug(
                    "[MCTS] Group skip: parent=%d seq=%s group=%s R=%.4f",
                    parent.node_id, seq[:20], group_id, reward,
                )
                continue

            child = MCTSNode(
                node_id=self._new_id(),
                sequence=seq,
                logits=_cpu_detached(parent.logits),
                coordinate=_cpu_detached(result.coordinate[0:1] if result.coordinate is not None else None),
                L_real=float(result.total_loss),
                loss_vector=loss_data,
                tree_reward=reward,
                value_sum=reward,
                visit_count=1,
                prior=max(prior, 1e-12),
                edit_from_parent=edits,
                status_flags=status_flags,
                structural_signature=structural,
                group_id=group_id,
                parent=parent,
                children=[],
                actions=[],
                action_sequence_index={},
                depth=parent.depth + 1,
            )
            self._register_node(child)
            parent.children.append(child)
            if action is not None:
                action.status = "expanded"
                action.child_node_id = child.node_id
                action.prior = max(prior, 1e-12)
            new_nodes.append(child)
            self._prune_pareto_dominated_leaves(group_id, child)
            self._enforce_group_representatives(group_id)
            self.update_elite_archive(child)
            logger.debug(
                "[MCTS] Expand: parent=%d -> child=%d seq=%s L=%.4f R=%.4f P=%.4f group=%s",
                parent.node_id, child.node_id, child.sequence[:20],
                child.L_real, child.tree_reward, child.prior, child.group_id,
            )

        return [node for node in new_nodes if node.node_id in self._nodes]

    def backpropagate(
        self,
        node: MCTSNode,
        rewards: List[float],
    ) -> None:
        """
        将 child rewards 沿 selected node -> root 回传。
        """
        if not rewards:
            return

        K = len(rewards)
        sum_rewards = float(sum(rewards))
        for ancestor in node.get_path_to_root():
            ancestor.value_sum += sum_rewards
            ancestor.visit_count += K

        self.N_total += K
        logger.debug(
            "[MCTS] Backprop: parent=%d K=%d sum_R=%.4f N_total=%d",
            node.node_id, K, sum_rewards, self.N_total,
        )

    def backpropagate_action(
        self,
        selection: MCTSSelection,
        action: Optional[MCTSAction],
        reward: float,
    ) -> None:
        """
        Back up one verifier reward through both node and action statistics.
        """
        value = float(reward)
        for ancestor in selection.node_path:
            ancestor.value_sum += value
            ancestor.visit_count += 1

        for path_action in selection.action_path:
            path_action.value_sum += value
            path_action.visit_count += 1

        if action is not None:
            action.value_sum += value
            action.visit_count += 1

        self.N_total += 1
        logger.debug(
            "[MCTS] Action backprop: node=%d action=%s R=%.4f N_total=%d",
            selection.node.node_id,
            action.action_id if action is not None else "none",
            value,
            self.N_total,
        )

    def prune(self) -> int:
        """
        超过 max_nodes 时删除 Q 最低的非 root leaf。
        """
        if self.total_nodes <= self.max_nodes:
            return 0

        n_prune = self.total_nodes - self.max_nodes
        leaves = [
            n for n in self._nodes.values()
            if n.is_leaf() and not n.is_root()
        ]
        if not leaves:
            logger.warning(
                "[MCTS] prune: no prunable leaves (size=%d/%d)",
                self.total_nodes, self.max_nodes,
            )
            return 0

        leaves_sorted = sorted(leaves, key=lambda n: n.q_value)
        pruned = 0
        for leaf in leaves_sorted:
            if pruned >= n_prune:
                break
            self._remove_node(leaf)
            pruned += 1

        logger.info("[MCTS] Pruned %d nodes (tree_size=%d/%d)", pruned, self.total_nodes, self.max_nodes)
        return pruned

    def _remove_node(self, node: MCTSNode) -> None:
        if node.parent is not None and node in node.parent.children:
            node.parent.children.remove(node)
        elif node in self._roots:
            self._roots.remove(node)
        self._unregister_node(node)

    def get_best_leaf(self) -> MCTSNode:
        leaves = [n for n in self._nodes.values() if n.is_leaf()]
        if not leaves:
            raise RuntimeError("[MCTS] No leaf nodes in tree")
        return max(leaves, key=lambda n: n.tree_reward)

    def get_best_root(self) -> MCTSNode:
        if not self._roots:
            raise RuntimeError("[MCTS] No root nodes")
        return max(self._roots, key=lambda n: n.tree_reward)

    def get_global_best(self) -> MCTSNode:
        if not self._nodes:
            raise RuntimeError("[MCTS] No nodes in tree")
        return max(self._nodes.values(), key=lambda n: n.tree_reward)

    def result_reward(self, result: Any) -> float:
        loss_data = {k: float(v) for k, v in result.loss_dict.items()}
        reward, _ = self.compute_node_reward(
            float(result.total_loss),
            loss_data,
            observe=False,
        )
        return reward

    def get_node_stats(self) -> Dict[str, Any]:
        depths = [n.depth for n in self._nodes.values()]
        actions = [action for node in self._nodes.values() for action in node.actions]
        expanded_actions = [action for action in actions if action.is_expanded]
        unexpanded_actions = [action for action in actions if action.status == "unexpanded"]
        return {
            "total_nodes": self.total_nodes,
            "num_roots": len(self._roots),
            "num_leaves": sum(1 for n in self._nodes.values() if n.is_leaf()),
            "num_internal": sum(1 for n in self._nodes.values() if not n.is_leaf()),
            "num_actions": len(actions),
            "num_expanded_actions": len(expanded_actions),
            "num_unexpanded_actions": len(unexpanded_actions),
            "N_total": self.N_total,
            "max_depth": max(depths) if depths else 0,
            "C": self.C,
            "max_nodes": self.max_nodes,
            "progressive_widening_k": self.progressive_widening_k,
            "progressive_widening_alpha": self.progressive_widening_alpha,
            "max_children_per_node": self.max_children_per_node,
            "min_children_before_descend": self.min_children_before_descend,
            "descend_score_margin": self.descend_score_margin,
            "num_groups": len(self._group_to_node_ids),
            "use_loss_grouping": self.use_loss_grouping,
            "use_reward_normalization": self.use_reward_normalization,
            "use_pareto_pruning": self.use_pareto_pruning,
            "use_elite_archive": self.use_elite_archive,
            "use_structural_grouping": self.use_structural_grouping,
            "use_diversity_bonus": self.use_diversity_bonus,
            "diversity_bonus_weight": self.diversity_bonus_weight,
            "diversity_bonus_max": self.diversity_bonus_max,
            "mean_diversity_bonus": (
                sum(float(node.status_flags.get("diversity_bonus", 0.0)) for node in self._nodes.values())
                / max(len(self._nodes), 1)
            ),
            "max_diversity_bonus": max(
                [float(node.status_flags.get("diversity_bonus", 0.0)) for node in self._nodes.values()] or [0.0]
            ),
            "length_group_bin_size": self.length_group_bin_size,
            "reward_stats_count": self.reward_normalizer.observations,
            "representatives_per_group": self.representatives_per_group,
            "group_rejections": self.group_rejections,
            "group_pruned": self.group_pruned,
            "pareto_rejections": self.pareto_rejections,
            "pareto_pruned": self.pareto_pruned,
            "elite_archive_size": sum(len(entries) for entries in self._elite_archive.values()),
            "elite_archive_groups": len(self._elite_archive),
            "elite_archive_updates": self.elite_archive_updates,
            "elite_archive_rejections": self.elite_archive_rejections,
            "verifier_cache_size": len(self._verifier_cache),
            "verifier_cache_hits": self.verifier_cache_hits,
            "verifier_cache_stores": self.verifier_cache_stores,
            "duplicate_merges": self.duplicate_merges,
            "num_invalid": sum(
                1 for node in self._nodes.values()
                if bool(node.status_flags.get("hard_invalid", False))
            ),
        }

    def log_tree_summary(self) -> None:
        stats = self.get_node_stats()
        best = self.get_global_best()
        logger.info(
            "[MCTS] Tree summary: %d nodes, %d roots, %d leaves, max_depth=%d, "
            "N_total=%d, groups=%d, invalid=%d, group_skip=%d, group_pruned=%d, "
            "pareto_skip=%d, pareto_pruned=%d, "
            "elite=%d/%d elite_upd=%d, "
            "cache=%d hits=%d dup_merge=%d, div=%.4f/%.4f, "
            "best_seq=%s... L=%.4f R=%.4f Q=%.4f",
            stats["total_nodes"], stats["num_roots"], stats["num_leaves"],
            stats["max_depth"], stats["N_total"], stats["num_groups"],
            stats["num_invalid"], stats["group_rejections"], stats["group_pruned"],
            stats["pareto_rejections"], stats["pareto_pruned"],
            stats["elite_archive_size"], stats["elite_archive_groups"],
            stats["elite_archive_updates"],
            stats["verifier_cache_size"], stats["verifier_cache_hits"],
            stats["duplicate_merges"], stats["mean_diversity_bonus"],
            stats["max_diversity_bonus"],
            best.sequence[:20], best.L_real, best.tree_reward, best.q_value,
        )
