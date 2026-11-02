"""
Legacy BindCraft v2 MCTS controller.

This module intentionally preserves the GitHub main MCTS semantics for the
two-stage de novo workflow bootstrap phase:
  - loss minimization
  - whole-tree UCB selection
  - one expansion inserts all verifier candidates
  - W/V loss backpropagation

Do not mix these statistics with the newer reward/action-level PUCT tree.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

LEGACY_SELECTION_MODES = frozenset({"global_ucb", "hierarchical_ucb"})


@dataclass
class LegacyMCTSNode:
    """A verified sequence node using the original loss-minimizing UCB state."""

    node_id: int
    sequence: str
    logits: torch.Tensor
    coordinate: Optional[torch.Tensor]
    L_real: float
    W: float = field(default=0.0)
    visit_count: int = field(default=0)
    parent: Optional["LegacyMCTSNode"] = field(default=None, repr=False)
    children: List["LegacyMCTSNode"] = field(default_factory=list, repr=False)
    depth: int = field(default=0)
    soft_step: int = field(default=0)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_path_to_root(self) -> List["LegacyMCTSNode"]:
        path = []
        node: Optional["LegacyMCTSNode"] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path


class LegacyMCTSTree:
    """Original whole-tree UCB MCTS used as a coarse bootstrap stage."""

    def __init__(
        self,
        C: float = 1.0,
        max_nodes: int = 5000,
        mcts_tau: float = 4.0,
        selection_mode: str = "global_ucb",
        min_root_expansions: int = 0,
    ):
        self.C = C
        self.max_nodes = max_nodes
        self.mcts_tau = mcts_tau
        normalized_mode = str(selection_mode).strip().lower()
        if normalized_mode not in LEGACY_SELECTION_MODES:
            raise ValueError(
                "selection_mode must be one of: "
                + ", ".join(sorted(LEGACY_SELECTION_MODES))
            )
        self.selection_mode = normalized_mode
        self.min_root_expansions = max(int(min_root_expansions), 0)

        self._nodes: Dict[int, LegacyMCTSNode] = {}
        self._next_id: int = 0
        self._roots: List[LegacyMCTSNode] = []
        self._root_expansion_counts: Dict[int, int] = {}
        self._last_selection: Dict[str, Any] = {}
        self.N_total: int = 0

    def _new_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def _register_node(self, node: LegacyMCTSNode) -> None:
        self._nodes[node.node_id] = node

    def add_root_node(
        self,
        sequence: str,
        logits: torch.Tensor,
        L_real: float,
        coordinate: Optional[torch.Tensor] = None,
    ) -> LegacyMCTSNode:
        node = LegacyMCTSNode(
            node_id=self._new_id(),
            sequence=sequence,
            logits=logits.detach().clone(),
            coordinate=coordinate.detach().clone() if coordinate is not None else None,
            L_real=float(L_real),
            W=float(L_real),
            visit_count=1,
            parent=None,
            children=[],
            depth=0,
            soft_step=0,
        )
        self._register_node(node)
        self._roots.append(node)
        self._root_expansion_counts[node.node_id] = 0
        self.N_total += 1
        logger.debug(
            "[LegacyMCTS] Root added: id=%d seq=%s L=%.4f N_total=%d",
            node.node_id,
            sequence[:20],
            node.L_real,
            self.N_total,
        )
        return node

    @property
    def roots(self) -> List[LegacyMCTSNode]:
        return self._roots

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    def ucb_score(self, node: LegacyMCTSNode) -> float:
        if self.N_total == 0:
            exploration = float("inf")
        else:
            exploration = self.C * math.sqrt(math.log(self.N_total) / node.visit_count)
        exploitation = -node.W / node.visit_count
        return exploitation + exploration

    @staticmethod
    def root_for_node(node: LegacyMCTSNode) -> LegacyMCTSNode:
        root = node
        while root.parent is not None:
            root = root.parent
        return root

    @property
    def root_expansion_counts(self) -> Dict[int, int]:
        return dict(self._root_expansion_counts)

    @property
    def last_selection(self) -> Dict[str, Any]:
        return dict(self._last_selection)

    def root_ucb_score(self, root: LegacyMCTSNode) -> float:
        if all(candidate is not root for candidate in self._roots):
            raise ValueError("root_ucb_score requires a registered root node")
        expansion_count = self._root_expansion_counts.get(root.node_id, 0)
        if expansion_count == 0:
            return float("inf")
        total_expansions = max(sum(self._root_expansion_counts.values()), 1)
        exploitation = -root.W / max(root.visit_count, 1)
        exploration = self.C * math.sqrt(
            math.log1p(total_expansions) / expansion_count
        )
        return exploitation + exploration

    def _select_hierarchical_root(
        self,
    ) -> tuple[LegacyMCTSNode, str, Optional[float]]:
        under_warmup = [
            root
            for root in self._roots
            if self._root_expansion_counts.get(root.node_id, 0)
            < self.min_root_expansions
        ]
        if under_warmup:
            min_count = min(
                self._root_expansion_counts.get(root.node_id, 0)
                for root in under_warmup
            )
            root = min(
                (
                    candidate
                    for candidate in under_warmup
                    if self._root_expansion_counts.get(candidate.node_id, 0)
                    == min_count
                ),
                key=lambda candidate: (candidate.L_real, candidate.node_id),
            )
            return root, "root_warmup", None

        root = max(
            self._roots,
            key=lambda candidate: (
                self.root_ucb_score(candidate),
                -candidate.node_id,
            ),
        )
        return root, "root_ucb", self.root_ucb_score(root)

    def _record_selection(
        self,
        node: LegacyMCTSNode,
        *,
        phase: str,
        root_score: Optional[float],
    ) -> None:
        root = self.root_for_node(node)
        self._root_expansion_counts[root.node_id] = (
            self._root_expansion_counts.get(root.node_id, 0) + 1
        )
        self._last_selection = {
            "selection_mode": self.selection_mode,
            "selection_phase": phase,
            "selected_root_id": root.node_id,
            "selected_node_id": node.node_id,
            "selected_root_expansions": self._root_expansion_counts[root.node_id],
            "root_score": root_score,
            "root_expansion_counts": self.root_expansion_counts,
        }

    def select(self) -> LegacyMCTSNode:
        if not self._nodes:
            raise RuntimeError("[LegacyMCTS] Tree is empty, cannot select")

        if self.selection_mode == "hierarchical_ucb":
            selected_root, phase, root_score = self._select_hierarchical_root()
            root_nodes = [
                node
                for node in self._nodes.values()
                if self.root_for_node(node) is selected_root
            ]
            best_node = max(
                root_nodes,
                key=lambda node: (self.ucb_score(node), -node.node_id),
            )
            self._record_selection(
                best_node,
                phase=phase,
                root_score=root_score,
            )
        else:
            best_node = max(self._nodes.values(), key=self.ucb_score)
            self._record_selection(
                best_node,
                phase="global_ucb",
                root_score=None,
            )
        logger.debug(
            "[LegacyMCTS] Selected: mode=%s root=%d id=%d depth=%d "
            "seq=%s L=%.4f W=%.4f V=%d UCB=%.4f root_count=%d",
            self.selection_mode,
            self._last_selection["selected_root_id"],
            best_node.node_id,
            best_node.depth,
            best_node.sequence[:20],
            best_node.L_real,
            best_node.W,
            best_node.visit_count,
            self.ucb_score(best_node),
            self._last_selection["selected_root_expansions"],
        )
        return best_node

    def expand(
        self,
        parent: LegacyMCTSNode,
        candidates: List[Any],
    ) -> List[LegacyMCTSNode]:
        if not candidates:
            return []

        new_nodes: List[LegacyMCTSNode] = []
        for cand in candidates:
            child = LegacyMCTSNode(
                node_id=self._new_id(),
                sequence=cand.binder_seq,
                logits=parent.logits.detach().clone(),
                coordinate=cand.coordinate[0:1].detach().clone() if cand.coordinate is not None else None,
                L_real=float(cand.total_loss),
                W=float(cand.total_loss),
                visit_count=1,
                parent=parent,
                children=[],
                depth=parent.depth + 1,
                soft_step=parent.soft_step,
            )
            self._register_node(child)
            parent.children.append(child)
            new_nodes.append(child)
            logger.debug(
                "[LegacyMCTS] Expand: parent=%d -> child=%d seq=%s L=%.4f",
                parent.node_id,
                child.node_id,
                child.sequence[:20],
                child.L_real,
            )

        return new_nodes

    def backpropagate(
        self,
        node: LegacyMCTSNode,
        losses: List[float],
    ) -> None:
        count = len(losses)
        if count == 0:
            return
        sum_losses = float(sum(losses))

        for ancestor in node.get_path_to_root():
            ancestor.W += sum_losses
            ancestor.visit_count += count

        self.N_total += count
        logger.debug(
            "[LegacyMCTS] Backprop: parent=%d K=%d sum_L=%.4f N_total=%d",
            node.node_id,
            count,
            sum_losses,
            self.N_total,
        )

    def prune(self) -> int:
        if self.total_nodes <= self.max_nodes:
            return 0

        n_prune = self.total_nodes - self.max_nodes
        leaves = [
            node for node in self._nodes.values()
            if node.is_leaf() and not node.is_root()
        ]
        if not leaves:
            logger.warning(
                "[LegacyMCTS] prune: no prunable leaves (size=%d/%d)",
                self.total_nodes,
                self.max_nodes,
            )
            return 0

        leaves_sorted = sorted(
            leaves,
            key=lambda node: node.W / max(node.visit_count, 1),
            reverse=True,
        )

        pruned = 0
        for leaf in leaves_sorted:
            if pruned >= n_prune:
                break
            self._remove_node(leaf)
            pruned += 1

        logger.info(
            "[LegacyMCTS] Pruned %d nodes (tree_size=%d/%d)",
            pruned,
            self.total_nodes,
            self.max_nodes,
        )
        return pruned

    def _remove_node(self, node: LegacyMCTSNode) -> None:
        if node.parent is not None:
            if node in node.parent.children:
                node.parent.children.remove(node)
        elif node in self._roots:
            self._roots.remove(node)
        del self._nodes[node.node_id]

    def get_best_leaf(self) -> LegacyMCTSNode:
        leaves = [node for node in self._nodes.values() if node.is_leaf()]
        if not leaves:
            raise RuntimeError("[LegacyMCTS] No leaf nodes in tree")
        return min(leaves, key=lambda node: node.W / max(node.visit_count, 1))

    def get_best_root(self) -> LegacyMCTSNode:
        if not self._roots:
            raise RuntimeError("[LegacyMCTS] No root nodes")
        return min(self._roots, key=lambda node: node.L_real)

    def get_global_best(self) -> LegacyMCTSNode:
        if not self._nodes:
            raise RuntimeError("[LegacyMCTS] No nodes in tree")
        return min(self._nodes.values(), key=lambda node: node.W / max(node.visit_count, 1))

    def get_node_stats(self) -> Dict[str, Any]:
        depths = [node.depth for node in self._nodes.values()]
        return {
            "total_nodes": self.total_nodes,
            "num_roots": len(self._roots),
            "num_leaves": sum(1 for node in self._nodes.values() if node.is_leaf()),
            "num_internal": sum(1 for node in self._nodes.values() if not node.is_leaf()),
            "N_total": self.N_total,
            "max_depth": max(depths) if depths else 0,
            "C": self.C,
            "max_nodes": self.max_nodes,
            "selection_mode": self.selection_mode,
            "min_root_expansions": self.min_root_expansions,
            "root_expansion_counts": self.root_expansion_counts,
        }

    def log_tree_summary(self) -> None:
        stats = self.get_node_stats()
        best = self.get_global_best()
        logger.info(
            "[LegacyMCTS] Tree summary: %d nodes, %d roots, %d leaves, "
            "max_depth=%d, N_total=%d, best_seq=%s L=%.4f",
            stats["total_nodes"],
            stats["num_roots"],
            stats["num_leaves"],
            stats["max_depth"],
            stats["N_total"],
            best.sequence[:20],
            best.L_real,
        )
