"""
BindCraft v2 - 梯度引导的离散候选生成器

根据一阶梯度矩阵 G = nabla_Theta L，通过一阶泰勒近似计算每个位置
从当前氨基酸突变为其他氨基酸的收益，选取 Top-K 候选突变，
并采样组合出 B 条离散候选序列。

对应算法指导文档：
  - 算法指导2.md 第六章 Step 2
  - Bindcraft v2零阶更新.md 第五章 5.3 Step 2

核心公式：
  Delta_L[i, c->a] ≈ dL/dTheta[i,a] - dL/dTheta[i,c]
  P_mut(i, a) = Softmax(-Delta_L[i, c->a] / T_sample)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

from proteocraft.utils import (
    PROTEIN_RESIDUE_NAMES,
    AA_1LETTER,
    NUM_AMINO_ACIDS,
)

logger = logging.getLogger(__name__)

AA_1_TO_IDX = {AA_1LETTER[name]: idx for idx, name in enumerate(PROTEIN_RESIDUE_NAMES)}
MCTS_PRIOR_MODES = ("full", "uniform", "no_gradient", "no_group", "no_sequence")
MCTS_PROPOSAL_METHODS = ("one_step_fixed", "one_step_adaptive", "two_step_truncated")


def _aa_indices(letters: str) -> List[int]:
    return [AA_1_TO_IDX[aa] for aa in letters if aa in AA_1_TO_IDX]


SMALL_AA = set(_aa_indices("AGST"))
HYDROPHOBIC_AA = set(_aa_indices("AVLIMFWY"))
BULKY_AA = set(_aa_indices("FWYILMRK"))
AROMATIC_AA = set(_aa_indices("FWYH"))
POLAR_CHARGED_AA = set(_aa_indices("RKDEQNHSTY"))
CHARGED_AA = set(_aa_indices("RKDE"))
RISKY_AA = set(_aa_indices("CPG"))

# ── BLOSUM62 替换矩阵（20×20，对应 PROTEIN_RESIDUE_NAMES 顺序）─────────────
# 行 = 目标氨基酸 (to)，列 = 源氨基酸 (from)
# 正值：演化上保守的替换（高分）；负值：罕见替换（低分）
# 矩阵来源：NCBI BLAST BLOSUM62，对角线为最大正值，非对角线反映替换频率
# 顺序：ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
_BLOSUM62_ROWS = [
    # A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
    [ 4,  -1,  -2,  -2,   0,  -1,  -1,   0,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   0,  -3,  -2,   0],  # A
    [-1,   5,   0,  -2,  -3,   1,   0,  -2,   0,  -3,  -2,   2,  -1,  -3,  -2,  -1,  -1,  -3,  -2,  -3],  # R
    [-2,   0,   6,   1,  -3,   0,   0,   0,   1,  -3,  -3,   0,  -2,  -3,  -2,   1,   0,  -4,  -2,  -3],  # N
    [-2,  -2,   1,   6,  -3,   0,   2,  -1,  -1,  -3,  -4,  -1,  -3,  -3,  -1,   0,  -1,  -4,  -3,  -3],  # D
    [ 0,  -3,  -3,  -3,   9,  -3,  -4,  -3,  -3,  -1,  -1,  -3,  -1,  -2,  -3,  -1,  -1,  -2,  -2,  -1],  # C
    [-1,   1,   0,   0,  -3,   5,   2,  -2,   0,  -3,  -2,   1,   0,  -3,  -1,   0,  -1,  -2,  -1,  -2],  # Q
    [-1,   0,   0,   2,  -4,   2,   5,  -2,   0,  -3,  -3,   1,  -2,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # E
    [ 0,  -2,   0,  -1,  -3,  -2,  -2,   6,  -2,  -4,  -4,  -2,  -3,  -3,  -2,   0,  -2,  -2,  -3,  -3],  # G
    [-2,   0,   1,  -1,  -3,   0,   0,  -2,   8,  -3,  -3,  -1,  -2,  -1,  -2,  -1,  -2,  -2,   2,  -3],  # H
    [-1,  -3,  -3,  -3,  -1,  -3,  -3,  -4,  -3,   4,   2,  -3,   1,   0,  -3,  -2,  -1,  -3,  -1,   3],  # I
    [-1,  -2,  -3,  -4,  -1,  -2,  -3,  -4,  -3,   2,   4,  -2,   2,   0,  -3,  -2,  -1,  -2,  -1,   1],  # L
    [-1,   2,   0,  -1,  -3,   1,   1,  -2,  -1,  -3,  -2,   5,  -1,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # K
    [-1,  -1,  -2,  -3,  -1,   0,  -2,  -3,  -2,   1,   2,  -1,   5,   0,  -2,  -1,  -1,  -1,  -1,   1],  # M
    [-2,  -3,  -3,  -3,  -2,  -3,  -3,  -3,  -1,   0,   0,  -3,   0,   6,  -4,  -2,  -2,   1,   3,  -1],  # F
    [-1,  -2,  -2,  -1,  -3,  -1,  -1,  -2,  -2,  -3,  -3,  -1,  -2,  -4,   7,  -1,  -1,  -4,  -3,  -2],  # P
    [ 1,  -1,   1,   0,  -1,   0,   0,   0,  -1,  -2,  -2,   0,  -1,  -2,  -1,   4,   1,  -3,  -2,  -2],  # S
    [ 0,  -1,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   5,  -2,  -2,   0],  # T
    [-3,  -3,  -4,  -4,  -2,  -2,  -3,  -2,  -2,  -3,  -2,  -3,  -1,   1,  -4,  -3,  -2,  11,   2,  -3],  # W
    [-2,  -2,  -2,  -3,  -2,  -1,  -2,  -3,   2,  -1,  -1,  -2,  -1,   3,  -3,  -2,  -2,   2,   7,  -1],  # Y
    [ 0,  -3,  -3,  -3,  -1,  -2,  -2,  -3,  -3,   3,   1,  -2,   1,  -1,  -2,  -2,   0,  -3,  -1,   4],  # V
]
# to_idx = row (目标氨基酸 to), from_idx = col (源氨基酸 from)
# BLOSUM62[a, c] = 从 c 替换为 a 的得分（c→a）
BLOSUM62: np.ndarray = np.array(_BLOSUM62_ROWS, dtype=np.float32)


@dataclass(frozen=True)
class MutationProposal:
    """MCTS 用的离散 edit proposal。"""

    sequence: str
    edits: List[Tuple[int, str, str]]
    prior: float
    benefit: float
    action_group: str = "unrouted"
    group_prior: float = 1.0


def normalize_mcts_prior_mode(mode: str) -> str:
    """Normalize the StageB prior ablation mode used by MCTS."""
    normalized = str(mode or "full").strip().lower()
    if normalized not in MCTS_PRIOR_MODES:
        raise ValueError(
            f"Unknown mcts_prior_mode={mode!r}; expected one of {', '.join(MCTS_PRIOR_MODES)}"
        )
    return normalized


def normalize_mcts_proposal_method(method: str) -> str:
    """Normalize the StageB truncated-gradient proposal ablation method."""
    normalized = str(method or "one_step_fixed").strip().lower()
    if normalized not in MCTS_PROPOSAL_METHODS:
        raise ValueError(
            "Unknown mcts_proposal_method="
            f"{method!r}; expected one of {', '.join(MCTS_PROPOSAL_METHODS)}"
        )
    return normalized


def should_use_sequence_prior(prior_mode: str, use_prior: bool) -> bool:
    """Return whether BLOSUM62 sequence prior should be active."""
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    return bool(use_prior) and prior_mode not in {"uniform", "no_sequence"}


def should_use_gradient_prior(prior_mode: str) -> bool:
    """Return whether AF3 mutation-gradient signal should guide StageB proposals."""
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    return prior_mode not in {"uniform", "no_gradient"}


DE_NOVO_ACTION_GROUPS = (
    "core_stabilization",
    "compactness_repair",
    "interface_formation",
    "clash_repair",
    "specificity_repair",
    "solubility_repair",
)

ACTION_GROUP_LOSS_SCALES = {
    "plddt": 1.0,
    "pae": 1.0,
    "pde": 1.0,
    "con": 1.0,
    "i_pae": 1.0,
    "i_pde": 1.0,
    "i_con": 1.0,
    "rg": 1.0,
    "clash": 1.0,
    "i_clash": 1.0,
    "bond_violation": 0.5,
    "chainbreak": 0.5,
    "background_contact": 1.0,
    "decoy_margin": 1.0,
    "specificity": 1.0,
    "exposed_hydrophobicity": 1.0,
    "sequence_prior": 1.0,
    "solubility": 1.0,
}

ACTION_GROUP_LOSS_KEYS = {
    "core_stabilization": ("plddt", "pae", "con", "pde"),
    "compactness_repair": ("rg",),
    "interface_formation": ("i_pae", "i_con", "i_pde"),
    "clash_repair": ("clash", "i_clash"),
    "specificity_repair": ("background_contact", "decoy_margin", "specificity"),
    "solubility_repair": ("exposed_hydrophobicity", "sequence_prior", "solubility"),
}


def _loss_value(loss_vector: Dict[str, float], *keys: str) -> float:
    for key in keys:
        if key in loss_vector:
            return max(float(loss_vector[key]), 0.0)
    return 0.0


def _loss_weight(loss_weights: Optional[Dict[str, float]], key: str) -> float:
    if loss_weights is None:
        return 1.0
    return max(float(loss_weights.get(key, 0.0)), 0.0)


def _loss_scale(key: str, binder_length: Optional[int] = None) -> float:
    if key == "rg" and binder_length is not None and int(binder_length) > 0:
        return 2.38 * (float(binder_length) ** 0.365)
    return ACTION_GROUP_LOSS_SCALES.get(key, 1.0)


def _scaled_loss_value(
    loss_vector: Dict[str, float],
    key: str,
    weight: float = 1.0,
    binder_length: Optional[int] = None,
) -> float:
    value = _loss_value(loss_vector, key)
    scale = _loss_scale(key, binder_length=binder_length)
    return float(weight) * value / max(float(scale), 1e-8)


def _weighted_group_mean(
    loss_vector: Dict[str, float],
    keys: Tuple[str, ...],
    loss_weights: Optional[Dict[str, float]] = None,
    binder_length: Optional[int] = None,
) -> Optional[float]:
    values = []
    for key in keys:
        if key not in loss_vector:
            continue
        weight = _loss_weight(loss_weights, key)
        if weight <= 0.0:
            continue
        values.append(
            _scaled_loss_value(
                loss_vector,
                key,
                weight=weight,
                binder_length=binder_length,
            )
        )
    if not values or sum(values) <= 1e-8:
        return None
    return float(sum(values) / len(values))


def compute_de_novo_action_group_priors(
    loss_vector: Dict[str, float],
    temperature: float = 1.0,
    loss_weights: Optional[Dict[str, float]] = None,
    binder_length: Optional[int] = None,
) -> Dict[str, float]:
    """
    根据 de novo verifier loss vector 计算 action group prior。

    Stage 2 只做轻量 routing：使用当前 total loss 权重下的 normalized
    residual contribution，并在每个 group 内部取 mean，避免项数更多的
    group 天然占优。
    """
    raw_scores = {}
    for group in DE_NOVO_ACTION_GROUPS:
        score = _weighted_group_mean(
            loss_vector,
            ACTION_GROUP_LOSS_KEYS[group],
            loss_weights=loss_weights,
            binder_length=binder_length,
        )
        if score is not None:
            raw_scores[group] = score

    if not raw_scores:
        return {"unrouted": 1.0}

    scores = torch.tensor(
        [raw_scores[group] for group in raw_scores],
        dtype=torch.float32,
    )
    priors = torch.softmax(scores / max(temperature, 1e-8), dim=0)
    return {
        group: float(priors[i].item())
        for i, group in enumerate(raw_scores)
    }


def select_de_novo_action_group(
    loss_vector: Dict[str, float],
    temperature: float = 1.0,
    loss_weights: Optional[Dict[str, float]] = None,
    binder_length: Optional[int] = None,
) -> Dict[str, object]:
    """
    选择当前 de novo node 最需要修复的 action group。
    """
    priors = compute_de_novo_action_group_priors(
        loss_vector,
        temperature=temperature,
        loss_weights=loss_weights,
        binder_length=binder_length,
    )
    action_group = max(priors, key=priors.get)
    return {
        "action_group": action_group,
        "group_prior": priors[action_group],
        "group_priors": priors,
    }


def _seq_to_indices(seq: str) -> List[int]:
    """单字母序列 -> 索引列表"""
    return [AA_1_TO_IDX[aa] for aa in seq]


def _indices_to_seq(indices: List[int]) -> str:
    """索引列表 -> 单字母序列"""
    return "".join(AA_1LETTER[PROTEIN_RESIDUE_NAMES[i]] for i in indices)


def compute_mutation_benefit(
    grad_matrix: torch.Tensor,
    current_indices: List[int],
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> torch.Tensor:
    """
    计算突变收益矩阵 Delta_L，并可选地引入 BLOSUM62 生物物理先验。

    Delta_L[i, a] = G[i, a] - G[i, c_i]
    其中 c_i 是位置 i 当前的氨基酸索引。
    负值表示突变为 a 可降低 loss（有利突变）。

    当 use_prior=True 时，用 BLOSUM62 降低保守替换的有效 delta：
      adjusted[i, a] = delta_L[i, a] - prior_strength * BLOSUM62[a, c_i]
    BLOSUM62[a, c] 为从 c 替换为 a 的演化保守性得分（正值=保守）。
    这会使演化上合理的突变（高 BLOSUM62 分）更容易被选中，
    而不合理的突变（低分或负分）被抑制。

    Args:
        grad_matrix:      [L_binder, 20] 梯度矩阵
        current_indices:  长度 L_binder 的当前氨基酸索引列表
        use_prior:        是否应用 BLOSUM62 先验约束掩码
        prior_strength:   先验权重系数（推荐 0.05~0.2，太大则梯度信息被淹没）

    Returns:
        delta_L: [L_binder, 20] 突变收益矩阵；后续 benefit = -delta_L。
    """
    L = grad_matrix.shape[0]
    current_grad = grad_matrix[torch.arange(L), current_indices]  # [L]
    delta_L = grad_matrix - current_grad.unsqueeze(1)  # [L, 20]

    if use_prior:
        # BLOSUM62[current_a, current_c] — 从 c_i 替换为 a 的保守性得分
        current_indices_arr = np.array(current_indices, dtype=np.int64)  # [L]
        # 从 BLOSUM62 中按行（to）和列（from）索引
        prior_scores = BLOSUM62[:, current_indices_arr]  # [20, L]
        prior_scores = prior_scores.T  # [L, 20]
        delta_L = delta_L - prior_strength * torch.from_numpy(prior_scores).to(delta_L)
        # 保留自身突变 delta=0 的不变量，方便旧单元测试和下游推理。
        delta_L[torch.arange(L), current_indices] = 0.0

    return delta_L


def compute_mutation_benefit_legacy(
    grad_matrix: torch.Tensor,
    current_indices: List[int],
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> torch.Tensor:
    """
    GitHub main-compatible mutation benefit for legacy MCTS bootstrap.

    This intentionally keeps the original BLOSUM62 direction used by the old
    MCTS path. StageB uses compute_mutation_benefit(), which has the corrected
    document-style prior direction.
    """
    L = grad_matrix.shape[0]
    current_grad = grad_matrix[torch.arange(L), current_indices]
    delta_L = grad_matrix - current_grad.unsqueeze(1)

    if use_prior:
        current_indices_arr = np.array(current_indices, dtype=np.int64)
        prior_scores = BLOSUM62[:, current_indices_arr].T
        delta_L = delta_L + prior_strength * torch.from_numpy(prior_scores).to(delta_L)

    return delta_L


def _middle_position_profile(length: int) -> torch.Tensor:
    if length <= 1:
        return torch.ones(length, dtype=torch.float32)
    pos = torch.arange(length, dtype=torch.float32)
    center = (length - 1) / 2.0
    denom = max(center, 1.0)
    return 1.0 - torch.abs(pos - center) / denom


def _binary_aa_profile(indices: List[int], preferred: set) -> torch.Tensor:
    return torch.tensor(
        [1.0 if idx in preferred else 0.0 for idx in indices],
        dtype=torch.float32,
    )


def _aa_target_profile(preferred: set, discouraged: Optional[set] = None) -> torch.Tensor:
    profile = torch.zeros(NUM_AMINO_ACIDS, dtype=torch.float32)
    for idx in preferred:
        profile[idx] += 1.0
    for idx in (discouraged or set()):
        profile[idx] -= 1.0
    return profile


def _coerce_position_weights(raw_value: Any, length: int) -> Optional[torch.Tensor]:
    if raw_value is None or length <= 0:
        return None
    if isinstance(raw_value, torch.Tensor):
        values = raw_value.detach().cpu().flatten().float()
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text or text == "unknown":
            return None
        if set(text) <= {"0", "1"} and len(text) == length:
            values = torch.tensor([1.0 if ch == "1" else 0.0 for ch in text], dtype=torch.float32)
        else:
            parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
            try:
                indices = [int(part) for part in parts]
            except ValueError:
                return None
            values = torch.zeros(length, dtype=torch.float32)
            for idx in indices:
                if 0 <= idx < length:
                    values[idx] = 1.0
    elif isinstance(raw_value, dict):
        values = torch.zeros(length, dtype=torch.float32)
        for key, value in raw_value.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < length:
                values[idx] = float(value)
    elif isinstance(raw_value, (list, tuple, set, np.ndarray)):
        raw_list = list(raw_value)
        if len(raw_list) == length and all(isinstance(v, (bool, int, float)) for v in raw_list):
            values = torch.tensor([float(v) for v in raw_list], dtype=torch.float32)
        else:
            values = torch.zeros(length, dtype=torch.float32)
            for item in raw_list:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < length:
                    values[idx] = 1.0
    else:
        return None

    if values.numel() != length:
        return None
    values = torch.clamp(values, min=0.0)
    max_value = float(values.max().item()) if values.numel() else 0.0
    if max_value <= 0.0:
        return None
    return values / max_value


ACTION_GROUP_POSITION_KEYS = {
    "core_stabilization": (
        "core_residue_indices",
        "buried_residue_indices",
        "non_interface_residue_indices",
    ),
    "compactness_repair": (
        "non_interface_residue_indices",
        "core_residue_indices",
        "helix_bundle_residue_indices",
    ),
    "interface_formation": (
        "target_facing_residue_indices",
        "interface_residue_indices",
        "hotspot_facing_residue_indices",
        "antigen_facing_residue_indices",
        "surface_residue_indices",
    ),
    "clash_repair": (
        "clash_residue_indices",
        "clash_near_residue_indices",
        "steric_conflict_residue_indices",
    ),
    "specificity_repair": (
        "background_contact_residue_indices",
        "non_hotspot_contact_residue_indices",
        "wrong_contact_residue_indices",
        "interface_residue_indices",
    ),
    "solubility_repair": (
        "surface_residue_indices",
        "exposed_hydrophobic_residue_indices",
        "solubility_residue_indices",
    ),
}


def extract_action_group_position_weights(
    sequence: str,
    action_group: Optional[str],
    structural_signature: Optional[Dict[str, Any]] = None,
) -> Optional[torch.Tensor]:
    """
    Convert structural routing metadata into per-residue mutation weights.

    The keys mirror the MCTS document's preferred mutation regions. If the
    current verifier does not provide residue-level maps yet, this returns None
    and the caller falls back to the existing sequence-level soft bias.
    """
    signature = structural_signature or {}
    length = len(sequence)
    for key in ACTION_GROUP_POSITION_KEYS.get(action_group or "unrouted", ()):
        weights = _coerce_position_weights(signature.get(key), length)
        if weights is not None:
            return weights
    return None


def apply_position_weight_bias(
    delta_L: torch.Tensor,
    position_weights: Optional[torch.Tensor],
    strength: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if position_weights is None or strength <= 0.0:
        return delta_L, {"position_bias_max": 0.0, "position_bias_mean": 0.0}
    weights = position_weights.detach().cpu().flatten().float()
    if weights.numel() != delta_L.shape[0] or float(weights.max().item()) <= 0.0:
        return delta_L, {"position_bias_max": 0.0, "position_bias_mean": 0.0}
    weights = torch.clamp(weights, min=0.0)
    weights = weights / max(float(weights.max().item()), 1e-8)
    bonus = strength * weights.unsqueeze(1)
    adjusted = delta_L - bonus.to(delta_L)
    return adjusted, {
        "position_bias_max": float(bonus.max().item()) if bonus.numel() else 0.0,
        "position_bias_mean": float(bonus.mean().item()) if bonus.numel() else 0.0,
    }


def apply_position_weight_mask(
    delta_L: torch.Tensor,
    position_weights: Optional[torch.Tensor],
    min_active_positions: int = 1,
    mask_penalty: float = 1.0e6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Restrict mutation proposals to structurally routed positions when available.

    The document's action groups are region-specific: interface repair should
    spend verifier budget on target-facing/interface-shell residues, clash
    repair near clashes, and core repair in core/buried residues. This helper
    makes that routing strict only when the verifier supplied a non-empty
    residue map; otherwise callers keep the existing sequence-level soft bias.
    """
    if position_weights is None:
        return delta_L, {"position_mask_active": 0.0, "position_mask_fraction": 0.0}

    weights = position_weights.detach().cpu().flatten().float()
    if weights.numel() != delta_L.shape[0]:
        return delta_L, {"position_mask_active": 0.0, "position_mask_fraction": 0.0}

    active = weights > 0.0
    active_count = int(active.sum().item())
    if active_count < max(int(min_active_positions), 1):
        return delta_L, {"position_mask_active": 0.0, "position_mask_fraction": 0.0}

    adjusted = delta_L.clone()
    adjusted[~active.to(adjusted.device), :] = float(mask_penalty)
    return adjusted, {
        "position_mask_active": float(active_count),
        "position_mask_fraction": float(active_count) / max(int(weights.numel()), 1),
    }


def apply_action_group_mutation_bias(
    delta_L: torch.Tensor,
    current_indices: List[int],
    action_group: Optional[str],
    strength: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Softly bias mutation scores according to the selected action group.

    This is intentionally a soft bonus rather than a hard mask. The current
    de novo path does not yet expose residue-level hotspot or clash atom maps,
    so the bias uses conservative sequence/position proxies:
      - clash repair: mutate bulky current residues toward smaller residues;
      - interface/specificity/solubility: prefer exposed-like polar/charged
        and aromatic/polar replacements;
      - core/compactness: prefer internal positions and compact hydrophobics.

    Later structural routing can replace the position profile while keeping this
    helper as the single place where group-specific mutation priors enter.
    """
    group = action_group or "unrouted"
    if group == "unrouted" or strength <= 0.0:
        return delta_L, {"routing_bias_max": 0.0, "routing_bias_mean": 0.0}

    L = delta_L.shape[0]
    position_bonus = torch.zeros(L, dtype=torch.float32)
    aa_bonus = torch.zeros(NUM_AMINO_ACIDS, dtype=torch.float32)
    current_profile = torch.zeros(L, dtype=torch.float32)
    middle_profile = _middle_position_profile(L)

    if group == "core_stabilization":
        position_bonus = 0.6 * middle_profile
        aa_bonus = 0.45 * _aa_target_profile(HYDROPHOBIC_AA, discouraged=CHARGED_AA | RISKY_AA)
    elif group == "compactness_repair":
        position_bonus = 0.5 * middle_profile
        aa_bonus = 0.35 * _aa_target_profile(HYDROPHOBIC_AA | SMALL_AA, discouraged=CHARGED_AA | {AA_1_TO_IDX["P"]})
    elif group == "interface_formation":
        current_profile = _binary_aa_profile(current_indices, POLAR_CHARGED_AA | AROMATIC_AA)
        position_bonus = 0.35 * current_profile
        aa_bonus = 0.45 * _aa_target_profile(POLAR_CHARGED_AA | AROMATIC_AA, discouraged={AA_1_TO_IDX["C"]})
    elif group == "clash_repair":
        current_profile = _binary_aa_profile(current_indices, BULKY_AA)
        position_bonus = 0.7 * current_profile
        aa_bonus = 0.75 * _aa_target_profile(SMALL_AA, discouraged=BULKY_AA | {AA_1_TO_IDX["P"]})
    elif group == "specificity_repair":
        current_profile = _binary_aa_profile(current_indices, HYDROPHOBIC_AA)
        position_bonus = 0.35 * current_profile
        aa_bonus = 0.45 * _aa_target_profile(POLAR_CHARGED_AA | AROMATIC_AA, discouraged={AA_1_TO_IDX["C"]})
    elif group == "solubility_repair":
        current_profile = _binary_aa_profile(current_indices, HYDROPHOBIC_AA | {AA_1_TO_IDX["C"]})
        position_bonus = 0.55 * current_profile
        aa_bonus = 0.60 * _aa_target_profile(POLAR_CHARGED_AA | SMALL_AA, discouraged=HYDROPHOBIC_AA | {AA_1_TO_IDX["C"]})
    else:
        return delta_L, {"routing_bias_max": 0.0, "routing_bias_mean": 0.0}

    bonus = strength * (position_bonus.unsqueeze(1) + aa_bonus.unsqueeze(0))
    adjusted = delta_L - bonus.to(delta_L)
    adjusted[torch.arange(L), current_indices] = 0.0
    return adjusted, {
        "routing_bias_max": float(bonus.max().item()) if bonus.numel() else 0.0,
        "routing_bias_mean": float(bonus.mean().item()) if bonus.numel() else 0.0,
    }


def select_top_k_mutations(
    delta_L: torch.Tensor,
    current_indices: List[int],
    K_top: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    选取收益最大的 Top-K 候选突变（排除自身突变）。

    Args:
        delta_L: [L_binder, 20] 突变收益矩阵
        current_indices: 当前氨基酸索引
        K_top: 选取数量

    Returns:
        positions: [K] 突变位置
        new_aas:   [K] 突变后的氨基酸索引
        benefits:  [K] 对应的 -delta_L 值（越大越好）
    """
    L = delta_L.shape[0]
    benefit = -delta_L.clone()  # [L, 20]，正值 = 有利

    # 将自身突变的收益设为 -inf，使其不会被选中
    for i, c in enumerate(current_indices):
        benefit[i, c] = float("-inf")

    flat_benefit = benefit.reshape(-1)  # [L * 20]
    K_actual = min(K_top, int((flat_benefit > float("-inf")).sum().item()))
    if K_actual == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([])

    topk_vals, topk_flat_idx = torch.topk(flat_benefit, K_actual)
    positions = topk_flat_idx // NUM_AMINO_ACIDS
    new_aas = topk_flat_idx % NUM_AMINO_ACIDS

    return positions, new_aas, topk_vals


def sample_candidate_sequences(
    current_indices: List[int],
    positions: torch.Tensor,
    new_aas: torch.Tensor,
    benefits: torch.Tensor,
    B_candidates: int,
    T_sample: float,
    max_mutations_per_seq: int,
) -> List[List[int]]:
    """
    从 Top-K 候选突变中采样组合，生成 B 条候选序列。

    采样策略：
    1. 按 Softmax(benefits / T_sample) 构造突变采样分布
    2. 每条候选随机选择 1~max_mutations_per_seq 个突变
    3. 同一位置只保留最后一个突变（避免冲突）

    Args:
        current_indices: 当前序列索引
        positions:   [K] 突变位置
        new_aas:     [K] 突变氨基酸
        benefits:    [K] 突变收益
        B_candidates: 生成候选数
        T_sample:     采样温度
        max_mutations_per_seq: 每条候选最大突变数

    Returns:
        candidates: B 条候选序列（索引列表的列表）
    """
    K = positions.shape[0]
    if K == 0:
        return [list(current_indices)] * B_candidates

    probs = F.softmax(benefits / max(T_sample, 1e-8), dim=0)  # [K]

    candidates = []
    seen = set()

    for _ in range(B_candidates * 5):  # 多试几次以去重
        if len(candidates) >= B_candidates:
            break

        n_mut = torch.randint(1, max_mutations_per_seq + 1, (1,)).item()
        n_mut = min(n_mut, K)

        chosen_idx = torch.multinomial(probs, n_mut, replacement=False)

        seq = list(current_indices)
        for ci in chosen_idx.tolist():
            pos = positions[ci].item()
            aa = new_aas[ci].item()
            seq[pos] = aa

        key = tuple(seq)
        if key not in seen and key != tuple(current_indices):
            seen.add(key)
            candidates.append(seq)

    # 如果去重后不够 B 条，用已有的填充
    while len(candidates) < B_candidates:
        if candidates:
            candidates.append(list(candidates[len(candidates) % len(candidates)]))
        else:
            candidates.append(list(current_indices))

    return candidates[:B_candidates]


def sample_mutation_actions(
    current_indices: List[int],
    positions: torch.Tensor,
    new_aas: torch.Tensor,
    benefits: torch.Tensor,
    B_candidates: int,
    T_sample: float,
    max_mutations_per_seq: int,
    action_group: str = "unrouted",
    group_prior: float = 1.0,
    prior_mode: str = "full",
) -> List[MutationProposal]:
    """
    从 Top-K 突变中采样 MCTS action，并返回序列、edit、prior、benefit。

    prior 在最终返回前会按候选集合重新归一化，供 PUCT 使用。
    no_gradient/uniform 模式下，PUCT action prior 不使用 gradient benefit。
    """
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    K = positions.shape[0]
    if K == 0:
        return []

    probs = F.softmax(benefits / max(T_sample, 1e-8), dim=0)
    proposals = []
    seen = set()

    for _ in range(B_candidates * 8):
        if len(proposals) >= B_candidates:
            break

        n_mut = torch.randint(1, max_mutations_per_seq + 1, (1,)).item()
        n_mut = min(n_mut, K)
        chosen_idx = torch.multinomial(probs, n_mut, replacement=False).tolist()

        seq = list(current_indices)
        edit_by_pos = {}
        raw_prior = 0.0
        total_benefit = 0.0
        for ci in chosen_idx:
            pos = int(positions[ci].item())
            aa = int(new_aas[ci].item())
            old = current_indices[pos]
            if aa == old:
                continue
            seq[pos] = aa
            edit_by_pos[pos] = (old, aa)
            if prior_mode not in {"uniform", "no_gradient"}:
                raw_prior += float(probs[ci].item())
            total_benefit += float(benefits[ci].item())

        key = tuple(seq)
        if not edit_by_pos or key == tuple(current_indices) or key in seen:
            continue
        if prior_mode in {"uniform", "no_gradient"}:
            raw_prior = 1.0

        seen.add(key)
        edits = [
            (
                pos,
                AA_1LETTER[PROTEIN_RESIDUE_NAMES[old]],
                AA_1LETTER[PROTEIN_RESIDUE_NAMES[new]],
            )
            for pos, (old, new) in sorted(edit_by_pos.items())
        ]
        proposals.append(
            MutationProposal(
                sequence=_indices_to_seq(seq),
                edits=edits,
                prior=max(raw_prior, 1e-12),
                benefit=total_benefit,
                action_group=action_group,
                group_prior=group_prior,
            )
        )

    while len(proposals) < B_candidates and proposals:
        proposals.append(proposals[len(proposals) % len(proposals)])

    if proposals:
        prior_sum = sum(p.prior for p in proposals)
        proposals = [
            MutationProposal(
                sequence=p.sequence,
                edits=p.edits,
                prior=float(group_prior * p.prior / max(prior_sum, 1e-12)),
                benefit=p.benefit,
                action_group=p.action_group,
                group_prior=p.group_prior,
            )
            for p in proposals[:B_candidates]
        ]

    return proposals[:B_candidates]


def propose_mutation_actions(
    grad_matrix: torch.Tensor,
    current_seq: str,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    use_prior: bool = True,
    prior_strength: float = 0.1,
    action_group: Optional[str] = None,
    group_prior: float = 1.0,
    routing_strength: float = 0.25,
    structural_signature: Optional[Dict[str, Any]] = None,
    position_weights: Optional[torch.Tensor] = None,
    position_bias_strength: float = 0.5,
    enforce_position_mask: bool = True,
    prior_mode: str = "full",
) -> List[MutationProposal]:
    """
    MCTS 专用候选生成：返回 sequence + edit metadata + PUCT prior。
    """
    prior_mode = normalize_mcts_prior_mode(prior_mode)
    current_indices = _seq_to_indices(current_seq)
    grad_cpu = grad_matrix.detach().cpu()
    if not should_use_gradient_prior(prior_mode):
        grad_cpu = torch.zeros_like(grad_cpu)

    delta_L = compute_mutation_benefit(
        grad_cpu, current_indices,
        use_prior=should_use_sequence_prior(prior_mode, use_prior),
        prior_strength=prior_strength,
    )
    delta_L, routing_stats = apply_action_group_mutation_bias(
        delta_L,
        current_indices,
        action_group=action_group,
        strength=routing_strength,
    )
    if position_weights is None:
        position_weights = extract_action_group_position_weights(
            current_seq,
            action_group,
            structural_signature=structural_signature,
        )
    delta_L, position_stats = apply_position_weight_bias(
        delta_L,
        position_weights,
        strength=position_bias_strength,
    )
    mask_stats = {"position_mask_active": 0.0, "position_mask_fraction": 0.0}
    if enforce_position_mask and position_weights is not None:
        delta_L, mask_stats = apply_position_weight_mask(
            delta_L,
            position_weights,
        )
    candidate_k_top = K_top
    if prior_mode == "uniform":
        candidate_k_top = delta_L.shape[0] * max(NUM_AMINO_ACIDS - 1, 1)
    positions, new_aas, benefits = select_top_k_mutations(
        delta_L, current_indices, candidate_k_top
    )

    if positions.shape[0] == 0:
        logger.warning("No valid mutations found, returning no MCTS proposals")
        return []

    proposals = sample_mutation_actions(
        current_indices, positions, new_aas, benefits,
        B_candidates, T_sample, max_mutations_per_seq,
        action_group=action_group or "unrouted",
        group_prior=group_prior,
        prior_mode=prior_mode,
    )
    if proposals:
        logger.info(
            f"Proposed {len(proposals)} MCTS actions | "
            f"prior_mode={prior_mode} | "
            f"group={proposals[0].action_group} P_group={group_prior:.4f} | "
            f"Top-K={positions.shape[0]} mutations | "
            f"Best benefit={benefits[0]:.4f} at pos {positions[0].item()} | "
            f"best_prior={max(p.prior for p in proposals):.4f} | "
            f"routing_bias=max:{routing_stats['routing_bias_max']:.3f},"
            f"mean:{routing_stats['routing_bias_mean']:.3f} | "
            f"position_bias=max:{position_stats['position_bias_max']:.3f},"
            f"mean:{position_stats['position_bias_mean']:.3f} | "
            f"position_mask=active:{int(mask_stats['position_mask_active'])},"
            f"frac:{mask_stats['position_mask_fraction']:.3f}"
        )
    return proposals


def propose_mutations_legacy(
    grad_matrix: torch.Tensor,
    current_seq: str,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> List[str]:
    """
    GitHub main-compatible candidate generation for StageA legacy MCTS.

    Returns a plain List[str], evaluates up to B_candidates sequences, and does
    not attach action metadata or action-group routing.
    """
    current_indices = _seq_to_indices(current_seq)
    grad_cpu = grad_matrix.detach().cpu()

    delta_L = compute_mutation_benefit_legacy(
        grad_cpu,
        current_indices,
        use_prior=use_prior,
        prior_strength=prior_strength,
    )

    positions, new_aas, benefits = select_top_k_mutations(
        delta_L,
        current_indices,
        K_top,
    )

    if positions.shape[0] == 0:
        logger.warning("No valid legacy mutations found, returning current sequence copies")
        return [current_seq] * B_candidates

    candidate_indices = sample_candidate_sequences(
        current_indices,
        positions,
        new_aas,
        benefits,
        B_candidates,
        T_sample,
        max_mutations_per_seq,
    )
    candidate_seqs = [_indices_to_seq(idx_list) for idx_list in candidate_indices]

    logger.info(
        f"Proposed {len(candidate_seqs)} legacy candidates | "
        f"Top-K={positions.shape[0]} mutations | "
        f"Best benefit={benefits[0]:.4f} at pos {positions[0].item()}"
    )

    return candidate_seqs


def propose_mutations(
    grad_matrix: torch.Tensor,
    current_seq: str,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    use_prior: bool = True,
    prior_strength: float = 0.1,
) -> List[str]:
    """
    梯度引导的离散候选序列生成（含可选 BLOSUM62 先验约束）。

    完整流程：
    1. 将梯度矩阵转换为突变收益矩阵 Delta_L
    2. （可选）叠加 BLOSUM62 先验约束掩码 M_prior
    3. 选取 Top-K 候选突变
    4. 按收益分布采样组合，生成 B 条离散候选序列

    Args:
        grad_matrix:          [L_binder, 20] 梯度矩阵 G = nabla_Theta L
        current_seq:         当前离散序列（单字母 str）
        K_top:              选取 Top-K 候选突变数
        B_candidates:       生成 B 条候选序列
        T_sample:           采样温度（越小越确定性，越大越随机）
        max_mutations_per_seq: 每条候选的最大突变数
        use_prior:          是否启用 BLOSUM62 先验约束掩码（M_prior）
        prior_strength:     先验权重系数（推荐 0.05~0.2）

    Returns:
        B 条离散候选序列（单字母 str 列表）
    """
    proposals = propose_mutation_actions(
        grad_matrix=grad_matrix,
        current_seq=current_seq,
        K_top=K_top,
        B_candidates=B_candidates,
        T_sample=T_sample,
        max_mutations_per_seq=max_mutations_per_seq,
        use_prior=use_prior,
        prior_strength=prior_strength,
    )
    if not proposals:
        logger.warning("No valid mutations found, returning current sequence copies")
        return [current_seq] * B_candidates
    return [p.sequence for p in proposals]
