"""ESMFold2 proposal, candidate and verifier contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import torch


# ESMFold2 residue order from Biohub ESMFold2 constants:
# ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
# LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
ESM_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_ESM_IDX = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}
CYS_INDEX = AA_TO_ESM_IDX["C"]

PAIR_CHAIN_IPTM_KEYS = (
    "pair_chain_iptm",
    "pair_chain_i_ptm",
    "pair_chains_iptm",
    "chain_pair_iptm",
    "binder_target_pair_iptm",
)
ESMFOLD2_ACTION_GROUP_ROUTING_KEYS = {
    "interface_formation": "pair_chain_iptm_loss",
    "core_stabilization": "binder_confidence_loss",
    "compactness_repair": "binder_rg_excess_loss",
}
ESMFOLD2_ACTION_GROUP_PROPOSAL_LOSS_KEYS = {
    "interface_formation": {"inter_contact_loss": 1.0},
    "core_stabilization": {"intra_contact_loss": 1.0},
    "compactness_repair": {"glob_loss": 1.0},
}
AUTO_ESMC_LAMBDA_BY_TASK = {
    "denovo": 0.15,
    "de_novo": 0.15,
    "minibinder": 0.15,
    "binder": 0.15,
    "tcr": 0.05,
    "cdr": 0.05,
    "cdr_redesign": 0.05,
    "vhh_redesign": 0.05,
    "antibody": 0.05,
    "antibody_like": 0.05,
}


@dataclass(frozen=True)
class ESMFold2ProposalConfig:
    """Configuration for the first ESMFold2 Stage-A proposal backend."""

    experimental_model_name: str = "biohub/ESMFold2-Experimental-Fast"
    verifier_model_name: str = "biohub/ESMFold2"
    optimization_root: Optional[str] = None
    proposal_optimization: str = "off"
    verifier_optimization: str = "off"
    esmc_optimization: str = "off"
    optimization_report_path: Optional[str] = None
    optimization_timing: bool = False
    optimization_graph_max_tokens: Optional[int] = None
    memory_mode: str = "resident"
    memory_token_threshold: int = 1024
    memory_min_free_gib: Optional[float] = None
    memory_saved_tensor_pin_memory: bool = True
    memory_prior_residency: str = "per_step"
    verifier_num_loops: int = 20
    verifier_num_sampling_steps: int = 200
    verifier_structure_mode: str = "sequence_only"
    soft_steps: int = 30
    learning_rate: float = 0.05
    temperature_min: float = 1e-2
    temperature_max: float = 1.0
    lambda_esmc: Union[str, float] = "auto"
    esmc_n_passes: int = 4
    esmc_batch_size: int = 4
    esmc_activation_checkpointing: bool = False
    num_loops: int = 1
    num_sampling_steps: int = 1
    proposal_load_internal_esmc: bool = True
    proposal_lm_dropout: Optional[float] = 0.5
    force_proposal_lm_dropout: bool = True
    one_step_diffusion_enabled: bool = True
    one_step_diffusion_mode: str = "truncated_trajectory"
    one_step_diffusion_tau: float = 1.0
    one_step_diffusion_truncated_num_sampling_steps: int = 50
    one_step_diffusion_truncated_schedule_mode: str = "official_clipped"
    one_step_diffusion_truncated_dense_num_sampling_steps: int = 41
    one_step_diffusion_weight: float = 0.25
    one_step_diffusion_rg_weight: float = 0.1
    one_step_diffusion_interval: int = 5
    one_step_diffusion_recycle_coordinates: bool = False
    one_step_diffusion_hotspot_enabled: bool = False
    distogram_hotspot_enabled: bool = True
    verifier_loss_mode: str = "pair_chain_iptm"
    hotspot_weight: float = 0.0
    hotspot_satisfied_proposal_weight: float = 0.0
    hotspot_cutoff: float = 8.0
    hotspot_interface_mode: str = "additive"
    hotspot_verify_weight: float = 0.0
    hotspot_verify_near_cutoff: float = 8.0
    hotspot_verify_far_cutoff: float = 20.0
    hotspot_verify_min_distance: float = 2.5
    hotspot_verify_too_close_weight: float = 0.0
    stageb_group_weight: float = 1.0
    track_proposal_confidence: bool = False
    proposal_confidence_temperature: float = 0.05
    proposal_confidence_interval: int = 1
    proposal_confidence_num_loops: int = 1
    proposal_confidence_num_sampling_steps: int = 50
    proposal_confidence_max_candidates: int = 8
    proposal_confidence_tail_steps: int = 0
    target_chain_roles: Optional[Tuple[str, ...]] = None
    # Explicit opt-in: legacy single-/multi-chain modes retain their scoring.
    target_assembly: str = "monomer"
    target_interface_mode: str = "all"
    target_interface_contact_cutoff: float = 5.0
    target_interface_min_contacts: int = 5
    binder_mutable_mask: Optional[Tuple[bool, ...]] = None
    template_structure_path: Optional[str] = None
    template_chain_ids: Optional[Tuple[str, ...]] = None
    template_chain_sequences: Optional[Tuple[str, ...]] = None
    binder_structure_regeneration_mask: Optional[Tuple[bool, ...]] = None

    def __post_init__(self) -> None:
        if self.target_assembly not in {"monomer", "homotrimer"}:
            raise ValueError("target_assembly must be monomer or homotrimer")
        if self.target_interface_mode not in {"all", "contacted"}:
            raise ValueError("target_interface_mode must be all or contacted")
        if not math.isfinite(float(self.target_interface_contact_cutoff)) or float(self.target_interface_contact_cutoff) <= 0:
            raise ValueError("target_interface_contact_cutoff must be finite and positive")
        if (isinstance(self.target_interface_min_contacts, bool)
                or not isinstance(self.target_interface_min_contacts, int)
                or self.target_interface_min_contacts < 1):
            raise ValueError("target_interface_min_contacts must be a positive integer")
        if self.target_interface_mode == "contacted" or self.target_assembly == "homotrimer":
            if self.target_assembly != "homotrimer" or self.target_interface_mode != "contacted":
                raise ValueError("homotrimer and contacted interface scoring must be enabled together")
            if self.verifier_loss_mode != "aligned_common":
                raise ValueError("contacted interface scoring requires aligned_common verifier raw output")
            if self.target_chain_roles is not None and tuple(self.target_chain_roles) != ("target",) * 3:
                raise ValueError("homotrimer requires three target chain roles")
        cap = self.optimization_graph_max_tokens
        if self.memory_mode not in {"resident", "auto", "phase_offload"}:
            raise ValueError("memory_mode must be resident, auto or phase_offload")
        if not isinstance(self.memory_saved_tensor_pin_memory, bool):
            raise ValueError("memory_saved_tensor_pin_memory must be a boolean")
        if self.memory_prior_residency not in ("per_step", "proposal", "between_diffusion"):
            raise ValueError("memory_prior_residency must be per_step, proposal or between_diffusion")
        if (isinstance(self.memory_token_threshold, bool)
                or not isinstance(self.memory_token_threshold, int)
                or self.memory_token_threshold < 1):
            raise ValueError("memory_token_threshold must be a positive integer")
        if self.memory_min_free_gib is not None and (
                isinstance(self.memory_min_free_gib, bool)
                or not isinstance(self.memory_min_free_gib, (int, float))
                or not math.isfinite(float(self.memory_min_free_gib))
                or float(self.memory_min_free_gib) <= 0):
            raise ValueError("memory_min_free_gib must be finite and positive or None")
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError("optimization_graph_max_tokens must be a nonnegative integer or None")
        for name in ("proposal_optimization", "verifier_optimization"):
            if getattr(self, name) not in {"off", "exact", "fast", "big"}:
                raise ValueError(f"{name} must be off, exact, fast or big")
        if self.esmc_optimization not in {"off", "exact"}:
            raise ValueError("esmc_optimization must be off or exact")
        if int(self.verifier_num_loops) <= 0:
            raise ValueError("verifier_num_loops must be positive")
        if int(self.verifier_num_sampling_steps) <= 0:
            raise ValueError("verifier_num_sampling_steps must be positive")
        verifier_structure_mode = str(self.verifier_structure_mode).strip().lower()
        if verifier_structure_mode not in {"sequence_only", "scaffold_consistency"}:
            raise ValueError(
                "verifier_structure_mode must be 'sequence_only' or "
                "'scaffold_consistency'"
            )
        if int(self.soft_steps) <= 0:
            raise ValueError("soft_steps must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.temperature_min) <= 0.0:
            raise ValueError("temperature_min must be positive")
        if float(self.temperature_max) < float(self.temperature_min):
            raise ValueError("temperature_max must be >= temperature_min")
        if float(self.stageb_group_weight) < 0.0:
            raise ValueError("stageb_group_weight must be non-negative")
        if float(self.hotspot_weight) < 0.0:
            raise ValueError("hotspot_weight must be non-negative")
        if float(self.hotspot_satisfied_proposal_weight) < 0.0:
            raise ValueError(
                "hotspot_satisfied_proposal_weight must be non-negative"
            )
        if float(self.hotspot_cutoff) <= 0.0:
            raise ValueError("hotspot_cutoff must be positive")
        hotspot_interface_mode = str(self.hotspot_interface_mode).strip().lower()
        if hotspot_interface_mode not in {"additive", "localized"}:
            raise ValueError(
                "hotspot_interface_mode must be 'additive' or 'localized'"
            )
        if hotspot_interface_mode == "localized" and float(self.hotspot_weight) > 1.0:
            raise ValueError(
                "localized hotspot_interface_mode requires hotspot_weight in [0, 1]"
            )
        if (
            hotspot_interface_mode == "localized"
            and float(self.hotspot_satisfied_proposal_weight) > 1.0
        ):
            raise ValueError(
                "localized hotspot_interface_mode requires "
                "hotspot_satisfied_proposal_weight in [0, 1]"
            )
        if float(self.hotspot_verify_weight) < 0.0:
            raise ValueError("hotspot_verify_weight must be non-negative")
        if float(self.hotspot_verify_near_cutoff) < 0.0:
            raise ValueError("hotspot_verify_near_cutoff must be non-negative")
        if float(self.hotspot_verify_far_cutoff) <= float(
            self.hotspot_verify_near_cutoff
        ):
            raise ValueError(
                "hotspot_verify_far_cutoff must be greater than "
                "hotspot_verify_near_cutoff"
            )
        if float(self.hotspot_verify_min_distance) < 0.0:
            raise ValueError("hotspot_verify_min_distance must be non-negative")
        if float(self.hotspot_verify_min_distance) > float(
            self.hotspot_verify_near_cutoff
        ):
            raise ValueError(
                "hotspot_verify_min_distance must not exceed "
                "hotspot_verify_near_cutoff"
            )
        if float(self.hotspot_verify_too_close_weight) < 0.0:
            raise ValueError(
                "hotspot_verify_too_close_weight must be non-negative"
            )
        if self.proposal_lm_dropout is not None:
            dropout = float(self.proposal_lm_dropout)
            if dropout < 0.0 or dropout >= 1.0:
                raise ValueError("proposal_lm_dropout must be in [0, 1)")
        if float(self.one_step_diffusion_tau) <= 0.0:
            raise ValueError("one_step_diffusion_tau must be positive")
        one_step_mode = str(self.one_step_diffusion_mode).strip().lower()
        supported_one_step_modes = {
            "reference_noising",
            "truncated_trajectory",
            "template_anchored_truncated_trajectory",
        }
        if one_step_mode not in supported_one_step_modes:
            raise ValueError(
                "one_step_diffusion_mode must be 'reference_noising' or "
                "'truncated_trajectory' or "
                "'template_anchored_truncated_trajectory'"
            )
        if int(self.one_step_diffusion_truncated_num_sampling_steps) <= 1:
            raise ValueError(
                "one_step_diffusion_truncated_num_sampling_steps must be greater than 1"
            )
        truncated_schedule_mode = str(
            self.one_step_diffusion_truncated_schedule_mode
        ).strip().lower()
        if truncated_schedule_mode not in {"official_clipped", "dense"}:
            raise ValueError(
                "one_step_diffusion_truncated_schedule_mode must be "
                "'official_clipped' or 'dense'"
            )
        if int(
            self.one_step_diffusion_truncated_dense_num_sampling_steps
        ) <= 1:
            raise ValueError(
                "one_step_diffusion_truncated_dense_num_sampling_steps must be "
                "greater than 1"
            )
        if (
            truncated_schedule_mode != "official_clipped"
            and one_step_mode == "reference_noising"
        ):
            raise ValueError(
                "one_step_diffusion_truncated_schedule_mode only applies to "
                "truncated trajectory modes"
            )
        if float(self.one_step_diffusion_weight) < 0.0:
            raise ValueError("one_step_diffusion_weight must be non-negative")
        if float(self.one_step_diffusion_rg_weight) < 0.0:
            raise ValueError(
                "one_step_diffusion_rg_weight must be non-negative"
            )
        if int(self.one_step_diffusion_interval) < 0:
            raise ValueError(
                "one_step_diffusion_interval must be non-negative"
            )
        if (
            int(self.one_step_diffusion_interval) > 0
            and not bool(self.one_step_diffusion_enabled)
        ):
            raise ValueError(
                "a positive one_step_diffusion_interval requires "
                "one_step_diffusion_enabled"
            )
        if (
            bool(self.one_step_diffusion_recycle_coordinates)
            and (
                not bool(self.one_step_diffusion_enabled)
                or int(self.one_step_diffusion_interval) <= 0
            )
        ):
            raise ValueError(
                "one_step_diffusion_recycle_coordinates requires a positive "
                "one_step_diffusion_interval"
            )
        if (
            bool(self.one_step_diffusion_recycle_coordinates)
            and one_step_mode != "reference_noising"
        ):
            raise ValueError(
                "one_step_diffusion_recycle_coordinates is only supported by "
                "reference_noising mode"
            )
        verifier_loss_mode = str(self.verifier_loss_mode).strip().lower()
        if verifier_loss_mode not in {"pair_chain_iptm", "aligned_common"}:
            raise ValueError(
                "verifier_loss_mode must be 'pair_chain_iptm' or 'aligned_common'"
            )
        if float(self.proposal_confidence_temperature) <= 0.0:
            raise ValueError("proposal_confidence_temperature must be positive")
        if int(self.proposal_confidence_interval) <= 0:
            raise ValueError("proposal_confidence_interval must be positive")
        if int(self.proposal_confidence_num_loops) <= 0:
            raise ValueError("proposal_confidence_num_loops must be positive")
        if int(self.proposal_confidence_num_sampling_steps) <= 0:
            raise ValueError("proposal_confidence_num_sampling_steps must be positive")
        if int(self.proposal_confidence_max_candidates) <= 0:
            raise ValueError("proposal_confidence_max_candidates must be positive")
        if int(self.proposal_confidence_tail_steps) < 0:
            raise ValueError("proposal_confidence_tail_steps must be non-negative")
        if self.target_chain_roles is not None:
            roles = tuple(
                str(role).strip().lower() for role in self.target_chain_roles
            )
            if not roles:
                raise ValueError("target_chain_roles cannot be empty")
            invalid_roles = sorted(set(roles) - {"target", "scaffold"})
            if invalid_roles:
                raise ValueError(
                    "target_chain_roles only supports target/scaffold; "
                    f"got {invalid_roles}"
                )
            if "target" not in roles:
                raise ValueError("target_chain_roles must contain at least one target")
        if self.binder_mutable_mask is not None:
            mutable_mask = tuple(bool(value) for value in self.binder_mutable_mask)
            if not mutable_mask:
                raise ValueError("binder_mutable_mask cannot be empty")
            if not any(mutable_mask):
                raise ValueError("binder_mutable_mask must select at least one residue")
        if (
            one_step_mode == "template_anchored_truncated_trajectory"
            or verifier_structure_mode == "scaffold_consistency"
        ):
            from proteocraft.hallucination.esmfold2_template import (
                validate_structure_regeneration_mask,
            )

            if not self.template_structure_path:
                raise ValueError(
                    "template-anchored mode requires template_structure_path"
                )
            template_path = Path(self.template_structure_path).expanduser()
            if not template_path.is_file():
                raise ValueError(
                    f"template_structure_path does not exist: {template_path}"
                )
            if not self.template_chain_ids or not self.template_chain_sequences:
                raise ValueError(
                    "template-anchored mode requires template chain ids and "
                    "reference sequences"
                )
            if len(self.template_chain_ids) != len(self.template_chain_sequences):
                raise ValueError(
                    "template_chain_ids and template_chain_sequences must match"
                )
            if len(set(self.template_chain_ids)) != len(self.template_chain_ids):
                raise ValueError("template_chain_ids must be unique")
            if self.binder_structure_regeneration_mask is None:
                raise ValueError(
                    "template-anchored mode requires "
                    "binder_structure_regeneration_mask"
                )
            structure_mask = validate_structure_regeneration_mask(
                self.binder_structure_regeneration_mask
            )
            if len(structure_mask) != len(self.template_chain_sequences[-1]):
                raise ValueError(
                    "binder_structure_regeneration_mask length must match the "
                    "template binder sequence"
                )
            if (
                self.binder_mutable_mask is not None
                and len(self.binder_mutable_mask) != len(structure_mask)
            ):
                raise ValueError(
                    "binder mutable and structure-regeneration masks must have "
                    "the same length"
                )
            if self.binder_mutable_mask is not None and any(
                bool(is_mutable) and not structure_mask[index]
                for index, is_mutable in enumerate(self.binder_mutable_mask)
            ):
                raise ValueError(
                    "binder_structure_regeneration_mask must include every "
                    "mutable residue"
                )


@dataclass
class ESMFold2VerifierResult:
    """MCTS-compatible verifier result for ESMFold2 candidates."""

    binder_seq: str
    target_seq: Union[str, Sequence[str]]
    total_loss: float
    loss_dict: Dict[str, float]
    summary_confidence: Dict[str, Any]
    coordinate: Optional[torch.Tensor] = field(default=None, repr=False)
    plddt_logits: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class ESMFold2DiscreteCandidateBatch:
    """Discrete sequences plus diagnostics from one soft-logit update."""

    sequences: Sequence[str]
    candidate_metadata: Sequence[Mapping[str, Any]] = field(default_factory=list)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequences = [str(sequence) for sequence in self.sequences]
        self.candidate_metadata = [dict(item) for item in self.candidate_metadata]
        self.diagnostics = dict(self.diagnostics)


@dataclass
class ESMFold2ProposalBatch:
    """Mutation proposal output plus optional persistent soft-logit state."""

    sequences: Sequence[str]
    logits_before: Optional[torch.Tensor] = field(default=None, repr=False)
    logits_after: Optional[torch.Tensor] = field(default=None, repr=False)
    confidence_candidates: Sequence[Mapping[str, Any]] = field(default_factory=list)
    soft_step_metrics: Sequence[Mapping[str, Any]] = field(default_factory=list)
    candidate_metadata: Sequence[Mapping[str, Any]] = field(default_factory=list)
    proposal_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequences = [str(sequence) for sequence in self.sequences]
        self.candidate_metadata = [dict(item) for item in self.candidate_metadata]
        self.proposal_diagnostics = dict(self.proposal_diagnostics)

    def __iter__(self) -> Iterator[str]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        return self.sequences[index]

    def __contains__(self, sequence: object) -> bool:
        return sequence in self.sequences


def _as_bool_mask(
    mutable_mask: Optional[Sequence[bool]],
    length: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if mutable_mask is None:
        return torch.ones(length, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mutable_mask, dtype=torch.bool, device=device)
    if mask.numel() != length:
        raise ValueError(
            f"mutable_mask length {mask.numel()} does not match sequence length {length}"
        )
    return mask.reshape(length)


def _sequence_indices(sequence: str) -> torch.Tensor:
    missing = sorted({aa for aa in sequence if aa not in AA_TO_ESM_IDX})
    if missing:
        raise ValueError(f"Unsupported amino acid(s) for ESMFold2 backend: {missing}")
    return torch.tensor([AA_TO_ESM_IDX[aa] for aa in sequence], dtype=torch.long)


def _finite_float(value: Any) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _first_finite_metric(
    summary_confidence: Mapping[str, Any],
    keys: Sequence[str],
) -> Tuple[Optional[float], Optional[str]]:
    for key in keys:
        if key not in summary_confidence:
            continue
        scalar = _finite_float(summary_confidence[key])
        if scalar is not None:
            return scalar, key
    return None, None


def resolve_esm_verifier_loss(
    summary_confidence: Mapping[str, Any],
) -> Tuple[float, str]:
    """Convert ESMFold2 confidence metrics to a scalar MCTS loss.

    The search value is strictly based on binder-target pair-chain iPTM.  A
    missing value indicates a broken chain mapping or incomplete verifier output
    and is therefore an error rather than a reason to silently change metrics.
    """

    score, source = _first_finite_metric(summary_confidence, PAIR_CHAIN_IPTM_KEYS)
    if score is None or source is None:
        raise ValueError(
            "ESMFold2 verifier summary must contain binder-target pair-chain iPTM"
        )
    score = min(max(score, 0.0), 1.0)
    return 1.0 - score, source


def build_esm_verifier_result(
    *,
    binder_seq: str,
    target_seq: Union[str, Sequence[str]],
    summary_confidence: Mapping[str, Any],
    coordinate: Optional[torch.Tensor] = None,
    plddt_logits: Optional[torch.Tensor] = None,
    extra_loss_dict: Optional[Mapping[str, float]] = None,
    hotspot_verify_loss: Optional[float] = None,
    hotspot_verify_weight: float = 0.0,
    hotspot_too_close_loss: Optional[float] = None,
    hotspot_too_close_weight: float = 0.0,
    aligned_loss_dict: Optional[Mapping[str, float]] = None,
) -> ESMFold2VerifierResult:
    """Build a verifier result object accepted by the existing MCTS cache."""

    global_loss, source = resolve_esm_verifier_loss(summary_confidence)
    resolved_aligned_losses: Dict[str, float] = {}
    if aligned_loss_dict is not None:
        for key, value in aligned_loss_dict.items():
            scalar = _finite_float(value)
            if scalar is not None:
                resolved_aligned_losses[str(key)] = scalar
        if "total" not in resolved_aligned_losses:
            raise ValueError("aligned_loss_dict must contain a finite total")
        total_loss = float(resolved_aligned_losses["total"])
        source = "esmfold2_aligned_common"
    else:
        total_loss = float(global_loss)
    base_verifier_loss = float(total_loss)
    hotspot_loss = _finite_float(hotspot_verify_loss)
    hotspot_weight = float(hotspot_verify_weight)
    if hotspot_weight < 0.0:
        raise ValueError("hotspot_verify_weight must be non-negative")
    if hotspot_weight > 0.0 and hotspot_loss is None:
        raise ValueError(
            "positive hotspot_verify_weight requires hotspot_verify_loss"
        )
    hotspot_enabled = hotspot_loss is not None and hotspot_weight > 0.0
    if hotspot_enabled:
        if hotspot_loss < 0.0 or hotspot_loss > 1.0:
            raise ValueError("hotspot_verify_loss must be in [0, 1]")
        total_loss += hotspot_weight * hotspot_loss
    too_close_loss = _finite_float(hotspot_too_close_loss)
    too_close_weight = float(hotspot_too_close_weight)
    if too_close_weight < 0.0:
        raise ValueError("hotspot_too_close_weight must be non-negative")
    if too_close_weight > 0.0 and too_close_loss is None:
        raise ValueError(
            "positive hotspot_too_close_weight requires hotspot_too_close_loss"
        )
    too_close_enabled = too_close_loss is not None and too_close_weight > 0.0
    if too_close_enabled:
        if too_close_loss < 0.0 or too_close_loss > 1.0:
            raise ValueError("hotspot_too_close_loss must be in [0, 1]")
        total_loss += too_close_weight * too_close_loss
    loss_dict: Dict[str, float] = {
        "total": float(total_loss),
        "esm_verify_loss": float(total_loss),
    }
    loss_dict["pair_chain_iptm_loss"] = float(global_loss)
    if resolved_aligned_losses:
        for key, value in resolved_aligned_losses.items():
            if key != "total":
                loss_dict[key] = float(value)
    if hotspot_enabled or too_close_enabled:
        loss_dict["esm_verify_global_loss"] = base_verifier_loss
    if hotspot_enabled:
        loss_dict.update(
            {
                "hotspot_verify_distance_loss": float(hotspot_loss),
                "hotspot_verify_weight": hotspot_weight,
                "hotspot_verify_weighted_contribution": hotspot_weight
                * float(hotspot_loss),
            }
        )
    if too_close_enabled:
        loss_dict.update(
            {
                "hotspot_too_close_loss": float(too_close_loss),
                "hotspot_too_close_weight": too_close_weight,
                "hotspot_too_close_weighted_contribution": too_close_weight
                * float(too_close_loss),
            }
        )
    plddt = _finite_float(summary_confidence.get("binder_plddt"))
    if plddt is not None:
        plddt_unit = plddt / 100.0 if plddt > 1.0 else plddt
        loss_dict["binder_confidence_loss"] = 1.0 - min(max(plddt_unit, 0.0), 1.0)
    rg_excess = _finite_float(summary_confidence.get("binder_rg_excess"))
    if rg_excess is not None:
        loss_dict["binder_rg_excess_loss"] = min(max(rg_excess, 0.0), 1.0)
    ptm = _finite_float(summary_confidence.get("ptm"))
    if ptm is not None:
        loss_dict["global_confidence_loss"] = 1.0 - min(max(ptm, 0.0), 1.0)
    if extra_loss_dict:
        for key, value in extra_loss_dict.items():
            scalar = _finite_float(value)
            if scalar is not None:
                loss_dict[str(key)] = scalar
    summary = dict(summary_confidence)
    summary["esm_verify_loss"] = float(total_loss)
    summary["esm_verify_source"] = source
    if resolved_aligned_losses:
        summary["esm_verify_aligned_common_loss"] = float(
            resolved_aligned_losses["total"]
        )
    if hotspot_enabled or too_close_enabled:
        summary["esm_verify_global_loss"] = base_verifier_loss
    if hotspot_enabled:
        summary.update(
            {
                "hotspot_verify_distance_loss": float(hotspot_loss),
                "hotspot_verify_weight": hotspot_weight,
                "hotspot_verify_weighted_contribution": hotspot_weight
                * float(hotspot_loss),
            }
        )
    if too_close_enabled:
        summary.update(
            {
                "hotspot_too_close_loss": float(too_close_loss),
                "hotspot_too_close_weight": too_close_weight,
                "hotspot_too_close_weighted_contribution": too_close_weight
                * float(too_close_loss),
            }
        )
    return ESMFold2VerifierResult(
        binder_seq=binder_seq,
        target_seq=target_seq,
        total_loss=float(total_loss),
        loss_dict=loss_dict,
        summary_confidence=summary,
        coordinate=coordinate,
        plddt_logits=plddt_logits,
    )


def compute_esmfold2_action_group_routing(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Compute normalized ESMFold2 Stage-B routing scores and priors."""

    raw_scores: Dict[str, float] = {}
    term_details: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing = []
    for group, key in ESMFOLD2_ACTION_GROUP_ROUTING_KEYS.items():
        value = _finite_float(loss_vector.get(key))
        if value is None:
            missing.append(key)
            continue
        score = min(max(float(value), 0.0), 1.0)
        raw_scores[group] = score
        term_details[group] = {
            key: {
                "raw": float(value),
                "scale": 1.0,
                "normalized": score,
                "weight": 1.0,
            }
        }
    if missing:
        raise ValueError(
            "ESMFold2 Stage-B routing requires full-verifier metrics: "
            + ", ".join(missing)
        )

    groups = list(raw_scores)
    scores = torch.tensor([raw_scores[group] for group in groups], dtype=torch.float32)
    priors = torch.softmax(scores / max(float(temperature), 1e-8), dim=0)
    return {
        "group_priors": {
            group: float(priors[idx].item())
            for idx, group in enumerate(groups)
        },
        "group_scores": raw_scores,
        "group_terms": term_details,
    }


def compute_esmfold2_action_group_priors(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute ESMFold2-native Stage-B routing priors from real loss terms."""

    return dict(
        compute_esmfold2_action_group_routing(
            loss_vector,
            temperature=temperature,
        )["group_priors"]
    )


def select_esmfold2_action_group(
    loss_vector: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """Select the ESMFold2 Stage-B repair group for the current verified node."""

    routing = compute_esmfold2_action_group_routing(loss_vector, temperature=temperature)
    priors = dict(routing["group_priors"])
    action_group = max(priors, key=priors.get)
    return {
        "action_group": action_group,
        "group_prior": priors[action_group],
        "group_priors": priors,
        "group_scores": routing["group_scores"],
        "group_terms": routing["group_terms"],
    }


def init_binder_logits(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    batch_size: int = 1,
    random_scale: float = 0.01,
    fixed_logit: float = 10.0,
    suppress_cys: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Initialize ESMFold2 soft binder logits for a current discrete sequence."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device) if device is not None else None
    length = len(sequence)
    current_idx = _sequence_indices(sequence).to(device=device)
    mutable = _as_bool_mask(mutable_mask, length, device=device)

    if random_scale > 0.0:
        logits = torch.randn(
            (batch_size, length, len(ESM_AA_ORDER)),
            device=device,
            dtype=dtype,
            generator=generator,
        ) * float(random_scale)
    else:
        logits = torch.zeros(
            (batch_size, length, len(ESM_AA_ORDER)),
            device=device,
            dtype=dtype,
        )

    fixed_positions = (~mutable).nonzero(as_tuple=False).flatten()
    if fixed_positions.numel() > 0:
        logits[:, fixed_positions, :] = 0.0
        logits[:, fixed_positions, current_idx[fixed_positions]] = float(fixed_logit)

    if suppress_cys:
        mutable_positions = mutable.nonzero(as_tuple=False).flatten()
        if mutable_positions.numel() > 0:
            logits[:, mutable_positions, CYS_INDEX] = float("-inf")

    logits.requires_grad_(requires_grad)
    return logits


def anchor_binder_logits_to_sequence(
    logits: torch.Tensor,
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    anchor_logit: float = 2.0,
    fixed_logit: float = 10.0,
    anchor_margin: float = 0.1,
    suppress_cys: bool = True,
    clone: bool = True,
) -> torch.Tensor:
    """Return node-local logits anchored to a verified discrete sequence.

    The ESMFold2 proposal state should remain logits, not probabilities.  This
    helper preserves the existing soft-logit landscape while making the node's
    verified sequence the row-wise argmax, so child nodes do not drift away from
    the sequence that was actually verifier-scored.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("node-local ESM logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("node-local ESM logits must have shape [L, 20] or [1, L, 20]")

    length = len(sequence)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match sequence shape "
            f"({length}, {len(ESM_AA_ORDER)})"
        )

    anchored = logits_2d.detach().clone() if clone else logits_2d
    current_idx = _sequence_indices(sequence).to(device=anchored.device)
    mutable = _as_bool_mask(mutable_mask, length, device=anchored.device)

    if suppress_cys:
        anchored[:, CYS_INDEX] = float("-inf")

    for pos in range(length):
        aa_idx = int(current_idx[pos].item())
        if not bool(mutable[pos].item()):
            anchored[pos, :] = 0.0
            if suppress_cys:
                anchored[pos, CYS_INDEX] = float("-inf")
            anchored[pos, aa_idx] = float(fixed_logit)
            continue

        row = anchored[pos]
        finite_row = row[torch.isfinite(row)]
        row_max = float(finite_row.max().item()) if finite_row.numel() else 0.0
        desired = max(float(row[aa_idx].item()), float(anchor_logit))
        if desired < row_max:
            desired = row_max + float(anchor_margin)
        row[aa_idx] = desired

    return anchored.detach().cpu().clone()


def minimally_align_binder_logits_to_sequence(
    logits: torch.Tensor,
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    fixed_logit: float = 10.0,
    anchor_margin: float = 0.1,
    suppress_cys: bool = True,
) -> torch.Tensor:
    """Align a verified sequence while preserving its saved soft-logit state.

    Mutable rows that already select the verified residue are left unchanged.
    Only mismatched or tied rows are lifted just above their strongest allowed
    competitor. Fixed rows retain the existing hard sequence constraint.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("node-local ESM logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("node-local ESM logits must have shape [L, 20] or [1, L, 20]")

    sequence = str(sequence)
    length = len(sequence)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match sequence shape "
            f"({length}, {len(ESM_AA_ORDER)})"
        )
    if float(anchor_margin) <= 0.0:
        raise ValueError("anchor_margin must be positive")

    aligned = logits_2d.detach().clone()
    sequence_idx = _sequence_indices(sequence).to(device=aligned.device)
    mutable = _as_bool_mask(mutable_mask, length, device=aligned.device)

    for pos in range(length):
        aa_idx = int(sequence_idx[pos].item())
        if not bool(mutable[pos].item()):
            aligned[pos, :] = 0.0
            if suppress_cys:
                aligned[pos, CYS_INDEX] = float("-inf")
            aligned[pos, aa_idx] = float(fixed_logit)
            continue
        if suppress_cys and aa_idx == CYS_INDEX:
            raise ValueError("cysteine mutations are disabled for ESMFold2 proposals")

        row = aligned[pos]
        if suppress_cys:
            row[CYS_INDEX] = float("-inf")
        competitor_mask = torch.ones(
            len(ESM_AA_ORDER),
            dtype=torch.bool,
            device=row.device,
        )
        competitor_mask[aa_idx] = False
        competitors = row[competitor_mask]
        finite_competitors = competitors[torch.isfinite(competitors)]
        competitor_max = (
            float(finite_competitors.max().item())
            if finite_competitors.numel()
            else 0.0
        )
        current = float(row[aa_idx].item())
        if not torch.isfinite(row[aa_idx]) or current <= competitor_max:
            row[aa_idx] = competitor_max + float(anchor_margin)

    return aligned.detach().cpu().clone()


def softly_anchor_mutation_logits(
    logits: torch.Tensor,
    parent_sequence: str,
    child_sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    anchor_margin: float = 0.1,
    suppress_cys: bool = True,
) -> torch.Tensor:
    """Make only explicit child mutations the argmax of their logit rows.

    Stage-A mutation candidates branch from a shared soft-logit update.  A full
    sequence anchor would rewrite every row where the candidate differs from
    the raw argmax trajectory, discarding useful uncertainty.  This helper
    changes only positions that differ between the verified parent and child;
    all other mutable-position logits remain bitwise unchanged.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("node-local ESM logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("node-local ESM logits must have shape [L, 20] or [1, L, 20]")

    parent_sequence = str(parent_sequence)
    child_sequence = str(child_sequence)
    if len(parent_sequence) != len(child_sequence):
        raise ValueError("parent and child sequences must have the same length")
    length = len(child_sequence)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match sequence shape "
            f"({length}, {len(ESM_AA_ORDER)})"
        )
    if float(anchor_margin) <= 0.0:
        raise ValueError("anchor_margin must be positive")

    anchored = logits_2d.detach().clone()
    mutable = _as_bool_mask(mutable_mask, length, device=anchored.device)
    child_idx = _sequence_indices(child_sequence).to(device=anchored.device)

    for pos, (parent_aa, child_aa) in enumerate(zip(parent_sequence, child_sequence)):
        if parent_aa == child_aa:
            continue
        if not bool(mutable[pos].item()):
            raise ValueError(f"child mutation at fixed binder position {pos}")
        aa_idx = int(child_idx[pos].item())
        if suppress_cys and aa_idx == CYS_INDEX:
            raise ValueError("cysteine mutations are disabled for ESMFold2 proposals")
        row = anchored[pos]
        finite_row = row[torch.isfinite(row)]
        row_max = float(finite_row.max().item()) if finite_row.numel() else 0.0
        if not torch.isfinite(row[aa_idx]) or float(row[aa_idx].item()) <= row_max:
            row[aa_idx] = row_max + float(anchor_margin)

    return anchored.detach().cpu().clone()


def build_gradient_mask(
    sequence: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    *,
    batch_size: int = 1,
    suppress_cys: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a [B, L, 20] mask for ESMFold2 binder-logit gradients."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device) if device is not None else None
    length = len(sequence)
    _sequence_indices(sequence)
    mutable = _as_bool_mask(mutable_mask, length, device=device)
    mask = torch.zeros(
        (batch_size, length, len(ESM_AA_ORDER)),
        device=device,
        dtype=dtype,
    )
    if mutable.any():
        mask[:, mutable, :] = 1.0
    if suppress_cys:
        mask[:, :, CYS_INDEX] = 0.0
    return mask


def normalized_gradient_tensor(
    gradient: torch.Tensor,
    gradient_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a masked gradient tensor in the style of ESMFold2 design code."""

    if gradient.shape != gradient_mask.shape:
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} does not match mask shape "
            f"{tuple(gradient_mask.shape)}"
        )
    masked = torch.nan_to_num(gradient) * gradient_mask
    output = torch.zeros_like(masked)
    if masked.numel() == 0:
        return output

    norms = torch.linalg.vector_norm(masked, dim=(-1, -2))
    active_positions = torch.square(masked).sum(dim=-1) > 0
    effective_lengths = active_positions.sum(dim=-1).to(masked.dtype)
    safe = (effective_lengths > 0) & (norms > eps)
    if safe.any():
        scale = torch.sqrt(torch.clamp(effective_lengths, min=1.0))
        normalized = masked / torch.clamp(norms, min=eps).view(-1, 1, 1)
        normalized = normalized * scale.view(-1, 1, 1)
        output = torch.where(safe.view(-1, 1, 1), normalized, torch.zeros_like(masked))
    return output * gradient_mask


def combine_normalized_gradients(
    structure_grad: torch.Tensor,
    esmc_grad: Optional[torch.Tensor],
    *,
    gradient_mask: torch.Tensor,
    lambda_esmc: float,
) -> torch.Tensor:
    """Combine ESMFold2 structural gradient and optional ESMC sequence prior."""

    combined = normalized_gradient_tensor(structure_grad, gradient_mask)
    if esmc_grad is not None and float(lambda_esmc) != 0.0:
        combined = combined + float(lambda_esmc) * normalized_gradient_tensor(
            esmc_grad,
            gradient_mask,
        )
    return combined * gradient_mask


def resolve_lambda_esmc(
    value: Union[str, float, int, None],
    *,
    task_type: Optional[str] = None,
) -> float:
    """Resolve the ESMC sequence-prior weight.

    ``auto`` follows the Biohub binder-design convention: stronger sequence
    prior for de novo/minibinder design and weaker prior for antibody-like
    redesign tasks.
    """

    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"off", "none", "false", "0"}:
        return 0.0
    if normalized == "auto":
        task = str(task_type or "denovo").strip().lower()
        return AUTO_ESMC_LAMBDA_BY_TASK.get(task, AUTO_ESMC_LAMBDA_BY_TASK["denovo"])
    return float(normalized)


def esm_temperature(
    step: int,
    *,
    total_steps: int,
    temperature_min: float = 1e-2,
    temperature_max: float = 1.0,
) -> float:
    """Cosine temperature schedule used by the short ESMFold2 soft update."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    clamped_step = min(max(int(step), 0), int(total_steps) - 1)
    t = (clamped_step + 1) / max(int(total_steps), 1)
    remaining = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(temperature_min + (temperature_max - temperature_min) * remaining)


def logit_delta_mutation_scores(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    suppress_cys: bool = True,
) -> torch.Tensor:
    """Score discrete mutations from a short ESMFold2/ESMC soft-logit update.

    score(i, aa) =
      [after(i, aa) - before(i, aa)]
      - [after(i, current_aa) - before(i, current_aa)]

    Higher score means the soft update pushed that amino acid up more than the
    current residue at the same position.
    """

    if logits_before.shape != logits_after.shape:
        raise ValueError(
            f"logits_before shape {tuple(logits_before.shape)} does not match "
            f"logits_after shape {tuple(logits_after.shape)}"
        )
    if logits_before.ndim != 2 or logits_before.shape[1] != len(ESM_AA_ORDER):
        raise ValueError("logits tensors must have shape [L, 20]")
    length = len(current_seq)
    if logits_before.shape[0] != length:
        raise ValueError(
            f"logits length {logits_before.shape[0]} does not match sequence length {length}"
        )

    current_idx = _sequence_indices(current_seq).to(device=logits_before.device)
    mutable = _as_bool_mask(mutable_mask, length, device=logits_before.device)
    delta = torch.nan_to_num(
        logits_after - logits_before,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    current_delta = delta[torch.arange(length, device=delta.device), current_idx]
    scores = delta - current_delta.unsqueeze(-1)

    invalid = torch.zeros_like(scores, dtype=torch.bool)
    invalid |= ~mutable.unsqueeze(-1)
    invalid[torch.arange(length, device=delta.device), current_idx] = True
    if suppress_cys:
        invalid[:, CYS_INDEX] = True
    scores = scores.masked_fill(invalid, float("-inf"))
    return scores


def sequence_from_logits_argmax(
    logits: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    suppress_cys: bool = True,
) -> str:
    """Convert binder logits to a discrete sequence by argmax.

    Fixed positions are copied from ``current_seq`` so CDR/TCR masks stay
    model-independent and cannot leak edits into protected regions.
    """

    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("batched logits must have batch size 1")
        logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("logits must have shape [L, 20] or [1, L, 20]")

    length = len(current_seq)
    if logits_2d.shape != (length, len(ESM_AA_ORDER)):
        raise ValueError(
            f"logits shape {tuple(logits_2d.shape)} does not match "
            f"({length}, {len(ESM_AA_ORDER)})"
        )

    scores = torch.nan_to_num(
        logits_2d.detach().clone(),
        nan=float("-inf"),
        posinf=1.0e9,
        neginf=float("-inf"),
    )
    if suppress_cys:
        scores[:, CYS_INDEX] = float("-inf")
    indices = torch.argmax(scores, dim=-1).detach().cpu().tolist()
    mutable = [bool(item) for item in _as_bool_mask(mutable_mask, length).cpu().tolist()]

    chars = list(str(current_seq))
    for idx, aa_index in enumerate(indices):
        if mutable[idx]:
            chars[idx] = ESM_AA_ORDER[int(aa_index)]
    return "".join(chars)


def _scores_to_gradient_like_matrix(scores: torch.Tensor) -> torch.Tensor:
    """Convert high-is-good mutation scores to legacy low-is-good delta matrix."""

    converted = torch.where(
        torch.isneginf(scores),
        torch.full_like(scores, 1.0e9),
        -scores,
    )
    return converted


def _candidate_metadata_from_sequence(
    *,
    current_seq: str,
    sequence: str,
    scores: torch.Tensor,
    source: str,
) -> Dict[str, Any]:
    positions = [
        index
        for index, (current_aa, new_aa) in enumerate(zip(current_seq, sequence))
        if current_aa != new_aa
    ]
    mutation_scores = []
    changes = []
    for position in positions:
        new_aa = sequence[position]
        score = scores[position, AA_TO_ESM_IDX[new_aa]]
        mutation_scores.append(
            float(score.item()) if torch.isfinite(score) else None
        )
        changes.append(f"{current_seq[position]}{position + 1}{new_aa}")
    return {
        "sequence": sequence,
        "source": source,
        "mutation_count": len(positions),
        "mutation_positions": positions,
        "mutation_changes": changes,
        "mutation_scores": mutation_scores,
    }


def validate_dominance_guard(gap: float, count: int) -> None:
    if not math.isfinite(float(gap)) or float(gap) <= 0:
        raise ValueError("dominance_gap must be finite and positive")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("protected_mutation_candidates must be a positive integer")


def dominant_current_residue_mask(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    dominance_gap: float = 0.5,
) -> torch.Tensor:
    """Protect a current residue only if it leads before and after the update."""
    validate_dominance_guard(dominance_gap, 1)
    protected = _as_bool_mask(mutable_mask, len(current_seq)).cpu().clone()
    indices = _sequence_indices(current_seq).cpu()
    rows = torch.arange(len(current_seq))
    for logits in (logits_before, logits_after):
        values = logits.detach().cpu()
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.shape != (len(current_seq), len(ESM_AA_ORDER)):
            raise ValueError("guard logits must have shape [L, 20] or [1, L, 20]")
        current = values[rows, indices]
        competitors = values.clone()
        competitors[:, CYS_INDEX] = float("-inf")
        competitors[rows, indices] = float("-inf")
        # -inf denotes a disabled option; NaN/+inf must not imply dominance.
        valid = ~(torch.isnan(competitors) | torch.isposinf(competitors)).any(dim=-1)
        best = competitors.max(dim=-1).values
        protected &= valid & torch.isfinite(current) & (indices != CYS_INDEX)
        protected &= (current - best) >= float(dominance_gap)
    return protected


def _guard_weighted_candidates(
    batch: ESMFold2DiscreteCandidateBatch,
    *,
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]],
    dominance_gap: float,
    protected_mutation_candidates: int,
) -> ESMFold2DiscreteCandidateBatch:
    protected = dominant_current_residue_mask(
        logits_before, logits_after, current_seq=current_seq,
        mutable_mask=mutable_mask, dominance_gap=dominance_gap,
    ).tolist()
    scores = logit_delta_mutation_scores(
        logits_before, logits_after, current_seq=current_seq, mutable_mask=mutable_mask,
    ).detach().cpu()
    sequences = list(batch.sequences)
    metadata = [dict(item) for item in batch.candidate_metadata]
    slots = [i for i, item in enumerate(metadata)
             if item["source"] == "positive_score_weighted"][-protected_mutation_candidates:]
    changed_count = 0
    fallback_count = 0
    for slot in slots:
        original = sequences[slot]
        restored = [i for i, (old, new) in enumerate(zip(current_seq, original))
                    if protected[i] and old != new]
        candidate = "".join(old if protected[i] else new
                            for i, (old, new) in enumerate(zip(current_seq, original)))
        reason = None
        if candidate == current_seq:
            reason = "parent_sequence"
        elif any(candidate == other for i, other in enumerate(sequences) if i != slot):
            reason = "duplicate_candidate"
        if reason is not None:
            fallback_count += 1
        elif candidate != original:
            sequences[slot] = candidate
            metadata[slot] = _candidate_metadata_from_sequence(
                current_seq=current_seq, sequence=candidate, scores=scores,
                source="positive_score_weighted_guarded",
            )
            changed_count += 1
        metadata[slot]["dominance_guard"] = {
            "original_sequence": original,
            "attempted_restore_positions": restored,
            "restored_positions": [] if reason else restored,
            "fallback_reason": reason,
            "status": "fallback" if reason else ("applied" if restored else "unchanged"),
        }
    diagnostics = dict(batch.diagnostics)
    diagnostics.update(
        candidate_strategy="positive_score_guarded",
        dominance_gap=float(dominance_gap),
        protected_mutation_candidates=protected_mutation_candidates,
        protected_positions=[i for i, value in enumerate(protected) if value],
        guard_selected_slots=slots,
        guard_changed_candidate_count=changed_count,
        guard_fallback_count=fallback_count,
        mutation_count_contract="unprotected_exact_protected_at_most",
    )
    return ESMFold2DiscreteCandidateBatch(
        sequences=sequences, candidate_metadata=metadata, diagnostics=diagnostics,
    )


def build_logit_delta_candidate_batch(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    mutation_fraction: Optional[float] = None,
    include_logit_candidates: bool = False,
    candidate_strategy: str = "legacy",
    dominance_gap: float = 0.5,
    protected_mutation_candidates: int = 2,
) -> ESMFold2DiscreteCandidateBatch:
    """Generate discrete candidates and proposal diagnostics.

    ``legacy`` preserves the original Top-K action sampler. ``positive_score``
    keeps only beneficial per-position logit-delta edits, uses one deterministic
    top-K candidate, then samples the remaining candidates with exactly the same
    number of distinct positions.
    ``positive_score_guarded`` restores dominant residues in the final weighted
    slots after sampling, without extra random draws or replacement mutations.
    """

    from proteocraft.hallucination.mutation_proposer import (
        AA_1LETTER,
        PROTEIN_RESIDUE_NAMES,
        sample_candidate_sequences,
        select_top_k_mutations,
    )

    strategy = str(candidate_strategy or "legacy").strip().lower()
    if strategy == "positive_score_guarded":
        validate_dominance_guard(dominance_gap, protected_mutation_candidates)
        # Reuse the original sampler without extra random draws or budget changes.
        batch = build_logit_delta_candidate_batch(
            logits_before, logits_after, current_seq=current_seq,
            mutable_mask=mutable_mask, K_top=K_top, B_candidates=B_candidates,
            T_sample=T_sample, max_mutations_per_seq=max_mutations_per_seq,
            mutation_fraction=mutation_fraction, include_logit_candidates=include_logit_candidates,
            candidate_strategy="positive_score",
        )
        return _guard_weighted_candidates(
            batch, logits_before=logits_before, logits_after=logits_after,
            current_seq=current_seq, mutable_mask=mutable_mask,
            dominance_gap=dominance_gap,
            protected_mutation_candidates=protected_mutation_candidates,
        )
    if strategy not in {"legacy", "positive_score"}:
        raise ValueError(
            "candidate_strategy must be 'legacy' or 'positive_score'"
        )
    if strategy == "positive_score" and mutation_fraction is None:
        raise ValueError(
            "positive_score candidate_strategy requires mutation_fraction"
        )

    candidate_limit = max(int(B_candidates), 1)
    current_indices = [AA_TO_ESM_IDX[aa] for aa in current_seq]
    scores = logit_delta_mutation_scores(
        logits_before,
        logits_after,
        current_seq=current_seq,
        mutable_mask=mutable_mask,
    ).detach().cpu()
    candidates: list[str] = []
    candidate_metadata: list[Dict[str, Any]] = []

    def append_candidate(sequence: str, source: str) -> bool:
        sequence = str(sequence)
        if sequence == current_seq or sequence in candidates:
            return False
        if len(candidates) >= candidate_limit:
            return False
        candidates.append(sequence)
        candidate_metadata.append(
            _candidate_metadata_from_sequence(
                current_seq=current_seq,
                sequence=sequence,
                scores=scores,
                source=source,
            )
        )
        return True

    argmax_seq = None
    if include_logit_candidates:
        argmax_seq = sequence_from_logits_argmax(
            logits_after,
            current_seq=current_seq,
            mutable_mask=mutable_mask,
        )
        append_candidate(argmax_seq, "unrestricted_argmax")

    if strategy == "positive_score":
        fraction = float(mutation_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("mutation_fraction must be in (0, 1]")

        mutable = _as_bool_mask(mutable_mask, len(current_seq)).cpu()
        mutable_count = int(mutable.sum().item())
        mutation_cap = (
            max(1, math.ceil(fraction * mutable_count))
            if mutable_count > 0
            else 0
        )
        best_benefits, best_aas = scores.max(dim=-1)
        positive_mask = (
            mutable
            & torch.isfinite(best_benefits)
            & (best_benefits > 0.0)
        )
        positive_positions = torch.nonzero(
            positive_mask,
            as_tuple=False,
        ).flatten()
        positive_benefits = best_benefits[positive_positions]
        actual_mutation_count = min(
            int(positive_positions.numel()),
            int(mutation_cap),
        )
        diagnostics: Dict[str, Any] = {
            "candidate_strategy": strategy,
            "include_argmax_candidate": bool(include_logit_candidates),
            "mutable_residue_count": mutable_count,
            "mutation_fraction": fraction,
            "mutation_cap": mutation_cap,
            "positive_position_count": int(positive_positions.numel()),
            "actual_mutation_count": actual_mutation_count,
            "positive_score_min": None,
            "positive_score_mean": None,
            "positive_score_max": None,
        }
        if positive_benefits.numel() > 0:
            diagnostics.update(
                positive_score_min=float(positive_benefits.min().item()),
                positive_score_mean=float(positive_benefits.mean().item()),
                positive_score_max=float(positive_benefits.max().item()),
            )

        if actual_mutation_count > 0 and len(candidates) < candidate_limit:
            _, deterministic_indices = torch.topk(
                positive_benefits,
                actual_mutation_count,
            )
            deterministic_positions = positive_positions[deterministic_indices]
            deterministic_sequence = list(current_seq)
            for position in deterministic_positions.tolist():
                deterministic_sequence[position] = ESM_AA_ORDER[
                    int(best_aas[position].item())
                ]
            append_candidate(
                "".join(deterministic_sequence),
                "positive_score_topk",
            )

            temperature = max(float(T_sample), 1.0e-8)
            position_probabilities = torch.softmax(
                positive_benefits / temperature,
                dim=0,
            )
            max_attempts = max(candidate_limit * 20, 20)
            for _ in range(max_attempts):
                if len(candidates) >= candidate_limit:
                    break
                chosen_local = torch.multinomial(
                    position_probabilities,
                    actual_mutation_count,
                    replacement=False,
                )
                chosen_positions = positive_positions[chosen_local]
                sampled_sequence = list(current_seq)
                for position in chosen_positions.tolist():
                    aa_scores = scores[position]
                    positive_aa_indices = torch.nonzero(
                        torch.isfinite(aa_scores) & (aa_scores > 0.0),
                        as_tuple=False,
                    ).flatten()
                    aa_probabilities = torch.softmax(
                        aa_scores[positive_aa_indices] / temperature,
                        dim=0,
                    )
                    chosen_aa_local = int(
                        torch.multinomial(
                            aa_probabilities,
                            1,
                            replacement=False,
                        ).item()
                    )
                    chosen_aa = int(
                        positive_aa_indices[chosen_aa_local].item()
                    )
                    sampled_sequence[position] = ESM_AA_ORDER[chosen_aa]
                append_candidate(
                    "".join(sampled_sequence),
                    "positive_score_weighted",
                )

        diagnostics["generated_candidate_count"] = len(candidates)
        diagnostics["generated_unique_candidate_count"] = len(set(candidates))
        return ESMFold2DiscreteCandidateBatch(
            sequences=candidates,
            candidate_metadata=candidate_metadata,
            diagnostics=diagnostics,
        )

    if mutation_fraction is None:
        delta_like = _scores_to_gradient_like_matrix(scores)
        positions, new_aas, benefits = select_top_k_mutations(
            delta_like,
            current_indices,
            K_top,
        )
        sampled_mutation_cap = int(max_mutations_per_seq)
    else:
        fraction = float(mutation_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("mutation_fraction must be in (0, 1]")

        mutable = _as_bool_mask(mutable_mask, len(current_seq)).cpu()
        mutable_count = int(mutable.sum().item())
        sampled_mutation_cap = (
            max(1, math.ceil(fraction * mutable_count))
            if mutable_count > 0
            else 0
        )

        # Fraction-based experiments count distinct residue positions. Keep only
        # the strongest amino-acid edit per mutable residue before sampling.
        best_benefits, best_aas = scores.max(dim=-1)
        valid_positions = torch.nonzero(
            mutable & torch.isfinite(best_benefits),
            as_tuple=False,
        ).flatten()
        if valid_positions.numel() == 0:
            positions = torch.empty(0, dtype=torch.long)
            new_aas = torch.empty(0, dtype=torch.long)
            benefits = torch.empty(0, dtype=scores.dtype)
        else:
            position_benefits = best_benefits[valid_positions]
            pool_size = min(
                int(valid_positions.numel()),
                max(int(K_top), int(sampled_mutation_cap)),
            )
            top_benefits, top_indices = torch.topk(
                position_benefits,
                pool_size,
            )
            positions = valid_positions[top_indices]
            new_aas = best_aas[positions]
            benefits = top_benefits
    if positions.numel() == 0:
        while len(candidates) < candidate_limit:
            candidates.append(current_seq)
            candidate_metadata.append(
                _candidate_metadata_from_sequence(
                    current_seq=current_seq,
                    sequence=current_seq,
                    scores=scores,
                    source="legacy_no_mutation",
                )
            )
        diagnostics = {
            "candidate_strategy": strategy,
            "include_argmax_candidate": bool(include_logit_candidates),
            "mutation_fraction": mutation_fraction,
            "mutation_cap": sampled_mutation_cap,
            "generated_candidate_count": len(candidates),
            "generated_unique_candidate_count": len(set(candidates)),
        }
        return ESMFold2DiscreteCandidateBatch(
            sequences=candidates,
            candidate_metadata=candidate_metadata,
            diagnostics=diagnostics,
        )

    candidate_indices = sample_candidate_sequences(
        current_indices,
        positions,
        new_aas,
        benefits,
        candidate_limit,
        T_sample,
        sampled_mutation_cap,
    )

    def indices_to_sequence(indices: Sequence[int]) -> str:
        return "".join(AA_1LETTER[PROTEIN_RESIDUE_NAMES[int(idx)]] for idx in indices)

    sampled_sequences = [indices_to_sequence(indices) for indices in candidate_indices]
    if not include_logit_candidates:
        candidates = sampled_sequences[:candidate_limit]
        candidate_metadata = [
            _candidate_metadata_from_sequence(
                current_seq=current_seq,
                sequence=sequence,
                scores=scores,
                source="legacy_weighted",
            )
            for sequence in candidates
        ]
    else:
        for sequence in sampled_sequences:
            append_candidate(sequence, "legacy_weighted")
        sampled_index = 0
        while len(candidates) < candidate_limit and sampled_sequences:
            sequence = sampled_sequences[sampled_index % len(sampled_sequences)]
            candidates.append(sequence)
            candidate_metadata.append(
                _candidate_metadata_from_sequence(
                    current_seq=current_seq,
                    sequence=sequence,
                    scores=scores,
                    source="legacy_weighted_repeat",
                )
            )
            sampled_index += 1
        while len(candidates) < candidate_limit:
            candidates.append(current_seq)
            candidate_metadata.append(
                _candidate_metadata_from_sequence(
                    current_seq=current_seq,
                    sequence=current_seq,
                    scores=scores,
                    source="legacy_no_mutation",
                )
            )

    diagnostics = {
        "candidate_strategy": strategy,
        "include_argmax_candidate": bool(include_logit_candidates),
        "mutation_fraction": mutation_fraction,
        "mutation_cap": sampled_mutation_cap,
        "generated_candidate_count": len(candidates),
        "generated_unique_candidate_count": len(set(candidates)),
    }
    return ESMFold2DiscreteCandidateBatch(
        sequences=candidates,
        candidate_metadata=candidate_metadata,
        diagnostics=diagnostics,
    )


def propose_sequences_from_logit_delta(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    current_seq: str,
    mutable_mask: Optional[Sequence[bool]] = None,
    K_top: int = 10,
    B_candidates: int = 8,
    T_sample: float = 1.0,
    max_mutations_per_seq: int = 3,
    mutation_fraction: Optional[float] = None,
    include_logit_candidates: bool = False,
    candidate_strategy: str = "legacy",
    dominance_gap: float = 0.5,
    protected_mutation_candidates: int = 2,
) -> list[str]:
    """Return only sequence strings for the legacy public helper contract."""

    batch = build_logit_delta_candidate_batch(
        logits_before,
        logits_after,
        current_seq=current_seq,
        mutable_mask=mutable_mask,
        K_top=K_top,
        B_candidates=B_candidates,
        T_sample=T_sample,
        max_mutations_per_seq=max_mutations_per_seq,
        mutation_fraction=mutation_fraction,
        include_logit_candidates=include_logit_candidates,
        candidate_strategy=candidate_strategy,
        dominance_gap=dominance_gap,
        protected_mutation_candidates=protected_mutation_candidates,
    )
    return list(batch.sequences)
