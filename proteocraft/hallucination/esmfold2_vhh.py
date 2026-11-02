"""Sequence-only all-CDR VHH redesign adapter for the ESMFold2 backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
IMGT_CDR_NAMES = ("CDR1", "CDR2", "CDR3")


@dataclass(frozen=True)
class ESMFold2VHHCase:
    """Validated inputs for one sequence-only VHH all-CDR redesign case."""

    case_id: str
    target_sequences: tuple[str, ...]
    target_chain_roles: tuple[str, ...]
    initial_vhh_sequence: str
    mutable_mask: tuple[bool, ...]
    cdr_regions_1based: tuple[tuple[str, tuple[int, ...]], ...]
    target_hotspots_1based: tuple[int, ...]
    numbering_scheme: str
    numbering_tool: Optional[str]
    numbering_tool_version: Optional[str]
    scaffold_pdb_id: Optional[str]
    scaffold_chain_id: Optional[str]
    reference_structure_path: Optional[str]

    @property
    def target_sequence_argument(self) -> str:
        return "|".join(self.target_sequences)

    @property
    def target_chain_roles_argument(self) -> str:
        return ",".join(self.target_chain_roles)

    @property
    def mutable_mask_argument(self) -> str:
        return "".join("1" if value else "0" for value in self.mutable_mask)

    @property
    def target_hotspots_argument(self) -> Optional[str]:
        if not self.target_hotspots_1based:
            return None
        return ",".join(str(index) for index in self.target_hotspots_1based)

    @property
    def mutable_positions_1based(self) -> tuple[int, ...]:
        return tuple(
            index + 1 for index, is_mutable in enumerate(self.mutable_mask) if is_mutable
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "target_sequence_argument": self.target_sequence_argument,
                "target_chain_roles_argument": self.target_chain_roles_argument,
                "mutable_mask_argument": self.mutable_mask_argument,
                "mutable_positions_1based": list(self.mutable_positions_1based),
                "mutable_residue_count": sum(self.mutable_mask),
                "cdr_regions_1based": {
                    name: list(positions)
                    for name, positions in self.cdr_regions_1based
                },
            }
        )
        return payload


def _normalize_sequence(value: Any, *, field_name: str) -> str:
    sequence = str(value).strip().upper()
    if not sequence:
        raise ValueError(f"{field_name} must not be empty")
    invalid = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"{field_name} has unsupported amino acids: {invalid}")
    return sequence


def _resolve_path(raw_path: Optional[str], *, base_dir: Path) -> Optional[Path]:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def _normalize_positions(
    raw_region: Any,
    *,
    region_name: str,
    sequence_length: int,
) -> tuple[int, ...]:
    raw_positions = (
        raw_region.get("sequence_indices_1based")
        if isinstance(raw_region, Mapping)
        else raw_region
    )
    if isinstance(raw_positions, (str, bytes)) or not isinstance(
        raw_positions, Sequence
    ):
        raise ValueError(
            f"{region_name} must provide sequence_indices_1based as a list"
        )
    positions: list[int] = []
    for raw_position in raw_positions:
        if isinstance(raw_position, bool):
            raise ValueError(f"{region_name} positions must be integers")
        try:
            position = int(raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{region_name} positions must be integers") from exc
        if position < 1 or position > sequence_length:
            raise ValueError(
                f"{region_name} position out of range: {position}; "
                f"VHH length is {sequence_length}"
            )
        positions.append(position)
    if not positions:
        raise ValueError(f"{region_name} must contain at least one position")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{region_name} contains duplicate positions")
    return tuple(positions)


def make_imgt_all_cdr_mask(
    sequence_length: int,
    cdr_regions: Mapping[str, Any],
) -> tuple[tuple[bool, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Build a mask from all three IMGT CDR regions using sequence indices."""

    normalized_regions = {
        str(name).upper(): value for name, value in cdr_regions.items()
    }
    if len(normalized_regions) != len(cdr_regions):
        raise ValueError("cdr_regions contains duplicate names after normalization")
    expected = set(IMGT_CDR_NAMES)
    actual = set(normalized_regions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "all-CDR VHH redesign requires exactly CDR1, CDR2 and CDR3; "
            f"missing={missing}, extra={extra}"
        )

    ordered_regions: list[tuple[str, tuple[int, ...]]] = []
    owner_by_position: dict[int, str] = {}
    for name in IMGT_CDR_NAMES:
        positions = _normalize_positions(
            normalized_regions[name],
            region_name=name,
            sequence_length=sequence_length,
        )
        for position in positions:
            previous = owner_by_position.get(position)
            if previous is not None:
                raise ValueError(
                    f"CDR regions overlap at VHH sequence position {position}: "
                    f"{previous} and {name}"
                )
            owner_by_position[position] = name
        ordered_regions.append((name, positions))

    mask = tuple(index in owner_by_position for index in range(1, sequence_length + 1))
    return mask, tuple(ordered_regions)


def _parse_mutable_mask(raw_mask: Any, *, expected_length: int) -> tuple[bool, ...]:
    if isinstance(raw_mask, str):
        normalized = "".join(
            character
            for character in raw_mask
            if character != "," and not character.isspace()
        )
        if set(normalized) - {"0", "1"}:
            raise ValueError("mutable_mask string must contain only 0 and 1")
        mask = tuple(character == "1" for character in normalized)
    elif isinstance(raw_mask, Sequence):
        values: list[bool] = []
        for value in raw_mask:
            if value in (0, False, "0"):
                values.append(False)
            elif value in (1, True, "1"):
                values.append(True)
            else:
                raise ValueError("mutable_mask values must be binary")
        mask = tuple(values)
    else:
        raise ValueError("mutable_mask must be a binary string or list")
    if len(mask) != expected_length:
        raise ValueError(
            "mutable_mask length must match initial_vhh_sequence: "
            f"{len(mask)} != {expected_length}"
        )
    return mask


def _normalize_hotspots(raw_hotspots: Any, *, target_length: int) -> tuple[int, ...]:
    if raw_hotspots is None:
        return ()
    if isinstance(raw_hotspots, (str, bytes)) or not isinstance(
        raw_hotspots, Sequence
    ):
        raise ValueError("target_hotspots_1based must be a list of integers")
    hotspots: list[int] = []
    for raw_position in raw_hotspots:
        if isinstance(raw_position, bool) or (
            isinstance(raw_position, float) and not raw_position.is_integer()
        ):
            raise ValueError("target_hotspots_1based must contain integers")
        try:
            position = int(raw_position)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "target_hotspots_1based must contain integers"
            ) from exc
        if position < 1 or position > target_length:
            raise ValueError(
                f"target hotspot out of range: {position}; "
                f"combined target length is {target_length}"
            )
        hotspots.append(position)
    if len(set(hotspots)) != len(hotspots):
        raise ValueError("target_hotspots_1based contains duplicate positions")
    return tuple(hotspots)


def prepare_esmfold2_vhh_case(
    case: Mapping[str, Any],
    *,
    input_dir: Path,
) -> ESMFold2VHHCase:
    """Validate a VHH case and derive its all-CDR mutable mask."""

    case_id = str(case["id"])
    raw_targets = case.get("target_sequences")
    if isinstance(raw_targets, (str, bytes)) or not isinstance(
        raw_targets, Sequence
    ):
        raise ValueError(f"{case_id}: target_sequences must be a non-empty list")
    target_sequences = tuple(
        _normalize_sequence(sequence, field_name=f"{case_id}: target_sequences")
        for sequence in raw_targets
    )
    if not target_sequences:
        raise ValueError(f"{case_id}: target_sequences must be a non-empty list")

    raw_roles = case.get("target_chain_roles")
    if raw_roles is None:
        target_chain_roles = ("target",) * len(target_sequences)
    else:
        if isinstance(raw_roles, (str, bytes)) or not isinstance(
            raw_roles, Sequence
        ):
            raise ValueError(f"{case_id}: target_chain_roles must be a list")
        target_chain_roles = tuple(str(role).strip().lower() for role in raw_roles)
        if len(target_chain_roles) != len(target_sequences):
            raise ValueError(
                f"{case_id}: target_chain_roles length must match target_sequences"
            )
        invalid_roles = sorted(set(target_chain_roles) - {"target", "scaffold"})
        if invalid_roles:
            raise ValueError(
                f"{case_id}: target_chain_roles only supports target/scaffold; "
                f"got {invalid_roles}"
            )
        if "target" not in target_chain_roles:
            raise ValueError(f"{case_id}: at least one target role is required")

    initial_vhh_sequence = _normalize_sequence(
        case["initial_vhh_sequence"],
        field_name=f"{case_id}: initial_vhh_sequence",
    )
    numbering = case.get("numbering")
    if not isinstance(numbering, Mapping):
        raise ValueError(f"{case_id}: numbering metadata is required")
    numbering_scheme = str(numbering.get("scheme", "")).strip().lower()
    if numbering_scheme != "imgt":
        raise ValueError(
            f"{case_id}: VHH all-CDR mode requires IMGT numbering; "
            f"got {numbering_scheme or 'missing'}"
        )
    raw_cdr_regions = case.get("cdr_regions")
    if not isinstance(raw_cdr_regions, Mapping):
        raise ValueError(f"{case_id}: cdr_regions mapping is required")
    mutable_mask, cdr_regions = make_imgt_all_cdr_mask(
        len(initial_vhh_sequence),
        raw_cdr_regions,
    )
    if "mutable_mask" in case:
        declared_mask = _parse_mutable_mask(
            case["mutable_mask"],
            expected_length=len(initial_vhh_sequence),
        )
        if declared_mask != mutable_mask:
            raise ValueError(
                f"{case_id}: mutable_mask does not match the union of IMGT CDR1/2/3"
            )

    target_hotspots = _normalize_hotspots(
        case.get("target_hotspots_1based"),
        target_length=sum(len(sequence) for sequence in target_sequences),
    )
    position_roles = [role for sequence, role in zip(target_sequences, target_chain_roles)
                      for _ in sequence]
    if any(position_roles[position - 1] != "target" for position in target_hotspots):
        raise ValueError(f"{case_id}: target hotspots must belong to a chain with role=target")
    reference_path = _resolve_path(
        case.get("reference_structure_path"),
        base_dir=input_dir,
    )
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(
            f"{case_id}: reference_structure_path not found: {reference_path}"
        )
    scaffold = case.get("scaffold") or {}
    if not isinstance(scaffold, Mapping):
        raise ValueError(f"{case_id}: scaffold must be a mapping")

    return ESMFold2VHHCase(
        case_id=case_id,
        target_sequences=target_sequences,
        target_chain_roles=target_chain_roles,
        initial_vhh_sequence=initial_vhh_sequence,
        mutable_mask=mutable_mask,
        cdr_regions_1based=cdr_regions,
        target_hotspots_1based=target_hotspots,
        numbering_scheme=numbering_scheme,
        numbering_tool=(
            None if numbering.get("tool") is None else str(numbering["tool"])
        ),
        numbering_tool_version=(
            None
            if numbering.get("tool_version") is None
            else str(numbering["tool_version"])
        ),
        scaffold_pdb_id=(
            None if scaffold.get("pdb_id") is None else str(scaffold["pdb_id"])
        ),
        scaffold_chain_id=(
            None if scaffold.get("chain_id") is None else str(scaffold["chain_id"])
        ),
        reference_structure_path=(
            None if reference_path is None else str(reference_path)
        ),
    )
