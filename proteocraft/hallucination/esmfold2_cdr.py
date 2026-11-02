"""CDR redesign input adapter for sequence-only and template-anchored ESMFold2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ESMFold2CDRCase:
    case_id: str
    fasta_path: str
    template_path: Optional[str]
    binder_chain: str
    target_chains: tuple[str, ...]
    scaffold_chains: tuple[str, ...]
    binder_sequence: str
    partner_sequences: tuple[str, ...]
    target_chain_roles: tuple[str, ...]
    cdr_motif: Optional[str]
    mutable_mask: tuple[bool, ...]
    cdr_start_0based: Optional[int]
    cdr_end_0based_exclusive: Optional[int]
    redesign_regions_1based: tuple[tuple[int, int], ...] = ()

    @property
    def target_sequence_argument(self) -> str:
        return "|".join(self.partner_sequences)

    @property
    def target_chain_roles_argument(self) -> str:
        return ",".join(self.target_chain_roles)

    @property
    def mutable_mask_argument(self) -> str:
        return "".join("1" if value else "0" for value in self.mutable_mask)

    @property
    def template_chain_ids(self) -> tuple[str, ...]:
        return (*self.target_chains, *self.scaffold_chains, self.binder_chain)

    @property
    def template_chain_ids_argument(self) -> str:
        return ",".join(self.template_chain_ids)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Preserve the legacy record shape for existing motif-based experiments.
        payload.pop("redesign_regions_1based")
        if self.redesign_regions_1based:
            payload.update(
                redesign_regions_1based=self.redesign_regions_1based,
                redesign_region_sequences=tuple(
                    self.binder_sequence[start - 1 : end]
                    for start, end in self.redesign_regions_1based
                ),
                region_numbering="sequence_1based_inclusive",
            )
        payload.update(
            {
                "target_sequence_argument": self.target_sequence_argument,
                "target_chain_roles_argument": self.target_chain_roles_argument,
                "mutable_mask_argument": self.mutable_mask_argument,
                "template_chain_ids": self.template_chain_ids,
                "template_chain_ids_argument": self.template_chain_ids_argument,
                "cdr_start_1based": (
                    None if self.cdr_start_0based is None else self.cdr_start_0based + 1
                ),
                "cdr_end_1based": self.cdr_end_0based_exclusive,
            }
        )
        return payload


def _resolve_path(raw_path: Optional[str], *, base_dir: Path) -> Optional[Path]:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def read_chain_fasta(fasta_path: Path) -> Dict[str, str]:
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)

    records: Dict[str, str] = {}
    header: Optional[str] = None
    sequence_parts: List[str] = []

    def flush() -> None:
        if header is None:
            return
        chain_id = next(
            (
                part.replace("Chain ", "", 1).strip()
                for part in header.split("|")
                if part.strip().startswith("Chain ")
            ),
            None,
        )
        if not chain_id:
            raise ValueError(f"Could not parse chain id from FASTA header: {header}")
        sequence = "".join(sequence_parts).upper()
        invalid = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                f"{fasta_path} chain {chain_id} has unsupported amino acids: {invalid}"
            )
        if chain_id in records:
            raise ValueError(f"Duplicate chain {chain_id} in {fasta_path}")
        records[chain_id] = sequence

    for raw_line in fasta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:]
            sequence_parts = []
        else:
            sequence_parts.append(line)
    flush()
    return records


def make_cdr_mutable_mask(sequence: str, motif: str) -> tuple[tuple[bool, ...], int, int]:
    normalized_motif = str(motif).strip().upper()
    start = sequence.find(normalized_motif)
    if start < 0:
        raise ValueError(f"CDR motif not found: {normalized_motif}")
    if sequence.find(normalized_motif, start + 1) >= 0:
        raise ValueError(f"CDR motif is not unique: {normalized_motif}")
    end = start + len(normalized_motif)
    mask = tuple(start <= index < end for index in range(len(sequence)))
    return mask, start, end


def make_redesign_mutable_mask(
    sequence: str,
    regions: Sequence[Sequence[int]],
) -> tuple[tuple[bool, ...], tuple[tuple[int, int], ...]]:
    """Select non-overlapping, 1-based inclusive intervals in the input sequence."""

    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence) or not regions:
        raise ValueError("redesign_regions must be a non-empty list of [start, end] pairs")
    intervals: list[tuple[int, int]] = []
    for region in regions:
        if (
            isinstance(region, (str, bytes))
            or not isinstance(region, Sequence)
            or len(region) != 2
        ):
            raise ValueError("each redesign_regions entry must be a [start, end] pair")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in region):
            raise ValueError("redesign_regions endpoints must be integers")
        start, end = map(int, region)
        if not 1 <= start <= end <= len(sequence):
            raise ValueError(
                f"redesign_regions out of range: [{start}, {end}]; sequence length is {len(sequence)}"
            )
        intervals.append((start, end))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise ValueError(f"redesign_regions overlap: {previous} and {current}")
    mask = [False] * len(sequence)
    for start, end in intervals:
        mask[start - 1 : end] = [True] * (end - start + 1)
    return tuple(mask), tuple(intervals)


def prepare_esmfold2_cdr_case(
    case: Mapping[str, Any],
    *,
    input_dir: Path,
) -> ESMFold2CDRCase:
    case_id = str(case["id"])
    fasta_path = _resolve_path(str(case["fasta_path"]), base_dir=input_dir)
    if fasta_path is None:
        raise ValueError(f"{case_id}: fasta_path is required")
    template_path = _resolve_path(case.get("template_path"), base_dir=input_dir)
    if template_path is not None and not template_path.is_file():
        raise FileNotFoundError(f"{case_id}: template_path not found: {template_path}")

    records = read_chain_fasta(fasta_path)
    binder_chain = str(case["binder_chain"])
    target_chains = tuple(str(chain) for chain in case["target_chains"])
    scaffold_chains = tuple(str(chain) for chain in case.get("scaffold_chains", ()))
    partner_chains = (*target_chains, *scaffold_chains)
    if not target_chains:
        raise ValueError(f"{case_id}: at least one target chain is required")
    if len(set(partner_chains)) != len(partner_chains):
        raise ValueError(f"{case_id}: target/scaffold chains overlap")
    if binder_chain in partner_chains:
        raise ValueError(f"{case_id}: binder chain cannot be target/scaffold")
    missing = [
        chain for chain in (binder_chain, *partner_chains) if chain not in records
    ]
    if missing:
        raise ValueError(f"{case_id}: chains missing from FASTA: {missing}")

    binder_sequence = records[binder_chain]
    if ("cdr_motif" in case) == ("redesign_regions" in case):
        raise ValueError(f"{case_id}: specify exactly one of cdr_motif or redesign_regions")
    regions: tuple[tuple[int, int], ...] = ()
    if "redesign_regions" in case:
        motif = None
        mutable_mask, regions = make_redesign_mutable_mask(
            binder_sequence, case["redesign_regions"]
        )
        # A bounding interval would incorrectly imply the intervening framework is mutable.
        cdr_start, cdr_end = (regions[0][0] - 1, regions[0][1]) if len(regions) == 1 else (None, None)
    else:
        motif = str(case["cdr_motif"]).strip().upper()
        mutable_mask, cdr_start, cdr_end = make_cdr_mutable_mask(binder_sequence, motif)
    return ESMFold2CDRCase(
        case_id=case_id,
        fasta_path=str(fasta_path),
        template_path=None if template_path is None else str(template_path),
        binder_chain=binder_chain,
        target_chains=target_chains,
        scaffold_chains=scaffold_chains,
        binder_sequence=binder_sequence,
        partner_sequences=tuple(records[chain] for chain in partner_chains),
        target_chain_roles=(
            *("target" for _ in target_chains),
            *("scaffold" for _ in scaffold_chains),
        ),
        cdr_motif=motif,
        mutable_mask=mutable_mask,
        cdr_start_0based=cdr_start,
        cdr_end_0based_exclusive=cdr_end,
        redesign_regions_1based=regions,
    )


def framework_unchanged(
    reference_sequence: str,
    designed_sequence: str,
    mutable_mask: Sequence[bool],
) -> bool:
    if len(reference_sequence) != len(designed_sequence):
        return False
    if len(reference_sequence) != len(mutable_mask):
        return False
    return all(
        bool(is_mutable) or reference_sequence[index] == designed_sequence[index]
        for index, is_mutable in enumerate(mutable_mask)
    )
