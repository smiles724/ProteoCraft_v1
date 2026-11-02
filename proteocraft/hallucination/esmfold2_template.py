"""External-template atom mapping for local ESMFold2 redesign.

The template is mapped onto ESMFold2's padded atom layout by explicit chain,
residue, atom-name, and element identity.  This module only prepares immutable
coordinates and masks; diffusion-time frame handling lives in
``esmfold2_one_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
REQUIRED_BACKBONE_ATOMS = frozenset({"N", "CA", "C"})
_ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
    "SE": 34,
}


@dataclass(frozen=True)
class TemplateAtomRecord:
    chain_index: int
    residue_index: int
    atom_name: str
    atomic_number: int
    coordinate: tuple[float, float, float]


@dataclass(frozen=True)
class ParsedTemplate:
    atoms: tuple[TemplateAtomRecord, ...]
    mapped_residue_indices: tuple[tuple[int, ...], ...]
    terminal_unresolved_residues: tuple[tuple[int, ...], ...]
    internal_unresolved_residues: tuple[tuple[int, ...], ...]


@dataclass
class TemplateClamp:
    coordinates: torch.Tensor
    fixed_atom_mask: torch.Tensor
    report: dict[str, Any]


def _feature0(features: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = features[name]
    if value.ndim > 1 and value.shape[0] == 1:
        return value[0]
    return value


def _decode_atom_name(encoded: torch.Tensor) -> str:
    return "".join(
        chr(int(value) + 32) if int(value) > 0 else " "
        for value in encoded.tolist()
    ).strip()


def _residue_letter(residue: Any) -> str:
    from Bio.SeqUtils import seq1

    return seq1(
        str(residue.resname),
        custom_map={"MSE": "M", "SEC": "U", "PYL": "O"},
        undef_code="X",
    )


def _aligned_residue_map(
    expected_sequence: str,
    observed_residues: Sequence[Any],
) -> tuple[dict[int, Any], tuple[int, ...], tuple[int, ...]]:
    from Bio.Align import PairwiseAligner

    observed_sequence = "".join(_residue_letter(residue) for residue in observed_residues)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -4.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    alignments = aligner.align(expected_sequence, observed_sequence)
    if len(alignments) == 0:
        raise ValueError("template chain could not be aligned to its reference sequence")

    residue_by_index: dict[int, Any] = {}
    alignment = alignments[0]
    for expected_block, observed_block in zip(
        alignment.aligned[0],
        alignment.aligned[1],
        strict=True,
    ):
        expected_start, expected_end = map(int, expected_block)
        observed_start, observed_end = map(int, observed_block)
        if expected_end - expected_start != observed_end - observed_start:
            raise ValueError("template alignment produced incompatible aligned blocks")
        for offset in range(expected_end - expected_start):
            expected_index = expected_start + offset
            observed_index = observed_start + offset
            if expected_sequence[expected_index] != observed_sequence[observed_index]:
                continue
            residue_by_index[expected_index] = observed_residues[observed_index]

    if not residue_by_index:
        raise ValueError("template chain has no sequence-identical mapped residues")
    mapped = sorted(residue_by_index)
    internal_missing = tuple(
        index
        for index in range(mapped[0], mapped[-1] + 1)
        if index not in residue_by_index
    )
    terminal_missing = tuple(range(0, mapped[0])) + tuple(
        range(mapped[-1] + 1, len(expected_sequence))
    )
    return residue_by_index, terminal_missing, internal_missing


def _select_atoms(residue: Any) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for atom in residue.get_unpacked_list():
        name = str(atom.get_name()).strip().upper()
        occupancy = atom.get_occupancy()
        occupancy_value = float(occupancy) if occupancy is not None else 0.0
        previous = selected.get(name)
        if previous is None:
            selected[name] = atom
            continue
        previous_occupancy = previous.get_occupancy()
        previous_value = (
            float(previous_occupancy) if previous_occupancy is not None else 0.0
        )
        if occupancy_value > previous_value:
            selected[name] = atom
    return selected


@lru_cache(maxsize=16)
def _load_parsed_template(
    template_path: str,
    chain_ids: tuple[str, ...],
    chain_sequences: tuple[str, ...],
) -> ParsedTemplate:
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(template_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if len(chain_ids) != len(chain_sequences):
        raise ValueError("template_chain_ids and template_chain_sequences must match")
    parser = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    model = parser.get_structure(path.stem, str(path))[0]

    records: list[TemplateAtomRecord] = []
    mapped_indices: list[tuple[int, ...]] = []
    terminal_missing_by_chain: list[tuple[int, ...]] = []
    internal_missing_by_chain: list[tuple[int, ...]] = []
    available_chain_ids = {str(chain.id) for chain in model}
    for chain_index, (chain_id, sequence) in enumerate(
        zip(chain_ids, chain_sequences, strict=True)
    ):
        if chain_id not in available_chain_ids:
            raise ValueError(
                f"template chain {chain_id!r} is missing from {path.name}"
            )
        chain = model[chain_id]
        observed_residues = [
            residue
            for residue in chain
            if str(residue.id[0]).strip() in {"", "H_MSE", "H_SEC"}
        ]
        residue_by_index, terminal_missing, internal_missing = _aligned_residue_map(
            str(sequence).upper(),
            observed_residues,
        )
        mapped_indices.append(tuple(sorted(residue_by_index)))
        terminal_missing_by_chain.append(terminal_missing)
        internal_missing_by_chain.append(internal_missing)
        for residue_index, residue in residue_by_index.items():
            for atom_name, atom in _select_atoms(residue).items():
                element = str(getattr(atom, "element", "") or "").strip().upper()
                atomic_number = _ELEMENT_TO_ATOMIC_NUMBER.get(element)
                if atomic_number is None:
                    continue
                coordinate = tuple(float(value) for value in atom.coord)
                records.append(
                    TemplateAtomRecord(
                        chain_index=chain_index,
                        residue_index=residue_index,
                        atom_name=atom_name,
                        atomic_number=atomic_number,
                        coordinate=coordinate,
                    )
                )
    return ParsedTemplate(
        atoms=tuple(records),
        mapped_residue_indices=tuple(mapped_indices),
        terminal_unresolved_residues=tuple(terminal_missing_by_chain),
        internal_unresolved_residues=tuple(internal_missing_by_chain),
    )


def validate_structure_regeneration_mask(
    mask: Sequence[bool],
    *,
    minimum_framework_anchors: int = 2,
) -> tuple[bool, ...]:
    """Validate local redesign intervals and their fixed framework anchors."""

    normalized = tuple(bool(value) for value in mask)
    if not normalized or not any(normalized):
        raise ValueError("binder_structure_regeneration_mask must select residues")
    index = 0
    while index < len(normalized):
        if not normalized[index]:
            index += 1
            continue
        start = index
        while index < len(normalized) and normalized[index]:
            index += 1
        end = index
        anchors = int(minimum_framework_anchors)
        if (
            start < anchors
            or len(normalized) - end < anchors
            or any(normalized[start - anchors : start])
            or any(normalized[end : end + anchors])
        ):
            raise ValueError(
                "each structure-regeneration interval requires at least "
                f"{minimum_framework_anchors} fixed framework residues on both sides"
            )
    return normalized


def build_template_clamp(
    *,
    features: Mapping[str, torch.Tensor],
    template_path: str,
    template_chain_ids: Sequence[str],
    template_chain_sequences: Sequence[str],
    binder_structure_regeneration_mask: Sequence[bool],
) -> TemplateClamp:
    """Map a complex template into the current ESMFold2 atom layout."""

    chain_ids = tuple(str(chain_id).strip() for chain_id in template_chain_ids)
    chain_sequences = tuple(
        str(sequence).strip().upper() for sequence in template_chain_sequences
    )
    structure_mask = validate_structure_regeneration_mask(
        binder_structure_regeneration_mask
    )
    if len(chain_ids) != len(chain_sequences):
        raise ValueError("template chain ids and sequences must have equal lengths")
    if len(structure_mask) != len(chain_sequences[-1]):
        raise ValueError(
            "binder_structure_regeneration_mask length must match binder sequence"
        )

    parsed = _load_parsed_template(
        str(Path(template_path).expanduser().resolve()),
        chain_ids,
        chain_sequences,
    )
    atom_by_identity = {
        (
            atom.chain_index,
            atom.residue_index,
            atom.atom_name,
            atom.atomic_number,
        ): atom
        for atom in parsed.atoms
    }

    atom_mask = _feature0(features, "atom_attention_mask").bool()
    atom_to_token = _feature0(features, "atom_to_token").long()
    atom_names = _feature0(features, "ref_atom_name_chars").long()
    elements = _feature0(features, "ref_element").long()
    token_mask = _feature0(features, "token_attention_mask").bool()
    asym_id = _feature0(features, "asym_id").long()
    if not (
        atom_mask.shape[0]
        == atom_to_token.shape[0]
        == atom_names.shape[0]
        == elements.shape[0]
    ):
        raise ValueError("ESMFold2 atom identity features have inconsistent sizes")

    chain_values: list[int] = []
    for token_index in token_mask.nonzero(as_tuple=True)[0].tolist():
        value = int(asym_id[token_index].item())
        if value not in chain_values:
            chain_values.append(value)
    if len(chain_values) != len(chain_ids):
        raise ValueError(
            "template chain count does not match ESMFold2 input chain count: "
            f"template={len(chain_ids)}, model={len(chain_values)}"
        )

    token_identity: dict[int, tuple[int, int]] = {}
    for chain_index, chain_value in enumerate(chain_values):
        chain_tokens = [
            token_index
            for token_index in token_mask.nonzero(as_tuple=True)[0].tolist()
            if int(asym_id[token_index].item()) == chain_value
        ]
        if len(chain_tokens) != len(chain_sequences[chain_index]):
            raise ValueError(
                f"model chain {chain_index} token count does not match template "
                f"reference sequence: {len(chain_tokens)} != "
                f"{len(chain_sequences[chain_index])}"
            )
        token_identity.update(
            {
                token_index: (chain_index, residue_index)
                for residue_index, token_index in enumerate(chain_tokens)
            }
        )

    coordinates = torch.zeros(
        1,
        atom_mask.shape[0],
        3,
        dtype=torch.float32,
        device=atom_mask.device,
    )
    fixed_mask = torch.zeros(
        1,
        atom_mask.shape[0],
        dtype=torch.bool,
        device=atom_mask.device,
    )
    available_names_by_residue: dict[tuple[int, int], set[str]] = {}
    missing_optional: list[str] = []
    mapped_atoms = 0
    binder_chain_index = len(chain_ids) - 1
    for atom_index in atom_mask.nonzero(as_tuple=True)[0].tolist():
        token_index = int(atom_to_token[atom_index].item())
        chain_index, residue_index = token_identity[token_index]
        atom_name = _decode_atom_name(atom_names[atom_index])
        atomic_number = int(elements[atom_index].item())
        is_fixed = chain_index != binder_chain_index or (
            not structure_mask[residue_index] and atom_name in BACKBONE_ATOMS
        )
        if not is_fixed:
            continue
        key = (chain_index, residue_index, atom_name, atomic_number)
        record = atom_by_identity.get(key)
        if record is None:
            if residue_index in parsed.mapped_residue_indices[chain_index]:
                missing_optional.append(
                    f"{chain_ids[chain_index]}:{residue_index + 1}:{atom_name}"
                )
            continue
        coordinates[0, atom_index] = torch.tensor(
            record.coordinate,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        fixed_mask[0, atom_index] = True
        available_names_by_residue.setdefault((chain_index, residue_index), set()).add(
            atom_name
        )
        mapped_atoms += 1

    missing_required: list[str] = []
    unresolved_fixed_residues: list[str] = []
    for chain_index, _sequence in enumerate(chain_sequences):
        for residue_index in parsed.internal_unresolved_residues[chain_index]:
            if chain_index == binder_chain_index and structure_mask[residue_index]:
                continue
            unresolved_fixed_residues.append(
                f"{chain_ids[chain_index]}:{residue_index + 1}"
            )
        for residue_index in parsed.mapped_residue_indices[chain_index]:
            if chain_index == binder_chain_index and structure_mask[residue_index]:
                continue
            names = available_names_by_residue.get((chain_index, residue_index), set())
            missing = sorted(REQUIRED_BACKBONE_ATOMS - names)
            if missing:
                missing_required.append(
                    f"{chain_ids[chain_index]}:{residue_index + 1}:"
                    + "/".join(missing)
                )
    if unresolved_fixed_residues:
        raise ValueError(
            "template has unresolved or sequence-mismatched internal fixed "
            "residues: " + ", ".join(unresolved_fixed_residues[:12])
        )
    if missing_required:
        raise ValueError(
            "template fixed residues are missing required backbone atoms: "
            + ", ".join(missing_required[:12])
        )
    if mapped_atoms == 0:
        raise ValueError("template mapping produced no fixed atoms")

    terminal_unresolved = {
        chain_ids[index]: [position + 1 for position in positions]
        for index, positions in enumerate(parsed.terminal_unresolved_residues)
        if positions
    }
    internal_unresolved_free = {
        chain_ids[index]: [position + 1 for position in positions]
        for index, positions in enumerate(parsed.internal_unresolved_residues)
        if positions
    }
    report = {
        "template_path": str(Path(template_path).expanduser().resolve()),
        "template_chain_ids": list(chain_ids),
        "fixed_atom_count": int(fixed_mask.sum().item()),
        "free_atom_count": int((atom_mask.unsqueeze(0) & ~fixed_mask).sum().item()),
        "structure_regeneration_residue_count": int(sum(structure_mask)),
        "terminal_unresolved_residues_1based": terminal_unresolved,
        "internal_unresolved_free_residues_1based": internal_unresolved_free,
        "missing_optional_atoms": sorted(set(missing_optional)),
    }
    return TemplateClamp(
        coordinates=coordinates,
        fixed_atom_mask=fixed_mask,
        report=report,
    )
