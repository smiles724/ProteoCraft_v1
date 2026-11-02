from pathlib import Path

import torch
import yaml

from proteocraft.hallucination.esmfold2_cdr import prepare_esmfold2_cdr_case
from proteocraft.hallucination.esmfold2_template import (
    build_template_clamp,
    validate_structure_regeneration_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _encoded_atom_name(name: str) -> list[int]:
    return [ord(char) - 32 if char != " " else 0 for char in name.ljust(4)]


def _backbone_features(sequences: tuple[str, ...]) -> dict[str, torch.Tensor]:
    atom_names = ("N", "CA", "C", "O")
    token_asym: list[int] = []
    atom_to_token: list[int] = []
    encoded_names: list[list[int]] = []
    elements: list[int] = []
    token_index = 0
    for chain_index, sequence in enumerate(sequences):
        for _ in sequence:
            token_asym.append(chain_index)
            for atom_name in atom_names:
                atom_to_token.append(token_index)
                encoded_names.append(_encoded_atom_name(atom_name))
                elements.append(7 if atom_name == "N" else 8 if atom_name == "O" else 6)
            token_index += 1
    return {
        "atom_attention_mask": torch.ones(1, len(atom_to_token), dtype=torch.bool),
        "atom_to_token": torch.tensor([atom_to_token], dtype=torch.long),
        "ref_atom_name_chars": torch.tensor([encoded_names], dtype=torch.long),
        "ref_element": torch.tensor([elements], dtype=torch.long),
        "token_attention_mask": torch.ones(1, len(token_asym), dtype=torch.bool),
        "asym_id": torch.tensor([token_asym], dtype=torch.long),
    }


def test_structure_regeneration_mask_requires_two_fixed_anchors():
    assert validate_structure_regeneration_mask(
        (False, False, True, True, False, False)
    ) == (False, False, True, True, False, False)

    try:
        validate_structure_regeneration_mask((False, True, False, False))
    except ValueError as exc:
        assert "fixed framework residues on both sides" in str(exc)
    else:
        raise AssertionError("mask without two anchors should fail")


def test_real_tcr_template_maps_complex_and_frees_only_cdr_structure():
    config_path = (
        PROJECT_ROOT
        / "benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_dir = PROJECT_ROOT / config["input_dir"]
    case = prepare_esmfold2_cdr_case(config["cases"][0], input_dir=input_dir)
    chain_sequences = (*case.partner_sequences, case.binder_sequence)
    features = _backbone_features(chain_sequences)

    clamp = build_template_clamp(
        features=features,
        template_path=case.template_path,
        template_chain_ids=case.template_chain_ids,
        template_chain_sequences=chain_sequences,
        binder_structure_regeneration_mask=case.mutable_mask,
    )

    binder_token_start = sum(len(sequence) for sequence in case.partner_sequences)
    binder_atom_start = binder_token_start * 4
    binder_mask = clamp.fixed_atom_mask[0, binder_atom_start:].reshape(-1, 4)
    assert binder_mask[case.cdr_start_0based : case.cdr_end_0based_exclusive].sum() == 0
    assert binder_mask[: case.cdr_start_0based].all()
    assert binder_mask[case.cdr_end_0based_exclusive :].all()
    assert clamp.report["terminal_unresolved_residues_1based"] == {"A": [1]}
    assert clamp.report["fixed_atom_count"] == int(clamp.fixed_atom_mask.sum())
