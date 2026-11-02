from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from proteocraft.evaluation import esmfold2_runner


@dataclass(frozen=True)
class FakeConfig:
    target_chain_roles: tuple[str, ...] | None = None
    verifier_model_name: str = "biohub/ESMFold2"
    verifier_num_loops: int = 20
    verifier_num_sampling_steps: int = 200
    verifier_loss_mode: str = "aligned_common"


class FakeBackend:
    def __init__(self):
        self.config = FakeConfig()
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append((kwargs, self.config.target_chain_roles))
        return SimpleNamespace(
            total_loss=0.18,
            summary_confidence={
                "pair_chain_iptm": 0.82,
                "iptm": 0.91,
                "ptm": 0.88,
                "plddt": 0.86,
                "binder_plddt": 0.89,
                "binder_rg_excess": 0.02,
                "binder_radius_gyration": 14.2,
                "hotspot_mean_min_heavy_atom_distance": 6.1,
                "hotspot_heavy_atom_coverage": 0.75,
                "hotspot_heavy_atom_coverage_5a": 0.50,
                "mmcif": "data_test\n#\n",
            },
        )








def test_parse_multichain_roles_and_hotspots():
    assert esmfold2_runner.parse_target_sequences("AAAA:CCCC") == ["AAAA", "CCCC"]
    assert esmfold2_runner.parse_target_chain_roles("target+scaffold", 2) == (
        "target",
        "scaffold",
    )
    assert esmfold2_runner.parse_hotspot_indices("[3, 1, 3]") == [3, 1]
    with pytest.raises(ValueError, match="length must match"):
        esmfold2_runner.parse_target_chain_roles("target", 2)
