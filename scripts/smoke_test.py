#!/usr/bin/env python3
"""Run a four-round accelerated Bet v 1 search and check its outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from proteocraft.hallucination.profile_loader import write_case_profile  # noqa: E402

# Start from the mainline profile; keep losses, acceleration and 4x50 verification.
SMOKE_OVERRIDES = {
    "n_steps": 4, "mcts_bootstrap_steps": 2, "mcts_bootstrap_max_steps": 2,
    "mcts_num_beams": 1, "mcts_handoff_top_k": 1,
    "esmfold2_stagea_soft_steps": 5, "esmfold2_stageb_soft_steps": 5,
    "b_candidates": 2, "esmfold2_stageb_verifier_candidates": 2,
    "esmfold2_stageb_rerank_pool_size": 4,
}


def check_outputs(case_dir: Path) -> None:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    artifacts = case_dir / "esmfold2_artifacts"
    summary = json.loads((artifacts / "run_summary.json").read_text())
    if summary["n_steps"] != 4 or summary["strategy"] != "two_stage":
        raise ValueError("Smoke search did not complete the four-round two-stage budget")
    handoff = (artifacts / "mcts_handoff.jsonl").read_text().strip()
    if not handoff or "stageb_selection_surfaces" not in summary:
        raise ValueError("Smoke search did not reach Stage B")
    checkpoint = artifacts / "checkpoints/final/step_0004"
    meta = json.loads((checkpoint / "step_0004_best_meta.json").read_text())
    if (meta["binder_seq"] != summary["best_sequence"]
            or meta["total_loss"] != summary["best_loss"]
            or not math.isfinite(meta["total_loss"])):
        raise ValueError("Selected sequence/loss does not match the final structure metadata")
    if meta["structure_status"] != "ok" or meta["structure_file"] != "step_0004_best.cif":
        raise ValueError("The selected final structure was not saved")
    structure = MMCIF2Dict(str(checkpoint / "step_0004_best.cif"))
    for axis in ("x", "y", "z"):
        coordinates = structure[f"_atom_site.Cartn_{axis}"]
        if not coordinates or not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError("Final structure contains missing or non-finite coordinates")
    report = json.loads((case_dir / "optimization_report.json").read_text())
    if report["requested"] != {"proposal": "fast", "verifier": "fast", "esmc": "exact"}:
        raise ValueError("Smoke run did not retain default acceleration")
    for phase in ("proposal", "verifier"):
        if report["applied"].get(phase, {}).get("mode") != "fast":
            raise ValueError(f"{phase} acceleration was not applied")
    for label in ("proposal_esmc", "verifier_esmc", "prior_esmc"):
        if report["applied"].get(label, {}).get("mode") != "exact":
            raise ValueError(f"{label} acceleration was not applied")
    for phase in ("proposal_update", "proposal_forward", "proposal_diffusion_gradient",
                  "esmc_prior_forward", "verifier"):
        if report["counters"].get(phase + ".successes", 0) < 1 or report["counters"].get(phase + ".errors", 0):
            raise ValueError(f"Smoke run has no successful {phase} call, or recorded an error")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/smoke")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare inputs and profile on CPU")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    case_dir = output / "denovo_BetV1"
    if case_dir.exists():
        raise FileExistsError(f"{case_dir} already exists; choose a new --output-dir")
    if not args.prepare_only:
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / "environment/esmfold2/check_environment.py"),
            "--require-cuda", "--require-models", "--device", args.device,
        ], check=True)
    profile = write_case_profile("esmfold2_d", SMOKE_OVERRIDES, output / "smoke_profile.yaml")
    command = [
        sys.executable, str(PROJECT_ROOT / "benchmark/script/run_esmfold2_59.py"),
        "--case", "denovo_BetV1", "--profile", str(profile),
        "--output-dir", str(output), "--device", args.device,
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.prepare_only:
        print("Inputs prepared; GPU search has not run.")
    else:
        check_outputs(case_dir)
        print(f"Smoke test passed: Stage A (2) -> Stage B (2), finite structure and matching metrics.\nResults: {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
