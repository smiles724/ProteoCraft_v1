# ProteoCraft

Protein binder design with **ESMFold2**, gradient-guided sequence proposals and
two-stage Monte Carlo Tree Search (MCTS). The structure models stay frozen;
ProteoCraft optimizes the binder sequence and verifies discrete candidates with
full complex prediction.

This is the ESMFold2-only distribution of
[ProteoCraft](https://github.com/smiles724/ProteoCraft). It provides four tasks:

| Task | Fixed input | Designed sequence |
|---|---|---|
| De novo binder design | One or more target chains | Entire binder |
| TCR binder design | TCR target chains | Entire binder |
| Local/CDR redesign | Antigen, partner chains, starting binder | Explicit mutable positions |
| VHH redesign | Target and starting VHH | All three IMGT CDRs |

## Install

GPU execution uses **Linux x86-64, Python 3.12 and an NVIDIA CUDA GPU**. The current
accelerated defaults require an SM80/SM90 GPU and the pinned optimization stack.
CPU installation is useful for unit tests, argument validation and benchmark
preparation; it does not run the full model workflow.

```bash
git clone https://github.com/smiles724/ProteoCraft_v1.git
cd ProteoCraft_v1

git clone https://github.com/anthropics/uplifting-biomolecular-modeling.git ../uplifting-biomolecular-modeling
git -C ../uplifting-biomolecular-modeling checkout --detach f4f62fa6592ae4938d49b1757bea0cfeff9f468e
export PROTEOCRAFT_UPLIFT_ROOT="$(cd ../uplifting-biomolecular-modeling && pwd)"

bash environment/esmfold2/install_optimization.sh /absolute/path/to/proteocraft-env
source /absolute/path/to/proteocraft-env/bin/activate
python environment/esmfold2/download_models.py --cache-dir /absolute/path/to/hf_cache
```

Export the cache and CCD variables printed by the download command. It downloads
three pinned model revisions, including ESMC-6B; provision GPU/host memory and disk
for these models. The installer installs this package into the environment.
See [installation details](docs/INSTALLATION.md) for the stock environment and
the precise dependency boundaries.

## Design a binder

```bash
proteocraft \
  --target GIVEQCCTSICSLYQLENYCN \
  --binder_length 80 \
  --output_dir outputs/first_design \
  --device cuda:0
```

This is a syntax example with a short target, not a validated design benchmark.
The bundled `esmfold2_d` profile loads automatically. Add
`--target_hotspots "4,8-10"` to designate target residues. For multiple target
chains, join their sequences with `|`; hotspot positions are **1-based over their
concatenation**. Explicit CLI options override profile values.

The mainline search settings are preserved: seed 43, 8 starting roots, 70/40
Stage A/B outer steps, 30/20 soft steps per proposal, guarded positive-score
Stage A candidates, single-substitution Stage B candidates, and a 4-loop/50-step
search verifier. Proposal/verifier/ESMC optimization defaults are `fast/fast/exact`.
These settings are operational defaults, not evidence of universally optimal
design quality.

Outputs include `hallucination_result.txt`, plus
`esmfold2_artifacts/run_summary.json`, `candidates.jsonl` and `trajectory.jsonl`.
Use `--esmfold2_save_verified_candidate_cif true` to retain verified structures.

See [task examples and outputs](docs/USAGE.md),
[the core algorithm](docs/ALGORITHM.md), or `proteocraft --help` for advanced
settings, restart controls, memory management and model options.

## Prepare the reproducible benchmark

The original portable inputs remain available: **14 de novo + 41 TCR + 4 CDR
redesign cases**. No model or GPU is needed for preparation:

```bash
python benchmark/script/run_esmfold2_59.py --prepare-only
# A real GPU design run, when you choose to launch it:
python benchmark/script/run_esmfold2_59.py --case denovo_BetV1 --device cuda:0
```

These commands require an installed checkout. Full benchmark inputs are in the
source repository, and are not included in the Python wheel.

## Independent rescoring

`proteocraft-rescore` freezes candidates and a separate unoptimized ESMFold2
evaluation protocol (20 loops × 200 sampling steps and three evaluation seeds).
It rejects accelerated evaluation settings and incomplete seed matrices.

```bash
proteocraft-rescore prepare --manifest /path/to/candidates.jsonl --out-dir outputs/rescore
proteocraft-rescore run --out-dir outputs/rescore --device cuda:0
proteocraft-rescore report --out-dir outputs/rescore
```

See [the candidate manifest format](docs/USAGE.md#independent-rescoring). Search
confidence and rescoring confidence are computational evidence; they do not
establish experimental binding, specificity or affinity.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

The CPU regression suite uses small deterministic tensors and explicit fake
backends to test search contracts; it does not download model weights or claim
GPU/kernel parity. See [migration scope and validation](docs/MIGRATION.md).

## License and attribution

Software: [Apache 2.0](LICENSE). Original notices are retained in [NOTICE](NOTICE).
External model weights and optimization sources retain their own licenses.
