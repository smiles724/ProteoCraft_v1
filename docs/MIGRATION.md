# ESMFold2-only migration

Prepared on 2026-11-02 from the 2026-11-01 mainline. The integrated main commit
was `935e1bca4182cc54f9364ef14bd49c8ad9387e2c`; every retained source file is also
present with identical original bytes in public main commit
`60fcbfdca59e54ff40bb209ac1bd3037ffec41aa` of
[smiles724/ProteoCraft](https://github.com/smiles724/ProteoCraft).
`MIGRATION_SOURCE.json` records the original paths and SHA-256 hashes. This is a
fresh distribution and does not copy the old Git history or local uncommitted
experiments.

## Retained

- ESMFold2 experimental proposals, frozen full verification and ESMC prior.
- Two-stage MCTS, mutation sampling, dominant-residue protection, local masks,
  target/scaffold chain roles, hotspots, checkpoint and restart contracts.
- De novo, TCR binder, CDR/local redesign and VHH all-CDR redesign tasks.
- Explicit optional acceleration and model residency controls, including their
  source/device checks and diagnostics.
- Frozen independent stock ESMFold2 rescoring, with complete-seed requirements
  and structure/source checksums.
- The portable 59-case inputs and native reference structures, for reproducible
  preparation and later evaluation. These account for most remaining file bytes.
- Applicable regression tests and the original Apache license/attribution.

## Removed

- Vendored `protenix/`, the old `runner/` and Protenix V1/V2 launch profiles.
- Protenix structure loading, atom-feature/STE bridges, old training/fine-tuning
  and model-inference demos, MSA preparation and databases.
- The original `protenix` package metadata, its console command and unrelated
  training/chemical/CUDA dependencies in the root installation requirements.
- Historical target-specific experiments, cluster submission scripts, debugging
  launchers, legacy notebooks, large promotional assets and upstream PDFs.
- Multi-backend evaluation infrastructure for AtlasFold/AF3/older ESMFold,
  benchmark campaign/report-generation tools and visualization applications.
- Separate experimental dual-target/fusion launchers. The retained sequence
  design API still accepts multiple fixed context chains.
- The test-wide fake Protenix modules. Missing backend imports are tested in a
  fresh process that actively blocks both legacy backends and model SDKs.
- The copied upstream PyPI publishing workflow and legacy Docker instructions.

These components remain available in the original repository; they are not
required for this distribution's four task workflows.

## Intentional interface changes

- The Python package is `proteocraft`, and the command is `proteocraft`.
- ESMFold2 and MCTS are fixed, so `--model_backend`, `--model_name`, `--mode`
  and Protenix-only loss/precision arguments are rejected by argparse.
- The mainline `esmfold2_d` profile is applied automatically. Its numerical
  values are unchanged; only the redundant `mode`/`model_backend` routing fields
  were removed. Explicit CLI overrides retain precedence.
- Profiles and evaluation configuration are package data, so wheel installations
  can find them outside a repository checkout.
- Batch launchers use the new package; the three supported batch tools retain
  their familiar source-tree paths.
- Independent rescoring uses `proteocraft-rescore` and dispatches its worker with
  `python -m proteocraft.evaluation.cli`, including from installed wheels.
- The accelerated installer now installs this package itself. The inherited
  helper previously installed its dependencies without installing ProteoCraft.
  Installer failures propagate; the stock installer no longer masks uninstall
  failures with `|| true`.

## Validation scope

CPU regression tests cover candidate sampling, MCTS budgets and handoff,
framework/mutable masks, chain roles, hotspot/interface scoring, gradient and
diffusion helpers, model residency, optional optimization guards, independent
rescoring and fresh-process imports without Protenix. Real model calls in these
tests are replaced with explicit test backends; CUDA/external-kit tests skip
when their requirements are absent.

During the initial extraction, the 14 retained computational/search modules were compared by Python AST after
the `bindcraft` → `proteocraft` namespace change and docstring updates; their
algorithm bodies match the source. Profile lookup, package initialization,
CLI/backend routing and evaluator file locations were deliberately changed.
The stock and accelerated package/model version locks remain inherited pins.

The initial extraction was validated with 504 CPU tests and 12 skips. The
following local simplification preserved the default `esmfold2_d` numerical
recipe and removed Best-of-N, chunked Best-of-N, continuous/raw-lineage drivers
and final polish. Only the two-stage search strategy remains. The active trees
were renamed `stage_a_tree.py` and `stage_b_tree.py`; their ASTs are unchanged
apart from names. The public CLI now has 32 options; algorithm settings are
validated in maintainer YAML profiles. The ineffective `save_checkpoint_dir`
and `checkpoint_save_policy` options were removed.

The cleanup checks completed with **517 passed and 12 skipped**, plus an
explicit check that empty Stage B iterations still consume budget. All source
inputs and reference structures were preserved. Critical Ruff checks, package
builds and installed-wheel checks passed. The wheel excludes tests, benchmark
inputs, run outputs and the removed tree modules. Batch launchers reject removed
algorithm options and preserve existing profiles when skipping completed cases.
The repository's [GitHub Actions workflow](https://github.com/smiles724/ProteoCraft_v1/actions/workflows/ci.yml)
runs the CPU suite, package builds and installed-wheel checks.

Subsequent local checks corrected boolean configuration parsing, saved the final
selected structure together with its matching metrics, rejected handoff recovery
against a different fixed binder framework, and corrected boolean overrides in
the batch launcher. These changes passed CPU regression tests and installed-wheel
checks. The latest source including these boundary fixes has not yet repeated
the complete GPU search described below.

A real IFNAR2 VHH case was run with both the frozen pre-cleanup package and the
simplified package on H200 GPUs. Both completed 110 proposal calls and 2,900 soft
updates, with the 70/40 Stage A/B budget and unchanged framework sequences. Saved
structures and handoff states were validated. Delivered sequences from these
runs and the historical run were then rescored using the same unoptimized
20-loop/200-step protocol and three seeds: all 9 scores completed successfully.

The first review harness incorrectly equated trajectory rows with budget
iterations. Empty Stage B iterations append loss history without emitting an
action trajectory row. The original failed harness receipts were retained;
independent validation used all 110 contiguous history entries, the 70-round
handoff checkpoint, actual model-call counts and the saved structures.

This review also reproduced a pre-existing numerical-repeatability limitation:
fixed sequence/seed 4x50 predictions vary in both the pre-cleanup and simplified
packages, including with external acceleration disabled. Repeated input
preparation produced identical tensor features for this case. The inference
output variability has not been root-caused or changed in this cleanup. One
full case establishes the tested workflow and its constraints; it does not
establish general design-quality equivalence or experimental binding.

The existing Linux/CUDA environment and cached model revisions were reused.
Fresh Linux installation and low-memory/offload modes were not validated in
this round. No new commit or push was made during the local simplification.


## First-run workflow update, 2026-11-03

The stock Transformers URL now uses the public Hugging Face repository at the
same immutable commit. A filtered Git fetch resolved that commit successfully.
Both environment locks now use zstd 1.5.7.2, the maintainer's recommendation
after the 1.5.7.3 thread-safety withdrawal. On macOS/Python 3.12, 128 concurrent
round-trip reads of compressed NumPy/pickle data passed with the new version.

The accelerated installer now runs its matching dependency/source/GPU check.
Model download and offline checks validate pinned file lengths, safetensors
headers and ESMC shard indexes. The reusable activation script restores all
paths. A four-round Bet v 1 smoke command checks both stages, saved outputs,
applied acceleration and completed forward/gradient calls. The complete IFNAR2
VHH example copies inputs from the cleanup regression; its source checksum is
recorded in the YAML.

Local CPU tests completed with **531 passed and 12 skipped**. Regression cases
cover missing/truncated/corrupt weights, inconsistent shard indexes, editable
source detection, shell activation, the short search budget and output checks.
Lint, package builds, installed-wheel checks and input preparation passed.
The main algorithm, its default profile and model-resolution behavior were
unchanged. Fresh Linux installation, full model downloads and the new GPU smoke
command remain untested while 15122 is unavailable.
