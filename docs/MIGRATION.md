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

The 14 retained computational/search modules were compared by Python AST after
the `bindcraft` → `proteocraft` namespace change and docstring updates; their
algorithm bodies match the source. Profile lookup, package initialization,
CLI/backend routing and evaluator file locations were deliberately changed.
The stock and accelerated package/model version locks remain inherited pins.

The cleanup was validated on macOS with Python 3.12 CPU tests, package builds,
installed command checks, all 59 benchmark inputs and all four CDR batch inputs.
The final CPU run completed with **504 passed and 12 skipped**. All 25 original
benchmark input-file SHA-256 checks passed. Wheel installation was checked in a
separate environment outside the source checkout, including bundled profile and
evaluator-source lookup and model-free rescoring preparation. Critical Ruff
checks and shell syntax checks passed. The new GitHub Actions workflow has been
prepared; it has not run on GitHub during local validation.
This work did **not** install the Linux CUDA environment, download model weights,
run new GPU inference, measure acceleration, or evaluate wet-lab design quality.
GPU execution of the cleaned distribution remains unverified.
