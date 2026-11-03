# Distribution and validation

## Source and attribution

This repository provides the ESMFold2 binder-design workflow from
[ProteoCraft](https://github.com/smiles724/ProteoCraft), including two-stage MCTS,
four task adapters and independent rescoring.

[NOTICE](../NOTICE) records the upstream attribution.
[MIGRATION_SOURCE.json](../MIGRATION_SOURCE.json) lists the original file paths
and source hashes. The software uses [Apache 2.0](../LICENSE); external model
weights and optimization sources retain their own licenses.

## Interfaces

- `proteocraft` runs the two-stage design workflow with the bundled `esmfold2_d`
  profile. Advanced search settings use validated YAML profiles.
- `proteocraft-rescore` evaluates fixed sequences with the independent protocol.
- Benchmark inputs, batch launchers and environment helpers are included in the
  source checkout. The wheel contains the core package, commands and profiles.

See [installation](INSTALLATION.md) and [task examples](USAGE.md) for commands.

## Validation scope

| Check | Result |
|---|---|
| CPU regression | 531 passed, 12 skipped |
| Packaging | Ruff, source/wheel builds and installed-wheel command checks passed |
| Model downloads | Three pinned snapshots verified, including weight shards, tokenizer files and CCD; hashes matched official repository metadata |
| Clean Linux installation | Python 3.12 and the pinned accelerated environment installed successfully |
| Bet v 1 GPU smoke test | Stage A: 2 rounds; Stage B: 2 rounds; default `fast / fast / exact` acceleration and 4×50 verification; gradients, saved structure and matching metrics checked |
| Full IFNAR2 VHH search | 70 Stage A rounds and 40 Stage B rounds completed on H200 using an earlier revision; framework sequences preserved |

The current code has passed the Bet v 1 GPU smoke test. A full 110-round rerun
of the current code remains pending. GPU validation used resident memory mode.

Repeated predictions can vary even with fixed inputs and seeds. Compare
candidates using matched model revisions, inputs and evaluation protocols.

[GitHub Actions](https://github.com/smiles724/ProteoCraft_v1/actions/workflows/ci.yml)
runs the CPU suite and package checks.
