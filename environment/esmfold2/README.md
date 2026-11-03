# ESMFold2 environment

See [installation instructions](../../docs/INSTALLATION.md) for the stock and accelerated environments, model revisions and dependency boundaries.

- `install_optimization.sh`: pinned accelerated Python 3.12 venv; checks dependencies, sources and GPU, then writes `activate_proteocraft.sh`.
- `install.sh`: stock Conda environment.
- `download_models.py`: download and validate the three pinned model revisions; `--write-env` saves all activation paths in one script.
- `check_environment.py`: check the accelerated environment by default, or select `--stack stock`; `--require-models` checks all pinned weight files and shard indexes offline.
- `requirements-runtime.txt`, `requirements-biohub.txt`: stock environment locks.

These helpers do not launch design or training jobs.
After installation, `python scripts/smoke_test.py` runs the short two-stage GPU search.
