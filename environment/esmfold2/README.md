# ESMFold2 environment

See [installation instructions](../../docs/INSTALLATION.md) for the stock and accelerated environments, model revisions and dependency boundaries.

- `install_optimization.sh`: pinned accelerated Python 3.12 venv; requires the pinned external optimization checkout.
- `install.sh`: stock Conda environment.
- `download_models.py`: download the three pinned model revisions and print cache/CCD exports.
- `check_environment.py`: check stock environment pins and optional CUDA/model availability.
- `requirements-runtime.txt`, `requirements-biohub.txt`: stock environment locks.

These helpers do not launch design or training jobs.
