#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="${1:-proteocraft_esmfold2_py312}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniforge or Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  conda env create --name "$ENV_NAME" --file "$SCRIPT_DIR/environment.yml"
fi

run_in_env() {
  conda run --no-capture-output --name "$ENV_NAME" "$@"
}

run_in_env python -m pip install --upgrade pip setuptools wheel

# fair-esm and Biohub esm expose the same Python namespace. The ESMFold2 path
# requires the Biohub package and must use a separate environment from Protenix.
run_in_env python -m pip uninstall --yes fair-esm
run_in_env python -m pip install --requirement "$SCRIPT_DIR/requirements-runtime.txt"
run_in_env python -m pip install --no-deps \
  --requirement "$SCRIPT_DIR/requirements-biohub.txt"
run_in_env python -m pip install --editable "$REPO_ROOT" --no-deps

run_in_env python "$SCRIPT_DIR/check_environment.py" --require-cuda

cat <<EOF

Environment installed: $ENV_NAME

Next steps:
  conda activate $ENV_NAME
  python environment/esmfold2/download_models.py --cache-dir /path/to/hf_cache

Then export the paths printed by download_models.py before launching a run.
EOF
