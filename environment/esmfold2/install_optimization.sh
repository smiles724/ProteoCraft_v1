#!/usr/bin/env bash
# A portable adapter environment using the ef2inv kernel stack pins.
set -euo pipefail
: "${PROTEOCRAFT_UPLIFT_ROOT:?Set PROTEOCRAFT_UPLIFT_ROOT to the optimization checkout}"
ESMOPT_ENV="${1:?Usage: install_optimization.sh /absolute/path/to/new/environment}"
ESMOPT_BASE_PYTHON="${ESMOPT_BASE_PYTHON:-python3.12}"
ESMOPT_ESM_COMMIT="d0207ea3c8cbf072679ece3bb332787d25e06852"
ESMOPT_TORCH_URL="https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997"
ESMOPT_TRITON_URL="https://download.pytorch.org/whl/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=6f5928e6d44c34a97bbe164cceddc0ef2007121c89ebcfba5415cf452de7ee9f"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ESMOPT_UPLIFT_COMMIT="f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
if [[ "$(git -C "$PROTEOCRAFT_UPLIFT_ROOT" rev-parse HEAD)" != "$ESMOPT_UPLIFT_COMMIT" ]]; then
  echo "Optimization checkout must be pinned to $ESMOPT_UPLIFT_COMMIT" >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This pinned wheel environment requires Linux x86-64" >&2
  exit 1
fi
"$ESMOPT_BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ -e "$ESMOPT_ENV" ]]; then
  echo "Choose a new environment path; refusing to modify $ESMOPT_ENV" >&2
  exit 1
fi
"$ESMOPT_BASE_PYTHON" -m venv "$ESMOPT_ENV"
"$ESMOPT_ENV/bin/python" -m pip install \
  "torch @ $ESMOPT_TORCH_URL" "triton @ $ESMOPT_TRITON_URL"
# requirements-runtime.txt remains the baseline ESMFold2 lock. Keep its
# non-engine dependencies, while the optimization environment uses the newer
# torch/triton pair required by ef2inv/stock/PINS.json.
ESMOPT_RUNTIME_REQ="$(mktemp)"
trap 'rm -f "$ESMOPT_RUNTIME_REQ"' EXIT
awk '!/^torch==/ && !/^triton==/' \
  "$REPO_ROOT/environment/esmfold2/requirements-runtime.txt" > "$ESMOPT_RUNTIME_REQ"
"$ESMOPT_ENV/bin/python" -m pip install \
  -r "$ESMOPT_RUNTIME_REQ"
# The wheel is part of the kit's pinned base stack. flash-attn and
# transformer-engine need locally compiled, torch-specific wheels and remain
# optional for this ProteoCraft adapter.
"$ESMOPT_ENV/bin/python" -m pip install --no-deps \
  'xformers==0.0.35' 'regex==2026.7.19'
"$ESMOPT_ENV/bin/python" -m pip install --no-deps \
  "$PROTEOCRAFT_UPLIFT_ROOT/esmfold2/stock/transformers-ef32577f.tar.gz"
# ef2inv/stock/esm-d0207ea3.tar.gz is intentionally a partial audit archive,
# not an installable copy of the esm package.  The kit pins an editable full
# checkout because ESMFold2 and the internal ESMC route share that package.
ESMOPT_ESM_SOURCE="${ESMOPT_ENV}.src/esm"
git clone --filter=blob:none https://github.com/Biohub/esm.git "$ESMOPT_ESM_SOURCE"
git -C "$ESMOPT_ESM_SOURCE" checkout --detach "$ESMOPT_ESM_COMMIT"
"$ESMOPT_ENV/bin/python" -m pip install --no-deps \
  --editable "$ESMOPT_ESM_SOURCE"
"$ESMOPT_ENV/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"
"$ESMOPT_ENV/bin/python" -m pip freeze > "$ESMOPT_ENV/requirements.receipt.txt"
printf 'Environment ready: %s\nRun from: %s\n' "$ESMOPT_ENV" "$REPO_ROOT"
