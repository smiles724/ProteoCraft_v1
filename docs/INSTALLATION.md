# Installation

## System prerequisites

Use Linux x86-64, Python 3.12 and an NVIDIA GPU. The default fast kernels accept
compute capability 8.0 (A100) and 9.0 (H100/H200); the full regression was run on
H200. Check the installed NVIDIA driver with `nvidia-smi` before installation.
For Ubuntu 24.04, install the Python venv tools, Git and a C compiler:

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv build-essential
```

Set `PROTEOCRAFT_WORKDIR` in the README commands to a writable disk with room for
the model cache (about 25.6 GiB), the Python environment and pip's download cache.

## Accelerated mainline environment

Use the pinned Linux x86-64/Python 3.12 installer in the README. It uses Torch
2.11.0+cu128, Triton 3.6.0, Biohub ESM commit
`d0207ea3c8cbf072679ece3bb332787d25e06852`, and optimization-source commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
The installer automatically checks the selected dependencies, pinned model and
optimization sources, API imports and GPU support. Model download checks each
required file's pinned size and safetensors headers, including all six ESMC
shards and their index. These checks read metadata without loading model tensors.

The README's `--write-env` command saves the venv, optimization checkout, cache,
CCD and offline settings in one script. In a new shell, from the repository root:

```bash
source "$HOME/proteocraft-runtime/env/activate_proteocraft.sh"
```

Use your chosen `PROTEOCRAFT_WORKDIR` directory if different. To diagnose an
installation manually, run:

```bash
python environment/esmfold2/check_environment.py --require-cuda --require-models
```

This command checks the accelerated stack by default and performs no downloads.
It checks file completeness and safetensors structure, rather than hashing all
25.6 GiB on each invocation.

The helper installs selected kernels used by the ProteoCraft adapter. It does
not install every optional Biohub SDK feature. In particular, external ESM
metadata declares DockQ and CUDA-13 cuEquivariance dependencies that this pinned
adapter environment does not install. `pip check` therefore reports these
missing SDK dependencies; this is a known limitation of the inherited stack.
The adapter does not use those SDK paths. The mainline was exercised on H200
using the existing pinned environment and cached models; a fresh installer run
was not part of that validation.

For a full 110-round IFNAR2 VHH run (128-residue binder, 37 mutable residues),
resident mode peaked at approximately **73.15 GiB allocated / 78.18 GiB reserved**
by Torch. These are measurements for this input, not a universal memory minimum.
Low-memory/offload modes and other GPU types need separate runtime validation.

Unset upstream `ESMFOLD2_OPT`, `ESMC_OPT`, `EF2_FAST_KIT` and `ESMC_KIT` to avoid
implicit model patching outside the recorded adapter settings. Run different
optimization modes in separate processes.

## Stock model environment

The separate stock environment retains the original pinned Torch 2.7.1 and
Biohub source revisions. With Conda installed:

```bash
bash environment/esmfold2/install.sh proteocraft_esmfold2_py312
conda activate proteocraft_esmfold2_py312
python environment/esmfold2/download_models.py --cache-dir /absolute/path/to/hf_cache
```

Export the variables printed by the download helper, then check:

```bash
python environment/esmfold2/check_environment.py --stack stock --require-cuda --require-models
```

The stock environment requires all three acceleration switches to be explicitly
off, including when using the default profile:

```bash
proteocraft --target GIVEQCCTSICSLYQLENYCN --binder_length 80 \
  --esmfold2_proposal_optimization off \
  --esmfold2_verifier_optimization off \
  --esmfold2_esmc_optimization off \
  --output_dir outputs/stock --device cuda:0
```

`--stack stock` selects the stock pins; the default acceleration settings remain
unchanged. Both stacks use `zstd==1.5.7.2`, the maintainer's recommended version
after 1.5.7.3 was yanked for thread-safety defects.

## Models

`download_models.py` fixes the following Hugging Face revisions:

| Model | Revision |
|---|---|
| biohub/ESMFold2 | `1ebf0e3481a5184eb6171d40615c79e384b48796` |
| biohub/ESMFold2-Experimental-Fast | `04dec820ede9283c9893e318e5ca5a9ac2ab93bc` |
| biohub/ESMC-6B | `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` |

With `--write-env`, the downloader saves `HF_HOME`, offline flags and
`ESMCFOLD_CCD_PATH` in the activation script; otherwise it prints their exports.
The runtime uses the model identifiers in the profile and the cache selected
by these variables. Preserve the exact cached revisions when comparing runs.

## Python package boundaries

`pip install -e .` installs the lightweight package dependencies and console
commands. It does **not** install Biohub ESM/Transformers, model weights, CUDA
kernels or the external optimization kit. Use the supplied environment helpers
for GPU execution. Do not install `fair-esm` into this environment: it shares the
`esm` namespace with Biohub ESM.

The Python wheel contains the core package, console commands and bundled
profiles. Benchmark inputs and environment scripts remain source-checkout tools.
