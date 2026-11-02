# Installation

## Accelerated mainline environment

Use the pinned Linux x86-64/Python 3.12 installer in the README. It uses Torch
2.11.0+cu128, Triton 3.6.0, Biohub ESM commit
`d0207ea3c8cbf072679ece3bb332787d25e06852`, and optimization-source commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
Set `PROTEOCRAFT_UPLIFT_ROOT` to that checkout in each new shell before running
the default search. Fast/big kernels require SM80/SM90; the adapter rejects
unsupported devices. Missing optimization sources fail explicitly.

The helper installs selected kernels used by the ProteoCraft adapter. It does
not install every optional Biohub SDK feature. In particular, external ESM
metadata declares DockQ and CUDA-13 cuEquivariance dependencies that this pinned
adapter environment does not install. `pip check` therefore reports these
missing SDK dependencies; this is a known limitation of the inherited stack.
The adapter does not use those SDK paths. GPU execution of this cleaned
distribution must still be validated on the intended machine.

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
python environment/esmfold2/check_environment.py --require-cuda --require-models
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

`check_environment.py` checks the stock package pins, not the newer accelerated
stack. The default acceleration settings are never silently changed to stock.

## Models

`download_models.py` fixes the following Hugging Face revisions:

| Model | Revision |
|---|---|
| biohub/ESMFold2 | `1ebf0e3481a5184eb6171d40615c79e384b48796` |
| biohub/ESMFold2-Experimental-Fast | `04dec820ede9283c9893e318e5ca5a9ac2ab93bc` |
| biohub/ESMC-6B | `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` |

The downloader prints `HF_HOME`, offline flags and `ESMCFOLD_CCD_PATH`.
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
