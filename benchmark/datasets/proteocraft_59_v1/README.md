# ProteoCraft 59-case ESMFold2 benchmark

This directory contains the portable inputs and run registry for the 59-case
benchmark:

| Task | Cases | Unique source complexes/targets |
|---|---:|---:|
| De novo binder design | 14 | 8 |
| TCR binder design | 41 | 5 |
| TCR CDR redesign | 4 | 2 |
| **Total** | **59** | **15** |

`manifest.tsv` is the source of truth for the 59 run definitions. Repeated
hotspot variants share the same sequence or structure input instead of storing
duplicate files. `SHA256SUMS.txt` records checksums for all copied FASTA,
CIF, and PDB inputs.

## Configuration

`benchmark.yaml` points the benchmark launcher to `manifest.tsv` and to the
current recommended `esmfold2_d` launch profile. The profile itself lives at:

```text
proteocraft/configs/proteocraft_profiles/esmfold2_d.yaml
```

The manifest preserves the case and hotspot registry used by the historical
59-run benchmark, while the actual search algorithm comes from the current
ESMFold2 profile. Historical Protenix step counts, loss weights, and absolute
server paths are intentionally not copied into this portable configuration.

## Input semantics

- De novo cases read one target sequence from FASTA. `target_hotspots` already
  uses that sequence's 1-based indices.
- TCR cases extract the specified alpha and beta chains from the source PDB.
  Chain-local PDB residue ranges are mapped to 1-based indices in the
  concatenated `alpha|beta` ESMFold2 input. This is important when a PDB chain
  has missing residue numbers.
- CDR cases read the full multichain FASTA. The antigen is assigned the
  `target` role, the fixed partner chains are assigned `scaffold`, and only the
  listed CDR motif is marked mutable.
- ESMFold2 is sequence-only in this workflow. CIF/PDB files are retained for
  sequence extraction, provenance, visualization, and downstream evaluation;
  they are not passed to ESMFold2 as structural templates.

## Validate without using a GPU

From the repository root:

```bash
python benchmark/script/run_esmfold2_59.py --prepare-only
```

This validates all 59 rows, input files, amino-acid alphabets, hotspot bounds,
TCR residue-number mappings, and CDR mutable masks, then prints the exact
commands without loading a model.

Run one case or one task family:

```bash
python benchmark/script/run_esmfold2_59.py \
  --case denovo_BetV1 \
  --device cuda:0

python benchmark/script/run_esmfold2_59.py \
  --task-type cdr_redesign \
  --device cuda:0
```

## Multi-GPU sharding

Each worker receives a deterministic, non-overlapping shard. For example,
launching eight workers with directly visible GPU indices:

```bash
for i in 0 1 2 3 4 5 6 7; do
  python benchmark/script/run_esmfold2_59.py \
    --num-shards 8 \
    --shard-index "$i" \
    --device "cuda:$i" \
    --skip-existing &
done
wait
```

If each process receives one GPU through `CUDA_VISIBLE_DEVICES`, pass
`--device cuda:0` inside every worker instead.

Algorithm options can be overridden without editing the manifest. For a smoke
test, for example:

```bash
python benchmark/script/run_esmfold2_59.py \
  --case denovo_BetV1 \
  --set n_steps=3 \
  --device cuda:0
```

Benchmark input fields such as target sequence, hotspot, mutable mask, task
type, and output directory cannot be changed through `--set`.

## Output

Each case writes to `<output_dir>/<case_id>/` and records:

- `case_spec.json`: resolved sequence, hotspot, chain-role, and provenance data;
- `command.txt`: exact invocation;
- `esmfold2_artifacts/`: normal ProteoCraft search artifacts;
- `checkpoints/`: search checkpoints;
- `run.log`: case log.

For one worker, the output root contains `benchmark_run_summary.json`. A
multi-GPU launch writes one non-conflicting
`benchmark_run_summary.shard-XX-of-YY.json` per worker. Search-time verification
follows the profile default (`4 x 50`); final `20 x 200` rescoring is a separate
evaluation step.
