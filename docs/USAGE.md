# Task examples

[English](USAGE.md) | [简体中文](USAGE.zh-CN.md)

Install and configure models using the [README](../README.md#install). Run these
commands from the repository root. Custom-input commands use shell variables
such as `TARGET_SEQ`; set them to your amino-acid sequences first.

`--target` takes sequences directly, separated by `|` for multiple chains.
`target` and `scaffold` roles identify binding targets and fixed context chains;
both preserve the supplied sequence. The binder is supplied separately.
Hotspots are optional, 1-based positions over all chains in `--target`
concatenated in input order. If chain 1 has 100 residues, position 101 starts chain 2.
Batch scripts read sequences from FASTA/PDB inputs; default structure prediction
uses sequences, with PDB/CIF files serving as references and provenance.

## De novo binder design

Provide target sequence(s) and the desired binder length. The search designs
the entire binder.

```bash
proteocraft --target "$TARGET_SEQ" --binder_length 120 \
  --output_dir outputs/denovo --device cuda:0
```

Try the bundled Bet v 1 input, which supplies the target, hotspot and binder length:

```bash
python benchmark/script/run_esmfold2_59.py --case denovo_BetV1 --prepare-only
python benchmark/script/run_esmfold2_59.py --case denovo_BetV1 --device cuda:0
```

`--prepare-only` checks inputs and prints the command. The second command starts GPU design.

## TCR binder design

Provide TCR α/β sequences and the new binder's length. The TCR stays fixed in
sequence; the search designs a separate binder that targets it.

```bash
proteocraft --esmfold2_task_type tcr \
  --target "${TCR_ALPHA_SEQ}|${TCR_BETA_SEQ}" \
  --esmfold2_target_chain_roles target,target \
  --binder_length 120 --output_dir outputs/tcr --device cuda:0
```

The bundled 1QRN case extracts α/β sequences from its PDB:

```bash
python benchmark/script/run_esmfold2_59.py --case tcr_1qrn_no_hotspot --prepare-only
python benchmark/script/run_esmfold2_59.py --case tcr_1qrn_no_hotspot --device cuda:0
```

To modify a TCR's own CDRs against its antigen, use local/CDR redesign.

## Local/CDR redesign

Provide the target, complete starting binder and mutable positions. Add fixed
partner chains when needed. Supply an equal-length binary mutable mask:
`1` allows substitution, `0` preserves the amino acid. At least one position must
be mutable. For a 12-residue sequence, `000111000000` selects positions 4–6.
The following command includes one antigen and one fixed partner chain:

```bash
proteocraft --esmfold2_task_type cdr_redesign \
  --target "${ANTIGEN_SEQ}|${PARTNER_TCR_SEQ}" \
  --esmfold2_target_chain_roles target,scaffold \
  --binder_length "${#REDESIGN_CHAIN_SEQ}" \
  --init_binder_seq "$REDESIGN_CHAIN_SEQ" \
  --esmfold2_mutable_mask "$CDR_MASK" \
  --output_dir outputs/cdr --device cuda:0
```

The batch launcher also accepts multiple `redesign_regions` as 1-based inclusive
positions in the complete sequence. They are sequence indices, not PDB author
or IMGT numbers. The framework must remain unchanged.

```bash
python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml \
  --case 9NMW_alpha --prepare-only

python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml \
  --case 9NMW_alpha --device cuda:0
```

The 9NMW example supplies the antigen, fixed context chains, starting TCR α
sequence and CDR motif. For your own batch input, copy the
[CDR YAML](../benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml) and set:

| Field | Value |
|---|---|
| `input_dir`, `fasta_path` | Input directory and FASTA path relative to it |
| `binder_chain` | Chain to redesign |
| `target_chains`, `scaffold_chains` | Binding target and fixed context chain IDs |
| `cdr_motif` **or** `redesign_regions` | Unique sequence motif, or 1-based inclusive intervals such as `[[26, 33], [51, 58]]` |

Choose exactly one region format and set positions for your sequence. Multichain
FASTA headers identify chains, for example `>my_complex|Chain A|target`.
Use `scaffold_chains: []` when there are no fixed context chains.
Replace or omit the optional `template_path` reference for the default sequence workflow.

## VHH all-CDR redesign

Start with the fully specified [IFNAR2 example](../benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml):

```bash
python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml --prepare-only

python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml --device cuda:0
```

It uses the 217-residue IFNAR2 input from the cleanup regression and a
128-residue starting VHH from 7EOW chain B. All 37 CDR positions are mutable;
the remaining 91 positions stay fixed. The current `esmfold2_d` profile supplies
the search settings. Hotspots refer to positions in this case's IFNAR2 sequence.
Results are under `outputs/vhh_ifnar2/IFNAR2/`.

| CDR | 1-based positions in the starting VHH |
|---|---|
| CDR1 | 26–33 |
| CDR2 | 51–58 |
| CDR3 | 97–117 |

For your own input, edit the template as described below.

Provide a target, complete starting VHH and IMGT-derived CDR1/2/3 positions.
The adapter generates their combined mutable mask and preserves the framework.
Copy the [VHH template](../benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml):

```bash
cp benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml my_vhh.yaml
```

Edit the copied file before preparing the run:

| Field | What to enter |
|---|---|
| `cases[].target_sequences` | Replace `REPLACE_WITH_TARGET_SEQUENCE` with target sequence(s) |
| `initial_vhh_sequence` | Full VHH sequence; the template supplies the 7EOW starting VHH |
| `numbering` | `scheme: imgt` and the numbering tool/version used for your sequence |
| `cdr_regions` | CDR1/2/3 as `sequence_indices_1based` lists |
| `target_hotspots_1based` | Optional hotspot positions, or `[]` |
| `output_dir` | Run output directory |

First number the VHH using IMGT, then map its CDR labels back to 1-based
positions in the supplied sequence. For example, `[26, 27, 28]` selects its
26th–28th amino acids. The launcher reads this mapping; `numbering` records
its source. When replacing the VHH, update all three CDR lists and source
metadata. Remove optional `mutable_mask` to derive it from the CDR lists, or
provide a matching mask. Update or omit the template's optional `scaffold`
metadata for your starting VHH.

```bash
python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config my_vhh.yaml \
  --prepare-only

python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config my_vhh.yaml --device cuda:0
```

Preparation checks CDR bounds, overlaps and mask consistency. CDR and VHH
launchers write `case_spec.json`, `command.txt` and `provenance.json` under
`<output_dir>/<case_id>/` for inspection before GPU design.

## Outputs, checkpoints and memory

- `hallucination_result.txt`: best/final binder and resolved run settings.
- `esmfold2_artifacts/run_summary.json`: machine-readable summary.
- `esmfold2_artifacts/candidates.jsonl`: verified discrete candidates and metrics.
- `esmfold2_artifacts/trajectory.jsonl`: search trajectory.
- `esmfold2_artifacts/checkpoints/`: structures and metrics saved during search.
- `esmfold2_artifacts/checkpoints/final/step_NNNN/step_NNNN_best.cif`: the selected final verifier structure, with matching metrics in `step_NNNN_best_meta.json`; `NNNN` is the completed search budget, padded to four digits.
- `esmfold2_artifacts/handoff_checkpoints/stagea_handoff_final.pt`: final Stage A state for an explicit restart into Stage B; intermediate handoffs are saved every 10 Stage A rounds.
- `--esmfold2_resume_handoff_checkpoint /path/to/checkpoint`: explicit Stage A/B handoff restart.
- `--esmfold2_save_verified_candidate_cif true`: save candidate structures.

When a starting binder is supplied during handoff recovery, its fixed positions
must match the checkpoint. Changes at mutable positions do not replace the
checkpoint's saved search state.

`run_summary.json` reports consumed budget through `n_steps`. An empty Stage B
iteration still consumes budget and appends loss history, but has no action
trajectory row. Use the full loss history and the handoff checkpoint's
`completed_stagea_steps` when auditing the Stage A/B budget; trajectory row
count alone is insufficient.

Resident model memory management remains the default. For long complexes,
`--esmfold2_memory_mode phase_offload` offloads inactive components between
phases; `auto` uses the explicit token/free-memory settings. This changes model
placement and runtime costs. It is not a guarantee that a given input fits.

## Algorithm configuration for maintainers

The public CLI accepts task inputs and deployment controls. The algorithm is
the two-stage ESMFold2 search in the bundled `esmfold2_d` profile: eight initial
roots, 70 Stage A rounds with 30 soft updates per proposal, then 40 Stage B
rounds with 20 soft updates per proposal. Full verification uses 4 loops and
50 diffusion steps. Ordinary runs do not need to choose these settings.

For controlled development experiments, copy
`proteocraft/configs/proteocraft_profiles/esmfold2_d.yaml` and pass
`--profile /absolute/path/to/profile.yaml`. Algorithm settings such as
`n_steps` and `esmfold2_stagea_soft_steps` belong in its `defaults` mapping;
they are no longer CLI flags. Unknown profile keys and removed search
strategies fail explicitly. Batch launchers freeze explicit algorithm overrides
in a per-case `search_profile.yaml` for reproducibility.

Best-of-N, chunked Best-of-N, continuous/raw-lineage search drivers and final
polish have been removed. Stage A and Stage B remain two parts of one algorithm;
their separate tree implementations serve different search states and rewards.

## Independent rescoring

The input is JSONL, one candidate per row, for example:

```json
{"case_id":"demo","candidate_id":"demo_1","target_sequences":["GIVEQCCTSICSLYQLENYCN"],"binder_seq":"AAAAAAAAAA","target_chain_roles":["target"],"target_hotspot_concat_indices":[]}
```

This is a format example, not a proposed binder. Candidate IDs must be unique.
`target_hotspot_concat_indices` in this evaluator manifest are **zero-based** over concatenated
target chains. Do not copy 1-based CLI hotspots without converting them.
For completed search runs use `--from-search` and the completed-run manifest
schema checked by the evaluator.

```bash
proteocraft-rescore prepare --manifest candidates.jsonl --out-dir outputs/rescore
proteocraft-rescore run --out-dir outputs/rescore --device cuda:0
proteocraft-rescore report --out-dir outputs/rescore
```

Preparation freezes candidates, model revisions, source hashes, 20×200 stock
inference and evaluation seeds. Inference runs in a fresh process. The evaluator
checks structures and score records, and refuses incompatible resumed outputs.
The default evaluation configuration is bundled in
`proteocraft/configs/evaluation/esmfold2_official_rescore.yaml`.
