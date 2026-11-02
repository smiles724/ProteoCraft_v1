# Task examples

Install the checkout and configure models as described in the README first.
The placeholders below denote user-provided sequences; they are not runnable
experimental examples. Chain order determines concatenated hotspot numbering.

## De novo and TCR binder design

```bash
proteocraft --target "$TARGET_SEQ" --binder_length 120 \
  --output_dir outputs/denovo --device cuda:0

proteocraft --esmfold2_task_type tcr \
  --target "${TCR_ALPHA_SEQ}|${TCR_BETA_SEQ}" \
  --esmfold2_target_chain_roles target,target \
  --binder_length 120 --output_dir outputs/tcr --device cuda:0
```

Target chains are fixed in sequence. Hotspots use 1-based positions over the
target chains in input order. `--target_hotspots "24,35,48-50"` is valid only
if all positions exist in that concatenation.

## Local/CDR redesign

Supply the complete starting binder and an equal-length binary mutable mask:
`1` allows substitution, `0` preserves the amino acid. At least one position must
be mutable. A target/scaffold role labels each input target/context chain.

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
  --case 9NMW_alpha --device cuda:0
```

## VHH all-CDR redesign

Copy and edit the provided YAML with your target, VHH sequence and IMGT CDR
mapping. The adapter requires all three CDRs, disjoint valid positions and an
unchanged framework; it does not install or automatically run a numbering tool.

```bash
python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml \
  --prepare-only
```

After editing and validating the configuration, omit `--prepare-only` and supply
`--device cuda:0` for a real run. The default redesign path is sequence-only.
Reference structures record provenance and support evaluation; they are not
automatically model templates.

## Outputs, checkpoints and memory

- `hallucination_result.txt`: best/final binder and resolved run settings.
- `esmfold2_artifacts/run_summary.json`: machine-readable summary.
- `esmfold2_artifacts/candidates.jsonl`: verified discrete candidates and metrics.
- `esmfold2_artifacts/trajectory.jsonl`: search trajectory.
- `--save_checkpoint_dir /path/to/checkpoints`: save search checkpoints.
- `--esmfold2_resume_handoff_checkpoint /path/to/checkpoint`: explicit Stage A/B handoff restart.
- `--esmfold2_save_verified_candidate_cif true`: save candidate structures.

Resident model memory management remains the default. For long complexes,
`--esmfold2_memory_mode phase_offload` offloads inactive components between
phases; `auto` uses the explicit token/free-memory settings. This changes model
placement and runtime costs. It is not a guarantee that a given input fits.

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
