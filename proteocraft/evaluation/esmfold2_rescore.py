"""Frozen, sequence-only ESMFold2 rescoring, independent of search campaigns.

Preparation and reporting need no GPU. Inference is dispatched by the CLI into
a fresh process; no search backend or patched model is accepted as input.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "proteocraft/configs/evaluation/esmfold2_official_rescore.yaml"
SWITCHES = ("proposal_optimization", "verifier_optimization", "esmc_optimization")
# Include the actual scorer and its local dependencies, not only the adapters.
LOCAL_EVALUATOR_SOURCES = (
    "proteocraft/evaluation/esmfold2_rescore.py",
    "proteocraft/evaluation/esmfold2_runner.py",
    "proteocraft/hallucination/esmfold2_runtime.py",
    "proteocraft/hallucination/esmfold2_backend.py",
    "proteocraft/hallucination/esmfold2_optimizations.py",
    "proteocraft/hallucination/esmfold2_one_step.py",
    "proteocraft/hallucination/esmfold2_template.py",
    "proteocraft/model_sources.py",
    "proteocraft/losses/geometry.py",
)
METRICS = ("total_loss", "pair_chain_iptm", "binder_plddt", "iptm", "ptm",
           "hotspot_heavy_atom_coverage", "hotspot_heavy_atom_coverage_5a",
           "hotspot_mean_min_heavy_atom_distance", "binder_radius_gyration",
           "binder_rg_excess", "raw_common_clash", "raw_common_i_clash")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(encoded(value) + b"\n")
    temp.replace(path)


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".rescore.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another rescore process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    expected = {"schema_version", "evaluator", "structure_mode", "model", "model_revision",
                "esmc_model", "esmc_revision", *SWITCHES, "num_loops", "num_sampling_steps",
                "loss_mode", "seeds", "selection", "ranking"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Evaluation config has missing or unsupported fields")
    for key, value in {"schema_version": 1, "evaluator": "unoptimized_esmfold2",
                       "structure_mode": "sequence_only", "loss_mode": "aligned_common",
                       "model": "biohub/ESMFold2", "esmc_model": "biohub/ESMC-6B",
                       **dict.fromkeys(SWITCHES, "off")}.items():
        if config[key] != value:
            raise ValueError(f"Independent evaluator requires {key}={value!r}")
    for key in ("model_revision", "esmc_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(config[key])):
            raise ValueError(f"{key} must be an immutable 40-character revision")
    for key in ("num_loops", "num_sampling_steps"):
        positive_int(config[key], key)
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    for seed in seeds:
        positive_int(seed, "seed", 0)
        if seed >= 2**32:
            raise ValueError("seed must fit the NumPy seed range")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")
    selection = config["selection"]
    if set(selection) != {"top_k", "metric", "direction"} or selection["metric"] != "total_loss" or selection["direction"] != "ascending":
        raise ValueError("Selection must use ascending search total_loss")
    positive_int(selection["top_k"], "top_k")
    if config["ranking"] != {"metric": "pair_chain_iptm", "aggregation": "mean",
                             "direction": "descending", "require_all_seeds": True}:
        raise ValueError("Ranking requires all seeds and descending mean pair_chain_iptm")
    return config


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected non-empty JSONL objects: {path}")
    return rows


def normalize_candidate(row):
    row = dict(row)
    for field in ("case_id", "candidate_id"):
        if not isinstance(row.get(field), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row[field]):
            raise ValueError(f"Invalid {field}; use a filesystem-safe identifier")
    def sequence(value):
        if not isinstance(value, str) or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", value):
            raise ValueError("Sequences must use uppercase canonical amino acids")
        return value
    row["binder_seq"] = sequence(row.get("binder_seq"))
    targets = row.get("target_sequences")
    if not isinstance(targets, list) or not targets:
        raise ValueError("target_sequences must be an ordered non-empty list of chains")
    row["target_sequences"] = [sequence(value) for value in targets]
    roles = row.get("target_chain_roles")
    if not isinstance(roles, list) or len(roles) != len(targets) or "target" not in roles or set(roles) - {"target", "scaffold"}:
        raise ValueError("Explicit target_chain_roles must match the ordered chains")
    hotspots = row.get("target_hotspot_concat_indices")
    if not isinstance(hotspots, list):
        raise ValueError("Explicit zero-based target_hotspot_concat_indices required (use [] if absent)")
    for index in hotspots:
        positive_int(index, "hotspot index", 0)
        if index >= sum(map(len, targets)):
            raise ValueError("Hotspot index outside concatenated target chains")
    row["target_hotspot_concat_indices"] = sorted(set(hotspots))
    if "search_seed" in row:
        positive_int(row["search_seed"], "search_seed", 0)
    if "search_loss" in row and (type(row["search_loss"]) not in (int, float) or not math.isfinite(row["search_loss"])):
        raise ValueError("search_loss must be finite")
    return row


def candidates_from_runs(rows, manifest, top_k):
    """Select distinct sequences within each completed run, not across contexts."""
    candidates, sources = [], []
    for item in rows:
        run = Path(item["run_dir"]).expanduser()
        run = (manifest.parent / run).resolve() if not run.is_absolute() else run.resolve()
        summary = run / "esmfold2_artifacts/run_summary.json"
        receipt = run / "experiment_run.json"
        if not summary.is_file() or (receipt.exists() and json.loads(receipt.read_text()).get("returncode") != 0):
            raise ValueError(f"Search not successfully complete: {run}")
        source = run / "esmfold2_artifacts/candidates.jsonl"
        unique = {}
        for row in read_jsonl(source):
            loss = row.get("total_loss")
            if type(loss) not in (int, float) or not math.isfinite(loss):
                raise ValueError(f"Nonfinite or missing search loss: {source}")
            seq = row["sequence"]
            if seq not in unique or loss < unique[seq]["total_loss"]:
                unique[seq] = row
        selected = sorted(unique.values(), key=lambda r: (r["total_loss"], r["sequence"]))[:top_k]
        for rank, row in enumerate(selected, 1):
            candidate = {k: v for k, v in item.items() if k not in {"run_dir", "candidate_prefix"}}
            candidate.update(candidate_id=f"{item['candidate_prefix']}-rank{rank:03d}",
                             binder_seq=row["sequence"], search_loss=row["total_loss"],
                             search_rank=rank, source_run=str(run))
            candidates.append(normalize_candidate(candidate))
        sources.append({"run_dir": str(run), "candidates_sha256": file_hash(source),
                        "summary_sha256": file_hash(summary),
                        "receipt_sha256": file_hash(receipt) if receipt.exists() else None,
                        "unique_sequences": len(unique), "selected": len(selected)})
    return candidates, sources


def freeze(manifest, config_path, output, *, from_search=False):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    config = load_config(config_path)
    rows = read_jsonl(manifest)
    candidates, sources = (candidates_from_runs(rows, manifest, config["selection"]["top_k"])
                           if from_search else ([normalize_candidate(row) for row in rows], []))
    keys = [(row["case_id"], row["candidate_id"]) for row in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case/candidate identity")
    contexts = {}
    for row in candidates:
        if row.get("search_seed") in config["seeds"]:
            raise ValueError("Evaluation seeds must differ from search seeds")
        context = [row[k] for k in ("target_sequences", "target_chain_roles", "target_hotspot_concat_indices")]
        if contexts.setdefault(row["case_id"], context) != context:
            raise ValueError("One case_id cannot mix different target contexts")
    contract = {"schema_version": 1, "config": config, "candidates": candidates,
                "selection": "top_k_per_completed_run" if from_search else "explicit_manifest",
                "input_sha256": file_hash(manifest), "config_sha256": file_hash(config_path),
                "sources": sources}
    contract["contract_hash"] = digest(contract)
    with locked(output):
        path = output / "evaluation_contract.json"
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Frozen evaluation contract changed; use a new output directory")
            load_contract(output)
        else:
            if any(p.name != ".rescore.lock" for p in output.iterdir()):
                raise ValueError("New evaluation requires an empty output directory")
            # Contract is the commit marker, written after the candidate manifest.
            (output / "candidates.jsonl").write_bytes(b"".join(encoded(row) + b"\n" for row in candidates))
            write_json(path, contract)
    return contract


def load_contract(output):
    output = Path(output)
    contract = json.loads((output / "evaluation_contract.json").read_text())
    if digest({k: v for k, v in contract.items() if k != "contract_hash"}) != contract["contract_hash"]:
        raise ValueError("Evaluation contract hash mismatch")
    if read_jsonl(output / "candidates.jsonl") != contract["candidates"]:
        raise ValueError("Frozen candidate manifest changed")
    return contract


def score_path(root, candidate, seed):
    return Path(root) / "structures" / candidate["case_id"] / candidate["candidate_id"] / f"seed_{seed}" / "score.json"


def validate_score(score, candidate, seed, contract, path):
    for key, value in {"contract_hash": contract["contract_hash"], "candidate_hash": digest(candidate),
                       "case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"], "seed": seed}.items():
        if score.get(key) != value:
            raise ValueError(f"Saved score identity mismatch: {key}")
    if score.get("status") not in {"ok", "failed"}:
        raise ValueError("Invalid saved score status")
    if score["status"] == "ok":
        for metric in ("total_loss", "pair_chain_iptm", "binder_plddt"):
            if not isinstance(score.get("metrics", {}).get(metric), (int, float)) or not math.isfinite(score["metrics"][metric]):
                raise ValueError(f"Missing/nonfinite required metric: {metric}")
        cif = Path(path).parent / "esmfold2.cif"
        if not cif.is_file() or file_hash(cif) != score.get("structure_sha256"):
            raise ValueError("Missing or changed structure for saved score")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def report(output):
    output = Path(output)
    contract = load_contract(output)
    environment = output / "runtime_environment.json"
    runtime_hash = digest(json.loads(environment.read_text())) if environment.exists() else None
    expected_paths, scores, summaries = set(), [], []
    for candidate in contract["candidates"]:
        good, failed, missing = [], 0, 0
        for seed in contract["config"]["seeds"]:
            path = score_path(output, candidate, seed)
            expected_paths.add(path)
            row = {"case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"], "seed": seed}
            if not path.exists():
                row["status"] = "missing"
                missing += 1
            else:
                score = json.loads(path.read_text())
                validate_score(score, candidate, seed, contract, path)
                if runtime_hash is None or score.get("runtime_hash") != runtime_hash:
                    raise ValueError("Missing or changed runtime environment receipt")
                row.update(status=score["status"], runtime_s=score.get("runtime_s"),
                           error=score.get("error", ""), **score.get("metrics", {}))
                if score["status"] == "ok":
                    good.append(score["metrics"])
                else:
                    failed += 1
            scores.append(row)
        complete = not failed and not missing
        summary = {"case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"],
                   "binder_seq": candidate["binder_seq"], "search_loss": candidate.get("search_loss"),
                   "expected_seeds": len(contract["config"]["seeds"]), "successful_seeds": len(good),
                   "failed_seeds": failed, "missing_seeds": missing, "complete": complete, "rank": ""}
        # No survivor-only aggregate: partial candidates are visible but unranked.
        for metric in METRICS:
            values = [m[metric] for m in good if m.get(metric) is not None]
            for statistic in ("mean", "min", "max", "std"):
                summary[f"{metric}_{statistic}"] = ""
            if complete and len(values) == len(good):
                for statistic, fn in (("mean", statistics.mean), ("min", min), ("max", max), ("std", statistics.pstdev)):
                    summary[f"{metric}_{statistic}"] = fn(values)
        summaries.append(summary)
    if set((output / "structures").rglob("score.json")) - expected_paths:
        raise ValueError("Unexpected score records in evaluation directory")
    for case in {r["case_id"] for r in summaries}:
        eligible = sorted((r for r in summaries if r["case_id"] == case and r["complete"]),
                          key=lambda r: (-r["pair_chain_iptm_mean"], r["candidate_id"]))
        for rank, row in enumerate(eligible, 1):
            row["rank"] = rank
    write_csv(output / "scores.csv", scores)
    write_csv(output / "summary.csv", summaries)
    status = {"contract_hash": contract["contract_hash"], "expected_scores": len(scores),
              "successful_scores": sum(r["status"] == "ok" for r in scores),
              "failed_scores": sum(r["status"] == "failed" for r in scores),
              "missing_scores": sum(r["status"] == "missing" for r in scores),
              "complete": all(r["complete"] for r in summaries)}
    write_json(output / "evaluation_status.json", status)
    return status


def assert_clean_process():
    for name in ("ESMFOLD2_OPT", "ESMC_OPT", "EF2_FAST_KIT", "ESMC_KIT"):
        if os.environ.get(name, "").lower() not in ("", "off", "0"):
            raise RuntimeError(f"Unset {name} before stock evaluation")
    for name in sys.modules:
        if name in {"ef2_opt", "ef2_autograd_kernels", "ef2_trimul", "ef2_esmc_rope"} or name.startswith("esmc_opt."):
            raise RuntimeError("Optimization kit already imported; use a fresh evaluation process")


def stock_backend(config, device):
    """Resolve immutable cached snapshots; never download models implicitly."""
    assert_clean_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from .esmfold2_runner import build_backend
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    model_path = snapshot_download(config["model"], revision=config["model_revision"], local_files_only=True)
    esmc_path = snapshot_download(config["esmc_model"], revision=config["esmc_revision"], local_files_only=True)
    ccd = Path(model_path) / "ccd.pkl"
    if not ccd.is_file():
        raise FileNotFoundError(f"Missing pinned model CCD cache: {ccd}")
    os.environ["ESMCFOLD_CCD_PATH"] = str(ccd)
    # Pin the verifier's internal LM as well as its folding weights.
    model_config = ESMFold2Config.from_pretrained(model_path, local_files_only=True)
    model_config.esmc_id = esmc_path
    backend = build_backend(device=device, model_name=model_path, num_loops=config["num_loops"],
                            num_sampling_steps=config["num_sampling_steps"], loss_mode=config["loss_mode"])
    backend.verifier_model = ESMFold2Model.from_pretrained(
        model_path, config=model_config, esmc_precision="bf16", local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    return backend


def runtime_identity(backend):
    import torch
    from proteocraft.hallucination.esmfold2_one_step import (
        ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS, ESMFOLD2_VERIFIER_LOSS_WEIGHTS,
    )
    sources = {name: file_hash(ROOT / name) for name in LOCAL_EVALUATOR_SOURCES}
    sources["model_ccd"] = file_hash(os.environ["ESMCFOLD_CCD_PATH"])
    sources["model_config"] = file_hash(Path(backend.config.verifier_model_name) / "config.json")
    sources["esmc_config"] = file_hash(Path(backend.verifier_model.config.esmc_id) / "config.json")
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("transformers.models.esmfold2.", "transformers.models.esmc.", "esm.models.esmfold2")):
            path = getattr(module, "__file__", None)
            if path and Path(path).suffix == ".py":
                sources[name] = file_hash(path)
    versions = {}
    for name in ("torch", "triton", "transformers", "esm", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "versions": versions, "source_sha256": sources,
            "scoring_weights": {"bounded": dict(ESMFOLD2_VERIFIER_LOSS_WEIGHTS),
                                "raw_common": dict(ESMFOLD2_LEGACY_VERIFIER_LOSS_WEIGHTS)},
            "device": str(backend.device), "gpu": torch.cuda.get_device_name(backend.device),
            "cuda": torch.version.cuda,
            "optimization": {key: getattr(backend.config, key) for key in SWITCHES}}


def run_worker(output, device, *, retry_failed=False):
    """Only the CLI worker calls this; it constructs its own unpatched model."""
    from .esmfold2_runner import evaluate
    output = Path(output)
    assert_clean_process()
    with locked(output):
        contract = load_contract(output)
        status = report(output)  # Verify saved identities before loading weights.
        if status["complete"]:
            return 0
        backend = stock_backend(contract["config"], device)
        identity = runtime_identity(backend)
        receipt = output / "runtime_environment.json"
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise ValueError("Runtime environment changed; use a new output directory")
        write_json(receipt, identity)
        for candidate in contract["candidates"]:
            for seed in contract["config"]["seeds"]:
                path = score_path(output, candidate, seed)
                if path.exists():
                    old = json.loads(path.read_text())
                    if old["status"] == "ok" or not retry_failed:
                        continue
                    history = path.parent / "failed_attempts"
                    history.mkdir(exist_ok=True)
                    write_json(history / f"{time.time_ns()}.json", old)
                score = {"case_id": candidate["case_id"], "candidate_id": candidate["candidate_id"],
                         "seed": seed, "contract_hash": contract["contract_hash"],
                         "candidate_hash": digest(candidate), "runtime_hash": digest(identity)}
                start = time.perf_counter()
                try:
                    result = evaluate(backend, binder_seq=candidate["binder_seq"],
                                      target_seq=candidate["target_sequences"], design_dir=path.parent,
                                      target_chain_roles=candidate["target_chain_roles"],
                                      hotspot_indices=candidate["target_hotspot_concat_indices"], seed=seed)
                    raw = {**getattr(result.verifier_result, "loss_dict", {}),
                           **result.verifier_result.summary_confidence,
                           "total_loss": result.verifier_result.total_loss}
                    metrics = {k: float(raw[k]) if raw.get(k) is not None else None for k in METRICS}
                    if any(v is not None and not math.isfinite(v) for v in metrics.values()):
                        raise ValueError("Nonfinite evaluation metric")
                    score.update(status="ok", metrics=metrics, runtime_s=result.runtime_s,
                                 structure_sha256=file_hash(result.mmcif_path))
                    validate_score(score, candidate, seed, contract, path)
                except Exception as exc:
                    score.update(status="failed", metrics={}, runtime_s=time.perf_counter() - start,
                                 error=f"{type(exc).__name__}: {exc}")
                backend.optimizations.finalize()
                optimization = backend.optimizations.report()
                if optimization["applied"] or any(getattr(backend.config, key) != "off" for key in SWITCHES):
                    raise RuntimeError("Stock evaluator unexpectedly enabled optimizations")
                write_json(path, score)
                write_json(output / "optimization_report.json", optimization)
                report(output)
        return 0 if report(output)["complete"] else 1
