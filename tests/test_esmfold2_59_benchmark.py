from collections import Counter
from pathlib import Path

import pytest

from benchmark.script.run_esmfold2_59 import (
    DEFAULT_CONFIG,
    benchmark_summary_filename,
    build_hallucination_command,
    load_benchmark_config,
    load_manifest,
    load_prepared_cases,
    parse_overrides,
    run_cases,
    select_cases,
)
from proteocraft.cli import (
    parse_args_with_profile,
    parse_esmfold2_mutable_mask,
    parse_esmfold2_target_chain_roles,
    resolve_esmfold2_hotspot_defaults,
    resolve_search_defaults,
)


@pytest.fixture(scope="module")
def benchmark_config():
    return load_benchmark_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def manifest_rows(benchmark_config):
    return load_manifest(benchmark_config)


@pytest.fixture(scope="module")
def prepared_cases(benchmark_config):
    return load_prepared_cases(benchmark_config)


def _case(cases, case_id):
    return next(case for case in cases if case.case_id == case_id)


def test_skipping_completed_case_preserves_saved_search_profile(
    prepared_cases, benchmark_config, tmp_path, monkeypatch,
):
    import benchmark.script.run_esmfold2_59 as runner

    case = _case(prepared_cases, "denovo_BetV1")
    case_dir = tmp_path / case.case_id
    artifact = case_dir / "esmfold2_artifacts"
    artifact.mkdir(parents=True)
    (artifact / "run_summary.json").write_text("{}\n")
    profile = case_dir / "search_profile.yaml"
    saved = "schema_version: 1\nname: original\ndefaults:\n  n_steps: 110\n"
    profile.write_text(saved)

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("A skipped case must not launch a model process")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_launch)
    result = run_cases(
        [case], config=benchmark_config, profile="esmfold2_d", output_root=tmp_path,
        device="cuda:0", overrides=[("n_steps", "2")], prepare_only=False,
        skip_existing=True, continue_on_error=False, num_shards=1, shard_index=0,
    )
    assert result == 0
    assert profile.read_text() == saved


def test_manifest_is_portable_and_contains_exactly_59_cases(manifest_rows):
    assert len(manifest_rows) == 59
    assert Counter(row["task_type"] for row in manifest_rows) == {
        "denovo": 14,
        "tcr": 41,
        "cdr_redesign": 4,
    }
    assert len({row["case_id"] for row in manifest_rows}) == 59
    for row in manifest_rows:
        for field in ("sequence_file", "structure_file"):
            if row[field]:
                assert not Path(row[field]).is_absolute()
                assert ".." not in Path(row[field]).parts


def test_all_inputs_prepare_without_gpu(prepared_cases):
    assert len(prepared_cases) == 59
    assert all(case.target_sequence for case in prepared_cases)
    assert all(case.binder_length > 0 for case in prepared_cases)


def test_tcr_hotspots_use_concatenated_sequence_indices(prepared_cases):
    # 1QRN alpha chain has missing PDB residue numbers before residue 63.
    alpha_case = _case(prepared_cases, "tcr_1qrn_alpha_63_66_template")
    assert alpha_case.source_chain_hotspot == "alpha:63-66"
    assert alpha_case.target_hotspots_1based == (61, 62, 63, 64)

    # 9C3E beta starts after a 252-residue alpha chain and PDB numbering at 22.
    beta_case = _case(
        prepared_cases, "tcr_9c3e_beta_24_26_custom_loss_hotspot"
    )
    assert beta_case.source_chain_hotspot == "beta:24-26"
    assert beta_case.target_hotspots_1based == (255, 256, 257)


def test_no_hotspot_tcr_keeps_empty_hotspot_set(prepared_cases):
    case = _case(prepared_cases, "tcr_1qrn_no_hotspot")
    assert case.target_hotspots_1based == ()
    assert case.target_chain_roles == ("target", "target")


def test_cdr_case_preserves_chain_roles_and_exact_motif_mask(prepared_cases):
    case = _case(prepared_cases, "cdr_9NMW_alpha")
    assert case.target_chain_roles == ("target", "scaffold", "scaffold")
    assert case.initial_binder_sequence is not None
    assert case.mutable_mask is not None
    assert len(case.mutable_mask) == len(case.initial_binder_sequence)
    mutable_positions = [
        index for index, value in enumerate(case.mutable_mask) if value == "1"
    ]
    assert len(mutable_positions) == len("CAVNNARLMFGDGTQL")
    start = mutable_positions[0]
    end = mutable_positions[-1] + 1
    assert case.initial_binder_sequence[start:end] == "CAVNNARLMFGDGTQL"


@pytest.mark.parametrize(
    "case_id,task_type",
    (
        ("denovo_BetV1", "denovo"),
        ("tcr_1qrn_alpha_63_66_template", "tcr"),
        ("cdr_9NMW_alpha", "cdr_redesign"),
    ),
)
def test_commands_use_shared_profile_and_task_specific_inputs(
    prepared_cases,
    tmp_path,
    case_id,
    task_type,
):
    case = _case(prepared_cases, case_id)
    command = build_hallucination_command(
        case,
        profile="esmfold2_d",
        device="cuda:0",
        case_dir=tmp_path / case.case_id,
    )
    args, parser = parse_args_with_profile(command[2:])
    assert resolve_search_defaults(args, parser) == "esmfold2"
    assert args.launch_profile_name == "esmfold2_d"
    assert args.esmfold2_task_type == task_type
    assert args.mcts_strategy == "two_stage"
    assert args.seed == 43
    assert args.esmfold2_one_step_diffusion_mode == "truncated_trajectory"
    assert args.esmfold2_stagea_include_argmax_candidate is True
    assert args.esmfold2_stagea_mutation_fraction == pytest.approx(0.5)

    if task_type == "cdr_redesign":
        mutable_mask = parse_esmfold2_mutable_mask(
            args.esmfold2_mutable_mask,
            args.binder_length,
            parser,
        )
        roles = parse_esmfold2_target_chain_roles(
            args.esmfold2_target_chain_roles,
            n_target_chains=3,
            parser=parser,
        )
        assert sum(mutable_mask) == len("CAVNNARLMFGDGTQL")
        assert roles == ("target", "scaffold", "scaffold")


def test_hotspot_weight_resolves_for_hotspot_and_no_hotspot_cases(
    prepared_cases,
    tmp_path,
):
    hotspot_case = _case(prepared_cases, "denovo_BetV1")
    hotspot_command = build_hallucination_command(
        hotspot_case,
        profile="esmfold2_d",
        device="cuda:0",
        case_dir=tmp_path / hotspot_case.case_id,
    )
    hotspot_args, hotspot_parser = parse_args_with_profile(hotspot_command[2:])
    backend = resolve_search_defaults(hotspot_args, hotspot_parser)
    resolve_esmfold2_hotspot_defaults(
        hotspot_args,
        model_backend=backend,
        hotspot_params={"target_hotspot_concat_indices": [23]},
    )
    assert hotspot_args.esmfold2_hotspot_weight == pytest.approx(0.5)

    no_hotspot_case = _case(prepared_cases, "tcr_1qrn_no_hotspot")
    no_hotspot_command = build_hallucination_command(
        no_hotspot_case,
        profile="esmfold2_d",
        device="cuda:0",
        case_dir=tmp_path / no_hotspot_case.case_id,
    )
    no_hotspot_args, no_hotspot_parser = parse_args_with_profile(
        no_hotspot_command[2:]
    )
    backend = resolve_search_defaults(no_hotspot_args, no_hotspot_parser)
    resolve_esmfold2_hotspot_defaults(
        no_hotspot_args,
        model_backend=backend,
        hotspot_params=None,
    )
    assert no_hotspot_args.esmfold2_hotspot_weight == pytest.approx(0.0)
    assert no_hotspot_args.esmfold2_hotspot_verify_weight == pytest.approx(0.0)


def test_eight_shards_are_disjoint_and_cover_all_cases(prepared_cases):
    shards = [
        select_cases(prepared_cases, num_shards=8, shard_index=index)
        for index in range(8)
    ]
    flattened = [case.case_id for shard in shards for case in shard]
    assert len(flattened) == 59
    assert len(set(flattened)) == 59
    assert set(flattened) == {case.case_id for case in prepared_cases}


def test_each_multi_gpu_shard_uses_a_distinct_summary_file():
    assert benchmark_summary_filename(num_shards=1, shard_index=0) == (
        "benchmark_run_summary.json"
    )
    names = {
        benchmark_summary_filename(num_shards=8, shard_index=index)
        for index in range(8)
    }
    assert len(names) == 8
    assert "benchmark_run_summary.shard-00-of-08.json" in names
    assert "benchmark_run_summary.shard-07-of-08.json" in names


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
def test_boolean_batch_overrides_reach_the_real_parser(prepared_cases, tmp_path, value, expected):
    case = _case(prepared_cases, "denovo_BetV1")
    command = build_hallucination_command(
        case, profile="esmfold2_d", device="cuda:0", case_dir=tmp_path,
        overrides=parse_overrides([
            f"esmfold2_memory_saved_tensor_pin_memory={value}",
            f"esmfold2_optimization_timing={value}",
        ]),
    )
    args, parser = parse_args_with_profile(command[2:])
    resolve_search_defaults(args, parser)
    assert args.esmfold2_memory_saved_tensor_pin_memory is expected
    assert args.esmfold2_optimization_timing is expected


def test_overrides_cannot_change_benchmark_inputs():
    assert parse_overrides(["n_steps=3"]) == [("n_steps", "3")]
    with pytest.raises(ValueError, match="cannot override benchmark input"):
        parse_overrides(["target=AAAA"])
