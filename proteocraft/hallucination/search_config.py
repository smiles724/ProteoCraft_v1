"""Validated algorithm settings for maintainer YAML profiles.

Ordinary task inputs and deployment controls live in the public CLI. This
schema reuses argparse types and choices solely for profile validation.
"""
import argparse

from proteocraft.hallucination.profile_loader import parse_boolean


def build_search_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="优化步数；默认按 backend 解析：ESMFold2=110",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="Adam 学习率（默认：0.05）"
    )
    parser.add_argument(
        "--esmfold2_verifier_model",
        type=str,
        default="biohub/ESMFold2",
        help="ESMFold2 verifier 模型名（仅 --model_backend esmfold2 使用）",
    )
    parser.add_argument(
        "--esmfold2_experimental_model",
        type=str,
        default="biohub/ESMFold2-Experimental-Fast",
        help="ESMFold2Experimental proposal 模型名（默认：biohub/ESMFold2-Experimental-Fast；仅 --model_backend esmfold2 使用）",
    )
    parser.add_argument(
        "--esmfold2_stagea_soft_steps",
        type=int,
        default=None,
        help="ESMFold2 Stage A 每个 MCTS 节点的 soft gradient update 步数（默认：30）",
    )
    parser.add_argument(
        "--esmfold2_temperature_min",
        type=float,
        default=0.01,
        help="ESMFold2 soft optimization 最低温度（默认：0.01）",
    )
    parser.add_argument(
        "--esmfold2_temperature_max",
        type=float,
        default=1.0,
        help="ESMFold2 soft optimization 最高温度（默认：1.0；与最低温度相等时使用固定温度）",
    )
    parser.add_argument(
        "--esmfold2_stagea_tree_children",
        type=int,
        default=2,
        help="ESMFold2 Stage A 每轮 full verify 后进入 MCTS tree 的 child 数；默认 2（argmax soft trajectory + 其余候选中 verifier 最优者）",
    )
    parser.add_argument(
        "--esmfold2_stagea_root_selection_mode",
        type=str,
        choices=["global_ucb", "hierarchical_ucb"],
        default="hierarchical_ucb",
        help="ESMFold2 Stage A root/node 选择方式；global_ucb 保留原全树 UCB，hierarchical_ucb 先选 root 再在其子树内选 node（默认）",
    )
    parser.add_argument(
        "--esmfold2_stagea_min_root_expansions",
        type=int,
        default=2,
        help="hierarchical_ucb 下每个 Phase0 root 在自适应选择前至少获得的 outer-step 扩展次数（默认：2）",
    )
    parser.add_argument(
        "--esmfold2_stagea_preserve_soft_state",
        type=parse_boolean,
        default=True,
        help="Stage A 是否保留 child 的 soft logits：argmax child 不 anchor，mutation child 只在真实突变位做轻量 anchor（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stagea_cumulative_annealing",
        type=parse_boolean,
        default=False,
        help="Stage A 是否沿 MCTS 路径累计温度退火进度（默认：false）",
    )
    parser.add_argument(
        "--esmfold2_stagea_annealing_horizon",
        type=int,
        default=150,
        help="Stage A 路径累计温度退火的 soft-step horizon（默认：150）",
    )
    parser.add_argument(
        "--esmfold2_stagea_soft_anchor_margin",
        type=float,
        default=0.1,
        help="Stage A mutation child 在突变位超过当前最大 logit 的最小 margin（默认：0.1）",
    )
    parser.add_argument(
        "--esmfold2_stageb_soft_steps",
        type=int,
        default=None,
        help="ESMFold2 Stage B 每个 MCTS 节点的 soft gradient update 步数（默认：20）",
    )
    parser.add_argument(
        "--esmfold2_stagea_paired_verifier_seed",
        type=parse_boolean,
        default=True,
        help="Stage A 是否让所有 sequence 使用同一个 run-level verifier seed（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stagea_verifier_seed",
        type=int,
        default=None,
        help="可选的 Stage A 固定 verifier seed；未设置时由 run seed 和 target 推导",
    )
    parser.add_argument(
        "--esmfold2_stageb_discrete_rerank",
        type=parse_boolean,
        default=True,
        help="Stage B 是否用轻量 proposal forward 重排离散单点候选（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stageb_rerank_pool_size",
        type=int,
        default=20,
        help="Stage B 每步进入轻量重排序的候选池大小（默认：20）",
    )
    parser.add_argument(
        "--esmfold2_stageb_exploration_actions",
        type=int,
        default=1,
        help="Stage B 每步 full verifier 保留的探索候选数（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_stageb_verifier_candidates",
        type=int,
        default=8,
        help="Stage B 每步送入 full verifier 的候选数（默认：8）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_coverage_weight",
        type=float,
        default=0.25,
        help="Stage B reward 中连续 hotspot coverage loss 的权重（默认：0.25）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_distance_weight",
        type=float,
        default=0.25,
        help="Stage B reward 中 hotspot 距离 loss 的权重（默认：0.25）",
    )
    parser.add_argument(
        "--esmfold2_stageb_hotspot_miss_penalty",
        type=float,
        default=0.0,
        help="兼容旧实验的固定 hotspot miss penalty；V5 默认 0，改用连续 coverage loss",
    )
    parser.add_argument(
        "--esmfold2_stageb_min_pair_chain_iptm",
        type=float,
        default=0.0,
        help="Stage B reward 的 pair-chain iPTM 最低门槛；0 表示关闭（默认：0）",
    )
    parser.add_argument(
        "--esmfold2_stageb_handoff_quality_tolerance",
        type=float,
        default=0.05,
        help="Stage B branch 相对 handoff pair-chain iPTM 的最大允许下降（默认：0.05）",
    )
    parser.add_argument(
        "--esmfold2_stageb_paired_verifier_seed",
        type=parse_boolean,
        default=True,
        help="Stage B 是否让所有 sequence 使用同一个 run-level verifier seed（默认：true）",
    )
    parser.add_argument(
        "--esmfold2_stageb_verifier_seed",
        type=int,
        default=None,
        help="Stage B paired verifier 的显式 seed；默认按 run seed 和 target 派生",
    )
    parser.add_argument(
        "--esmfold2_lambda_esmc",
        type=str,
        default="auto",
        help="ESMC sequence prior 权重；auto 按任务类型选择，或传入数值字符串",
    )
    parser.add_argument(
        "--esmfold2_logit_init_mode",
        type=str,
        default="official_exact",
        choices=["official_free", "official_exact", "legacy_anchor"],
        help="ESMFold2 root logits 初始化方式；official_exact 用官方 seed 直接生成诊断初始 logits",
    )
    parser.add_argument(
        "--esmfold2_structure_regeneration_mask",
        type=str,
        default=None,
        help="template-anchored 局部结构重预测 mask；长度需等于 binder_length。第一版 TCR CDR 使用与 mutable mask 相同的区间",
    )
    parser.add_argument(
        "--esmfold2_template_chain_ids",
        type=str,
        default=None,
        help="外部复合物模板 chain ID，顺序必须与 --target 各链再加 binder 一致；例如 C,A,E,D",
    )
    parser.add_argument(
        "--esmfold2_verifier_num_loops",
        type=int,
        default=None,
        help="ESMFold2 verifier full fold loop 数；所有任务搜索期默认 4，可显式覆盖",
    )
    parser.add_argument(
        "--esmfold2_verifier_num_sampling_steps",
        type=int,
        default=None,
        help="ESMFold2 verifier diffusion sampling steps；所有任务搜索期默认 50，可显式覆盖",
    )
    parser.add_argument(
        "--esmfold2_proposal_num_loops",
        type=int,
        default=1,
        help="ESMFold2 proposal distogram forward loop 数（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_proposal_num_sampling_steps",
        type=int,
        default=1,
        help="ESMFold2 proposal distogram forward diffusion steps（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_proposal_load_internal_esmc",
        type=parse_boolean,
        default=True,
        help="是否为 ESMFold2Experimental proposal model 加载内部 ESMC（官方 binder design 为 true）",
    )
    parser.add_argument(
        "--esmfold2_proposal_lm_dropout",
        type=float,
        default=0.5,
        help="ESMFold2Experimental proposal LM dropout；官方 binder design inversion/proposal 使用 0.5",
    )
    parser.add_argument(
        "--esmfold2_force_proposal_lm_dropout",
        type=parse_boolean,
        default=True,
        help="是否在 eval/inference 状态强制启用 proposal LM dropout（官方 binder design 为 true）",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion",
        type=parse_boolean,
        default=True,
        help="是否启用可微 ESMFold2 one-step diffusion；默认 true。默认每 5 个 soft steps 联合更新一次",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_mode",
        choices=(
            "reference_noising",
            "truncated_trajectory",
            "template_anchored_truncated_trajectory",
        ),
        default="truncated_trajectory",
        help="One-step 坐标来源：reference_noising 使用 verifier 结构加噪；truncated_trajectory 从标准高噪声初态无梯度采样到 tau；template_anchored_truncated_trajectory 在同一路径固定外部模板区域",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_tau",
        type=float,
        default=1.0,
        help="ESMFold2 one-step diffusion 固定噪声尺度 tau（默认：1.0）",
    )
    parser.add_argument(
        "--esmfold2_one_step_truncated_num_sampling_steps",
        type=int,
        default=50,
        help="truncated_trajectory 构造官方 Karras schedule 的采样步数（默认：50）",
    )
    parser.add_argument(
        "--esmfold2_one_step_truncated_schedule_mode",
        choices=("official_clipped", "dense"),
        default="official_clipped",
        help="截断扩散 schedule：official_clipped 使用官方完整 Karras schedule 在 sigma=256 截断；dense 直接在 sigma=256 到 tau 之间加密",
    )
    parser.add_argument(
        "--esmfold2_one_step_truncated_dense_num_sampling_steps",
        type=int,
        default=41,
        help="dense schedule 从 sigma=256 到 tau 的 sigma 点数；默认 41，对应 40 次无梯度去噪和 1 次最终可微去噪",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_weight",
        type=float,
        default=0.25,
        help="归一化 one-step diffusion gradient 的组合权重（默认：0.25）",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_rg_weight",
        type=float,
        default=0.1,
        help="one-step diffusion common loss 中 Rg/compactness 项的权重（默认：0.1）",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_interval",
        type=int,
        default=5,
        help="在 soft optimization 内每隔多少步联合加入一次 one-step diffusion（默认：5）；0 表示仅在全部 soft steps 后额外更新一次",
    )
    parser.add_argument(
        "--esmfold2_one_step_recycle_coordinates",
        type=parse_boolean,
        default=False,
        help="间隔式 one-step 中是否将上一次去噪坐标作为下一次的参考结构；仅在 interval > 0 时有效",
    )
    parser.add_argument(
        "--esmfold2_one_step_diffusion_hotspot",
        type=parse_boolean,
        default=False,
        help="是否让 one-step diffusion 的 iPAE loss 使用 hotspot-aware localized 聚合；默认 false，保持现有 one-step 行为",
    )
    parser.add_argument(
        "--esmfold2_distogram_hotspot",
        type=parse_boolean,
        default=True,
        help="是否在 ESMFold2 distogram proposal loss 中启用 hotspot；默认 true。设为 false 可单独消融 distogram hotspot，不会关闭 one-step diffusion hotspot 或 full-verifier hotspot",
    )
    parser.add_argument(
        "--esmfold2_verifier_loss_mode",
        choices=["pair_chain_iptm", "aligned_common"],
        default="pair_chain_iptm",
        help="ESMFold2 full verifier loss：旧版 pair-chain iPTM，或不含 PDE 的 aligned common loss；默认保留旧版",
    )
    parser.add_argument(
        "--esmfold2_track_proposal_confidence",
        type=parse_boolean,
        default=None,
        help="ESMFold2 proposal 后期是否计算 confidence/iPTM 并加入候选（Best-of-N 默认 true，其余模式默认 false）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_temperature",
        type=float,
        default=0.05,
        help="ESMFold2 proposal temperature 低于该值时记录 confidence/iPTM（默认：0.05）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_interval",
        type=int,
        default=1,
        help="ESMFold2 proposal 后期每隔多少 local step 记录一次 confidence/iPTM（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_num_loops",
        type=int,
        default=1,
        help="ESMFold2 proposal confidence forward loop 数（默认：1）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_num_sampling_steps",
        type=int,
        default=50,
        help="ESMFold2 proposal confidence forward sampling steps（默认：50）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_max_candidates",
        type=int,
        default=8,
        help="每次 proposal 最多把多少个 late-confidence 序列加入 verifier 候选（默认：8）",
    )
    parser.add_argument(
        "--esmfold2_proposal_confidence_tail_steps",
        type=int,
        default=None,
        help="ESMFold2 proposal 固定记录最后 N 个 local steps；Best-of-N 默认 20，其余模式默认 0",
    )
    parser.add_argument(
        "--esmfold2_hotspot_weight",
        type=float,
        default=None,
        help="ESMFold2 proposal hotspot contact loss 权重；未显式设置时，ESMFold2 有 hotspot 为 0.5，否则为 0.0",
    )
    parser.add_argument(
        "--esmfold2_hotspot_cutoff",
        type=float,
        default=8.0,
        help="ESMFold2 proposal hotspot contact 距离阈值，单位 Å（默认：8.0）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_satisfied_proposal_weight",
        type=float,
        default=0.0,
        help="所有 hotspot 已在 verifier near cutoff 内命中后继续使用的 proposal hotspot 权重（默认：0.0，停止继续拉近）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_interface_mode",
        type=str,
        choices=["additive", "localized"],
        default=None,
        help="hotspot proposal 聚合方式：additive 保持旧版额外加权；localized 用 hotspot_weight 在全局 inter 与 hotspot inter 之间插值；未显式设置时，ESMFold2 有 hotspot 为 localized，否则为 additive",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_weight",
        type=float,
        default=None,
        help="ESMFold2 full verifier hotspot 距离 loss 权重；未显式设置时，ESMFold2 有 hotspot 为 0.25，否则为 0.0",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_near_cutoff",
        type=float,
        default=8.0,
        help="ESMFold2 hotspot verifier 满足接触的重原子距离，单位 Å（默认：8.0；同时记录严格 5 Å coverage）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_far_cutoff",
        type=float,
        default=20.0,
        help="ESMFold2 hotspot verifier 距离 loss 饱和上限，单位 Å（默认：20.0）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_min_distance",
        type=float,
        default=2.5,
        help="hotspot 与 binder 的最小安全重原子距离，单位 Å（默认：2.5）",
    )
    parser.add_argument(
        "--esmfold2_hotspot_verify_too_close_weight",
        type=float,
        default=None,
        help="低于最小安全距离时的 verifier 软惩罚权重；未显式设置时，ESMFold2 有 hotspot 为 0.25，否则为 0.0",
    )
    parser.add_argument(
        "--target_template_path",
        type=str,
        default=None,
        help="可选 target template 结构文件（.cif/.mmcif/.pdb），只作用于 target 链",
    )
    parser.add_argument(
        "--k_top", type=int, default=10, help="Top-K 突变候选数（默认：10）"
    )
    parser.add_argument(
        "--b_candidates", type=int, default=8, help="零阶验证候选序列数（默认：8）"
    )
    parser.add_argument(
        "--t_sample", type=float, default=1.0, help="突变采样温度（默认：1.0）"
    )
    parser.add_argument(
        "--max_mutations_per_seq",
        type=int,
        default=3,
        help="每条序列的最大突变数（默认：3）",
    )
    parser.add_argument(
        "--stagea_max_mutations_per_seq",
        type=int,
        default=None,
        help="ESMFold2/两阶段搜索 StageA 每条候选最大突变数；默认沿用 --max_mutations_per_seq",
    )
    parser.add_argument(
        "--esmfold2_stagea_mutation_fraction",
        type=float,
        default=0.5,
        help="ESMFold2 StageA sampled candidate 的最大唯一突变位点比例，按 mutable mask 长度计算（默认：0.5）",
    )
    parser.add_argument(
        "--esmfold2_stagea_candidate_strategy",
        type=str,
        choices=["legacy", "positive_score", "positive_score_guarded"],
        default="positive_score",
        help="StageA 离散候选策略；positive_score 仅使用正 logit-delta 突变，并要求设置 mutation fraction（默认：positive_score）",
    )
    parser.add_argument(
        "--esmfold2_stagea_dominance_gap",
        type=float,
        default=0.5,
        help="Guarded mode: minimum raw logit gap before and after update",
    )
    parser.add_argument(
        "--esmfold2_stagea_protected_mutation_candidates",
        type=int,
        default=2,
        help="Guarded mode: number of final weighted candidates to protect",
    )
    parser.add_argument(
        "--esmfold2_stagea_include_argmax_candidate",
        type=parse_boolean,
        default=True,
        help="StageA 是否加入不受突变比例限制的 logits argmax 候选（默认：true）",
    )
    parser.add_argument(
        "--stageb_max_mutations_per_seq",
        type=int,
        default=None,
        help="ESMFold2 StageB 每条候选最大突变数（默认：1）",
    )
    parser.add_argument(
        "--mcts_c", type=float, default=1.0, help="MCTS UCB 探索常数 C（默认：1.0）"
    )
    parser.add_argument(
        "--mcts_max_nodes",
        type=int,
        default=5000,
        help="MCTS 树最大节点数（默认：5000）",
    )
    parser.add_argument(
        "--mcts_num_beams",
        type=int,
        default=8,
        help="MCTS Phase0 根节点数 / beam（默认：8）",
    )
    parser.add_argument("--mcts_strategy", choices=["two_stage"], default="two_stage")
    parser.add_argument(
        "--mcts_bootstrap_steps",
        type=int,
        default=None,
        help="two_stage: StageA 至少运行步数；默认按 backend 解析：ESMFold2=70",
    )
    parser.add_argument(
        "--mcts_bootstrap_patience",
        type=int,
        default=10,
        help="two_stage: StageA 停滞 patience（默认：10）",
    )
    parser.add_argument(
        "--mcts_bootstrap_min_delta",
        type=float,
        default=0.02,
        help="two_stage: 判断 StageA 有效改善的 loss 阈值（默认：0.02）",
    )
    parser.add_argument(
        "--mcts_bootstrap_max_steps",
        type=int,
        default=None,
        help="two_stage: StageA 最大运行步数；默认按 backend 解析：ESMFold2=70",
    )
    parser.add_argument(
        "--mcts_handoff_top_k",
        type=int,
        default=None,
        help="two_stage: 交给 StageB 的 top 序列数（默认 2）",
    )
    parser.add_argument(
        "--mcts_expand_actions_per_step",
        type=int,
        default=None,
        help="MCTS StageB 每步最多扩展并验证的 action 数（默认：2）",
    )
    parser.add_argument(
        "--mcts_prior_mode",
        type=str,
        default=None,
        choices=[
            "full",
            "uniform",
            "no_gradient",
            "no_group",
            "no_group_matched_compute",
            "no_sequence",
        ],
        help="MCTS StageB prior 模式；默认按 backend 解析：ESMFold2=no_group；no_group_matched_compute 用三条 unified-loss proposal 对齐 Full StageB 的 proposal 计算量）",
    )
    return parser
