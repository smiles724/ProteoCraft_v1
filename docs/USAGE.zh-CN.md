# 输入与使用方式

[English](USAGE.md) | [简体中文](USAGE.zh-CN.md)

先按 [README](../README.zh-CN.md#安装) 安装环境和模型。以下命令从仓库根目录执行。
自定义命令中的 `$TARGET_SEQ` 等变量，先填入自己的氨基酸序列。

## 通用输入规则

- `--target` 直接填写序列，多条链用 `|` 连接。批量脚本从 FASTA/PDB 文件读取序列。
- `--esmfold2_target_chain_roles` 按 `--target` 中的链顺序填写角色：
  `target` 是结合靶标，`scaffold` 是固定的配对或环境链。两类链的序列都保持不变，
  待设计的 binder 另外提供。
- 热点是可选输入，按 `--target` 中所有链顺序拼接后，从 1 开始编号。
  例如第一条链长 100，编号 101 就是第二条链的第一个残基。

默认流程根据序列预测结构。基准案例中的 PDB/CIF 用于提取序列、保存参考结构和记录来源。

## 从头设计 binder

准备靶标序列和 binder 长度，程序设计整条 binder。

自己的靶标使用：

```bash
proteocraft --target "$TARGET_SEQ" --binder_length 120 \
  --output_dir outputs/denovo --device cuda:0
```

仓库中的 Bet v 1 案例已经包含靶标、热点和 binder 长度，可直接检查输入并运行：

```bash
python benchmark/script/run_esmfold2_59.py --case denovo_BetV1 --prepare-only
python benchmark/script/run_esmfold2_59.py --case denovo_BetV1 --device cuda:0
```

第一条命令检查输入并打印运行命令，第二条启动 GPU 设计。

## TCR binder 设计

准备 TCR α/β 两条链的序列，以及新 binder 的长度。这里以 TCR 为靶标，
设计一个新的结合分子；TCR 自身的序列保持不变。

```bash
proteocraft --esmfold2_task_type tcr \
  --target "${TCR_ALPHA_SEQ}|${TCR_BETA_SEQ}" \
  --esmfold2_target_chain_roles target,target \
  --binder_length 120 --output_dir outputs/tcr --device cuda:0
```

仓库中的 1QRN 案例由脚本从 PDB 提取 α/β 链序列：

```bash
python benchmark/script/run_esmfold2_59.py --case tcr_1qrn_no_hotspot --prepare-only
python benchmark/script/run_esmfold2_59.py --case tcr_1qrn_no_hotspot --device cuda:0
```

如果要修改 TCR 自身的 CDR，使其针对抗原进行重设计，使用下面的局部/CDR 模式。

## 局部/CDR 重设计

准备靶标、完整的起始 binder 和可变位置；复合物包含固定配对链时，一并提供这些链。
搜索修改指定位置，保留其余氨基酸。

`CDR_MASK` 是与 binder 等长的 0/1 字符串：`1` 允许修改，`0` 保持不变。
例如，12 位序列对应的 `000111000000` 表示只修改第 4–6 位；至少选择一个可变位置。

下面的命令包含一条抗原链和一条固定配对链：

```bash
proteocraft --esmfold2_task_type cdr_redesign \
  --target "${ANTIGEN_SEQ}|${PARTNER_TCR_SEQ}" \
  --esmfold2_target_chain_roles target,scaffold \
  --binder_length "${#REDESIGN_CHAIN_SEQ}" \
  --init_binder_seq "$REDESIGN_CHAIN_SEQ" \
  --esmfold2_mutable_mask "$CDR_MASK" \
  --output_dir outputs/cdr --device cuda:0
```

仓库中的 9NMW α 链案例已经包含抗原、固定环境链、起始 TCR α 序列和 CDR 片段：

```bash
python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml \
  --case 9NMW_alpha --prepare-only

python benchmark/script/run_esmfold2_cdr_redesign.py \
  --config benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml \
  --case 9NMW_alpha --device cuda:0
```

使用自己的批量输入时，复制 [CDR YAML](../benchmark/yaml/cdr_tcr/esmfold2_cdr_redesign_d_4cases.yaml)，
提供多链 FASTA。每条链的标题包含链名，例如 `>my_complex|Chain A|target`。
填写以下字段：

| 字段 | 填写内容 |
|---|---|
| `input_dir`、`fasta_path` | 输入目录，以及相对该目录的 FASTA 路径 |
| `binder_chain` | 要修改的链名 |
| `target_chains`、`scaffold_chains` | 结合靶标和固定环境链的链名列表 |
| `cdr_motif` **或** `redesign_regions` | 一个在序列中唯一出现的片段，或 `[[26, 33], [51, 58]]` 这样的区间列表 |

两种区域写法选择一种。区间按完整起始 binder 的序列从 1 开始编号，包含起点和终点；
请改成自己序列中的实际位置。区间之外的框架保持不变。
没有固定环境链时填 `scaffold_chains: []`。
默认序列流程中的 `template_path` 是可选参考文件，换成自己的参考文件或删去该字段。

## VHH 三个 CDR 重设计

仓库提供输入已填全的 [IFNAR2 示例](../benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml)：

```bash
python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml --prepare-only

python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config benchmark/yaml/vhh/esmfold2_vhh_ifnar2_example.yaml --device cuda:0
```

它沿用清理回归实验的 217 残基 IFNAR2 靶标输入，以及来自 7EOW B 链的
128 残基起始 VHH。三个 CDR 共 37 个位置可变，其余 91 个位置固定；
搜索设置由当前 `esmfold2_d` 配置提供。热点按案例中的 IFNAR2 输入序列编号。
结果位于 `outputs/vhh_ifnar2/IFNAR2/`。

| CDR | 起始 VHH 中的位置（从 1 开始） |
|---|---|
| CDR1 | 26–33 |
| CDR2 | 51–58 |
| CDR3 | 97–117 |

使用自己的输入时，按下面的方法编辑模板。

准备靶标、完整起始 VHH，以及通过 IMGT 编号确定的 CDR1、CDR2、CDR3 位置。
程序把三个 CDR 的并集作为可变区域，保留框架。

先复制 [VHH 模板](../benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml)：

```bash
cp benchmark/yaml/vhh/esmfold2_vhh_redesign_7eow_template.yaml my_vhh.yaml
```

准备运行前，编辑复制的文件：

| 字段 | 填写内容 |
|---|---|
| `cases[].target_sequences` | 把 `REPLACE_WITH_TARGET_SEQUENCE` 换成自己的靶标序列，一条链一项 |
| `initial_vhh_sequence` | 完整 VHH 序列；模板自带 7EOW 的起始 VHH |
| `numbering` | `scheme: imgt`，以及对该序列使用的编号工具和版本 |
| `cdr_regions` | CDR1、CDR2、CDR3 各自的 `sequence_indices_1based` 列表 |
| `target_hotspots_1based` | 可选热点位置；没有时填 `[]` |
| `output_dir` | 输出目录 |

先对 VHH 做 IMGT 编号，再把三个 CDR 映射回输入序列中从 1 开始的位置。
例如 `[26, 27, 28]` 表示输入序列的第 26、27、28 个氨基酸。
启动脚本读取这份映射，`numbering` 用来记录映射来源。
更换 VHH 时同步更新三个 CDR 列表和来源信息；可删去可选的 `mutable_mask` 字段，
让程序按 CDR 位置生成掩码，也可以填写与之相符的掩码。
模板中可选的 `scaffold` 来源信息也应换成自己的 VHH 信息或删去。

编辑完成后，先检查输入，再启动设计：

```bash
python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config my_vhh.yaml \
  --prepare-only

python benchmark/script/run_esmfold2_vhh_redesign.py \
  --config my_vhh.yaml --device cuda:0
```

准备步骤检查 CDR 位置范围、重叠和掩码一致性。CDR/VHH 批量脚本会在
`<output_dir>/<case_id>/` 写入 `case_spec.json`、`command.txt` 和 `provenance.json`，
可先查看解析后的输入，再启动 GPU 设计。

## 输出与恢复

直接运行的结果位于 `--output_dir`；批量案例位于 `<output_dir>/<case_id>/`。

| 文件 | 内容 |
|---|---|
| `hallucination_result.txt` | 选中的 binder 序列和实际配置 |
| `esmfold2_artifacts/checkpoints/final/step_NNNN/step_NNNN_best.cif` | 最终选中的复合物结构；默认 110 轮对应 `step_0110` |
| 同目录中的 `step_NNNN_best_meta.json` | 与该结构匹配的指标 |
| `esmfold2_artifacts/candidates.jsonl` | 已验证的候选及指标 |
| `esmfold2_artifacts/trajectory.jsonl` | 搜索轨迹 |
| `esmfold2_artifacts/run_summary.json` | 搜索摘要 |

最终结构默认保存；添加 `--esmfold2_save_verified_candidate_cif true` 可保存每个已验证候选。
Stage A 交接状态位于 `esmfold2_artifacts/handoff_checkpoints/stagea_handoff_final.pt`，
可用 `--esmfold2_resume_handoff_checkpoint /path/to/checkpoint` 恢复。
恢复时提供的起始 binder，其固定位置需与检查点一致。

默认内存模式为 `resident`；`--esmfold2_memory_mode phase_offload` 在阶段间卸载暂不用的模型。
具体配置见[内存与检查点说明](USAGE.md#outputs-checkpoints-and-memory)。
维护者配置见[算法设置](USAGE.md#algorithm-configuration-for-maintainers)，
候选清单和评估编号见[独立复评分](USAGE.md#independent-rescoring)。
