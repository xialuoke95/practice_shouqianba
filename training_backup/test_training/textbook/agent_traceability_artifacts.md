# Agent 可追溯性与证据包要求

## 1. 核心判断

Agent 有概率干不好，尤其是在 数据转换、label mask、checkpoint/export、推理模板、评测脚本这些环节。训练结果不能只靠一句“loss 下降了”下结论，必须留下足够 artifacts，让人可以从最终结论追溯回：

```text
run 配置 -> 数据版本 -> batch/mask -> 训练曲线 -> checkpoint -> 推理输出 -> 评测结果
```

## 2. 为什么必须保存证据

常见误判包括：

- `loss` 下降，但 user prompt 也被算进 loss，实际 mask 错了。
- dev NLL 没坏，但 checkpoint reload 或 adapter merge 后输出变了。
- 训练侧 chat template 正确，serving 侧模板不一致，导致线上输出坏。
- fixed generation 看起来正常，但 JSON/tool 格式已经退化。
- LoRA 配置写了 target modules，但实际没有命中目标模块或只命中很少模块。
- MoE/MLA 模型复用了 dense 模型的 target module 名字，训练范围不符合预期。
- 同一个数据集重新切分，导致不同 run 不可比。
- agent 生成的总结遗漏失败样本，导致人工复盘无法定位。

所以每个 run 必须先定义“证据包”，再谈结论。

## 3. 最小证据包

每个训练或检查 run 至少按五类保存 artifacts。每一类再拆成小子项，方便模型和人快速定位应该看哪个证据。

### 3.1 环境 artifacts

| 子项 | 必须记录 | 用途 |
|---|---|---|
| 代码版本 | git commit、repo dirty status、训练脚本路径 | 确认代码来源和是否可复现 |
| 镜像与系统 | docker image、OS/kernel、驱动版本 | 定位系统和容器差异 |
| CUDA 栈 | CUDA、NCCL、cuDNN、GPU 型号和数量 | 定位分布式和显存问题 |
| Python 依赖 | torch、transformers、ms-swift、peft、bnb、deepspeed、megatron 版本 | 定位训练框架差异 |
| 推理依赖 | vLLM、SGLang、推理侧 transformers/tokenizer 版本 | 定位 export/serving 差异 |
| 模型信息 | base model 路径、revision、tokenizer、chat template、special tokens | 确认输入输出协议 |

### 3.2 数据 artifacts

| 子项 | 必须记录 | 用途 |
|---|---|---|
| 原始数据 | 原始 JSONL、样本数、source、license/usage notes | 追溯数据来源 |
| 数据切分 | train/dev/test split、split seed、split hash | 避免泄漏和不可比 |
| Template 后数据 | template 后 JSONL、真实训练样本快照 | 检查 role、special token、tool serialization |
| 数据版本指纹 | manifest fingerprint、shard hashes/etags、数据生成脚本版本、shuffle seed、sample ranges | 唯一标识本次训练实际消费的数据，确认不同 run 是否可比 |
| token 统计 | token length 分布、assistant-token ratio、label-token ratio | 判断训练信号和长度风险 |
| 截断样本 | 被截断样本列表、截断前后 token 数、截断原因 | 排查长样本和 packing 问题 |
| mask 快照 | decoded input、decoded labels、有效 label token 数 | 证明 assistant-only loss 正确 |

### 3.3 训练 artifacts

| 子项 | 必须记录 | 用途 |
|---|---|---|
| 配置 | YAML/config、CLI、env vars、DeepSpeed/FSDP/Megatron 配置 | 复现训练行为 |
| LoRA/可训练参数 | target module regex、trainable params、trainable parameter list | 证明实际训练范围 |
| 训练曲线 | loss、dev NLL、test NLL、PPL | 判断目标是否被学习 |
| 数值状态 | grad norm、lr、overflow/skip、NaN/inf 记录 | 判断数值稳定性 |
| 性能状态 | tokens/sec、step time、显存、GPU utilization | 定位效率和显存问题 |
| 分布式日志 | 每 rank 日志、rank failure、通信错误 | 定位多卡/多机问题 |
| Checkpoint | checkpoint 路径、步数、大小、save manifest、reload 结果、resume 结果 | 确认保存恢复可用 |
| Export/merge | adapter 路径、merge 结果、export 路径、export 后 tokenizer/template | 确认训练产物能进入推理链路 |

### 3.4 MoE 专项 artifacts

| 子项 | 必须记录 | 用途 |
|---|---|---|
| 路由配置 | router/gate 是否冻结、router 参数是否可训练 | 确认 MoE 训练边界 |
| Expert 负载 | expert load、负载均衡统计、top/bottom expert | 判断 expert imbalance |
| Aux loss | aux loss 曲线、异常点 | 判断路由训练/负载约束是否正常 |
| Expert adapter | 各 expert adapter size、shared/routed expert 小计 | 确认 LoRA 是否命中预期 expert |
| Shard manifest | save/resume shard manifest、分片数量、每片大小 | 确认 MoE checkpoint 可恢复 |
| Expert 并行 | EP/TP/DP/PP/SP 配置、每 rank expert 分配 | 定位并行策略问题 |

### 3.5 评测 artifacts

| 子项 | 必须记录 | 用途 |
|---|---|---|
| Decoding 配置 | temperature、top_p、max tokens、stop、chat template | 保证生成可复现 |
| 原始输出 | base/sft raw outputs、fixed generation、sentinel、JSON/tool probes | 支持人工复盘 |
| 解析结果 | JSON/tool parse 结果、解析失败样例 | 判断格式是否退化 |
| 切片分数 | category/slice score、Dolly category、probe type | 判断问题集中在哪类样本 |
| 失败聚类 | failure taxonomy、代表样本、数量 | 支持定位和下一轮修正 |
| 推理一致性 | training-side generate、reload generate、merge/export/serving generate 对照 | 检查 checkpoint 和推理模板一致 |

其中三项是强制保存项，不能用摘要替代：

| 强制项 | 原因 |
|---|---|
| template 后的真实训练样本 | chat template、role、special token、tool serialization 最容易错，且错了 loss 也可能下降 |
| loss mask 可视化 | assistant-only mask 错时仍可能看到 loss 下降，必须保存 decoded labels |
| trainable parameter list | LoRA target module 写对不等于实际命中，必须保存实际可训练参数列表 |


## 4. Run card : 结论文件的题头

结论文件不归入上述五类 artifacts，而是五类证据的出口摘要，使用如下题头

```yaml
run_id:
date:
owner:
stage:
  - dolly_probe
  - qwen235b_lora_sft
model:
training_method:
  method: lora_sft
  precision: bf16
  seq_len:
  micro_batch_size:
  gradient_accumulation_steps:
  global_batch_size:
  learning_rate:
  max_steps:
  save_steps:
moe:
  enabled:
  router_or_gate_frozen:
  expert_load:
  aux_loss:
  expert_adapter_sizes:
  shard_manifest:

```

## 5. Artifact 目录结构

建议每个 run 使用固定目录结构：

```text
RUN_DIR/
  conclusion.md(含 run card)
  environment/
    env_summary.json
    pip_freeze.txt
    git_status.txt
    package_versions.json
  data/
    manifest.json
    raw_sample_preview.jsonl
    templated_sample_preview.jsonl
    token_stats.json
    truncation_report.jsonl
    mask_snapshots.jsonl
  training/
    train_config.yaml
    cli_args.txt
    env_vars.txt
    metrics.csv
    trainable_params_summary.json
    trainable_params_full.jsonl
    rank_logs/
    checkpoints_manifest.json
    export_manifest.json
  moe/
    moe_summary.json
    expert_load_summary.json
    expert_load_full.jsonl
    aux_loss.csv
    expert_adapter_sizes.json
    shard_manifest.json
  eval/
    decoding_config.json
    raw_outputs.jsonl
    parsed_outputs.jsonl
    parse_failures.jsonl
    slice_scores.json
    failure_clusters.json
    inference_consistency.json
```

如果不是 MoE 模型，`moe/` 可以省略，但 run_card 里要显式写 `moe.enabled: false`。

## 6. 渐进式披露规则

专项证据不要全部塞进结论文件。采用三层结构：

| 层级 | 内容 | 典型文件 |
|---|---|---|
| L1 summary | 人和模型首先读取的摘要，包含 pass/fail、关键统计、路径、hash | `*_summary.json`、`conclusion.md` |
| L2 diagnostic | 排查需要的中等规模诊断信息，包含代表样本、切片统计、失败聚类 | `mask_snapshots.jsonl`、`slice_scores.json`、`failure_clusters.json` |
| L3 full artifact | 大文件或全量明细，只在需要时打开 | `templated_full.jsonl`、`trainable_params_full.jsonl`、`expert_load_full.jsonl`、`raw_outputs.jsonl` |

大模型/MoE 场景必须使用渐进式披露：

- `trainable parameter list` 可能很长，结论里只放 summary 和 full path。
- `expert load` 可能按 step/layer/expert 爆炸，结论里只放均衡统计、异常 expert、full path。
- `template 后 JSONL` 可能很大，结论里只放 preview、hash、full path。
- `raw outputs` 可能很多，结论里只放失败聚类和代表样本，保留 full path。

## 7. 证据要求形式

### Mask 证据要求

必须保存 decoded mask snapshot。格式建议如下：

```json
{
  "sample_id": "dolly_train_000123",
  "raw_messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "templated_text": "...完整 chat template 后的训练文本...",
  "decoded_input": "...完整 chat template 后的输入...",
  "decoded_labels_non_ignore": "...只应该包含 assistant response + eos...",
  "num_input_tokens": 512,
  "num_label_tokens": 78,
  "label_policy": "assistant_only_with_eos",
  "padding_ignored": true
}
```

通过标准：

```text
system/user/context tokens -> label = -100
assistant response tokens -> label = token_id
assistant EOS -> label = eos_token_id
padding tokens -> label = -100
```

如果 decoded labels 里出现 instruction、context、user prompt，当前 run 直接判定为失败。

### Trainable parameter 证据要求

必须保存实际 trainable parameter list，而不是只保存 LoRA 配置。

建议拆成 summary 和 full 两个文件。

`trainable_params_summary.json`：

```json
{
  "target_module_regex": ".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
  "trainable_params_count": 123456789,
  "total_params_count": 235000000000,
  "trainable_params_ratio": 0.00052,
  "router_or_gate_frozen": true,
  "module_type_breakdown": {
    "attention": 45678901,
    "mlp": 67890123,
    "shared_experts": 1234567,
    "routed_experts": 8901234,
    "router_or_gate": 0
  },
  "full_list_path": "training/trainable_params_full.jsonl",
  "full_list_sha256": "..."
}
```

`trainable_params_full.jsonl`：

```json
{
  "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
  "shape": [16, 8192],
  "numel": 131072,
  "dtype": "bf16",
  "module_type": "attention",
  "layer": 0
}
```

对 MoE/MLA 模型，必须额外标注：

- 是否命中 shared experts。
- 是否命中 routed experts。
- 是否命中 router/gate。
- 是否命中 MLA attention projection。
- 每类模块的 trainable params 小计。

如果 target module regex 和实际 trainable parameter list 不一致，以实际 list 为准。

### MoE 专项证据要求

如果模型包含 MoE，必须单独保存 MoE artifacts。

建议拆成 summary 和 full 两层。

`moe_summary.json`：

```json
{
  "router_or_gate_frozen": true,
  "router_trainable_params": 0,
  "expert_load_summary": {
    "mean_load": 1024.0,
    "max_load": 1402,
    "min_load": 711,
    "imbalance_ratio": 1.97,
    "worst_layer": 12,
    "worst_step": 200
  },
  "aux_loss_summary": {
    "start": 0.12,
    "end": 0.09,
    "max": 0.18
  },
  "expert_adapter_total_params": 12345678,
  "expert_load_full_path": "moe/expert_load_full.jsonl",
  "aux_loss_path": "moe/aux_loss.csv",
  "expert_adapter_sizes_path": "moe/expert_adapter_sizes.json",
  "shard_manifest_path": "moe/shard_manifest.json"
}
```

`expert_load_full.jsonl`：

```json
{
  "step": 1,
  "layer": 0,
  "loads": [1024, 1011, 997, 1033],
  "mean": 1016.25,
  "max": 1033,
  "min": 997,
  "imbalance_ratio": 1.036
}
```

`expert_adapter_sizes.json`：

```json
{
  "shared_experts_total": 1234567,
  "routed_experts_total": 8901234,
  "entries": [
    {
      "name": "layer0.expert0",
      "type": "routed_expert",
      "adapter_params": 1048576
    }
  ]
}
```

必须能回答：

- router/gate 是否冻结。
- expert load 是否明显失衡。
- aux loss 是否记录。
- 每个 expert 的 adapter 大小是否符合预期。
- save/resume shard manifest 是否能解释 checkpoint 分片。

## 8. 失败分类

建议统一 failure taxonomy：

| 类型 | 含义 |
|---|---|
| `mask_error` | label 覆盖了非 assistant token，或 response/EOS 没进 loss |
| `template_error` | chat template、role、special token、tool serialization 错 |
| `eos_error` | 不会停、EOS 未训练、padding 参与 loss |
| `lora_target_error` | target module 没命中或命中范围不符合预期 |
| `trainable_param_error` | 实际 trainable parameter list 与设计不一致 |
| `data_truncation_error` | token length 或截断样本不符合预期 |
| `numerical_error` | loss NaN、grad 爆炸、BF16/FP8 数值异常 |
| `checkpoint_error` | save/load/resume/export/merge 后行为不一致 |
| `generation_regression` | fixed generation 明显坏于 base |
| `format_regression` | JSON/tool 输出格式错误率升高 |
| `moe_imbalance` | expert load、aux loss、router/gate 行为异常 |
| `eval_leakage` | train/dev/test 或 probe 混用 |
| `artifact_missing` | 关键证据缺失，无法追溯 |

## 9. 结论规则

不能接受的结论：

```text
loss 降了，所以训练成功。
生成看起来还行，所以模板没问题。
没有报错，所以 checkpoint 可用。
```

可以接受的结论：

```text
本 run 通过 Dolly SFT 基建检查：
已保存五类 artifacts；
template 后真实训练样本可复查；
mask snapshot 显示仅 assistant response + EOS 参与 loss；
trainable parameter list 与 LoRA target 设计一致；
train loss 下降；
dev/test assistant-only NLL 未明显劣化；
fixed generation、sentinel、JSON/tool probes 未出现格式退化；
checkpoint reload 后 NLL 与生成一致；
所有 artifacts 已保存到指定 run 目录。
```
