# Post-training 算法侧/工程侧训练复盘模板

本文件是模板文档，不是知识百科。它的用途是在跑完一次训练后，依照算法侧和工程侧知识点，把本次具体 training settings、观测结果、风险和结论写清楚。

## 1. 使用方式

每次训练后复制本模板，填成本次 run 的复盘文档。

目标不是解释所有概念，而是回答：

```text
这次训练具体用了什么 settings？
这些 settings 分别触及哪些算法/工程知识点？
这个知识点在本次训练里是什么意思、细节是什么样的？
观测到的证据是什么？
哪些地方还不能下结论？
```

所有原始 artifacts 引用 `docs/agent_traceability_artifacts.md` 规定的路径和 hash。

## 2. Run 摘要

```yaml
run_id:
date:
model:
base_model_path:
framework:
method:
seq_len:
micro_batch_size:
global_batch_size:
steps:
tokens_seen:
dataset:
checkpoint:
report_status:
```

一句话结论：

```text
例如：本 run 是 Qwen3-235B-A22B 在 Dolly probe 上的 BF16 LoRA SFT 基建检查；目标是验证 mask/template/checkpoint/推理一致性，不用于证明能力提升。
```

## 3. 算法侧知识点如何在本 run 中实例化

### 3.1 SFT objective / cross entropy / label mask

本次 setting：

```text
objective:
label policy:
EOS policy:
padding policy:
packing:
```

本次证据：

```text
mask snapshot path:
mask snapshot hash:
num checked samples:
assistant-token ratio:
```

解释本次具体情况：

```text
这里要把 SFT 目标解释清楚：模型训练时仍然是在预测 next token，但 loss 只应该落在 assistant answer 上。
如果 user prompt、system prompt、tool observation 也进入 labels，loss 下降并不能说明模型学会回答问题，可能只是学会复述输入格式。
如果 EOS 没有进入 labels，模型可能学到了回答内容但没有学到何时停止。
如果 padding 进入 labels，loss 会被无意义 token 污染。

例：本次只对 assistant response + EOS 计算 CE loss；user/context/tool observation 全部为 -100。mask snapshot 中 decoded labels 未出现 user prompt，且有效 label token 数与 response token 数一致。因此 train loss 下降可以解释为 assistant response 分布被学习，而不是模型在学习复述 prompt 或 padding。
```

不能下的结论：

```text
例：mask 正确不代表模型能力提升，只说明训练目标没有明显错位。
```

### 3.2 Chat template / tool-call serialization / packing

本次 setting：

```text
chat template:
system prompt:
tool-call format:
tool observation format:
packing enabled:
max seq len:
truncation policy:
```

本次证据：

```text
templated sample path:
token stats path:
truncation report path:
JSON/tool probe path:
```

解释本次具体情况：

```text
这里要说明训练时模型实际看到的文本是什么，而不是只说明原始 messages 长什么样。
Chat template 决定了 role token、assistant 起始位置、EOS、工具调用序列化方式。template 错时，训练 loss 仍可能下降，但推理时模型使用另一套格式就会坏。
Packing 需要额外说明样本边界是否清楚，EOS 是否阻断跨样本学习，position/attention 处理是否符合框架预期。

例：Dolly 样本被转成 user/assistant messages；context 为空时不插入空 Input 段。template 后样本显示 assistant 起始 token、response 文本和 EOS 位置符合 tokenizer 预期。由于 Dolly 不含 tool-call，JSON/tool probe 是额外格式探针，只能说明格式没有退化，不能说明真实 tool-use 能力提升。
```

风险：

```text
例：如果 serving 侧使用不同 chat template，训练侧 NLL 正常也不能说明线上推理正确。
```

### 3.3 数据切片与任务定位

本次 setting：

```text
dataset:
slices:
probe types:
train/dev/test split:
```

本次证据：

```text
manifest path:
slice score path:
failure cluster path:
```

解释本次具体情况：

```text
这里要把“数据训练了什么能力”和“报告能声明什么能力”分开。
Dolly 的 category 可以观察 instruction-following 的不同表面形式，例如 summarization、classification、open QA，但它不覆盖 repo navigation、patch generation、tool execution 或 agent benchmark。
Sentinel 是训练链路探针，主要证明梯度打到了 assistant answer，不代表通用能力。
JSON/tool probe 是格式防退化检查，不代表多轮工具任务成功率。

例：Dolly 在本 run 中是 SFT 基建探针，不是目标域能力数据。slice 主要按 Dolly category、sentinel、JSON/tool probes 切分。若 summarization 类输出变短，只能解释为模型向 Dolly response 风格靠近；不能据此声明 coding、Brazil localization 或 tau-bench retail 能力提升。
```

### 3.4 LoRA / target modules / trainable params

本次 setting：

```text
LoRA rank:
LoRA alpha:
LoRA dropout:
target module regex:
router/gate trainable:
expert trainable:
```

本次证据：

```text
trainable params summary path:
trainable params full path:
trainable params hash:
```

解释本次具体情况：

```text
这里要解释 LoRA 实际训练的是哪些参数，而不是只抄 rank/alpha。
LoRA 的效果取决于 target modules 是否命中真实模型结构；regex 写对不等于参数真的可训练。对 MoE/MLA 模型，attention projection、MLP、shared experts、routed experts、router/gate 的命中情况会显著影响训练含义。
trainable parameter list 是最终证据：它告诉我们哪些权重参与反向传播、参数量占比是多少、是否误训了 router/gate 或漏训了预期模块。

例：本次 target module regex 命中了 attention 和 MLP projection，但 router/gate 冻结。trainable parameter summary 显示 attention、MLP 的 LoRA 参数量符合预期，full list 中没有 router/gate 参数。因此可以排除“配置写了但 LoRA 没挂上”或“误训 router”的问题。
```

风险：

```text
例：Qwen3-235B-A22B 是 MoE/MLA 结构时，不能机械复用 dense Qwen3-4B 的 module 名称。必须以实际 trainable parameter list 为准。
```

### 3.5 QLoRA / FP8 / BF16 定位

本次 setting：

```text
precision:
base weight dtype:
optimizer dtype:
quantization:
FP8 enabled:
```

本次证据：

```text
config path:
package versions:
numerical logs:
```

解释本次具体情况：

```text
这里要解释精度选择对结论边界的影响。
BF16 路径通常更适合第一阶段排查，因为它减少了量化误差、scale、反量化和 FP8 kernel 兼容性带来的变量。
QLoRA 会引入 4-bit base、NF4、量化噪声和 adapter 训练可比性问题；FP8 会引入框架/kernel/scale 策略问题。
如果第一阶段目标是检查 SFT 基建，先用 BF16 可以把问题收敛到数据、mask、LoRA、checkpoint 和模板。

例：本次 BF16 起步，因此优先验证训练链路和 mask/template/checkpoint。没有把 FP8 或 QLoRA 引入第一阶段，避免把量化数值问题和 SFT 基建问题混在一起。若后续改成 FP8/QLoRA，需要重新解释数值稳定性、NLL 可比性和 export/serving 路径。
```

不能下的结论：

```text
例：BF16 run 通过，不代表 FP8/QLoRA 路径已经可用。
```

### 3.6 MoE / MLA / Kimi 迁移

本次 setting：

```text
MoE enabled:
MLA enabled:
router/gate frozen:
expert parallel:
expert LoRA:
```

本次证据：

```text
moe summary path:
expert load path:
aux loss path:
expert adapter size path:
shard manifest path:
```

解释本次具体情况：

```text
这里要解释 MoE/MLA 让“训练了哪些模块”这个问题变复杂。
MoE 模型有 router/gate、shared experts、routed experts，训练 LoRA 时必须说明 adapter 挂在哪些模块上。router 如果参与训练，可能改变样本进入 expert 的分布；router 如果冻结，训练主要改变 expert 内部变换。
Expert load 用来判断 token 是否过度集中到少数 expert；aux loss 用来观察负载均衡约束；expert adapter size 用来确认每个 expert 的 LoRA 参数是否符合预期；shard manifest 用来确认 checkpoint 能保存和恢复。
MLA 下 attention projection 不一定等同传统 q/k/v/o，因此 target module 必须按本模型结构审计。

例：本次如果是 Qwen3-235B-A22B，MoE 相关证据需要说明 router 是否冻结、expert load 是否失衡、expert adapter size 是否符合预期。迁移自小模型的是审计流程，不是 target module 名称和超参。
```

风险：

```text
例：如果没有 expert load 和 shard manifest，不能判断 MoE 分布式训练是否健康，也不能确认 checkpoint resume 风险。
```

## 4. 工程侧知识点如何在本 run 中实例化

### 4.1 分布式策略与显存结构

本次 setting：

```text
DP:
ZeRO/FSDP:
TP:
EP:
PP:
SP:
num nodes:
num GPUs:
```

本次证据：

```text
rank logs path:
memory logs path:
tokens/sec path:
step time path:
```

解释本次具体情况：

```text
这里要解释 batch、并行策略和显存/吞吐之间的关系。
Micro batch 决定每张卡单步 activation 压力；gradient accumulation 决定有效 global batch；ZeRO/FSDP 影响参数、梯度、优化器状态如何切分；TP/EP/PP/SP 影响通信模式和每 rank 负载。
显存高不一定是参数问题，也可能是 seq_len、activation、checkpointing、optimizer state 或通信 buffer。tokens/sec 低也不一定是模型慢，可能是数据加载、rank imbalance、NCCL 通信或 checkpoint 写盘。

例：micro_batch_size=1，global batch 通过 grad_accum 堆。本次观察 tokens/sec、step time、GPU memory、每 rank 日志和 OOM 位置，用来判断瓶颈是显存、通信、expert imbalance 还是数据加载。短 Dolly probe 的性能只能说明当前短序列设置，不应外推到 8k/32k 正式训练。
```

不能下的结论：

```text
例：短 Dolly probe 的 tokens/sec 不能直接外推到 8k/32k 长上下文正式训练。
```

### 4.2 框架选择与生命周期

本次 setting：

```text
framework:
launcher:
training entrypoint:
resume strategy:
export strategy:
serving engine:
```

本次证据：

```text
config path:
checkpoint manifest:
export manifest:
serving output path:
```

解释本次具体情况：

```text
这里要解释为什么选这个框架，以及该框架覆盖了训练生命周期的哪些环节。
LLaMA-Factory 适合快速 baseline 和数据格式 rehearsal，但它不能自动证明 Megatron-SWIFT 的 MoE、shard checkpoint、export 路径可用。
Megatron-SWIFT 更贴近 Qwen235B/MoE 主线，但排查成本更高，所以报告要明确哪些问题已经在当前框架上验证，哪些只是从小模型 rehearsal 迁移过来的假设。

例：本次使用 Megatron-SWIFT 是为了贴近 Qwen235B/MoE 主线；如果先在 LLaMA-Factory/Qwen3-4B rehearsal，则该 rehearsal 只能验证数据、template、mask 和通用 SFT 逻辑，不能覆盖 Qwen235B 的 MoE/MLA target module、分布式策略、shard checkpoint 和 export 风险。
```

### 4.3 Checkpoint / resume / export / merge / inference

本次 setting：

```text
save_steps:
save_total_limit:
checkpoint format:
resume tested:
export tested:
merge tested:
serving tested:
```

本次证据：

```text
checkpoint path:
reload metrics:
resume logs:
export path:
merge comparison path:
serving comparison path:
```

解释本次具体情况：

```text
这里要把 checkpoint、resume、export、merge、inference 分开解释。
Checkpoint reload 只说明保存的训练产物能被同一路径加载；resume 还需要 optimizer/scheduler/global step 状态正确；export/merge 说明 adapter 或分片能转成推理可用格式；serving 一致性说明线上引擎加载了正确权重和 template。
这些环节任何一个没测，都不能被前一个环节替代。

例：checkpoint reload 后 dev NLL 与训练结束时一致，fixed generation 差异在 decoding 随机性范围内，因此可以认为 checkpoint 可加载。若未测试 resume，就不能声明可中断恢复；若 serving 未测试，报告必须明确“不覆盖 serving template 一致性”。
```

### 4.4 观测与审核

本次 setting：

```text
artifact policy:
failure taxonomy:
review owner:
```

本次证据：

```text
run card:
conclusion:
failure clusters:
raw outputs:
```

解释本次具体情况：

```text
这里要解释 artifacts 如何支撑结论，而不是把所有原始文件塞进报告。
Summary artifacts 用来快速判断 pass/fail；diagnostic artifacts 用来定位异常；full artifacts 用于最终复查。这样可以让人和模型先读高信噪比摘要，再根据异常打开具体证据。
失败聚类要连接到 failure taxonomy，例如 mask_error、template_error、trainable_param_error、checkpoint_error、format_regression。否则“失败样本很多”不能转化成下一步行动。

例：本次报告只引用 summary artifacts；full trainable parameter list、raw outputs、expert load 以路径和 hash 形式保留，必要时再打开。这符合渐进式披露，避免结论文件塞入过多低信噪比内容，同时仍然能让人工追溯到原始证据。
```

## 5. 本 run 的指标解释

| 指标 | 观测 | 解释 | 限制 |
|---|---|---|---|
| train loss | | | |
| dev NLL | | | |
| test NLL | | | |
| grad norm | | | |
| tokens/sec | | | |
| GPU memory | | | |
| sentinel exact | | | |
| JSON valid rate | | | |
| fixed generation regressions | | | |

