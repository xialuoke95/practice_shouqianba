# Dolly SFT 探针任务流程与检查点

本文件只描述 Dolly 阶段特有流程；通用 artifact、目录结构、证据格式，统一看 `docs/agent_traceability_artifacts.md`。

## 1. 定位

Dolly 是第一阶段 SFT 基建探针，不是能力 benchmark。

用于检查：

- Dolly 数据能否被稳定 template 成目标模型 chat 格式。
- assistant-only loss、EOS、padding 是否正确。
- base dev/test assistant-only NLL 是否能计算。
- SFT 后 train loss 是否下降，dev/test NLL 是否不明显劣化。
- fixed generation probes 是否不坏。
- sentinel probes 是否能证明梯度打到 assistant tokens。
- JSON/tool probes 是否没有格式退化。
- checkpoint reload / export / serving 是否与训练侧模板一致。

不用于证明：

- Qwen3-235B-A22B 真实能力提升。
- agent 能力提升。


## 3. Dolly probe 数据

```text
dolly_probe_v1/
  dolly_train.jsonl
  dolly_dev.jsonl
  dolly_test.jsonl
  sentinel_train.jsonl
  sentinel_dev.jsonl
  generation_prompts.jsonl
  json_tool_probes.jsonl
  manifest.json
```

要求：

- Dolly split 固定，不要每次随机切。
- Dolly 原始字段 `instruction/context/response/category` 转成 messages。
- Sentinel 用 exact-string 任务验证训练信号。
- JSON/tool probes 用来补 Dolly 不覆盖的格式检查。

## 4. Base baseline

训练前先跑 baseline，并保存到 artifact 目录。

必须完成：

1. 记录模型、tokenizer、chat template、special tokens。
2. 保存 Dolly split manifest 和 hash。
3. 保存 template 后真实训练样本 preview。
4. 保存 loss mask 可视化，确认 labels 只包含 assistant response + EOS。
5. 计算 base dev/test assistant-only NLL 和 PPL。
6. 跑 fixed generation prompts，保存 base raw outputs。
7. 跑 sentinel probes，保存 base outputs。
8. 跑 JSON/tool probes，保存 base parse 结果。

注意：base dev/test NLL 不是自由生成，而是强制模型看完整 prompt + reference response，只在 response tokens 上算 loss。

## 5. SFT: 参照对应模型的 recipe；小步保存，用于 reload/export 检查

## 6. SFT 后重跑

训练后用同一 split、同一 prompts、同一 decoding config 重跑：

- dev/test assistant-only NLL 和 PPL。
- fixed generation prompts。
- sentinel probes。
- JSON/tool probes。
- checkpoint reload consistency。
- adapter/export/serving consistency，如果该阶段包含 serving。

不要在训练后更换 prompt、split、template 或 decoding 参数。

## 7. Dolly 阶段通过标准

必须通过：

- template 后样本可复查。
- loss mask 可视化正确。
- train loss 有稳定下降趋势。
- dev/test NLL 不明显劣化。
- fixed generation 不出现模板错误、停不下、明显拒答退化。
- sentinel seen 有明显学习信号。
- JSON/tool probes 格式错误率不高于 base。
- checkpoint reload 后 NLL/生成基本一致。

可接受：

- 强模型 Dolly NLL 只小幅改善或持平。
- fixed generation 没有明显变好。
- 小步数下 heldout sentinel exact match 不高。

不可接受：

- decoded labels 包含 user prompt。
- labels 全是 `-100` 或有效 label token 异常少。
- EOS 没进 loss 或 padding 进 loss。
- dev/test NLL 大幅劣化。
- 输出出现模板 token。
- JSON/tool 格式明显退化。
- checkpoint reload/export/serving 后行为明显不一致。

## 8. 最小报告内容

Dolly report 只需要摘要，原始证据引用 artifact 路径和 hash。

```markdown
# Dolly Probe Report

## Decision
- status:
- blockers:
- next step:

## Data
- manifest:
- split hash:
- templated sample preview:
- mask snapshot:

## Metrics
| metric | base | sft | delta |
|---|---:|---:|---:|
| dev NLL | | | |
| test NLL | | | |
| dev PPL | | | |
| test PPL | | | |
| JSON valid rate | | | |
| sentinel exact | | | |

## Consistency
- checkpoint reload:
- export/merge:
- serving:

## Artifact Links
- run card:
- environment summary:
- data manifest:
- trainable params summary:
- raw outputs:
- failure clusters:
......

```
