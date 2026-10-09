# Kimi 2.7 与 GLM 5.2 训练准备验证

## 结论

两套 10 条 smoke 数据及其 renderer adapter 已准备完成，本机闭环状态为 `PASS`。随后已于 2026-07-20 分别完成一次 Kimi 2.7 和 GLM 5.2 full-parameter B300 真实训练。

两个输出模型均为 `READY`，W&B run 均为 `finished`，训练资源均已删除释放。真实训练结果汇总见 `training_10_b300_completion_report.json`。

## 公共入口

- 通用训练入口：`/home/ubuntu/workspace/fireworks/data_alignment/run_alignment_sft.py`
- 公共训练实现：`/home/ubuntu/workspace/fireworks/data_alignment/alignment_sft_lib.py`
- 公共 converter：`/home/ubuntu/workspace/fireworks/data_alignment/core/convert_core_aligned.fixed_0702.py`
- readiness 验证：`/home/ubuntu/workspace/fireworks/data_alignment/validate_model_adapter_readiness.py`
- 机器可读报告：`/home/ubuntu/workspace/fireworks/data_alignment/model_adapter_readiness_report.json`

## Kimi 2.7

- Adapter：`renderer_adapters/kimi_k27_single_datum.py`
- YAML：`configs/kimi_k27_code_full_sft.yaml`
- 10 条数据：`/home/ubuntu/.remotelab/workspace/fireworks-runs/kimi-k27-adapter-validation-10-ready/kimi_k27_code_sft_10.jsonl`
- 完整产物目录：`/home/ubuntu/.remotelab/workspace/fireworks-runs/kimi-k27-adapter-validation-10-ready`
- Base model：`accounts/fireworks/models/kimi-k2p7-code`，API 状态 `READY`
- Shape：`accounts/fireworks/trainingShapes/kimi-k2p7-coder-256k/versions/jpk93g11`
- Context 上限：`262144`
- 数据覆盖：10 条均含 tools，其中 3 条含图片，3/3 图片成功转成 multimodal content 并完成 render
- 结果：10 条输入、10 个 datum、2 个 batch、0 renderer error、0 tool linkage error
- Parity：adapter 与 cookbook 官方 `KimiK27CodeRenderer` token exact；只把已具备的 preserve-history 行为声明为 extension property，从而避免按 user turn 拆分

## GLM 5.2

- Adapter：`renderer_adapters/glm52_alignment.py`
- YAML：`configs/glm52_fp8_full_sft.yaml`
- 10 条数据：`/home/ubuntu/.remotelab/workspace/fireworks-runs/glm52-adapter-validation-10-ready/glm52_fp8_sft_10.jsonl`
- 完整产物目录：`/home/ubuntu/.remotelab/workspace/fireworks-runs/glm52-adapter-validation-10-ready`
- Base model：`accounts/fireworks/models/glm-5p2-fp8`，API 状态 `READY`
- Shape：`accounts/fireworks/trainingShapes/glm-5p2-200k/versions/rfpesks1`
- Context 上限：`204736`
- 数据覆盖：10 条 text-only tools 数据，包含 1/2/4/6/9 user-turn case
- 结果：10 条输入、10 个 datum、2 个 batch、0 renderer error、0 unresolved tool call、0 orphan result
- 样本 29 最后一条是合法的 terminal `submit_plan` tool call，无后续 tool result；已单独审计，不视为 linkage error
- Parity：adapter 与 GLM 5.2 HF chat template 的 `clear_thinking=False` + tools generation prompt token exact

## Adapter 边界

- Kimi adapter 只负责 preserve-history extension contract；tools、图片和 Kimi 格式继续复用 cookbook 官方 renderer。
- GLM adapter 只负责 preserve historical thinking、GLM 官方 tools prefix 和 extension contract。
- `loss -> trainable`、tool call schema、tool response ID/name、图片 content parts 由公共 converter 负责。
- 自定义 adapter 使用 `render_workers: 1`，保证 YAML 注册的 renderer 在 cookbook 官方 SFT DataLoader 同一进程可见；不使用 monkey patch。
- Full-param、batch size、shape、学习率和 checkpoint 参数均由模型 YAML 配置。

## 验证命令

```bash
cd /home/ubuntu/workspace/fireworks/data_alignment
PY=/home/ubuntu/.remotelab/workspace/fw-ai-cookbook/training/.venv/bin/python

$PY run_alignment_sft.py --config configs/kimi_k27_code_full_sft.yaml --dry-run
$PY run_alignment_sft.py --config configs/glm52_fp8_full_sft.yaml --dry-run
$PY validate_model_adapter_readiness.py
```

## 训练记录

- W&B 认证会从环境变量或 `~/.bashrc` 中读取，当前已验证为 `online` 且 entity 可解析。
- Kimi 默认 project：`fireworks-kimi-k27-code-sft`。
- GLM 默认 project：`fireworks-glm52-fp8-sft`。
- runner 会在训练前固定 `WANDB_RUN_ID`，并把 dashboard 地址写入每次产物目录的 `wandb_run.json`；训练结束后 Cookbook 调用 `wandb.finish()` 不会丢失该链接。
- 本地始终保留 `sft_logs/`、`metrics.jsonl`、`status.json`、`metadata.json`、`render_samples.jsonl`、`output_model.json`；失败时另写 `run_failure.json`，资源清理结果写入 `cleanup_result.json`。

## 正式提交命令

以下命令会创建付费资源；不要附加 `--dry-run`：

```bash
cd /home/ubuntu/workspace/fireworks/data_alignment
PY=/home/ubuntu/.remotelab/workspace/fw-ai-cookbook/training/.venv/bin/python

$PY run_alignment_sft.py --config configs/kimi_k27_code_full_sft.yaml
$PY run_alignment_sft.py --config configs/glm52_fp8_full_sft.yaml
```

## 10 条 B300 实际训练结果

- Kimi 2.7：2 steps，checkpoint `step-2`，输出模型 `accounts/yuyu/models/kimi-k27-code-full-sft-20260720-193117`，状态 `READY`。
- GLM 5.2：2 steps，checkpoint `step-2`，输出模型 `accounts/yuyu/models/glm52-fp8-full-sft-20260720-193117`，状态 `READY`。
- 两个 trainer job 均已转为 `JOB_STATE_DELETED`，不再产生 B300 费用；checkpoint 保留至 2026-08-19。

## 非阻塞项

- 当前没有设置 `HF_TOKEN`；缓存的公开 tokenizer 可正常使用，只影响 Hugging Face 下载限流。
- 当前环境没有 `sentencepiece`；Kimi tokenizer 已通过 TikToken fallback 完成 token parity 和 DataLoader 验证。
- 10 条数据适合验证训练链路，不足以评估正式训练收益；扩大训练数据量不需要新增 adapter，只需调整 YAML 的 selection/limit 和数据源。
