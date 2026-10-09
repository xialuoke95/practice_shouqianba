# BytePlus SaaS SFT v0 Exact-47 Submission

## Task

- Task ID: `t-20260807020653-4r4cm`
- Task name: `train_saas_sft_v0_exact47_qwen36_27b_16gpu`
- Final state: `Success`
- Platform start: `2026-08-06T18:06:54Z`
- Platform end: `2026-08-06T18:49:06Z`
- Platform elapsed: 2,532 seconds (42m 12s)
- Both workers exited successfully with code 0
- Resources: `2 x ml.hpcpni3h.42xlarge` = 16 H100
- Model: Qwen3.6-27B full SFT

## Dataset

- Rows: 47 unique traces
- Selection: exact trace-ID membership in the renderer-approved dataset used by the completed Fireworks SaaS SFT v0 run
- Native no-think validation: passed
- Trainable assistant/tool-call messages: 2,621
- SHA256: `b8d936581953d8edf59d24906fa4b6a88911b0bec2da12183f2b8aa2ccaad139`

## Training shape

- Parallelism: TP4 / PP2 / CP2 / DP1
- Micro batch size: 1
- Global batch size: 2
- Epochs: 2
- Learning rate: `1e-5`, cosine decay, 5% warmup, minimum `1e-6`
- Max sequence length: 262,144
- Optimizer: Adam with CPU optimizer offload
- Checkpoint interval: 24 optimizer steps plus final checkpoint

## Verification status

- Local asset validator: passed
- Remote row count, SHA256, model directory, and shell syntax preflight: passed
- Platform task status: `Success`
- Training completed: 47/47 optimizer steps
- Fatal log scan: 0 matches across both worker logs for traceback, OOM, runtime error, or error markers
- Intermediate checkpoint: `checkpoint-24`
- Final checkpoint: `checkpoint-47`
- Final checkpoint contains 12/12 referenced safetensors shards, 1,184 indexed tensors, and no missing shard
- Indexed model weight size: 54,713,457,120 bytes
- Final checkpoint directory size: 54,733,754,624 bytes (about 51 GiB)
- `latest_checkpointed_iteration.txt`: `47`

## Training metrics

| Metric | Value |
| --- | ---: |
| Optimizer steps | 47 |
| Iteration time | 34m 23s |
| Final cumulative speed | 43.899 s/step |
| First-step loss | 0.670632 |
| Final-step loss | 0.061537 |
| First-to-final loss reduction | 90.82% |
| Overall mean loss | 0.200711 |
| Epoch 0 mean loss (23 steps) | 0.284825 |
| Epoch 1 mean loss (24 steps) | 0.120102 |
| Epoch mean loss reduction | 57.83% |
| First five steps mean loss | 0.311893 |
| Last five steps mean loss | 0.130133 |
| Min / max step loss | 0.031521 / 0.670632 |
| Mean grad norm | 3.013373 |
| Min / max grad norm | 1.006859 / 10.776342 |
| Peak reported GPU memory | 36.32 GiB |
| Final learning rate | 1e-6 |

These are training-only metrics. No validation split or manual heldout evaluation
was run, as requested, so they do not establish heldout task quality.

## Output

- Run directory: `/pfs-verdent/guoyinuo/coding_agent_models/output/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2_20260806/v0-20260806-181022`
- Final checkpoint: `/pfs-verdent/guoyinuo/coding_agent_models/output/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2_20260806/v0-20260806-181022/checkpoint-47`
- Metrics: `/pfs-verdent/guoyinuo/coding_agent_models/output/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2_20260806/v0-20260806-181022/logging.jsonl`
- Node logs: `/pfs-verdent/guoyinuo/coding_agent_models/output/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2_20260806/train_20260806_180709_node{0,1}.log`

The model was trained and checkpointed only. It was not deployed.
