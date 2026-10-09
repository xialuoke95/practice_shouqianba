# BytePlus MLP Training Submission Runbook

Last checked: 2026-06-16 UTC.

This note records how to submit and debug BytePlus/Volc ML Platform custom
training tasks from this workspace, plus the operational traps found during the
35A3B no-think SFT work.

## Current Platform Facts

- Region: `ap-southeast-1`
- API host: `open.volcengineapi.com`
- Service name for signing: `ml_platform`
- API version used by local helper script: `2024-07-01`
- Resource group: `r-20260128180700-np4h5`
- Resource queue: `q-20260128181205-z9l2h`
- Queue name: `ml_verdent_2026`
- Training flavor: `ml.hpcpni3h.42xlarge`
- Training flavor GPU shape: `8 x NVIDIA-H100-SXM5-80GB`
- Queue capacity: `10 x ml.hpcpni3h.42xlarge` = `80 H100`
- Shared storage: `pfs-verdent`
- Vepfs id: `vepfs-apse1754289b494ca`
- In-container mount path: `/pfs-verdent`
- Preset training image:
  `mlp-ap-southeast-1.cr.bytepluses.com/preset-images/pytorch:2.7.1-cu12.8.1-py3.12-ubuntu22.04`

The currently used shared project root is:

```text
/pfs-verdent/guoyinuo/coding_agent_models
```

## Credential Setup

The local `~/.bashrc` may have a non-interactive-shell early return before the
Volc credentials. Plain `source ~/.bashrc` can therefore leave the variables
unset for scripts.

Use this pattern from the workspace when running `volc` or the OpenAPI helper:

```bash
eval "$(grep '^export HuoShan' ~/.bashrc)"
export VOLC_ACCESSKEY="$HuoShanAccessKeyID"
export VOLC_SECRETKEY="$HuoShanSecretAccessKey"
```

Do not print or commit the credential values. When inspecting service
definitions, redact any `Credential`, `AccessKey`, or `SecretAccessKey` fields
before saving output.

## Resource Queries

Queue-level summary:

```bash
eval "$(grep '^export HuoShan' ~/.bashrc)"
python3 scripts/query_mlp_resource_summary.py
```

The helper intentionally does not print secrets. It calls:

- `ListResourceGroups`
- `ListResourceQueues`

For exact queue quota, including allocated GPUs:

```bash
eval "$(grep '^export HuoShan' ~/.bashrc)"
python3 -c '
import json, sys
sys.path.append("scripts")
import query_mlp_resource_summary as q
r = q.unwrap(q.call("GetResourceQueue", {"Id": q.RESOURCE_QUEUE_ID}))
print(json.dumps({
    "QuotaCapability": r.get("QuotaCapability"),
    "QuotaAllocated": r.get("QuotaAllocated"),
    "SystemQuotaAllocated": r.get("SystemQuotaAllocated"),
}, ensure_ascii=False, indent=2))
'
```

Running and queued custom tasks:

```bash
eval "$(grep '^export HuoShan' ~/.bashrc)"
export VOLC_ACCESSKEY="$HuoShanAccessKeyID"
export VOLC_SECRETKEY="$HuoShanSecretAccessKey"
volc ml_task list --output json --limit 100 --showAllTasks
```

Task instances:

```bash
volc ml_task instance list --id <task-id> --output json
```

Development instances in the same queue:

```bash
volc ml_devinstance list --queueIds q-20260128181205-z9l2h --limit 100
```

The CLI's dev-instance and service commands often open a TUI and exit with
status `255` in non-interactive runs. The output can still be useful, but for
automation prefer JSON-capable task commands or signed OpenAPI calls through
`scripts/query_mlp_resource_summary.py`.

Service/deployment details can be queried through OpenAPI actions such as
`ListServices`, `GetService`, `ListDeployments`, and `GetDeployment`. Redact
credentials before persisting any response.

## Interpreting Free GPU

MLP quota can report free GPUs that are not usable as a contiguous training
worker. Our full training workers use whole `ml.hpcpni3h.42xlarge` instances,
so each worker needs a non-fragmented block of `8 H100`.

Example from 2026-06-16 UTC:

```text
Queue capability: 80 H100
Queue allocated:  70 H100
Arithmetic free:  10 H100
Usable whole workers for ml.hpcpni3h.42xlarge: only 1 x 8 H100
```

A job requesting `RoleReplicas: 2` on `ml.hpcpni3h.42xlarge` needs `16 H100`
as two complete workers. It will remain in `Queue` when only one whole
8-GPU worker is available, even if quota arithmetic shows more than 8 GPUs
free.

The `QuotaAllocated` number includes jobs, services, and dev instances. In the
same snapshot:

- Running training job: `6 x ml.hpcpni3h.42xlarge` = `48 H100`
- Development instances: about `8 H100`
- Online services: about `14 H100`
- Total allocated: `70 H100`

## Training Script Pattern

Keep the executable training script on PFS and make the task YAML entrypoint
point at that remote path. If `UserCodePath` is empty, the platform will not
upload local scripts for you.

Sync a local script:

```bash
scp scripts/<script>.sh \
  gpu_dev_huoshan:/pfs-verdent/guoyinuo/coding_agent_models/scripts/may/<script>.sh
```

Inside the script, write logs to PFS explicitly:

```bash
OUTPUT_DIR=/pfs-verdent/guoyinuo/coding_agent_models/output/<run-name>
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
```

Use MLP-provided distributed variables:

```bash
python -m torch.distributed.launch \
  --nproc_per_node "$MLP_WORKER_GPU" \
  --nnodes "$MLP_WORKER_NUM" \
  --node_rank "$MLP_ROLE_INDEX" \
  --master_addr "$MLP_WORKER_0_HOST" \
  --master_port "$MLP_WORKER_0_PORT" \
  -m swift.cli._megatron.sft \
  ...
```

Common runtime setup used by the 35A3B runs:

```bash
export HF_DATASETS_CACHE=/pfs-verdent/subingtian/.cache
export TMPDIR=/pfs-verdent/subingtian/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=2400

pip install "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git" -U
pip install git+https://github.com/NVIDIA/Megatron-LM.git
pip install "mcore-bridge>=1.1.0" -U
pip install "transformer_engine[pytorch]==2.16.0" --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation
pip install flash-linear-attention
```

The flash-attn and transformer-engine pins were added after runtime failures
and should stay explicit unless a new image is validated.

## Task YAML Pattern

Minimal shape for this queue:

```yaml
TaskName: "train_or_diag_name"
Description: "What this run validates"
Entrypoint: "bash /pfs-verdent/guoyinuo/coding_agent_models/scripts/may/<script>.sh"
Tags: []
UserCodePath: ""
RemoteMountCodePath: ""
Envs: []
ResourceQueueID: "q-20260128181205-z9l2h"
Framework: "PyTorchDDP"
TaskRoleSpecs:
    - RoleName: "worker"
      RoleReplicas: 2
      Flavor: "ml.hpcpni3h.42xlarge"
ActiveDeadlineSeconds: 864000
EnableTensorBoard: true
TensorBoardStorage:
    Type: "Vepfs"
    SubPath: "guoyinuo/coding_agent_models/output/runs"
Storages:
    - Type: "Vepfs"
      MountPath: "/pfs-verdent"
      VepfsId: "vepfs-apse1754289b494ca"
      SubPath: ""
ImageUrl: "mlp-ap-southeast-1.cr.bytepluses.com/preset-images/pytorch:2.7.1-cu12.8.1-py3.12-ubuntu22.04"
CacheType: "Cloudfs"
RetryOptions:
    EnableRetry: false
    MaxRetryTimes: 5
    IntervalSeconds: 120
    PolicySets: []
DiagOptions:
    - Name: "EnvironmentalDiagnosis"
      Enable: false
    - Name: "PythonDetection"
      Enable: false
    - Name: "LogDetection"
      Enable: false
    - Name: "VecclHangDetection"
      Enable: false
```

Generated examples:

- `docs/generated/byteplus_no_think_sampled_24gpu_task.yaml`
- `docs/generated/byteplus_no_think_sampled_16gpu_diag_task.yaml`
- `docs/generated/byteplus_no_think_sampled_16gpu_te_fused_ce_diag_task.yaml`
- `docs/generated/byteplus_no_think_sampled_16gpu_len32k_noxentfusion_diag_task.yaml`
- `docs/generated/byteplus_no_think_short32_16gpu_len32k_noxentfusion_diag_task.yaml`

## Submit And Monitor

Submit:

```bash
eval "$(grep '^export HuoShan' ~/.bashrc)"
export VOLC_ACCESSKEY="$HuoShanAccessKeyID"
export VOLC_SECRETKEY="$HuoShanSecretAccessKey"
volc ml_task submit --conf docs/generated/<task>.yaml --output json
```

Get status:

```bash
volc ml_task get --id <task-id> --output json
volc ml_task list --output json --limit 100 --showAllTasks
volc ml_task instance list --id <task-id> --output json
```

Fetch platform logs when available:

```bash
volc ml_task logs --task <task-id>
```

The more reliable logs are the per-node PFS files written by the script:

```bash
ssh gpu_dev_huoshan \
  "ls -ltr /pfs-verdent/guoyinuo/coding_agent_models/output/<run-name>"
ssh gpu_dev_huoshan \
  "grep -n -E 'OutOfMemory|CUDA out of memory|Traceback|End time|Successfully saved' \
  /pfs-verdent/guoyinuo/coding_agent_models/output/<run-name>/train_*.log"
```

TensorBoard output is written under:

```text
/pfs-verdent/guoyinuo/coding_agent_models/output/runs
```

Training metrics may also appear in run-local `logging.jsonl` files under the
versioned output directory.

## Known 35A3B No-Think Runs

Reference long run:

- Task: `t-20260615232657-xf862`
- Script:
  `/pfs-verdent/guoyinuo/coding_agent_models/scripts/may/train_full_48gpu_day_01_29_drop_l4_l5_no_trunc_marker_35a3b_gbs72.sh`
- Output:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/train_full_48gpu_day_01_29_drop_l4_l5_no_trunc_marker_35a3b_gbs72/v1-20260615-153104`
- Shape: `48 H100`, `GBS=72`, `max_length=262144`
- CE config: `cross_entropy_loss_fusion=True`,
  `cross_entropy_fusion_impl='te'`
- Evidence: reached at least iteration `605/816` and saved
  `checkpoint-600`; no CE OOM keyword was found in train logs.

Native fused CE diagnostic:

- Task: `t-20260616174115-zgqzd`
- Shape: `16 H100`, `max_length=262144`, `train_iters=2`
- CE config: `cross_entropy_loss_fusion=True`,
  `cross_entropy_fusion_impl='native'`
- Result: failed in `fused_vocab_parallel_cross_entropy`; the log showed an
  attempted `50.48 GiB` CUDA allocation for a FP32 buffer shaped like
  `(s0, 1, 124160)`.

No-fused-CE short diagnostic:

- Task: `t-20260616185520-58fd9`
- Shape: `16 H100`, `max_length=32768`, short 32-row dataset
- CE config: `cross_entropy_loss_fusion=False`
- Result: completed `2/2` steps and saved checkpoint.

TE fused CE diagnostic:

- Task: `t-20260617010911-tztfv`
- Shape: `16 H100`, `max_length=262144`, `train_iters=2`
- CE config: `cross_entropy_loss_fusion=True`,
  `cross_entropy_fusion_impl='te'`
- Status when submitted: `Queue`; it needed two whole 8-GPU workers while only
  one whole worker was available.

## Pitfalls And Lessons

- `UserCodePath: ""` means local code is not uploaded. Always sync the script
  to `/pfs-verdent/.../scripts/may/` before submitting a YAML that points to
  that path.
- The platform's CLI can enter interactive TUI mode for `ml_service` and
  `ml_devinstance`; use `--output json` where supported, or signed OpenAPI
  calls for automation.
- Queue free GPU is not the same as schedulable workers. For
  `ml.hpcpni3h.42xlarge`, reason in whole 8-GPU units.
- Services and dev instances consume the same queue quota. A training queue can
  look idle in custom-task list while quota is occupied by online services or
  dev instances.
- Always write logs to PFS from inside the script. Platform logs are useful but
  not enough for postmortems, and CLI access does not provide a general way to
  enter arbitrary training containers.
- Keep output directories unique or cleanly versioned. Reusing an output prefix
  can mix old logs, checkpoints, and TensorBoard data with a new attempt.
- Keep dependency pins in the script. The preset image alone was not enough to
  reproduce the working Megatron/SWIFT path.
- For the 35A3B no-think 256K path, `native` fused CE can OOM in the loss
  kernel. `te` fused CE matches the reference run but Megatron warns about
  known stability issues. Disabling fused CE is expected to preserve the loss
  objective but can change performance and tiny numerical details.
- A short `max_length=32768` diagnostic is not representative of full 256K
  data. The 1000-row no-think sample has a long tail, including samples above
  32K and near/above 256K by approximate token count.
- Do not store raw service OpenAPI responses without redaction; service role
  specs may include embedded credentials.

