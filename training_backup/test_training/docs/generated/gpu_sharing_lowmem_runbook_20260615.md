# GPU Sharing + Low-Memory SGLang Debug Runbook

Cluster: `hp-cluster-hypd-0324` / `eks-cluster-hypd-0324`

Target inference deployment:

```text
namespace:  glm51-pd-b200-p1d1
deployment: qwen36-swift-r13-sglang
current pod after low-memory rollout:
            qwen36-swift-r13-sglang-684c4d95f4-hg4gx
node:       hyperpod-i-0da428942dd5a9cd9
```

Baseline facts from 2026-06-15 before lowering SGLang memory:

```text
qwen36 requests: 8 GPU, 8 EFA, 120 CPU, 1200Gi memory
node capacity:   8 GPU, 8 EFA, 192 CPU, ~2Ti memory
actual CPU:      ~7.6 cores
actual memory:   ~908Gi node, ~318Gi qwen36 pod
GPU memory:      ~157-159Gi / 183Gi per B200
GPU util:        ~13-21%
SGLang setting:  --mem-fraction-static 0.82
```

## Applied Status

Time-slicing was applied on 2026-06-15 with:

```bash
scripts/apply_gpu_sharing_timeslicing.sh apply
```

The device plugin loaded this config:

```json
"sharing": {
  "timeSlicing": {
    "resources": [
      {
        "name": "nvidia.com/gpu",
        "devices": "all",
        "replicas": 2
      }
    ]
  }
}
```

The qwen36 node now advertises:

```text
nvidia.com/gpu capacity:    16
nvidia.com/gpu allocatable: 16
qwen36 allocated:            8
```

The plugin in this cluster does not add the
`nvidia.com/gpu.sharing-strategy` node label, so use node capacity and device
plugin logs as the source of truth.

The colocated smoke job completed successfully on the qwen36 node:

```text
namespace: gpu-sharing-debug
job:       qwen36-colocated-gpu-smoke
pod:       qwen36-colocated-gpu-smoke-cz9sg
node:      hyperpod-i-0da428942dd5a9cd9
status:    Complete, 1/1, duration 12s
```

It saw all 8 B200 GPUs and completed a small CUDA matmul loop on every GPU.

Qwen36 remained available after the device plugin rollout and after the smoke
job:

```text
deployment/qwen36-swift-r13-sglang READY 1/1
pod/qwen36-swift-r13-sglang-6b94df68f9-v69l7 Running, restarts 0
```

The smoke job above was run before the SGLang memory reduction, so its
`nvidia-smi` output still showed the original ~157-159GiB/GPU footprint.

After that canary passed, qwen36 was rolled to:

```text
--mem-fraction-static 0.70
deployment/qwen36-swift-r13-sglang READY 1/1
pod/qwen36-swift-r13-sglang-79cf99b4bd-qqqrd Running
```

Post-rollout status from the qwen36 container:

```text
GPU0 used/free/total MiB: 136594 / 46039 / 183359
GPU1 used/free/total MiB: 135977 / 46655 / 183359
GPU2 used/free/total MiB: 135977 / 46655 / 183359
GPU3 used/free/total MiB: 135977 / 46655 / 183359
GPU4 used/free/total MiB: 135977 / 46655 / 183359
GPU5 used/free/total MiB: 135977 / 46655 / 183359
GPU6 used/free/total MiB: 135977 / 46655 / 183359
GPU7 used/free/total MiB: 135081 / 47551 / 183359
```

That frees roughly 22GiB more GPU memory per B200 compared with the 0.82
baseline. The qwen36 pod CPU was around 7.5 cores and pod memory around 35GiB
after the rollout settled.

## Important Constraint

Lowering SGLang memory only reduces physical GPU memory use. It does not create
new Kubernetes GPU resources. A second training pod can be scheduled onto the
same GPUs only after NVIDIA device plugin GPU sharing is enabled.

The current `hyperpod-dependencies-nvidia-device-plugin` DaemonSet is running
without a config file. The scripts in this folder add a ConfigMap and point the
DaemonSet at it. This is a cluster-level scheduling change for all GPU nodes in
this EKS cluster, because the current installation has no per-node config
manager.

## Dry-Run

```bash
scripts/apply_gpu_sharing_timeslicing.sh dry-run
MEM_FRACTION=0.60 scripts/patch_qwen36_sglang_mem_fraction.sh dry-run
env KUBECONFIG=/tmp/eks-cluster-hypd-0324.kubeconfig \
  kubectl apply --dry-run=client -f docs/generated/qwen36_colocated_gpu_smoke_job.yaml
```

## Apply Order

1. Enable time-slicing with two logical shares per physical GPU:

```bash
scripts/apply_gpu_sharing_timeslicing.sh apply
```

Expected result on the qwen36 node:

```text
nvidia.com/gpu capacity/allocatable changes from 8 to 16
device plugin logs show sharing.timeSlicing replicas=2
```

2. Reduce qwen36 SGLang static memory fraction. The first applied canary used
   0.70:

```bash
MEM_FRACTION=0.70 scripts/patch_qwen36_sglang_mem_fraction.sh apply
```

This restarts the qwen36 deployment. Verify readiness and GPU memory with:

```bash
scripts/patch_qwen36_sglang_mem_fraction.sh status
```

3. Run a small colocated 8-GPU smoke job on the same node:

```bash
env KUBECONFIG=/tmp/eks-cluster-hypd-0324.kubeconfig \
  kubectl apply -f docs/generated/qwen36_colocated_gpu_smoke_job.yaml
env KUBECONFIG=/tmp/eks-cluster-hypd-0324.kubeconfig \
  kubectl logs -n gpu-sharing-debug job/qwen36-colocated-gpu-smoke -f
```

This smoke job intentionally does not request EFA. It proves local GPU sharing
and basic Torch CUDA execution, not multi-node EFA training.

The 2026-06-16 canary below supersedes the earlier "try 0.60 next" note. It
validated a lower SGLang footprint with extra serve flags:

```bash
MEM_FRACTION=0.10 \
DISABLE_CUDA_GRAPH=true \
CHUNKED_PREFILL_SIZE=4096 \
KV_CACHE_DTYPE=fp8_e4m3 \
scripts/patch_qwen36_sglang_mem_fraction.sh dry-run

MEM_FRACTION=0.10 \
DISABLE_CUDA_GRAPH=true \
CHUNKED_PREFILL_SIZE=4096 \
KV_CACHE_DTYPE=fp8_e4m3 \
scripts/patch_qwen36_sglang_mem_fraction.sh apply
```

## 2026-06-16 0.10 SGLang + Full SFT Canary

Latest validated qwen36 serve args:

```text
--mem-fraction-static 0.10
--disable-cuda-graph
--chunked-prefill-size 4096
--kv-cache-dtype fp8_e4m3
```

Deployment status after rollout:

```text
namespace:  glm51-pd-b200-p1d1
deployment: qwen36-swift-r13-sglang
pod:        qwen36-swift-r13-sglang-684c4d95f4-hg4gx
node:       hyperpod-i-0da428942dd5a9cd9
status:     Ready 1/1
```

Qwen-only GPU memory after the 0.10 rollout:

```text
GPU0 used/free/total MiB: 23270 / 159363 / 183359
GPU1 used/free/total MiB: 22654 / 159979 / 183359
GPU2 used/free/total MiB: 22654 / 159979 / 183359
GPU3 used/free/total MiB: 22654 / 159979 / 183359
GPU4 used/free/total MiB: 22654 / 159979 / 183359
GPU5 used/free/total MiB: 22654 / 159979 / 183359
GPU6 used/free/total MiB: 22654 / 159979 / 183359
GPU7 used/free/total MiB: 21758 / 160875 / 183359
```

The full SFT canary used:

```text
manifest:  docs/generated/qwen36_full_sft_100_colocated_b200_job.yaml
namespace: glm51-pd-b200-p1d1
job:       qwen36-full-sft-100-colocated-b200-r7
pod:       qwen36-full-sft-100-colocated-b200-r7-vh5tq
node:      hyperpod-i-0da428942dd5a9cd9
status:    Complete, 1/1, duration about 14m
```

The run was a clean canary, not a resume:

```text
Checkpoint file not found in load directory .../checkpoints-r7.
Attempting to finetune with checkpoint in .../base_checkpoints/...
loading distributed checkpoint .../base_checkpoints/... at iteration 0
Starting training loop at iteration 0
```

Training result:

```text
iterations:        7/7
final train loss:  1.014585E+00
validation loss:   1.185183E+00
skipped/nan iters: 0 / 0
status summary:    "completed"
```

Observed 10s-sampled qwen+training GPU peak during the canary:

```text
GPU0 max used/free MiB: 166527 / 16106
GPU1 max used/free MiB: 163527 / 19106
GPU2 max used/free MiB: 163687 / 18946
GPU3 max used/free MiB: 163547 / 19086
GPU4 max used/free MiB: 163867 / 18766
GPU5 max used/free MiB: 163527 / 19106
GPU6 max used/free MiB: 163667 / 18966
GPU7 max used/free MiB: 162491 / 20142
```

After the training pod completed, qwen36 returned to the same 0.10 inference
baseline:

```text
deployment/qwen36-swift-r13-sglang READY 1/1
qwen pod memory: about 31Gi
qwen CPU:        about 7.4 cores
GPU0 used/free/total MiB: 23270 / 159363 / 183359
GPU1-6 used/free/total MiB: 22654 / 159979 / 183359
GPU7 used/free/total MiB: 21758 / 160875 / 183359
```

Operational lessons from this canary:

- A job that reuses `CHECKPOINT_SAVE="${NODE_ROOT}/checkpoints"` can silently
  resume an older short canary and produce a false pass. The invalid r6 run did
  exactly that by loading `iter_0000007` from the previous checkpoint. Use a
  fresh save directory such as `checkpoints-r7` for every clean canary, or
  explicitly remove the old checkpoint directory before submitting.
- Treat "loaded base checkpoint at iteration 0" plus "Starting training loop at
  iteration 0" as the minimum evidence for a clean full-SFT canary.
- Lowering SGLang memory to 0.10 is enough for this 100-sample, 8-GPU, full SFT
  canary, but the margin is not huge. The lowest sampled free memory was about
  16GiB on GPU0. Re-test before increasing sequence length, micro-batch size,
  model size, number of concurrent inference requests, or enabling features
  that increase KV/cache/workspace memory.
- `nvidia-smi --query-gpu ... -lms 10000` run through `kubectl exec` can exit
  with code 137 when the training pod exits. If the Kubernetes Job is
  `Complete` and logs show the final status JSON, this is not by itself a
  training failure.
- The final `status.json` may not be visible from the qwen pod's filesystem
  path even when it was written by the training pod. Use the completed training
  pod logs, or read the artifact while the training pod is still running.
- Keep checking qwen readiness and qwen-only GPU memory after the training pod
  exits. The expected 0.10 baseline is about 21.2-22.7GiB per B200.

Useful verification commands:

```bash
scripts/patch_qwen36_sglang_mem_fraction.sh status

env KUBECONFIG=/tmp/eks-cluster-hypd-0324.kubeconfig \
  kubectl get job qwen36-full-sft-100-colocated-b200-r7 \
  -n glm51-pd-b200-p1d1 -o wide

env KUBECONFIG=/tmp/eks-cluster-hypd-0324.kubeconfig \
  kubectl logs -n glm51-pd-b200-p1d1 \
  qwen36-full-sft-100-colocated-b200-r7-vh5tq --tail=120
```

## Rollback

Restore qwen36 to its previous `--mem-fraction-static 0.82`:

```bash
scripts/patch_qwen36_sglang_mem_fraction.sh rollback
```

Disable the device plugin config file:

```bash
scripts/apply_gpu_sharing_timeslicing.sh rollback
```

The rollback leaves the ConfigMap/volume mount in place but removes the
`CONFIG_FILE` env var, so the plugin returns to its default non-sharing config
after rollout.

## Risk Notes

- CUDA time-slicing does not isolate GPU memory. Qwen36 and the training smoke
  job can still OOM each other if SGLang memory is not reduced enough.
- Requesting multiple shared `nvidia.com/gpu` resources does not provide a
  guaranteed proportional compute share. CUDA time-slicing interleaves clients.
- MPS is closer to a true memory/compute fraction, but NVIDIA marks device
  plugin MPS support as experimental in this plugin family. Time-slicing is the
  lower-friction first canary.
- Existing qwen36 keeps using the devices it already has. The sharing change is
  mainly for newly scheduled pods.
