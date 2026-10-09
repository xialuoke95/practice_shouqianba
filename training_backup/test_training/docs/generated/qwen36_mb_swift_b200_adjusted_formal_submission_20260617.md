# Qwen36 Megatron Bridge B200 Adjusted Formal Submission

Date: 2026-06-17 UTC.

## Why This Was Not Submitted Earlier

The previous online target was to validate the SageMaker HyperPod CLI path, the
online management visibility, and the B200-adjusted training chain without
occupying a full B200 node for a long run. That is why the prior job used
bounded smoke settings:

- `TRAIN_ITERS_OVERRIDE=2`
- `MAX_TRAIN_SAMPLES=32`
- `SKIP_BASELINE_EVAL=true`
- `EXIT_AFTER_TRAIN=true`

The formal run below removes those smoke bounds and keeps the B200-adjusted
recipe running through the normal script path.

## Submitted Job

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal`

Run directory:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal/run/20260617T025010`

HyperPodPyTorchJob:

`dolly-probe-training/qwen36-mb-swift-b200-adjusted-formal`

S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_b200_adjusted_formal_20260617/`

Submission result:

- `/home/ubuntu/.local/bin/hyp validate` passed.
- `/home/ubuntu/.local/bin/hyp create` submitted successfully.
- `hyp list hyp-pytorch-job -n dolly-probe-training` showed the job as
  `Created`.
- Kubernetes created the `HyperPodPyTorchJob` CR and child Deployment.
- The first pod was `Running`, `Ready`, and had `0` restarts at the initial
  check.

Initial pod:

`qwen36-mb-swift-b200-adjusted-formal-pod-79db45bfb6-j9ph6`

Initial node:

`hyperpod-i-0cd50aacaabb9612d`

## Recipe Shape

Reference bare Kubernetes Job:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

The reference was B300-specific: `ml.p6-b300.48xlarge`, 8 GPU, 16 EFA, and
`CP=2`. This formal run targets current B200 capacity instead:

- `instance_type=ml.p6-b200.48xlarge`
- 8 GPU
- 8 EFA
- `CP=4`
- memory request/limit `1800Gi/1900Gi`

Preserved recipe shape:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `EP=8`
- `ETP=1`
- `EPOCHS=2`
- `DATASET_KEY=sft_opus_sonnet_swift_mbjson_v2`
- Image `nvcr.io/nvidia/nemo:26.04.00`

Formal behavior:

- No `TRAIN_ITERS_OVERRIDE`.
- No `MAX_TRAIN_SAMPLES`.
- No `EXIT_AFTER_TRAIN`.
- Baseline evaluation is not skipped.
- With the current dataset manifest, this computes `TRAIN_ITERS=1868`,
  baseline train eval iters `934`, and baseline test eval iters `105`.

## Initial Runtime Evidence

Checked at `2026-06-17T02:54:43Z` and again while the pod age was about
5 minutes:

- HyperPodPyTorchJob CR: present.
- Deployment: `1/1`.
- Pod: `Running`, `Ready`, `0` restarts.
- Scheduled on B200 node `hyperpod-i-0cd50aacaabb9612d`.
- S3 contained startup and preparation artifacts:
  - `logs/node0_01_overlay_runtime.log`
  - `logs/node0_02_dataset_helper_build.log`
  - `logs/node0_03_tokenizer_patch.log`
  - `run_context.json`
  - `base_import/node0/base_checkpoint_import_manifest.json`
  - `baseline_train_eval/baseline_eval_launch_manifest.json`
  - `tokenizer_patch_manifest.json`

Recent container stdout showed:

- Overlay source patching completed.
- Megatron Bridge/Core editable install completed.
- Transformer Engine attention patch applied.
- Dataset helper build completed.
- `run_context.json` written to the S3 root.
- Baseline train eval launch manifest was written.
- The latest stdout showed optimizer/scheduler setup followed by loading the
  local distributed base checkpoint for finetuning/baseline eval.

At the latest check, the job had entered baseline evaluation and had not yet
emitted formal training iteration lines.

## Training History Visibility

Checked again on 2026-06-17 UTC while the pod age was about 149 minutes:

- The Kubernetes pod was still `Running`/`Ready` with `0` restarts.
- Container stdout showed baseline train eval progressing through
  `Evaluating iter 503/934` to `Evaluating iter 622/934`.
- `/home/ubuntu/.local/bin/hyp list hyp-pytorch-job -n dolly-probe-training`
  still showed the job, but with status `Created`.
- The `HyperPodPyTorchJob` CR status had `Created=True` and
  `PodsRunning=False`.
- The podmanager log repeatedly failed to call
  `http://10.90.19.237:8080/status` with `connection refused`.

This means the training workload is alive, but the HyperPod Training Operator
does not see a ready HyperPod Elastic Agent status endpoint inside the worker
pod. The likely cause is that this custom manifest overrides
`container.command` to run the training script directly, so no HPEA status
server is running on port `8080`. A UI such as Training History that depends on
operator/HPEA status or metrics can therefore be empty even while the raw
Kubernetes workload is running.

For a future run that must appear in Training History, submit through a wrapper
that preserves the official HyperPod Elastic Agent lifecycle/status endpoint
instead of directly replacing the worker command, or explicitly add a compatible
agent/status integration.

## Stop Options

Stop completely and remove the HyperPod job:

```bash
/home/ubuntu/.local/bin/hyp delete hyp-pytorch-job \
  --job-name qwen36-mb-swift-b200-adjusted-formal \
  -n dolly-probe-training
```

Release GPU/EFA while keeping the HyperPodPyTorchJob CR visible:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  scale deployment/qwen36-mb-swift-b200-adjusted-formal-pod \
  -n dolly-probe-training --replicas=0
```

The formal wrapper intentionally sleeps after a successful script exit. This
avoids the HyperPod Deployment restart loop observed in the bounded online run.
After a successful formal completion, scale the Deployment to `0` or delete the
job to release B200 resources.
