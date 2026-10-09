# Qwen36 Megatron Bridge B200 Full-Sequence CP4 Smoke Validation

Date: 2026-06-16 UTC.

## Scope

This run validated that the current `eks-cluster-hypd-0324` B200 capacity can
execute the Qwen36 Megatron Bridge training path at the original full sequence
length for a one-iteration smoke.

It is not the exact full reproduction target. The exact reference remains:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

That bare Kubernetes Job used `ml.p6-b300.48xlarge`, 8 GPUs, 16 EFA devices,
and `CP=2`. The B200 smoke used current available B200 resources, 8 GPUs, 8
EFA devices, and `CP=4`.

## Submitted Scaffold

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_fullseq_cp4_smoke`

Rendered run:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_fullseq_cp4_smoke/run/20260616T214422`

Job:

`qwen36-mb-swift-fullseq-b200-cp4-smoke`

Namespace:

`dolly-probe-training`

S3 run root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_fullseq_b200_cp4_smoke_20260616/`

## Recipe Shape

Preserved from the target recipe:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `EP=8`
- `ETP=1`
- Base model `Qwen/Qwen3.6-35B-A3B`
- Image `nvcr.io/nvidia/nemo:26.04.00`
- Dataset key `sft_opus_sonnet_swift_mbjson_v2`

Smoke overrides:

- `TRAIN_ITERS_OVERRIDE=1`
- `MAX_TRAIN_SAMPLES=16`
- `SKIP_BASELINE_EVAL=true`
- `EXIT_AFTER_TRAIN=true`

B200-specific difference:

- `CP=4` instead of the B300 target's `CP=2`
- 8 EFA devices instead of 16 EFA devices
- `node.kubernetes.io/instance-type=ml.p6-b200.48xlarge`

## Result

The run completed successfully and wrote:

`status=diagnostic_train_completed`

Status evidence:

`/tmp/qwen36_mb_swift_b200_cp4_overlay_sft_status.json`

Key status fields:

- `seq_length: 128000`
- `cp: 4`
- `tp: 2`
- `ep: 8`
- `nnodes: 1`
- `nproc_per_node: 8`
- `global_batch_size: 16`
- `micro_batch_size: 1`
- `train_iters_override: "1"`
- `max_train_samples: "16"`

Training log evidence:

`/tmp/qwen36_mb_swift_b200_cp4_node0_20_train.log`

Important log lines:

- Base checkpoint loaded successfully.
- Training loop started at iteration 0.
- Iteration `1/1` completed with 16 consumed samples.
- Step time: `284.07s`.
- Train `lm loss: 3.103002E-01`.
- Grad norm: `2.214`.
- Skipped iterations: `0`.
- NaN iterations: `0`.
- Max allocated memory was about `94.498 GB`.
- Checkpoint saved at iteration 1.
- Validation `lm loss: 4.071109E-01`.
- Validation PPL: `1.502471E+00`.

## Cleanup

The HyperPod job was deleted after evidence collection.

Final check:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  get hyperpodpytorchjobs,pods -n dolly-probe-training
```

Result:

`No resources found in dolly-probe-training namespace.`

## Interpretation

This validates the HyperPod CLI submission path, operator reconciliation,
B200 scheduling, S3 PVC/ConfigMap dependencies, checkpoint import, tokenizer
patch, full 128k sequence training step, checkpoint save, validation, S3 output,
and cleanup.

It does not close the exact reproduction gate because the original target used
B300 capacity with `CP=2` and 16 EFA devices. Exact reproduction should be run
from:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro`

once a schedulable `ml.p6-b300.48xlarge` node is available, or after an
explicit decision to accept a B200-specific recipe as the new target.
