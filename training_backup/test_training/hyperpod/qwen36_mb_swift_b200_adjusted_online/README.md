# qwen36 B200 Adjusted Online HyperPod Run

This scaffold is the B200 adjusted online target for the current goal:

- Submit through SageMaker HyperPod CLI.
- Run on current `eks-cluster-hypd-0324` B200 capacity.
- Keep the training chain and recipe as close as practical to the reference
  bare Kubernetes Job.
- Keep the job visible through HyperPod/Kubernetes management.

Reference bare Kubernetes Job:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

The reference was B300-specific: `ml.p6-b300.48xlarge`, 8 GPU, 16 EFA, and
`CP=2`. The B200 adjusted recipe intentionally changes only the parts needed to
fit current B200 capacity:

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
- dataset `sft_opus_sonnet_swift_mbjson_v2`
- image `nvcr.io/nvidia/nemo:26.04.00`

Online verification bounds:

- `TRAIN_ITERS_OVERRIDE=2`
- `MAX_TRAIN_SAMPLES=32`
- `SKIP_BASELINE_EVAL=true`
- `EXIT_AFTER_TRAIN=true`

These bounds are for online submission and management-platform validation. The
previous B200 CP4 smoke already proved this shape can complete a full 128k
iteration.

Submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_online
/home/ubuntu/.local/bin/hyp validate
/home/ubuntu/.local/bin/hyp create
```

Expected S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_b200_adjusted_online_20260617/`

## 2026-06-17 Result

Run directory:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_online/run/20260617T020736`

The online run completed with:

- `status=diagnostic_train_completed`
- iteration `2/2`
- train `lm loss` at iteration 2: `3.439665E-01`
- validation `lm loss: 4.015621E-01`
- validation PPL: `1.494157E+00`
- checkpoint saved at iteration 2

The HyperPodPyTorchJob CR is retained for management visibility. The child
Deployment was scaled to `0` after completion to release B200 GPU/EFA resources,
because the operator's Deployment-based workload restarted the completed
container under `restartPolicy: Always`.

Validation note:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_b200_adjusted_online_validation_20260617.md`
