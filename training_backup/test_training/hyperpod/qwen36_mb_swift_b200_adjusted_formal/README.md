# qwen36 B200 Adjusted Formal HyperPod Run

This scaffold submits the formal B200-adjusted training run through SageMaker
HyperPod CLI.

Reference bare Kubernetes Job:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

The reference was B300-specific: `ml.p6-b300.48xlarge`, 8 GPU, 16 EFA, and
`CP=2`. This formal B200 run keeps the training recipe as close as practical and
adjusts only the B200 capacity-sensitive parts:

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

Formal behavior:

- No `TRAIN_ITERS_OVERRIDE`.
- No `MAX_TRAIN_SAMPLES`.
- No `EXIT_AFTER_TRAIN`.
- Baseline evaluation is not skipped.
- For the current dataset manifest this computes `TRAIN_ITERS=1868`.
- The wrapper sleeps after successful script completion to avoid a HyperPod
  Deployment restart loop.

Submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal
/home/ubuntu/.local/bin/hyp validate
/home/ubuntu/.local/bin/hyp create
```

Stop completely:

```bash
/home/ubuntu/.local/bin/hyp delete hyp-pytorch-job \
  --job-name qwen36-mb-swift-b200-adjusted-formal \
  -n dolly-probe-training
```

Release GPU/EFA but keep the HyperPodPyTorchJob CR visible:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  scale deployment/qwen36-mb-swift-b200-adjusted-formal-pod \
  -n dolly-probe-training --replicas=0
```

Expected S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_b200_adjusted_formal_20260617/`

## Submission Status

Submitted on 2026-06-17 UTC through:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal/run/20260617T025010`

Initial status:

- HyperPodPyTorchJob `dolly-probe-training/qwen36-mb-swift-b200-adjusted-formal`
  was created successfully.
- Child Deployment was `1/1`.
- Pod `qwen36-mb-swift-b200-adjusted-formal-pod-79db45bfb6-j9ph6` was
  `Running`, `Ready`, and at `0` restarts.
- Initial node was B200 node `hyperpod-i-0cd50aacaabb9612d`.
- Startup logs, base checkpoint import manifest, tokenizer patch log, tokenizer
  patch manifest, baseline train eval launch manifest, and `run_context.json`
  were written under the expected S3 root.

Detailed evidence:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_b200_adjusted_formal_submission_20260617.md`
