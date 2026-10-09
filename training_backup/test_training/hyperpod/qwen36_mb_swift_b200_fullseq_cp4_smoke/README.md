# qwen36 B200 Full-Sequence CP4 Smoke

This scaffold is a B200-only feasibility smoke for the current
`eks-cluster-hypd-0324` resource shape. It does not replace the exact B300
reproduction scaffold in:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro`

The goal is to keep full sequence length and the same dataset/optimizer shape,
while changing context parallelism enough to test whether current B200 nodes can
carry a 128k sequence within HyperPod's 8-accelerator B200 limit:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `NPROC_PER_NODE=8`
- `TP=2`
- `CP=4`
- `EP=8`
- 8 GPUs and 8 EFA devices on one `ml.p6-b200.48xlarge` node
- `TRAIN_ITERS_OVERRIDE=1`
- `MAX_TRAIN_SAMPLES=16`
- `SKIP_BASELINE_EVAL=true`
- `EXIT_AFTER_TRAIN=true`

Use this only as a full-sequence feasibility check. The exact reference job is
still `/tmp/test_training_evidence/21-train-swift-job.yaml`, which used B300,
8 GPUs, 16 EFA devices, and `CP=2`.

## 2026-06-16 Result

Run directory:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_fullseq_cp4_smoke/run/20260616T214422`

S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_fullseq_b200_cp4_smoke_20260616/`

The smoke completed with `status=diagnostic_train_completed`.

Observed highlights:

- Iteration `1/1` completed with 16 consumed samples.
- Train `lm loss: 3.103002E-01`.
- Validation `lm loss: 4.071109E-01`.
- Validation PPL: `1.502471E+00`.
- Max allocated memory was about `94.498 GB`.
- Checkpoint save at iteration 1 completed.

Validation note:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_b200_fullseq_cp4_smoke_validation_20260616.md`

Submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_fullseq_cp4_smoke
hyp validate
hyp create
```
