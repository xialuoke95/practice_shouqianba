# qwen36 Megatron Bridge HyperPod Smoke

This scaffold submits a `HyperPodPyTorchJob` through `hyp create` while keeping
the pod spec aligned with the earlier bare Kubernetes Job
`/tmp/test_training_evidence/21-train-swift-job.yaml`.

The smoke config targets `eks-cluster-hypd-0324` B200 nodes and intentionally
sets `SEQ_LENGTH=16384`, `SKIP_BASELINE_EVAL=true`,
`TRAIN_ITERS_OVERRIDE=1`, `MAX_TRAIN_SAMPLES=16`, and
`EXIT_AFTER_TRAIN=true` to validate submission, scheduling, S3 mount, script
execution, logging, one train iteration, checkpoint save, and validation before
running the full SFT.

The earlier `SEQ_LENGTH=128000` smoke preserved the original B300 recipe more
closely, but failed on B200 during the first backward pass with
`RuntimeError: Triton Error [CUDA]: out of memory`. The 16k smoke completed
with `status=diagnostic_train_completed`, but is therefore a
submission/runtime validation, not a full training-effect reproduction.

Validation record:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_smoke_validation_20260616.md`

Prerequisites:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply \
  -f /home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_prereqs.yaml

kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply \
  -f /tmp/test_training_evidence/qwen36-mb-overlay-scripts-r14.yaml
```

Render and submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_hypd_smoke
hyp validate
hyp create
```
