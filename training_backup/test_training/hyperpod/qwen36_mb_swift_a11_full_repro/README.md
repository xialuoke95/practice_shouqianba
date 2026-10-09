# qwen36 Megatron Bridge HyperPod Full Reproduction

This scaffold is the `hyp create` form of the earlier bare Kubernetes Job:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

It preserves the full training recipe:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `CP=2`
- `EP=8`
- `ETP=1`
- `ml.p6-b300.48xlarge`
- 8 GPUs and 16 EFA devices

Only HyperPod admission compatibility changes are applied:

- pod `restartPolicy` is `Always`
- reserved user env names are represented as `QWEN36_*` and exported back to
  `JOB_NAME`, `NNODES`, and `NPROC_PER_NODE` in the shell command

Prerequisites:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply \
  -f /home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_prereqs.yaml

kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply \
  -f /tmp/test_training_evidence/qwen36-mb-overlay-scripts-r14.yaml
```

Submit when a schedulable B300 node is present:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro
hyp validate
hyp create
```

As of 2026-06-16 UTC, `eks-cluster-hypd-0324` has only B200 GPU nodes and
`Algorithm-training` has no GPU worker nodes, so this exact reproduction
manifest passes HyperPod admission dry-run but cannot currently schedule.
