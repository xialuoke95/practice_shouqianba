# Qwen36 Megatron Bridge Full Reproduction Readiness

Date: 2026-06-16 UTC.

## Desired Reproduction Target

The effect-reproduction target is the earlier bare Kubernetes Job:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

That Job ran:

- `metadata.name: qwen36-mb-swift-aligned-a11`
- `metadata.namespace: dolly-probe-training`
- `node.kubernetes.io/instance-type: ml.p6-b300.48xlarge`
- 8 GPUs
- 16 EFA devices
- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `CP=2`
- `EP=8`
- `ETP=1`
- `RUN_ID=qwen36-mb-overlay-aligned-fullsft-20260602-211523`
- `JOB_NAME=qwen36_mb_swift_aligned_full_a11`

## HyperPod CLI Form

Full-reproduction scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro`

The scaffold is intended to be submitted with:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro
/home/ubuntu/.local/bin/hyp validate
/home/ubuntu/.local/bin/hyp create
```

The scaffold preserves the recipe and B300 resource shape from the bare Job.
The only compatibility changes are:

- `restartPolicy: Always`, required by the current HyperPod training operator.
- `JOB_NAME`, `NNODES`, and `NPROC_PER_NODE` are represented as `QWEN36_*`
  env vars and exported back in the shell command, because those names are
  reserved by HyperPod admission.

Validation performed:

- `/home/ubuntu/.local/bin/hyp validate` passed for the scaffold.
- The scaffold-rendered `HyperPodPyTorchJob` passed server-side dry-run on
  `eks-cluster-hypd-0324`.
- The scaffold-rendered `HyperPodPyTorchJob` passed server-side dry-run on
  `Algorithm-training`.

## Current Cluster Readiness

`eks-cluster-hypd-0324`:

- Operator/CRD: present and accepting the full-repro CR.
- Namespace/PVC/ConfigMap: present in `dolly-probe-training`.
- GPU nodes currently visible: seven `ml.p6-b200.48xlarge` nodes.
- B300 nodes currently visible: none.
- Exact full reproduction cannot schedule there at the moment because the
  target manifest requires `ml.p6-b300.48xlarge` and 16 EFA devices.

`Algorithm-training`:

- Operator/CRD: present and accepting the full-repro CR.
- Namespace/PVC/ConfigMap: present in `dolly-probe-training`.
- Nodes currently visible: one `ml.c5.2xlarge` node.
- B300 nodes currently visible: none.
- Exact full reproduction cannot schedule there at the moment.

## Related Runtime Evidence

The `eks-cluster-hypd-0324` B200 smoke path has been validated separately:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_smoke_validation_20260616.md`

That smoke completed one train iteration at `SEQ_LENGTH=16384` with
`status=diagnostic_train_completed`. It proves the HyperPod CLI/operator/S3
mount/checkpoint import/tokenizer patch/train-launch path, but it is not the
full training-effect reproduction.

An earlier B200 smoke that preserved `SEQ_LENGTH=128000` reached the first
backward pass and failed with a Triton CUDA OOM. This is why exact
effect-reproduction should wait for B300 capacity or use an explicitly designed
B200 recipe rather than silently changing the training recipe.

A later B200 full-sequence feasibility smoke completed successfully with
`SEQ_LENGTH=128000` after explicitly changing the B200 recipe to `CP=4`:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_b200_fullseq_cp4_smoke_validation_20260616.md`

That run completed one train iteration, saved a checkpoint, ran validation, and
wrote `status=diagnostic_train_completed`. It is useful evidence that the
current B200 cluster can carry a full 128k sequence with a B200-adjusted
parallelism plan, but it is still not the exact B300 `CP=2` reproduction target.

An attempted single-node B200 `CP=8` variant would have required 16
accelerators. HyperPod CLI rejected that request because its B200 instance
capacity table allows 8 accelerators for `ml.p6-b200.48xlarge`, even though the
kubelet currently advertises 16 `nvidia.com/gpu` resources on those nodes.

## Next Gate

Before submitting the full reproduction, verify at least one schedulable B300
node exists in the target cluster:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  get nodes -l node.kubernetes.io/instance-type=ml.p6-b300.48xlarge
```

If B300 capacity is added to `Algorithm-training`, use the equivalent
`/tmp/eks-algorithm-training.kubeconfig` check first, then point `hyp` at that
cluster context before running `hyp create`.

The current requirement-by-requirement status is tracked in:

`/home/ubuntu/workspace/test_training/docs/generated/hyperpod_goal_completion_audit_20260616.md`
