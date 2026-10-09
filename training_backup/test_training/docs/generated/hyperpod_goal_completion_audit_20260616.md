# HyperPod Training Goal Completion Audit

Date: 2026-06-16 UTC.

## Objective

User objective:

1. Run the submission path through SageMaker HyperPod CLI on
   `eks-cluster-hypd-0324`, including resource changes, namespace, PVC,
   ConfigMap, and testing.
2. Reproduce the training effect from the earlier bare Kubernetes submission
   and state which bare Kubernetes Job is the reference.

## Reference Bare Kubernetes Job

Authoritative reference:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

Reference shape:

- Job name: `qwen36-mb-swift-aligned-a11`
- Namespace: `dolly-probe-training`
- Image: `nvcr.io/nvidia/nemo:26.04.00`
- Instance type: `ml.p6-b300.48xlarge`
- GPU request/limit: 8
- EFA request/limit: 16
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
- ConfigMap: `qwen36-mb-overlay-scripts-r14`
- PVC: `s3-verdent-tech-use1`

## Current Evidence

### Submission path

Status: proven for HyperPod CLI and operator on `eks-cluster-hypd-0324`.

Evidence:

- `dolly-probe-training` namespace exists.
- PVC `s3-verdent-tech-use1` exists and was used by smoke jobs.
- ConfigMap `qwen36-mb-overlay-scripts-r14` exists and was used by smoke jobs.
- HyperPod training operator and
  `hyperpodpytorchjobs.sagemaker.amazonaws.com` CRD are installed.
- `hyp validate` for the full-repro scaffold reports
  `config.yaml is valid` and `Configuration validated successfully`.
- `docs/generated/qwen36_mb_swift_aligned_a11_hyperpod_pytorch_job.yaml`
  passes server-side dry-run on `eks-cluster-hypd-0324`.
- The same manifest passes server-side dry-run on `Algorithm-training`.
- B200 smoke job `qwen36-mb-swift-aligned-smoke-b200-s16k` completed through
  `hyp create`.
- B200 full-sequence smoke job
  `qwen36-mb-swift-fullseq-b200-cp4-smoke` completed through `hyp create`.

### Full-reproduction scaffold

Status: prepared and admission-valid, but not schedulable on current resources.

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro`

Generated manifest:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_aligned_a11_hyperpod_pytorch_job.yaml`

The scaffold preserves the reference recipe and B300 resource shape. The only
intentional compatibility changes are:

- `restartPolicy: Always`, required by the HyperPod training operator.
- Reserved env names `JOB_NAME`, `NNODES`, and `NPROC_PER_NODE` are stored as
  `QWEN36_*` and exported back before invoking
  `/scripts/run_qwen36_overlay_sft_job.sh`.

### Current cluster resources

Status: blocking exact reproduction.

Current `eks-cluster-hypd-0324` node inventory:

- Seven `ml.p6-b200.48xlarge` nodes.
- One `ml.c5.2xlarge` node.
- No `ml.p6-b300.48xlarge` node.

Current `Algorithm-training` node inventory:

- One `ml.c5.2xlarge` node.
- No GPU node.
- No `ml.p6-b300.48xlarge` node.

Current job state:

- `kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig get hyperpodpytorchjobs,pods -n dolly-probe-training`
  returns no resources.
- `hyp list hyp-pytorch-job -n dolly-probe-training` returns no jobs.

### Training-effect evidence

Status: partially proven on B200, not proven for the exact B300 target.

Validated smoke evidence:

- `docs/generated/qwen36_mb_swift_hypd_smoke_validation_20260616.md`
  proves the HyperPod CLI/operator/S3/checkpoint/tokenizer/train path at
  `SEQ_LENGTH=16384`.
- `docs/generated/qwen36_mb_swift_b200_fullseq_cp4_smoke_validation_20260616.md`
  proves a full `SEQ_LENGTH=128000` one-iteration run on B200 with `CP=4`.

The B200 CP4 full-sequence smoke completed:

- `status=diagnostic_train_completed`
- Iteration `1/1`
- Train `lm loss: 3.103002E-01`
- Validation `lm loss: 4.071109E-01`
- Validation PPL `1.502471E+00`
- Checkpoint saved at iteration 1

This is useful feasibility evidence, but it does not prove the exact target
because the reference recipe requires B300, 16 EFA devices, and `CP=2`.

## Completion State

| Requirement | Current state | Evidence |
| --- | --- | --- |
| Install/verify HyperPod CRD/controller | Complete | Operator notes and server dry-run success |
| Create/verify namespace | Complete | `dolly-probe-training` exists |
| Create/verify PVC | Complete | `s3-verdent-tech-use1` exists and smoke used it |
| Create/verify ConfigMap | Complete | `qwen36-mb-overlay-scripts-r14` exists and smoke used it |
| Convert bare K8s Job to HyperPod form | Complete | Full-repro scaffold and generated manifest |
| HyperPod CLI validation | Complete | `hyp validate` success |
| Server-side admission on target clusters | Complete | dry-run success on both checked clusters |
| Run a HyperPod training smoke | Complete | S16K smoke and B200 CP4 full-sequence smoke completed |
| State the reference bare K8s Job | Complete | `/tmp/test_training_evidence/21-train-swift-job.yaml` |
| Exact B300 training-effect reproduction | Blocked | No schedulable `ml.p6-b300.48xlarge` node exists |

## Exact Completion Gate

The remaining exact gate is:

1. Add a schedulable `ml.p6-b300.48xlarge` node to either
   `eks-cluster-hypd-0324` or `Algorithm-training`.
2. Verify:

   ```bash
   kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
     get nodes -l node.kubernetes.io/instance-type=ml.p6-b300.48xlarge
   ```

3. Submit:

   ```bash
   cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_a11_full_repro
   /home/ubuntu/.local/bin/hyp validate
   /home/ubuntu/.local/bin/hyp create
   ```

4. Prove completion with logs/status showing:

   - Base checkpoint loaded.
   - Training loop entered.
   - At least the requested full-repro training horizon completed, or the
     agreed smoke/full-run horizon completed if the user narrows the gate.
   - No skipped/nan failure.
   - Checkpoint save completed.
   - Validation completed if enabled by the recipe.
   - S3 run output exists under the expected training prefix.

Until that B300 run is submitted and verified, the full objective remains
incomplete even though the submission path is now validated.
