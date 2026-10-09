# HyperPod Training Operator Install Notes

Current IAM identity used for checks:

`arn:aws:sts::288761729453:assumed-role/eks-hypd-workspace-03240928-admin-role/i-0d6ac9306158cc9f6`

Local CLI status:

- `sagemaker-hyperpod==3.9.0` installed with `python3 -m pip install --user sagemaker-hyperpod`.
- `hyp` is available at `/home/ubuntu/.local/bin/hyp`.

`eks-cluster-hypd-0324` status:

- EKS cluster: `eks-cluster-hypd-0324`, region `us-west-2`, Kubernetes `1.33`.
- EKS add-on `amazon-sagemaker-hyperpod-training-operator` is `ACTIVE`.
- Add-on version: `v1.2.1-eksbuild.1`.
- `hyperpodpytorchjobs.sagemaker.amazonaws.com` CRD is installed.
- `sagemaker.amazonaws.com/v1` exposes `HyperPodPyTorchJob`.
- `aws-hyperpod` has `hp-training-operator-hp-training-controller-manager`
  running.
- `dolly-probe-training` now exists.
- `s3-verdent-tech-use1` PVC exists in `dolly-probe-training`.
- `qwen36-mb-overlay-scripts-r14` ConfigMap exists in
  `dolly-probe-training`.
- The B200 smoke scaffold submitted through `hyp create` completed one train
  iteration and wrote `status=diagnostic_train_completed`.
- The B200 full-sequence CP4 smoke also completed through `hyp create` at
  `SEQ_LENGTH=128000`, saved a checkpoint, ran validation, and wrote
  `status=diagnostic_train_completed`.
- The full B300 reproduction manifest passes server-side dry-run, but cannot
  currently schedule because the cluster has B200 nodes and no
  `ml.p6-b300.48xlarge` node.

Alternative checked cluster:

- Cluster: `eks-algorithm-training`, region `us-west-2`, Kubernetes `1.34`.
- Kubeconfig: `/tmp/eks-algorithm-training.kubeconfig`.
- Add-on version: `v1.2.1-eksbuild.1`.
- `hyperpodpytorchjobs.sagemaker.amazonaws.com` CRD is installed.
- `sagemaker.amazonaws.com/v1` exposes `HyperPodPyTorchJob`.
- `aws-hyperpod` has `hp-training-operator-hp-training-controller-manager`
  running.
- EKS add-ons include `amazon-sagemaker-hyperpod-training-operator`.
- `dolly-probe-training` namespace exists.
- `s3-verdent-tech-use1` PVC exists in `dolly-probe-training`.
- `qwen36-mb-overlay-scripts-r14` ConfigMap exists in
  `dolly-probe-training`.
- Current visible nodes include only one `ml.c5.2xlarge` node and no GPU
  worker node.
- The generated
  `docs/generated/qwen36_mb_swift_aligned_a11_hyperpod_pytorch_job.yaml`
  passed server-side dry-run on this cluster.

The generated manifest was adjusted for HyperPod admission rules:

- User-defined `JOB_NAME`, `NNODES`, and `NPROC_PER_NODE` environment variables
  were renamed to `QWEN36_*` names and exported in the shell command before
  invoking `/scripts/run_qwen36_overlay_sft_job.sh`.
- Pod `restartPolicy` was changed from `Never` to `Always`, which the
  HyperPod training operator requires.

AWS reports compatible default add-on version for EKS `1.33`:

`v1.2.1-eksbuild.1`

Installation command used when the add-on is absent:

```bash
aws eks create-addon \
  --cluster-name eks-cluster-hypd-0324 \
  --addon-name amazon-sagemaker-hyperpod-training-operator \
  --addon-version v1.2.1-eksbuild.1 \
  --region us-west-2
```

Verification commands:

```bash
aws eks describe-addon \
  --cluster-name eks-cluster-hypd-0324 \
  --addon-name amazon-sagemaker-hyperpod-training-operator \
  --region us-west-2

kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  get crd hyperpodpytorchjobs.sagemaker.amazonaws.com

kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  api-resources --api-group=sagemaker.amazonaws.com

kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  apply --dry-run=server \
  -f docs/generated/qwen36_mb_swift_aligned_a11_hyperpod_pytorch_job.yaml
```

Algorithm-training dry-run command:

```bash
kubectl --kubeconfig /tmp/eks-algorithm-training.kubeconfig \
  apply --dry-run=server \
  -f docs/generated/qwen36_mb_swift_aligned_a11_hyperpod_pytorch_job.yaml
```
