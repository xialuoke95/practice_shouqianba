# K8s IAM And S3 Access Snapshot

Checked: 2026-06-09 UTC.

## Workspace AWS Identity

The workspace has a usable default AWS identity when commands run outside the
local sandbox:

```text
arn:aws:sts::288761729453:assumed-role/eks-hypd-workspace-03240928-admin-role/i-0d6ac9306158cc9f6
```

This identity successfully listed both:

- Raw Megatron checkpoint:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`
- Exported HF/SGLang checkpoint:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Raw checkpoint list result:

- Objects: 15.
- Size: 70,312,965,895 bytes.
- Layout: Megatron `torch_dist`/`.distcp`.

Exported checkpoint list result:

- Objects: 34.
- Size: 70,234,598,273 bytes.
- Layout: HF/SGLang-loadable directory with config/tokenizer files and
  26 safetensors shards.

## GLM Namespace Service Accounts

Namespace: `glm51-pd-b200-p1d1`.

Observed service accounts:

- `default`: no AWS role annotation.
- `sglang-monitor`: no AWS role annotation.
- `sglang-router`: no AWS role annotation.
- `sglang-s3-access`:
  `eks.amazonaws.com/role-arn=arn:aws:iam::288761729453:role/glm51-pd-b200-p1d1-s3-reader-us-west-2`

Earlier conversion attempts confirmed the practical difference:

- A Job using the default service account failed with
  `Unable to locate credentials`.
- A Job using `sglang-s3-access` received AWS credentials, but the role was
  scoped to the existing GLM weight bucket and could not list
  `s3://verdent.tech/test_training/...`.

## EKS Pod Identity

AWS EKS Pod Identity associations for cluster `eks-cluster-hypd-0324` currently
exist only for:

- `aws-hyperpod/hp-training-operator-controller-manager`
- `hyperpod-observability/hyperpod-observability-operator-otel-collector`

There is no Pod Identity association for `glm51-pd-b200-p1d1`.

## Deployment Implication

The phrase "default IAM role" is true for this workspace, but not for arbitrary
pods launched in the GLM namespace.

For conversion or deployment pods that must read
`s3://verdent.tech/test_training/...`, use one of these proven paths:

- S3 CSI mount, which already succeeded for the raw checkpoint and export path.
- A service account/role with explicit read access to the target
  `verdent.tech/test_training` prefixes.

Do not assume the GLM namespace default service account can read the checkpoint.
