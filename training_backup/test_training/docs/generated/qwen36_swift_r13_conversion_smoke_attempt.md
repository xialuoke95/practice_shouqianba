# Qwen36 Swift R13 Conversion Smoke Attempt

Checked: 2026-06-08 UTC.

## Attempt 1

Manifest:

`docs/generated/qwen36_swift_r13_conversion_smoke_job.yaml` before the v2 fix.

Kubernetes Job:

`qwen36-swift-r13-conv-smoke-20260608` in namespace `glm51-pd-b200-p1d1`.

Result:

- The job was created on GLM node `hyperpod-i-060621000d6270d47`.
- The pod never ran because the image name was invalid.
- Kubernetes pod status showed `InvalidImageName`.
- The job then hit `activeDeadlineSeconds=300` and failed with
  `DeadlineExceeded`.

Cause:

The run-context image string included a non-Docker tag suffix:

`+source-megatron-bridge-v0.4.1-core-0.17.0rc0`

Historical K8s manifests show the actual valid base image was:

`763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39-soci`

Megatron-Bridge was installed at runtime by
`setup_aws_dlc_bridge_runtime.sh`, not by using a modified image tag.

## Corrected Manifests

Updated generated manifests now use the valid AWS DLC image and explicitly run
the archived runtime setup script before importing `megatron.bridge`.

Server dry-runs passed:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply --dry-run=server -f docs/generated/qwen36_swift_r13_conversion_job.yaml
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply --dry-run=server -f docs/generated/qwen36_swift_r13_conversion_smoke_job.yaml
```

Dry-run results:

- `job.batch/qwen36-swift-r13-hf-export created (server dry run)`
- `job.batch/qwen36-swift-r13-conv-smoke-v2-20260608 created (server dry run)`

After the dry-run, corrected v2/v3 smoke Jobs were launched and recorded below.
Both failed before conversion work started because the AWS CLI path did not have
the required S3 credentials/scope.

## Attempt 2

Kubernetes Job:

`qwen36-swift-r13-conv-smoke-v2-20260608` in namespace `glm51-pd-b200-p1d1`.

Result:

- The image pulled successfully on `hyperpod-i-060621000d6270d47`.
- The image size reported by kubelet was 12,966,869,492 bytes.
- The container started, then failed at the first S3 command.

Failure:

```text
Unable to locate credentials. You can configure credentials by running "aws configure".
```

Cause:

The v2 smoke Job used the namespace default service account. Existing GLM
serving deployments use `serviceAccountName: sglang-s3-access`, not the default
service account.

## Attempt 3

Kubernetes Job:

`qwen36-swift-r13-conv-smoke-v3-20260608` in namespace `glm51-pd-b200-p1d1`.

Change:

- Added `serviceAccountName: sglang-s3-access`.
- Added AWS region/STS environment variables matching the GLM deployments.

Result:

- IRSA/role injection worked.
- The pod assumed:
  `arn:aws:sts::288761729453:assumed-role/glm51-pd-b200-p1d1-s3-reader-us-west-2/...`
- The container failed at S3 access because the role is scoped only to the GLM
  weight bucket.

Failure:

```text
AccessDenied ... is not authorized to perform: s3:ListBucket on resource:
"arn:aws:s3:::verdent.tech"
```

The GLM role's existing inline policy only permits:

- `s3:ListBucket` / `s3:GetBucketLocation` on
  `arn:aws:s3:::llm-weight-files-us-west-2`, conditioned to
  `GLM-5.1-FP8` prefixes.
- `s3:GetObject` on
  `arn:aws:s3:::llm-weight-files-us-west-2/GLM-5.1-FP8/*`.

## IAM Policy Proposal

The minimal additional IAM policy needed for this conversion is stored at:

`docs/generated/glm_s3_reader_test_training_policy.json`

It is intentionally scoped to:

- read the target checkpoint prefix;
- read the target run's archived Bridge source and setup script;
- write only the proposed HF export prefix.

Applying it would persistently broaden the shared GLM S3 reader role, so it was
not applied without explicit user approval.
