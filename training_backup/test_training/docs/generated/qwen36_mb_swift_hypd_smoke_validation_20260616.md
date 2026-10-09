# Qwen36 Megatron Bridge HyperPod CLI Smoke Validation

Date: 2026-06-16 UTC.

## Goal

Validate that the qwen36 Megatron-Bridge SFT training flow can be submitted as a
`HyperPodPyTorchJob` through `sagemaker-hyperpod-cli` on
`eks-cluster-hypd-0324`, while preserving the behavior of the earlier bare
Kubernetes training job as closely as practical on available B200 capacity.

## Reference Bare Kubernetes Job

Reference manifest:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

Important reference settings:

- Kubernetes kind: `Job`.
- Name/namespace: `qwen36-mb-swift-aligned-a11` in `dolly-probe-training`.
- Image: `nvcr.io/nvidia/nemo:26.04.00`.
- Scripts ConfigMap: `qwen36-mb-overlay-scripts-r14`.
- S3 CSI PVC: `s3-verdent-tech-use1`.
- Original target hardware: `ml.p6-b300.48xlarge`, 8 GPU, 16 EFA.
- Original recipe shape: `SEQ_LENGTH=128000`, `GBS=16`, `MBS=1`,
  `TP=2`, `PP=1`, `CP=2`, `EP=8`, `ETP=1`.

## HyperPod CLI Scaffold

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_hypd_smoke`

Generated manifest:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_smoke_hyperpod_job.yaml`

Submitted run directory:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_hypd_smoke/run/20260616T211532`

Submission command:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_hypd_smoke
/home/ubuntu/.local/bin/hyp create
```

Submitted job:

`qwen36-mb-swift-aligned-smoke-b200-s16k`

## Cluster Prerequisites

Confirmed on `eks-cluster-hypd-0324`:

- `hyperpodpytorchjobs.sagemaker.amazonaws.com` CRD exists.
- HyperPod training operator controller is running in `aws-hyperpod`.
- Namespace `dolly-probe-training` exists.
- PVC `dolly-probe-training/s3-verdent-tech-use1` is `Bound`.
- ConfigMap `dolly-probe-training/qwen36-mb-overlay-scripts-r14` exists.
- Server-side dry-run accepted the generated `HyperPodPyTorchJob`.

Prerequisite manifest:

`/home/ubuntu/workspace/test_training/docs/generated/qwen36_mb_swift_hypd_prereqs.yaml`

## B200 Smoke Overrides

The first smoke preserved `SEQ_LENGTH=128000` and reached the training backward
pass, but failed on B200 with a Triton CUDA OOM. The successful smoke therefore
kept the same model, data, overlay scripts, checkpoint import, tokenizer patch,
parallelism, and optimizer settings, but used a reduced sequence-length smoke:

- Instance type: `ml.p6-b200.48xlarge`.
- Resources: 8 GPU, 8 EFA, 150 CPU, 1800Gi memory request, 1900Gi memory limit.
- `SEQ_LENGTH=16384`.
- `TRAIN_ITERS_OVERRIDE=1`.
- `MAX_TRAIN_SAMPLES=16`.
- `SKIP_BASELINE_EVAL=true`.
- `EXIT_AFTER_TRAIN=true`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

This is a submission/runtime smoke, not a full effect reproduction of the
original B300 `SEQ_LENGTH=128000` run.

## Result

S3 artifact root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_aligned_smoke_b200_s16k_20260616/`

Downloaded local evidence:

- `/tmp/qwen36_mb_swift_s16k_overlay_sft_status.json`
- `/tmp/qwen36_mb_swift_s16k_node0_20_train.log`

Status JSON result:

`status: diagnostic_train_completed`

Training log highlights:

- Iteration completed: `iteration 1/1`.
- Consumed samples: `16`.
- Step time: `189.94s`.
- Training lm loss: `5.504900E-01`.
- Grad norm: `3.355`.
- Skipped iterations: `0`.
- NaN iterations: `0`.
- Rank 0 memory allocated after one iteration: about `86.165GB`.
- Total parameters reported: `34.11B`.
- Checkpoint save completed for iteration `1`.
- Validation lm loss: `6.370878E-01`.
- Validation PPL: `1.890966E+00`.

The job briefly appeared as `CrashLoopBackOff` after completion because the
current HyperPod training operator defaulted this CR to
`runPolicy.workloadMode: Deployment` and required pod `restartPolicy: Always`.
After the script completed and exited, the Deployment controller restarted the
container. The authoritative run result is the S3 status file above. The
HyperPod job was then deleted with:

```bash
/home/ubuntu/.local/bin/hyp delete hyp-pytorch-job \
  --job-name qwen36-mb-swift-aligned-smoke-b200-s16k \
  -n dolly-probe-training
```

Cleanup verification:

`kubectl get hyperpodpytorchjobs,pods -n dolly-probe-training` returned no
resources after deletion.

## Implications

Validated end to end:

- `hyp create` submission path.
- HyperPod CRD/operator admission.
- B200 scheduling on `eks-cluster-hypd-0324`.
- PVC and ConfigMap dependencies.
- Overlay runtime setup.
- Base checkpoint import.
- Tokenizer chat-template patch.
- Torch distributed training launch.
- One train iteration, checkpoint save, and validation.
- S3 artifact/status reporting.
- Manual cleanup with `hyp delete`.

Not yet proven:

- Full training-effect reproduction of the original bare K8s B300 job at
  `SEQ_LENGTH=128000`.
- UI management from `https://hyperpod-west2.verdent.ai/cluster-management`.
  The CR was live in the cluster during the run, so it should be discoverable
  by any UI that watches the same cluster/API, but this was not browser-verified
  in this workspace.

For exact effect reproduction, prefer B300 capacity matching the original
resource shape. If only B200 is available, the recipe needs an explicit B200
plan rather than a literal `SEQ_LENGTH=128000` port.
