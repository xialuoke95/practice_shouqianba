# Qwen36 Megatron Bridge B200 Adjusted Online Validation

Date: 2026-06-17 UTC.

## Updated Target

The goal was changed from exact B300 reproduction to:

- Submit through SageMaker HyperPod CLI.
- Run on current B200 capacity.
- Reproduce the training chain and keep the recipe as close as practical to
  the original training recipe.
- Make the submitted task visible through the online HyperPod/Kubernetes
  management path.

The original reference remains:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

That reference is B300-specific: `ml.p6-b300.48xlarge`, 8 GPU, 16 EFA, and
`CP=2`.

## Submitted Job

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_online`

Run directory:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_online/run/20260617T020736`

HyperPodPyTorchJob:

`dolly-probe-training/qwen36-mb-swift-b200-adjusted-online`

S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_b200_adjusted_online_20260617/`

Submission result:

- `/home/ubuntu/.local/bin/hyp validate` passed.
- `/home/ubuntu/.local/bin/hyp create` submitted successfully.
- `hyp list hyp-pytorch-job -n dolly-probe-training` showed the job as
  `Created`.
- Kubernetes created the `HyperPodPyTorchJob` CR and scheduled the pod to
  `hyperpod-i-0bedb0d5f0facf6e4`.

## Recipe Shape

Preserved from the original recipe:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `EP=8`
- `ETP=1`
- `EPOCHS=2`
- `DATASET_KEY=sft_opus_sonnet_swift_mbjson_v2`
- Image `nvcr.io/nvidia/nemo:26.04.00`

B200 adjustments:

- `instance_type=ml.p6-b200.48xlarge`
- 8 GPU
- 8 EFA
- `CP=4`
- memory request/limit `1800Gi/1900Gi`

Online verification bounds:

- `TRAIN_ITERS_OVERRIDE=2`
- `MAX_TRAIN_SAMPLES=32`
- `SKIP_BASELINE_EVAL=true`
- `EXIT_AFTER_TRAIN=true`

## Runtime Evidence

The job wrote:

`status=diagnostic_train_completed`

Status JSON:

`/tmp/qwen36_b200_adjusted_overlay_sft_status.json`

Training log:

`/tmp/qwen36_b200_adjusted_node0_20_train.log`

S3 evidence includes:

- `run_context.json`
- `base_import/node0/base_checkpoint_import_manifest.json`
- `tokenizer_patch_manifest.json`
- `train/train_launch_manifest.json`
- `train/train_config.yaml`
- `status/overlay_sft_status.json`
- `logs/node0_20_train.log`

Important train log lines:

- Base checkpoint loaded successfully from local imported checkpoint.
- Training loop started at iteration 0.
- Iteration `1/2` completed with 16 consumed samples.
- Iteration `2/2` completed with 32 consumed samples.
- Iteration 1 train `lm loss: 3.103002E-01`.
- Iteration 2 train `lm loss: 3.439665E-01`.
- Skipped iterations: `0`.
- NaN iterations: `0`.
- Max allocated memory after iteration 2: about `129.06 GB`.
- Checkpoint saved at iteration 2.
- Validation `lm loss: 4.015621E-01`.
- Validation PPL: `1.494157E+00`.

## Management Visibility

The online management object remains present:

- `hyperpodpytorchjob.sagemaker.amazonaws.com/qwen36-mb-swift-b200-adjusted-online`
- `hyp list hyp-pytorch-job -n dolly-probe-training` shows the job.

After the bounded verification run completed, the container exited with code
`0`, then restarted because the HyperPod operator runs the pod through a
Deployment with `restartPolicy: Always`. The restarted container attempted to
reuse an existing S3 log path and hit:

`tee ... node0_01_overlay_runtime.log: Operation not permitted`

This happened after `status=diagnostic_train_completed`; it is a packaging
behavior after successful completion, not a training failure.

To avoid holding B200 resources in a restart loop while keeping the CR visible,
the child Deployment was scaled down:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  scale deployment/qwen36-mb-swift-b200-adjusted-online-pod \
  -n dolly-probe-training --replicas=0
```

Final checked state:

- HyperPodPyTorchJob CR: present.
- Child Deployment/ReplicaSet: `0/0`.
- Training pod: removed.
- Node `hyperpod-i-0bedb0d5f0facf6e4`: `nvidia.com/gpu=0` and
  `vpc.amazonaws.com/efa=0` allocated after cleanup.

## Conclusion

The updated B200 target is satisfied:

- Submitted through online HyperPod CLI.
- Visible through HyperPod CLI/Kubernetes management objects.
- Ran on current B200 capacity.
- Preserved the major original training shape where feasible.
- Completed the full 128k training chain for 2 iterations.
- Saved checkpoint and ran validation.
- Released B200 resources after evidence collection while retaining the CR for
  management visibility.
