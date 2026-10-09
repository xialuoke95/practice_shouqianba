# Megatron-Bridge West-US Training Environment Notes

Last checked: 2026-06-09 UTC.

## Working Baseline

The strongest local baseline evidence comes from:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/`

Target checkpoint:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`

Recorded runtime:

- Base model: `Qwen/Qwen3.6-35B-A3B`.
- GPU: 1 node, 8 GPUs.
- Runtime base image:
  `pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39-soci`.
- The run context recorded an additional descriptive
  `+source-megatron-bridge-v0.4.1-core-0.17.0rc0` suffix, but historical K8s
  manifests show that suffix was not part of the Docker image tag. The Bridge
  stack was installed at runtime from the archived source tarball.
- Python `3.12.10`.
- Torch `2.8.0+cu129`.
- Megatron-Bridge `0.4.1`.
- Megatron-Core `0.17.0rc0`.
- Transformer Engine was installed from source.
- Driver reported by the run: `580.126.09`, CUDA `13.0`.

Training parallelism:

- `TP=2`
- `PP=1`
- `CP=1`
- `EP=8`
- `nproc_per_node=8`
- `global_batch_size=16`
- `micro_batch_size=1`
- `seq_length=16384`
- `train_iters=1035`

Checkpoint settings:

- `ckpt_format=torch_dist`
- `fully_parallel_save=true`
- `fully_parallel_load=false`
- `load_optim=false`
- `load_rng=false`
- `save_optim=false`
- `save_rng=false`
- `save_tokenizer_assets=true`
- `use_checkpoint_args=false`
- `use_mp_args_from_checkpoint_args=false`
- `use_tokenizer_model_from_checkpoint_args=true`

## B300 Overlay Baseline

The separate West-US B300/NeMo overlay evidence comes from:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/`

The key B300 manifest is:

`artifacts/source/k8s/21-train-swift-job.yaml`

Local downloaded copy:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

This manifest records the practical B300 training shape:

- Kubernetes Job: `qwen36-mb-swift-aligned-a11`.
- Namespace: `dolly-probe-training`.
- Node selector: `node.kubernetes.io/instance-type=ml.p6-b300.48xlarge`.
- Image: `nvcr.io/nvidia/nemo:26.04.00`.
- Requests/limits: 8 GPUs and 16 EFA.
- CPU/memory request: 150 CPU, 2200Gi memory; memory limit 3800Gi.
- Scratch: `emptyDir` size limit 24000Gi mounted at `/scratch`.
- Shared memory: memory `emptyDir` size limit 128Gi mounted at `/dev/shm`.
- S3 access: PVC `s3-verdent-tech-use1` mounted at `/s3-verdent`.
- Scripts: ConfigMap `qwen36-mb-overlay-scripts-r14`, mounted read-only at
  `/scripts`.

The B300 overlay run used:

- `NNODES=1`, `NPROC_PER_NODE=8`.
- `TP=2`, `PP=1`, `CP=2`, `EP=8`, `ETP=1`.
- `SEQ_LENGTH=128000`, `GBS=16`, `MBS=1`.
- `CALCULATE_PER_TOKEN_LOSS=true`.
- `DDP_AVERAGE_IN_COLLECTIVE=false`.
- `MOE_AUX_LOSS_COEFF=0.0`.
- `TRUNCATION_METHOD=left`.
- `DATASET_KEY=sft_opus_sonnet_swift_mbjson_v2`.

Practical lesson:

- The working B300 recipe is not just "use a bigger GPU". It also depends on
  enough local scratch, a large `/dev/shm`, EFA, S3 CSI/PVC access, the NeMo
  26.04 base image, the overlay ConfigMap, and the CP/per-token-loss settings
  that match the loss-alignment recipe.
- For Qwen3.6 MoE SFT, keep `calculate_per_token_loss=true` paired with
  `ddp_average_in_collective=false`; otherwise Megatron-Bridge can reject the
  loss-reduction mode.

## Base Checkpoint Pattern

The training scripts consistently stage the base checkpoint locally before
training. Evidence references:

- Base checkpoint source:
  `/s3-verdent/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/base_checkpoints/qwen36_35b_a3b_megatron_mcore_r3`
- Training checkpoint output:
  `/scratch/node-work/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/qwen36_swift_all_full_r13/node0/checkpoints`
- Copied S3 checkpoint output:
  `training/qwen36_swift_all_full_r13/checkpoint_copy/`

Practical lesson:

- Treat `/scratch/node-work` as the high-throughput working path.
- Use S3 as source/archive, not as the direct write target for every checkpoint
  mutation.
- Copy finished checkpoints to S3 with a manifest so size/file-count mismatches
  are easy to detect later.

## Overlay Runtime Setup

The overlay run evidence includes:

- `setup_nemo2604_overlay_runtime.sh`
- `run_qwen36_overlay_sft_job.sh`
- `bridge_qwen36_train.py`
- `prepare_qwen36_sft_dataset.py`

Key overlay inputs:

- Megatron-LM commit tarball:
  `megatron-lm-378d81fbd32a3faa90726d0e59344f7893ce4b47.tar.gz`
- Megatron-Bridge commit tarball:
  `megatron-bridge-e00a65e73bcba924e208e18ff3b13c22e4b30e83.tar.gz`

The setup script extracts Megatron-Bridge, places Megatron-LM under
`3rdparty/Megatron-LM`, applies compatibility patches, installs both with
editable `pip install --no-build-isolation --no-deps -e`, and builds the
Megatron dataset helper.

Important patches observed in the setup:

- NVRx checkpointing compatibility in
  `dist_checkpointing/strategies/nvrx.py`.
- Encoded-chat dataset utility patch.
- `tools_json` handling for chat-template processing.
- Transformer Engine wrapper compatibility patch.
- SFT dataset left-truncation patch.
- Transformer Engine DPA eligibility patch for sm10.x/head-dim behavior.

## Environment Variables And Runtime Controls

The run script sets:

- `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Attention backend flags:
  - Flash path sets `NVTE_FLASH_ATTN=1`, `NVTE_FUSED_ATTN=0`,
    `NVTE_UNFUSED_ATTN=0`.
  - Non-flash path enables the other attention paths.

Common Bridge arguments pass:

- pretrained checkpoint path
- checkpoint save/load paths
- TP/PP/CP/EP topology
- attention backend
- recompute settings
- `moe_token_dispatcher_type=alltoall`

Practical lesson:

- Record the exact framework commits and pip freeze for every run.
- Keep the model-provider, checkpoint, and topology config together. A copied
  `torch_dist` checkpoint is not self-sufficient for serving or conversion.

## S3 Mountpoint Limitation

The loss-alignment runs observed a repeated operational issue:

- Writing an overwrite/update through S3 Mountpoint can fail.
- A successful eval can be followed by status-file overwrite failure.

Use one of these patterns instead:

- Write unique status filenames, for example `status_<timestamp>.json`.
- Write status locally and copy it to S3 using the S3 API.
- Treat status-write failure separately from training/eval success.
