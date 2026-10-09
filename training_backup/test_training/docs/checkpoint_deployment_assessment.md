# Checkpoint Deployment Assessment

Last checked: 2026-06-09 UTC.

## Summary

I did not wire the exported model into the live GLM route yet.

The requested object is a Megatron distributed checkpoint:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`

It is not a Hugging Face/SGLang model directory. The current GLM deployment
script expects a model directory containing files such as `config.json`,
tokenizer assets, and model weights loadable by `sglang serve`. The requested
checkpoint instead contains `torch_dist`/`distcp` shard files:

- `iter_0001035/__0_0.distcp` through `iter_0001035/__7_0.distcp`
- `iter_0001035/.metadata`
- `iter_0001035/common.pt`
- `iter_0001035/metadata.json`
- `iter_0001035/run_config.yaml`
- `latest_checkpointed_iteration.txt`

Trying to mount or sync this raw directory into the existing GLM SGLang launcher
would fail the current launcher's model-file checks and/or SGLang load step.

That raw checkpoint has now been exported to an HF/SGLang-loadable candidate:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

`deploy_model.py` preflight against the exported prefix reports
`verdict=ready_for_isolated_sglang_deploy`, `sglang_loadable=true`,
34 objects, 70,234,598,273 bytes, and no blockers. The one-node SGLang load
smoke has now passed, and the isolated Deployment/Service is Ready. The model
has not been wired into the existing GLM route.

## `deploy_model.py` Attempt

`deploy_model.py` has been converted into a deployment preflight/guard tool. It
now checks the S3 checkpoint layout, optional GLM Kubernetes status, and refuses
live mutation when the checkpoint is not already SGLang-loadable.

Command run:

```bash
python3 deploy_model.py --output-json docs/generated/qwen36_swift_r13_deploy_preflight.json
python3 deploy_model.py --apply --output-json docs/generated/qwen36_swift_r13_deploy_apply_attempt.json
```

Results:

- Preflight exit code: `0`.
- Apply-attempt exit code: `2`.
- Verdict: `blocked_before_live_deploy`.
- Blocker:
  `Checkpoint is Megatron torch_dist/distcp, not a Hugging Face/SGLang model directory.`
- The apply attempt refused live deployment before any Kubernetes mutation.

Generated evidence:

- `docs/generated/qwen36_swift_r13_deploy_preflight.json`
- `docs/generated/qwen36_swift_r13_deploy_apply_attempt.json`
- `docs/generated/tos_checkpoint854_metadata_preflight.json`
- `docs/generated/qwen36_swift_r13_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_conversion_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_conversion_smoke_attempt.md`
- `docs/generated/glm_s3_reader_test_training_policy.json`
- `docs/generated/qwen36_swift_r13_s3csi_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_runtime_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_export_preflight_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_conversion_attempt.md`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml`
- `docs/generated/run_qwen36_sglang_smoke_with_restore.sh`
- `docs/generated/qwen36_swift_r13_sglang_deployment.yaml`
- `docs/generated/qwen36_swift_r13_hf_export_preflight.json`

The script detected:

- `checkpoint_type=megatron_torch_dist`.
- `sglang_loadable=false`.
- `latest_checkpointed_iteration=1035`.
- `has_config_json=false`.
- `has_tokenizer_files=false`.
- `has_hf_weight_files=false`.
- `has_distcp_shards=true`.
- `sharded_backend=torch_dist`.

Related TOS metadata check:

- S3 prefix:
  `s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/artifacts/alignment_refs/source_artifacts_v4/pfs/train_output/checkpoint-854_metadata/`
- `deploy_model.py --skip-k8s` classified it as `unknown`, not deployable.
- It has `config.json`, tokenizer files, and `model.safetensors.index.json`.
- The index references 16 safetensors shards, but all 16 shard files are
  missing from that metadata prefix.
- This is useful for tokenizer/config alignment, but it is not a complete HF
  checkpoint directory for SGLang serving.

## Conversion Plan

Megatron-Bridge v0.4.1 source from the target run confirms the reverse export
API exists:

- Source package:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/env/Megatron-Bridge-v0.4.1-source.tar.gz`
- Example:
  `examples/conversion/convert_checkpoints.py export`
- API:
  `AutoBridge.from_hf_pretrained(...).export_ckpt(megatron_path, hf_path, show_progress=True, strict=False)`

Important constraints:

- `from_hf_pretrained()` can build a lazy HF state source from config/tokenizer
  metadata plus `model.safetensors.index.json`, but full export eventually
  needs real HF tensor data for shape/layout checks.
- The metadata-only TOS `checkpoint-854_metadata` prefix is therefore not enough
  as the HF base reference for a complete Megatron-to-HF export.
- Historical K8s jobs used the valid AWS DLC image
  `763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39-soci`
  and installed Megatron-Bridge at runtime via
  `setup_aws_dlc_bridge_runtime.sh`.
- The run-context string with
  `+source-megatron-bridge-v0.4.1-core-0.17.0rc0` is not a valid Docker image
  tag and caused the first smoke Job to fail with `InvalidImageName`.

Generated conversion manifest:

- `docs/generated/qwen36_swift_r13_conversion_job.yaml`
- Namespace: `glm51-pd-b200-p1d1`
- Node: `hyperpod-i-060621000d6270d47`
- Requests no GPU.
- Uses host NVMe at `/opt/dlami/nvme`.
- Downloads the archived Bridge source and setup script from S3.
- Syncs the Megatron checkpoint locally.
- Runs `AutoBridge.export_ckpt(...)`.
- Uploads the exported HF directory to:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Server dry-run passed for both:

- `docs/generated/qwen36_swift_r13_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_conversion_smoke_job.yaml`

Follow-up smoke results:

- v2 launched with the valid image, but used the default service account and
  failed with `Unable to locate credentials`.
- v3 launched with `serviceAccountName: sglang-s3-access`; credentials were
  injected, but the assumed role is scoped only to
  `s3://llm-weight-files-us-west-2/GLM-5.1-FP8/`.
- v3 failed on `s3://verdent.tech/...` with `AccessDenied` for
  `s3:ListBucket`.
- A minimal scoped policy proposal was written to
  `docs/generated/glm_s3_reader_test_training_policy.json`, but it was not
  applied because changing a shared IAM role requires explicit approval.

## S3 CSI Conversion Attempt

The cluster also has the AWS S3 CSI driver installed, which allowed a safer
prefix-scoped S3 access path without changing the shared GLM IAM role.

Generated S3 CSI manifests:

- `docs/generated/qwen36_swift_r13_s3csi_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_runtime_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_export_preflight_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml`

Results:

- Checkpoint mount smoke succeeded on
  `hyperpod-i-060621000d6270d47`, confirming 15 files,
  70,312,965,895 bytes, latest iteration `1035`, and `torch_dist` metadata.
- Runtime smoke succeeded with the archived Megatron-Bridge `0.4.1` source and
  reported Torch `2.8.0+cu129`.
- HF metadata preflight succeeded against the copied `checkpoint-854_metadata`
  prefix. It created a `SafeTensorsStateSource` with 1,026 keys and 16
  referenced shards, without loading full HF weights.
- The first full export failed during `AutoBridge.export_ckpt(...)` after
  loading the Megatron checkpoint and reaching about 3% of HF conversion.

The metadata-only export failure was:

```text
KeyError: "Keys not found in safetensors from .../hf_metadata:
{'model.language_model.layers.0.mlp.experts.down_proj'}"
```

This means the S3 CSI path, local NVMe copy, and Bridge runtime were good, but
the selected metadata-only HF source was incomplete. Bridge needs the
referenced safetensors shards, or another complete HF state source, for full
export.

The successful export used the public HF source model
`Qwen/Qwen3.6-35B-A3B` instead:

- HF Hub source probe succeeded and confirmed the repo is public/not gated,
  with 26 safetensors shards and 1,045 indexed keys.
- HF Hub source export completed local conversion to 100%:
  `Converting to HuggingFace 100% (21276/21276) Qwen35VLMoEBridge`.
- Bridge reported:
  `Success: All tensors from the original checkpoint were written.`
- The export-complete JSON reported 34 files and 70,234,598,273 bytes.
- The Kubernetes Job is marked `Failed` only because a final `cp -a` attempted
  to preserve directory timestamps on the S3 CSI mount after the files had
  copied. The manifest has been updated to use `cp -r`.

The exported HF/SGLang candidate prefix is:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Current S3 verification:

- Objects: `34`
- Total bytes: `70,234,598,273`
- Includes `config.json`, tokenizer assets, `model.safetensors.index.json`, and
  `model-00001-of-00026.safetensors` through
  `model-00026-of-00026.safetensors`.

Post-export evidence:

- `docs/generated/qwen36_swift_r13_s3csi_conversion_attempt.md`
- `docs/generated/qwen36_swift_r13_hf_export_preflight.json`

The post-export preflight now reports `ready_for_isolated_sglang_deploy`,
`sglang_loadable=true`, a complete 26-shard safetensors index, and
`blockers=[]`.

The generated smoke manifest uses a new model/cache name
`Qwen3.6-35B-A3B-Swift-R13` and targets
`hyperpod-i-060621000d6270d47`.

`deploy_model.py` supports:

```bash
python3 deploy_model.py \
  --checkpoint-s3-uri s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/ \
  --sglang-node hyperpod-i-060621000d6270d47 \
  --output-sglang-smoke-job-yaml docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml \
  --output-sglang-smoke-restore-script docs/generated/run_qwen36_sglang_smoke_with_restore.sh \
  --output-isolated-sglang-deployment-yaml docs/generated/qwen36_swift_r13_sglang_deployment.yaml
```

Live isolated SGLang result on 2026-06-09 UTC:

- Smoke pod: `qwen36-swift-r13-sglang-smoke-20260608-8prtb`
- Smoke node: `hyperpod-i-060621000d6270d47`
- SGLang argument added after an initial FlashInfer warmup autotune mismatch:
  `--disable-flashinfer-autotune`
- Smoke result: `/health` HTTP 200 and `/v1/chat/completions` HTTP 200.
- The restore wrapper left
  `glm51-pd-b200-p1d1/sglang-decode-anchor` at its original `0` replicas.
- The completed temporary smoke Job was deleted after evidence was recorded.

The isolated Deployment/Service was then applied:

- Deployment: `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`
- Ready pod: `qwen36-swift-r13-sglang-76f4c76b45-4vm8d`
- Node: `hyperpod-i-060621000d6270d47`
- Service ClusterIP: `172.20.17.137`
- Endpoint: `10.90.17.128:30000`
- In-pod `/health` returned HTTP 200.
- ClusterIP `/v1/chat/completions` returned HTTP 200 with
  `usage.total_tokens=17`.

The generated helper script
`docs/generated/run_qwen36_sglang_smoke_with_restore.sh` passed `bash -n`.
The generated smoke manifest passed Kubernetes server dry-run. The generated
isolated Deployment/Service manifest also passed Kubernetes server dry-run. It
uses independent labels and a ClusterIP service named
`qwen36-swift-r13-sglang`; it is not wired to the existing GLM router.

## GLM Deployment Shape

Current GLM namespace:

- Namespace: `glm51-pd-b200-p1d1`.
- Current model: `GLM-5.1-FP8`.
- Current model source: `s3://llm-weight-files-us-west-2/GLM-5.1-FP8/`.
- Original GLM serving deployments are currently scaled to `0/0`.
- Isolated Qwen36 deployment `qwen36-swift-r13-sglang` is `1/1`.
- Serving image:
  `288761729453.dkr.ecr.us-east-2.amazonaws.com/sglang-kimik2-detector-patch:20260602085214`.
- The container syncs model files to host NVMe:
  `/opt/dlami/nvme/models/${MODEL_NAME}`.
- The launcher checks for a local model directory and then starts:
  `sglang serve --model-path ${MODEL_LOCAL} --served-model-name GLM-5.1-FP8`.

Resource pattern when scaled up:

- Prefill pod requests 8 GPUs and 8 EFA.
- Decode pod requests 8 GPUs and 8 EFA.
- GLM historically used these two 8-GPU nodes:
  - `hyperpod-i-01e93a89cb27a5046`, currently occupied by
    `dpsk-v4-pro-standalone-b200-p1d1/sglang-standalone`
  - `hyperpod-i-060621000d6270d47`, currently occupied by
    `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`

Routing caveat:

- The default proxy ConfigMap currently has
  `PREFERRED_AWS_GLM51_BACKEND=b200`.
- Because the external proxy can still prefer B200, replacing the existing GLM
  pods would be a live-route risk even if product traffic is believed to be
  absent.

## Compatibility Gap

The requested checkpoint was saved from Megatron-Bridge with:

- `ckpt_format=torch_dist`
- `fully_parallel_save=true`
- `TP=2`
- `PP=1`
- `CP=1`
- `EP=8`
- `bf16=true`

The serving stack expects either:

- a Hugging Face-style model directory with `config.json`, tokenizer files, and
  safetensors/bin weights, or
- an SGLang-supported quantized/FP8 model layout.

Therefore the deployment path was not just copying S3 files. The checkpoint
needed an export/merge/convert path from Megatron `torch_dist` into an
SGLang-loadable model format. That export is now complete, and the cloned
isolated SGLang deployment has passed readiness and completion smokes. Routing
still must be changed deliberately if this model should receive live traffic.

## Current Deployment Path

1. Convert/export the Megatron checkpoint.

   Status: complete.

   Used the same Megatron-Bridge/Core versions recorded by the training run:
   Megatron-Bridge `0.4.1`, Megatron-Core `0.17.0rc0`, Torch `2.8.0+cu129`.
   Loaded the `torch_dist` checkpoint and exported to a
   Hugging Face/SGLang-readable model directory.

2. Validate the exported model directory.

   Status: complete.

   Completed checks:

   - `config.json` exists.
   - tokenizer files exist.
   - weight index and all referenced shard files exist.
   - `deploy_model.py` classified the prefix as SGLang-loadable.
   - a one-node SGLang load smoke reached readiness and served a small
     completion.

3. Upload the exported directory to a new S3 prefix.

   Status: complete.

   Do not reuse `s3://llm-weight-files-us-west-2/GLM-5.1-FP8/`.
   The exported candidate uses a distinct prefix and model name:
   `Qwen3.6-35B-A3B-Swift-R13`.

4. Clone GLM deployment manifests instead of overwriting the existing GLM model.

   Status: applied as an isolated service.

   `docs/generated/qwen36_swift_r13_sglang_deployment.yaml` uses independent
   labels, a separate ClusterIP service, and the model/cache name
   `Qwen3.6-35B-A3B-Swift-R13`. The ready pod is
   `qwen36-swift-r13-sglang-76f4c76b45-4vm8d` on
   `hyperpod-i-060621000d6270d47`.

5. Only after the isolated deployment is healthy, wire routing deliberately.

   Status: pending.

   If using the existing proxy, change the route/config explicitly and keep a
   rollback path to the old GLM backend.

## Safe Next Command Pattern

For inspection only:

```bash
aws s3 ls s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/ --recursive --summarize --human-readable --region us-west-2
```

For live deployment, do not point the existing GLM SGLang launcher at the raw
Megatron checkpoint prefix. Use the exported HF/SGLang candidate prefix, and
only wire traffic after an explicit route/config change with a rollback path.
