# Qwen36 Swift R13 S3 CSI Conversion Attempt

Checked: 2026-06-08 UTC.

## Access Path

The AWS S3 CSI driver is installed on the HyperPod/EKS cluster and can mount
`s3://verdent.tech` without broadening the shared GLM `sglang-s3-access` IAM
role.

Relevant manifests:

- `docs/generated/qwen36_swift_r13_s3csi_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_runtime_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_export_preflight_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml`
- `docs/generated/qwen36_swift_r13_sglang_deployment.yaml`

The S3 CSI mounts were prefix-scoped. Read-only mounts were used for the target
run/checkpoint and HF metadata template. The export mount was scoped only to the
planned HF export prefix:

`test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

## Successful Smokes

Checkpoint mount smoke:

- Job: `qwen36-swift-r13-s3csi-smoke-20260608`
- Namespace: `glm51-pd-b200-p1d1`
- Node: `hyperpod-i-060621000d6270d47`
- Result: `Complete`
- Confirmed target checkpoint file count: `15`
- Confirmed total bytes: `70,312,965,895`
- Confirmed latest iteration: `1035`
- Confirmed backend:
  `{"sharded_backend": "torch_dist", "common_backend": "torch"}`

Bridge runtime smoke:

- Job: `qwen36-swift-r13-s3csi-runtime-smoke-20260608`
- Result: `Complete`
- Runtime setup installed Megatron-Bridge `0.4.1` and Megatron-Core
  `0.17.0rc0` from the archived target-run source package.
- Import smoke result:
  `{"can_handle_qwen36": true, "cuda_available": true, "supported_models_count": 34, "torch": "2.8.0+cu129"}`

HF metadata export preflight:

- Job: `qwen36-swift-r13-s3csi-export-preflight-20260608`
- Result: `Complete`
- Metadata source:
  `s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/artifacts/alignment_refs/source_artifacts_v4/pfs/train_output/checkpoint-854_metadata/`
- Bridge detected config class: `Qwen3_5MoeConfig`
- State source: `SafeTensorsStateSource`
- Key count from `model.safetensors.index.json`: `1026`
- Shard count referenced by the index: `16`
- Full HF model weights were not loaded in the preflight:
  `has_loaded_model=false`

This proved the metadata-only prefix can build a Bridge/state-source object, but
it does not prove full export can finish.

## Metadata-Only Export Result

Metadata-only full export job:

- Job: `qwen36-swift-r13-s3csi-hf-export-20260608`
- Manifest:
  `docs/generated/qwen36_swift_r13_s3csi_conversion_job.yaml`
- Namespace: `glm51-pd-b200-p1d1`
- Node: `hyperpod-i-060621000d6270d47`
- Result: `Failed`
- Backoff limit: `0`

Progress before failure:

- Target checkpoint copied from the read-only S3 CSI mount to host NVMe.
- Bridge runtime setup completed.
- `AutoBridge.from_hf_pretrained(...)` succeeded against the metadata-only HF
  template.
- Bridge-ready JSON:
  `{"checkpoint_latest": "1035", "config_class": "Qwen3_5MoeConfig", "hf_key_count": 1026, "loaded_hf_model": false, "phase": "bridge_ready"}`
- Megatron checkpoint loading reached model construction and reported
  `35,107,181,936` parameters on rank `(0, 0)`.
- HF export reached about 3%:
  `Converting to HuggingFace 3% (547/21276) Qwen35VLMoEBridge`.

Failure:

```text
KeyError: "Keys not found in safetensors from
/host-nvme/test_training/qwen36_swift_r13_s3csi_hf_export_20260608/hf_metadata:
{'model.language_model.layers.0.mlp.experts.down_proj'}"
```

Interpretation:

- The S3 CSI access path is good.
- The Megatron checkpoint and archived Bridge runtime are usable on the GLM
  HyperPod node.
- The failure is at the HF state-source side. The selected
  `checkpoint-854_metadata` prefix has config/tokenizer/index files but not the
  16 referenced safetensors shards.
- During full export, Bridge needs to read at least some original HF tensors for
  shape/layout checks. Metadata-only index files are not enough.

## HF Hub Source Export Result

Because the metadata-only TOS checkpoint source was incomplete, the next export
used the public HF source model `Qwen/Qwen3.6-35B-A3B` as the Bridge state
source.

HF Hub source probe:

- Job: `qwen36-swift-r13-s3csi-hfhub-probe-20260608`
- Manifest:
  `docs/generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml`
- Result: `Complete`
- Confirmed the HF repo is public and not gated in this environment.
- Confirmed 26 safetensors shards and 1,045 indexed keys.
- Confirmed the key
  `model.language_model.layers.0.mlp.experts.down_proj` is backed by a real
  shard (`model-00002-of-00026.safetensors`), which was the missing key class
  in the metadata-only attempt.

HF Hub source export job:

- Job: `qwen36-swift-r13-s3csi-hfhub-export-20260608`
- Manifest:
  `docs/generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml`
- Namespace: `glm51-pd-b200-p1d1`
- Node: `hyperpod-i-060621000d6270d47`
- Kubernetes result: `Failed`
- Effective export result: local export completed and the S3 prefix is complete.

Important log evidence:

- Bridge-ready JSON:
  `{"checkpoint_latest": "1035", "config_class": "Qwen3_5MoeConfig", "hf_key_count": 1045, "hf_shard_count": 26, "loaded_hf_model": false, "model_id": "Qwen/Qwen3.6-35B-A3B", "phase": "bridge_ready", "state_source": "SafeTensorsStateSource"}`
- Megatron checkpoint loading reported `35,107,181,936` parameters on rank
  `(0, 0)`.
- Conversion reached 100%:
  `Converting to HuggingFace 100% (21276/21276) Qwen35VLMoEBridge`.
- Bridge printed:
  `Success: All tensors from the original checkpoint were written.`
- Export-complete JSON reported `file_count=34`, `has_config=true`,
  `has_index=true`, and `total_bytes=70,234,598,273`.

The Kubernetes Job status is `Failed` only because the final copy used
`cp -a ... /s3export/`, and S3 CSI does not allow preserving directory
timestamps on the mount:

```text
cp: preserving times for '/s3export/.': Operation not permitted
```

The generated HF Hub conversion manifest has been updated to use `cp -r` for
future reruns.

S3 verification for the exported prefix:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

- Objects: `34`
- Total bytes: `70,234,598,273`
- Includes `config.json`, tokenizer assets, `model.safetensors.index.json`, and
  `model-00001-of-00026.safetensors` through
  `model-00026-of-00026.safetensors`.

`deploy_model.py` post-export preflight was written to:

`docs/generated/qwen36_swift_r13_hf_export_preflight.json`

That preflight now reports:

- `verdict=ready_for_isolated_sglang_deploy`
- `checkpoint_type=hf_or_sglang_model_dir`
- `sglang_loadable=true`
- `object_count=34`
- `total_bytes=70,234,598,273`
- `has_complete_safetensors_index=true`
- `blockers=[]`

## Isolated SGLang Smoke and Deployment

On 2026-06-09 UTC, after `hyperpod-i-060621000d6270d47` was freed, the
isolated SGLang load smoke completed successfully.

Smoke Job:

- Job pod: `qwen36-swift-r13-sglang-smoke-20260608-8prtb`
- Namespace: `glm51-pd-b200-p1d1`
- Node: `hyperpod-i-060621000d6270d47`
- Model/cache name: `Qwen3.6-35B-A3B-Swift-R13`
- Cache path:
  `/opt/dlami/nvme/models/Qwen3.6-35B-A3B-Swift-R13`
- Result: health and completion smoke both succeeded.

Important runtime note:

- The first SGLang run hit a FlashInfer warmup shape mismatch during autotune.
- The generated smoke and deployment manifests now pass
  `--disable-flashinfer-autotune`.
- With that flag, SGLang completed CUDA graph capture, `/health` returned
  HTTP 200, and `/v1/chat/completions` returned HTTP 200.

Smoke evidence:

- The model was already staged locally and reused from the NVMe cache
  (`66G`).
- Completion response id observed during the smoke:
  `453b2f5b4bfb4ac6a6f6cdcdb7b470ec`.
- Completion usage: `prompt_tokens=13`, `completion_tokens=4`,
  `total_tokens=17`.
- The guarded runner restored the approved GLM deployment
  `glm51-pd-b200-p1d1/sglang-decode-anchor` to its original replica count,
  which was `0`.
- The completed temporary smoke Job was deleted after evidence was recorded.

The long-running isolated Deployment/Service was then applied:

- Deployment: `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`
- Service: `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`
- Ready pod: `qwen36-swift-r13-sglang-76f4c76b45-4vm8d`
- Node: `hyperpod-i-060621000d6270d47`
- Pod IP: `10.90.17.128`
- Service ClusterIP: `172.20.17.137`
- Endpoint: `10.90.17.128:30000`
- Rollout status: successfully rolled out.

Post-rollout verification:

- In-pod `GET http://127.0.0.1:30000/health` returned HTTP `200`.
- ClusterIP `POST /v1/chat/completions` returned HTTP 200 JSON with response
  id `11b23cd5d8384ebe916403efaef56d0f` and
  `prompt_tokens=13`, `completion_tokens=4`, `total_tokens=17`.
- `python3 -m py_compile deploy_model.py` passed.
- `bash -n docs/generated/run_qwen36_sglang_smoke_with_restore.sh` passed.
- Smoke Job manifest server dry-run:
  `job.batch/qwen36-swift-r13-sglang-smoke-20260608 created (server dry run)`.
- Deployment/Service manifest server dry-run:
  `deployment.apps/qwen36-swift-r13-sglang unchanged (server dry run)` and
  `service/qwen36-swift-r13-sglang unchanged (server dry run)`.

The isolated service is not wired to the existing GLM router or external proxy.
The original GLM serving deployments remain intentionally scaled to `0/0`.

The deployment manifest uses `strategy.type=Recreate` and 10 second startup and
readiness probe timeouts. `Recreate` avoids a one-replica, whole-node GPU
rolling-update deadlock where the replacement pod cannot be admitted until the
old 8-GPU pod exits.
