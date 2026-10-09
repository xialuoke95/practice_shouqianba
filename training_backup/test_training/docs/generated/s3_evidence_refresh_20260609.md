# S3 Evidence Refresh

Checked: 2026-06-09 UTC.

## Top-Level Evidence Roots

`s3://verdent.tech/test_training/` currently exposes these top-level prefixes:

- `docker_image_archive/`
- `dolly_probe/`
- `loss_alignment_full_repro/`
- `qwen36_cleanv5_lowdup_packed_sft/`
- `qwen36_cp2_qwen_text_sft/`
- `qwen36_cp_alignment/`
- `qwen36_cp_upgrade/`
- `qwen36_megatron_bridge_overlay_sft/`
- `qwen36_packaged_qwen_text_sft/`
- `qwen36_quality_swift500_sft/`
- `sft_opus_sonnet_swift/`

## B300 Megatron-Bridge Overlay Evidence

The strongest B300-specific training environment evidence is:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/artifacts/source/k8s/21-train-swift-job.yaml`

The local downloaded copy is:

`/tmp/test_training_evidence/21-train-swift-job.yaml`

Key properties from that manifest:

- Kubernetes Job: `qwen36-mb-swift-aligned-a11`
- Namespace: `dolly-probe-training`
- Node selector: `node.kubernetes.io/instance-type=ml.p6-b300.48xlarge`
- Image: `nvcr.io/nvidia/nemo:26.04.00`
- GPUs: `nvidia.com/gpu=8`
- EFA: `vpc.amazonaws.com/efa=16`
- Scratch: `emptyDir` size limit `24000Gi`
- `/dev/shm`: memory `emptyDir` size limit `128Gi`
- S3 mount: PVC `s3-verdent-tech-use1` at `/s3-verdent`
- Parallelism/config: `TP=2`, `PP=1`, `CP=2`, `EP=8`,
  `SEQ_LENGTH=128000`, `GBS=16`, `MBS=1`
- Loss flags: `CALCULATE_PER_TOKEN_LOSS=true`,
  `DDP_AVERAGE_IN_COLLECTIVE=false`
- Data: `sft_opus_sonnet_swift_mbjson_v2`

Related scripts and runtime artifacts:

- `artifacts/source/scripts/run_qwen36_overlay_sft_job.sh`
- `artifacts/source/scripts/setup_nemo2604_overlay_runtime.sh`
- `artifacts/source/scripts/bridge_qwen36_train.py`
- `artifacts/source/scripts/prepare_qwen36_sft_dataset.py`
- `artifacts/source/k8s/qwen36-mb-overlay-scripts-r14.yaml`

## Loss/Mask Alignment Evidence

The strongest cross-platform loss-alignment root is:

`s3://verdent.tech/test_training/loss_alignment_full_repro/qwen36-cleanv5-align-20260603/`

Important objects observed in the refresh:

- `evidence_20260603/reports/loss_alignment_update_20260603.md`
- `evidence_20260603/reports/loss_alignment_evidence_index_20260603.json`
- `evidence_20260603/source_artifacts_v4/generated/mask_snapshots.jsonl`
- `evidence_20260603/source_artifacts_v4/generated/mask_snapshots_summary.json`
- `evidence_20260603/run_evidence/pfs_run_args.json`
- `evidence_20260603/run_evidence/pfs_run_logging.jsonl`
- `evidence_20260603/scripts/prepare_swift_exact_packed_npy.py`
- `swift_exact_packed/first64_left16k_seqstartlist/manifest.json`
- `training/qwen36_cleanv5_swiftpacked_first64_r3/status/cleanv5_baseline_status_completed_20260603T115408463728Z.json`

Important values from the local evidence:

- Other-platform step-1 training loss: `0.64677322`
- Local exact SWIFT-mask packed Bridge eval: `0.6624455`
- Relative difference: about `2.4%`
- Mask rendering method:
  `AutoTokenizer.apply_chat_template` after role normalization
  `tool_call -> assistant.tool_calls`, `tool_response -> tool`
- Sampled trainable source counts: `assistant=341`, `tool_call=584`
- Sampled `loss=0` masked counts: `assistant=147`, `tool_call=233`
- Truncation: no sampled row truncated at `max_length=128000`

## Checkpoint Deployment Evidence

The requested raw checkpoint remains:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`

The exported HF/SGLang candidate remains:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Current `deploy_model.py` evidence:

- Raw prefix verdict: `blocked_before_live_deploy`
- Raw prefix reason: Megatron `torch_dist`/`.distcp`, not a direct SGLang
  model directory
- Exported prefix verdict: `ready_for_isolated_sglang_deploy`
- Exported prefix: 34 objects, 70,234,598,273 bytes, complete 26-shard
  safetensors index

Live isolated SGLang result:

- Smoke pod: `qwen36-swift-r13-sglang-smoke-20260608-8prtb`
- Smoke node: `hyperpod-i-060621000d6270d47`
- SGLang started with `--disable-flashinfer-autotune` after the first run
  exposed a FlashInfer warmup autotune shape mismatch.
- `/health` returned HTTP 200.
- `/v1/chat/completions` returned HTTP 200.
- The temporary smoke Job was deleted after evidence was recorded.

Current isolated service:

- Deployment: `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`, `1/1` Ready.
- Pod: `qwen36-swift-r13-sglang-76f4c76b45-4vm8d`.
- Node: `hyperpod-i-060621000d6270d47`.
- ClusterIP: `172.20.17.137`.
- Endpoint: `10.90.17.128:30000`.
- Not wired to the existing GLM router or external proxy.
