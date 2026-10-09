# Test Training Notes

Last checked: 2026-06-09 UTC.

Start here:

- [Evidence index](evidence_index.md): S3 roots, target checkpoint facts,
  IAM/Kubernetes access, HyperPod GPU nodes, and TOS access status.
- [Checkpoint deployment assessment](checkpoint_deployment_assessment.md):
  `deploy_model.py` preflight/apply evidence, why the requested Megatron
  `torch_dist` checkpoint was not safe to deploy directly into the current
  GLM/SGLang service, and the safe conversion path.
- [Deployment runbook](deployment_runbook.md):
  exact regenerate, dry-run, live smoke, isolated service, and rollback
  commands for the exported candidate.
- [Megatron-Bridge West-US training environment notes](megatron_bridge_west_us_training_env.md):
  working runtime, overlay setup, checkpoint handling, and operational lessons.
- [BytePlus MLP training submission runbook](byteplus_mlp_training_runbook.md):
  Volc CLI/OpenAPI setup, resource checks, custom-task YAML patterns, submit
  commands, monitoring, and 35A3B no-think SFT pitfalls.
- [Loss alignment, data cleaning, and mask lessons](loss_alignment_data_mask_lessons.md):
  clean_v5 data mismatch, SWIFT mask reproduction, packed Bridge findings, and
  future data-governance checklist.
- [GPU sharing + low-memory SGLang runbook](generated/gpu_sharing_lowmem_runbook_20260615.md):
  qwen36/B200 GPU sharing, SGLang memory reduction, colocated smoke tests, and
  the 2026-06-16 `--mem-fraction-static 0.10` full-SFT canary lessons.

Current deployment conclusion:

The requested checkpoint is present on S3 as a Megatron distributed checkpoint
(`torch_dist`/`.distcp`). It was not safe to point the live GLM SGLang launcher
at that raw prefix directly. Using S3 CSI on the GLM HyperPod/EKS cluster, the
checkpoint was copied to NVMe and exported with Megatron-Bridge using
`Qwen/Qwen3.6-35B-A3B` as the complete HF state source.

The exported HF/SGLang-loadable prefix is now complete:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

`deploy_model.py` preflight reports `ready_for_isolated_sglang_deploy` with
34 objects, 70,234,598,273 bytes, tokenizer/config files, and a complete
26-shard safetensors index. An isolated one-node SGLang load smoke under the
new model/cache name `Qwen3.6-35B-A3B-Swift-R13` has passed, and the isolated
Deployment/Service is now Ready in `glm51-pd-b200-p1d1`.

As of the 2026-06-09 UTC refresh, the live isolated service is
`qwen36-swift-r13-sglang`, backed by pod
`qwen36-swift-r13-sglang-76f4c76b45-4vm8d` on
`hyperpod-i-060621000d6270d47`. Its ClusterIP is `172.20.17.137` and endpoint
is `10.90.17.128:30000`. The original GLM serving deployments remain scaled to
`0/0`; this isolated service is not wired to the existing GLM router or
external proxy.

`deploy_model.py --apply` was run against the requested checkpoint and refused
live mutation because the preflight blocker is still present. Generated evidence:

- [Preflight JSON](generated/qwen36_swift_r13_deploy_preflight.json)
- [Apply-attempt JSON](generated/qwen36_swift_r13_deploy_apply_attempt.json)
- [TOS checkpoint-854 metadata preflight JSON](generated/tos_checkpoint854_metadata_preflight.json)
- [Conversion Job YAML](generated/qwen36_swift_r13_conversion_job.yaml)
- [Conversion Smoke Job YAML](generated/qwen36_swift_r13_conversion_smoke_job.yaml)
- [Conversion smoke attempt note](generated/qwen36_swift_r13_conversion_smoke_attempt.md)
- [Proposed GLM S3 role policy](generated/glm_s3_reader_test_training_policy.json)
- [S3 CSI checkpoint smoke Job YAML](generated/qwen36_swift_r13_s3csi_smoke_job.yaml)
- [S3 CSI runtime smoke Job YAML](generated/qwen36_swift_r13_s3csi_runtime_smoke_job.yaml)
- [S3 CSI HF metadata preflight Job YAML](generated/qwen36_swift_r13_s3csi_export_preflight_job.yaml)
- [S3 CSI full export Job YAML](generated/qwen36_swift_r13_s3csi_conversion_job.yaml)
- [S3 CSI HF Hub source probe Job YAML](generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml)
- [S3 CSI HF Hub export Job YAML](generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml)
- [Isolated SGLang smoke Job YAML](generated/qwen36_swift_r13_sglang_smoke_job.yaml)
- [SGLang smoke with GLM restore script](generated/run_qwen36_sglang_smoke_with_restore.sh)
- [Isolated SGLang Deployment/Service YAML](generated/qwen36_swift_r13_sglang_deployment.yaml)
- [S3 CSI conversion attempt note](generated/qwen36_swift_r13_s3csi_conversion_attempt.md)
- [Post-export preflight JSON](generated/qwen36_swift_r13_hf_export_preflight.json)
- [GPU allocation snapshot](generated/gpu_allocation_20260609.md)
- [K8s IAM and S3 access snapshot](generated/k8s_iam_s3_access_20260609.md)
- [S3 evidence refresh](generated/s3_evidence_refresh_20260609.md)
- [Goal completion audit](goal_completion_audit_20260609.md)
