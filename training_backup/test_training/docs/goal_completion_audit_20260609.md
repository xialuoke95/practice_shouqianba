# Goal Completion Audit

Checked: 2026-06-09 UTC.

## Objective Items

1. Use the GLM HyperPod/Kubernetes machines for checkpoint deployment, with
   low risk because GLM has no live traffic.
2. Use `deploy_model.py` to deploy the requested checkpoint:
   `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`
3. Search S3 for related evidence and write docs covering:
   - the earlier West-US training checkpoint;
   - the West-US B300 Megatron-Bridge training environment;
   - the West-US/B300 loss-alignment and data-governance lessons, especially
     data cleaning and mask alignment.

## Completion Evidence

Deployment:

- The requested raw checkpoint prefix was inspected and classified by
  `deploy_model.py` as Megatron `torch_dist`/`.distcp`, not a direct
  SGLang-loadable directory.
- Direct live mutation against the raw prefix was attempted through
  `deploy_model.py --apply` and refused before touching Kubernetes; evidence is
  in `docs/generated/qwen36_swift_r13_deploy_apply_attempt.json`.
- The raw checkpoint was converted/exported through the generated S3 CSI and
  Megatron-Bridge path into:
  `s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`
- The exported prefix has 34 objects, 70,234,598,273 bytes, tokenizer/config
  files, `model.safetensors.index.json`, and all 26 referenced safetensors
  shards.
- `deploy_model.py` generated the isolated smoke Job, restore wrapper, and
  Deployment/Service manifests for `Qwen3.6-35B-A3B-Swift-R13`.
- Live smoke passed on GLM node `hyperpod-i-060621000d6270d47`; `/health` and
  `/v1/chat/completions` both returned HTTP 200.
- The isolated Deployment/Service
  `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang` is `1/1` Ready, with endpoint
  `10.90.17.128:30000`.
- The service is intentionally isolated and not wired to the existing GLM
  router or external proxy.

S3 and documentation:

- `docs/evidence_index.md` records the S3 roots, IAM/Kubernetes access, target
  checkpoint facts, GPU machines, TOS access status, and live isolated
  deployment evidence.
- `docs/checkpoint_deployment_assessment.md` records the raw checkpoint
  incompatibility, conversion path, S3 CSI/HF Hub source export, and deployment
  result.
- `docs/deployment_runbook.md` records regenerate, dry-run, smoke, live
  service, and rollback commands.
- `docs/megatron_bridge_west_us_training_env.md` records the West-US training
  runtime, B300 overlay job, NeMo 26.04 setup, Megatron-Bridge/Core versions,
  topology, scratch/shared-memory requirements, S3 mount pattern, and runtime
  patches.
- `docs/loss_alignment_data_mask_lessons.md` records the cross-platform
  loss-alignment evidence, clean_v5 data changes, SWIFT/PFS mask semantics,
  `tool_call` handling, packed Bridge reproduction, CP alignment probes, and a
  future data-governance checklist.

## Remaining Deliberate Non-Action

Routing was not changed. The isolated service is healthy, but receiving live
traffic would require a separate route/config change and rollback plan.
