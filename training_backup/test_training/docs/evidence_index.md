# Evidence Index

Last checked: 2026-06-09 UTC.

This directory summarizes the S3 and Kubernetes evidence collected for the
Qwen3.6-35B-A3B checkpoint deployment and training/loss-alignment work.

Operational runbook:

- `docs/deployment_runbook.md`
- `docs/byteplus_mlp_training_runbook.md`
- `docs/generated/gpu_sharing_lowmem_runbook_20260615.md`
- `docs/generated/qwen36_mb_swift_hypd_smoke_validation_20260616.md`
- `docs/generated/qwen36_mb_swift_full_repro_readiness_20260616.md`
- `docs/generated/qwen36_mb_swift_b200_fullseq_cp4_smoke_validation_20260616.md`
- `docs/generated/hyperpod_goal_completion_audit_20260616.md`
- `docs/generated/qwen36_mb_swift_b200_adjusted_online_validation_20260617.md`
- `docs/generated/qwen36_mb_swift_b200_adjusted_formal_submission_20260617.md`

Current HyperPod training status as of 2026-06-16 UTC:

- The SageMaker HyperPod CLI submission path is validated on
  `eks-cluster-hypd-0324`.
- `dolly-probe-training`, PVC `s3-verdent-tech-use1`, and ConfigMap
  `qwen36-mb-overlay-scripts-r14` are present.
- The full B300 reproduction scaffold preserves the reference bare K8s Job
  recipe and passes server-side dry-run on both `eks-cluster-hypd-0324` and
  `Algorithm-training`.
- B200 smoke evidence includes both a S16K smoke and a 128K full-sequence CP4
  smoke with `status=diagnostic_train_completed`.
- Exact B300 reproduction is still blocked by current capacity:
  `eks-cluster-hypd-0324` has seven `ml.p6-b200.48xlarge` nodes and one
  `ml.c5.2xlarge`; `Algorithm-training` has one `ml.c5.2xlarge`; neither has a
  schedulable `ml.p6-b300.48xlarge` node.

Updated B200 online target status as of 2026-06-17 UTC:

- The goal was changed from exact B300 reproduction to a B200 adjusted online
  run.
- `qwen36-mb-swift-b200-adjusted-online` was submitted through
  `/home/ubuntu/.local/bin/hyp create`.
- The job ran on `eks-cluster-hypd-0324` B200 capacity with full
  `SEQ_LENGTH=128000`, `CP=4`, 8 GPU, and 8 EFA.
- It completed 2 iterations, saved a checkpoint, ran validation, and wrote
  `status=diagnostic_train_completed`.
- The HyperPodPyTorchJob CR remains present for management visibility; the
  child Deployment was scaled to `0` after completion to release B200 resources.

Formal B200 adjusted run status as of 2026-06-17 UTC:

- The formal run `qwen36-mb-swift-b200-adjusted-formal` was submitted through
  `/home/ubuntu/.local/bin/hyp create`.
- It removes the smoke bounds used by the online validation run: no
  `TRAIN_ITERS_OVERRIDE`, no `MAX_TRAIN_SAMPLES`, no `EXIT_AFTER_TRAIN`, and
  baseline evaluation is not skipped.
- With the current dataset manifest, the formal run computes
  `TRAIN_ITERS=1868`, baseline train eval iters `934`, and baseline test eval
  iters `105`.
- It is running on `eks-cluster-hypd-0324` B200 capacity with full
  `SEQ_LENGTH=128000`, `CP=4`, 8 GPU, and 8 EFA.
- Initial Kubernetes state was Deployment `1/1`, pod `Running`/`Ready`, and
  `0` restarts on node `hyperpod-i-0cd50aacaabb9612d`.
- Initial S3 artifacts include `run_context.json`,
  `base_import/node0/base_checkpoint_import_manifest.json`,
  `logs/node0_03_tokenizer_patch.log`, `tokenizer_patch_manifest.json`, and
  `baseline_train_eval/baseline_eval_launch_manifest.json`.
- Later live checks showed the pod still `Running`/`Ready` with `0` restarts
  and baseline train eval progressing, but the `HyperPodPyTorchJob` status
  stayed `Created` with `PodsRunning=False` because podmanager could not reach
  an HPEA `/status` endpoint on port `8080`.
- This explains why Training History can be empty even though the raw
  Kubernetes workload is running: the custom manifest runs the training command
  directly and does not expose the HyperPod Elastic Agent status API expected by
  the operator/UI.
- Full stop is available through `hyp delete hyp-pytorch-job`; GPU/EFA release
  while keeping the CR visible is available by scaling the child Deployment to
  `0`.

## AWS and Kubernetes Access

- The default IAM role is usable from this workspace when commands run outside
  the local sandbox.
- Observed identity:
  `arn:aws:sts::288761729453:assumed-role/eks-hypd-workspace-03240928-admin-role/i-0d6ac9306158cc9f6`.
- EKS kubeconfig used for inspection:
  `/tmp/eks-cluster-hypd-0324.kubeconfig`.
- EKS cluster: `eks-cluster-hypd-0324`, region `us-west-2`.
- The workspace IAM identity can list both the raw checkpoint prefix and the
  exported HF/SGLang prefix.
- The GLM namespace `default` service account has no AWS role annotation.
  `sglang-s3-access` has an IRSA role annotation, but earlier conversion
  attempts showed it is scoped to the existing GLM weight bucket rather than
  `s3://verdent.tech/test_training/...`.
- EKS Pod Identity associations are not configured for `glm51-pd-b200-p1d1`.
- Current IAM/S3 details are recorded in
  `docs/generated/k8s_iam_s3_access_20260609.md`.
- The 2026-06-09 S3 refresh, including B300 and loss/mask evidence paths, is
  recorded in `docs/generated/s3_evidence_refresh_20260609.md`.

## Target Checkpoint

Requested checkpoint:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`

Verified contents:

- Total objects: 15.
- Total size: 70,312,965,895 bytes, about 65.5 GiB as reported by `aws s3 ls`.
- Checkpoint iteration: `iter_0001035`.
- Backend metadata:
  `{"sharded_backend": "torch_dist", "common_backend": "torch"}`.
- Main shard files:
  `iter_0001035/__0_0.distcp` through `iter_0001035/__7_0.distcp`.
- Other files:
  `iter_0001035/.metadata`, `common.pt`, `metadata.json`,
  `run_config.yaml`, `train_state.pt`,
  `latest_checkpointed_iteration.txt`, `latest_train_state.pt`.

Small evidence files were copied to `/tmp/test_training_evidence/`:

- `target_run_config.yaml`
- `target_metadata.json`
- `target_checkpoint_copy_manifest.json`
- `target_run_context.json`
- `target_train_config.yaml`

Local generated deployment evidence:

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
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_probe_job.yaml`
- `docs/generated/qwen36_swift_r13_s3csi_hfhub_conversion_job.yaml`
- `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml`
- `docs/generated/run_qwen36_sglang_smoke_with_restore.sh`
- `docs/generated/qwen36_swift_r13_sglang_deployment.yaml`
- `docs/generated/gpu_allocation_20260609.md`
- `docs/generated/k8s_iam_s3_access_20260609.md`
- `docs/generated/s3_evidence_refresh_20260609.md`
- `docs/generated/qwen36_swift_r13_s3csi_conversion_attempt.md`
- `docs/generated/qwen36_swift_r13_hf_export_preflight.json`

The first two JSON files were produced by `python3 deploy_model.py` using the
default checkpoint URI and GLM namespace settings. The apply attempt exited
with code `2` after refusing live deployment because the checkpoint is not
SGLang-loadable.

The TOS metadata preflight was produced with `--skip-k8s` against the copied
`checkpoint-854_metadata` prefix. It found config/tokenizer/index files, but the
16 safetensors shards referenced by `model.safetensors.index.json` are missing
from that prefix.

The conversion manifests use the valid historical AWS DLC image and archived
runtime setup script. Both corrected manifests passed Kubernetes server dry-run.
The first smoke attempt failed because the run-context image string contained an
invalid `+source...` Docker tag suffix.

Later smoke attempts showed that the default service account has no AWS
credentials, while `sglang-s3-access` has credentials but is scoped only to the
GLM weight bucket. The proposed policy JSON documents the minimum extra S3
scope needed for this checkpoint conversion; it was not applied.

S3 CSI provided a safer read/write path without changing the shared GLM IAM
role:

- The cluster has `s3.csi.aws.com` installed, and the S3 CSI node role has S3
  access.
- A read-only prefix mount for the target run succeeded on
  `hyperpod-i-060621000d6270d47`.
- The smoke Job confirmed the requested checkpoint has 15 files,
  70,312,965,895 bytes, latest iteration `1035`, and `torch_dist` metadata.
- A runtime smoke installed the archived Megatron-Bridge stack and confirmed
  `AutoBridge.can_handle("Qwen/Qwen3.6-35B-A3B")`.
- A metadata preflight confirmed the copied `checkpoint-854_metadata` prefix can
  create a `SafeTensorsStateSource` with 1,026 keys and 16 referenced shards,
  without loading full HF weights.
- The first full S3 CSI export Job failed during `AutoBridge.export_ckpt(...)`
  because the metadata-only HF source lacks the referenced safetensors shards.
- A later HF Hub source probe confirmed `Qwen/Qwen3.6-35B-A3B` is accessible
  from the conversion pod with 26 safetensors shards and 1,045 indexed keys.
- The HF Hub source export completed locally, converted all 21,276 Bridge
  tensors, and wrote a complete HF export to S3. The Kubernetes Job is marked
  failed only because a final `cp -a` attempted to preserve directory
  timestamps on the S3 CSI mount after files had already copied. The manifest
  has since been updated to use `cp -r`.
- The exported prefix now has 34 objects, 70,234,598,273 bytes,
  `config.json`, tokenizer assets, `model.safetensors.index.json`, and all
  26 safetensors shards.
- `docs/generated/qwen36_swift_r13_hf_export_preflight.json` now reports
  `verdict=ready_for_isolated_sglang_deploy`, `sglang_loadable=true`, and
  `blockers=[]`.

Exported HF/SGLang candidate:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Isolated SGLang deployment evidence:

- `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml` was generated by
  `deploy_model.py` for model/cache name `Qwen3.6-35B-A3B-Swift-R13`.
- The restore wrapper
  `docs/generated/run_qwen36_sglang_smoke_with_restore.sh` saved and restored
  the approved deployment `glm51-pd-b200-p1d1/sglang-decode-anchor`; its
  original replica count was `0`.
- Live smoke completed on `hyperpod-i-060621000d6270d47`. SGLang reached
  `/health` HTTP 200 and returned HTTP 200 from `/v1/chat/completions`.
- The first run exposed a FlashInfer warmup autotune shape mismatch; the
  generated smoke and deployment manifests now pass
  `--disable-flashinfer-autotune`.
- The isolated long-running Deployment/Service manifest is generated by
  `deploy_model.py` with `--output-isolated-sglang-deployment-yaml`.
- `docs/generated/qwen36_swift_r13_sglang_deployment.yaml` is applied in
  namespace `glm51-pd-b200-p1d1`.
- Current live service:
  `qwen36-swift-r13-sglang`, ClusterIP `172.20.17.137`, endpoint
  `10.90.17.128:30000`.
- Ready pod:
  `qwen36-swift-r13-sglang-76f4c76b45-4vm8d` on
  `hyperpod-i-060621000d6270d47`.
- The isolated service is not wired to the existing GLM router or external
  proxy. Routing remains the only pending deployment step.
- GLM serving deployments remain intentionally scaled to `0/0`; see
  `docs/generated/gpu_allocation_20260609.md`.

Important target run facts:

- Base model: `Qwen/Qwen3.6-35B-A3B`.
- Training job: `qwen36_swift_all_full_r13`.
- Run id: `qwen36-swift-fullsft-20260527-205700`.
- Image:
  historical K8s manifests used
  `763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39-soci`.
  The run context also recorded a descriptive `+source-megatron-bridge...`
  suffix, but that suffix is not a valid Docker image tag.
- Runtime recorded H200 GPUs, driver `580.126.09`, CUDA `13.0`.
- Megatron-Bridge `0.4.1`, Megatron-Core `0.17.0rc0`.
- Parallelism: `TP=2`, `PP=1`, `CP=1`, `EP=8`.
- `global_batch_size=16`, `micro_batch_size=1`, `train_iters=1035`.
- `ckpt_format=torch_dist`, `fully_parallel_save=true`,
  `fully_parallel_load=false`.

## HyperPod / Kubernetes GPU Machines

Kubernetes reported five HyperPod nodes:

| Node | Internal IP | GPU | EFA | Notes |
| --- | --- | ---: | ---: | --- |
| `hyperpod-i-01e93a89cb27a5046` | `10.90.19.34` | 8 | 8 | currently `dpsk-v4-pro-standalone-b200-p1d1` standalone |
| `hyperpod-i-060621000d6270d47` | `10.90.18.143` | 8 | 8 | currently isolated Qwen36 SGLang |
| `hyperpod-i-0b7611eea4f8e7956` | `10.90.16.43` | 8 | 8 | Kimi prefill/router |
| `hyperpod-i-0da428942dd5a9cd9` | `10.90.18.207` | 8 | 8 | Kimi decode/monitor |
| `hyperpod-i-0e1a42052af8e595e` | `10.90.17.217` | 0 | 0 | small non-GPU/system node |

GLM namespace:

- Namespace: `glm51-pd-b200-p1d1`.
- Deployments: `qwen36-swift-r13-sglang` is `1/1`; `sglang-prefill`,
  `sglang-decode-anchor`, `sglang-decode-pool`, `sglang-router`, and
  `sglang-monitor` are `0/0`.
- The isolated Qwen36 service is not connected to the GLM router.
- Current GLM model: `GLM-5.1-FP8`.
- Current GLM model S3 URI: `s3://llm-weight-files-us-west-2/GLM-5.1-FP8/`.
- GLM historically used two 8-GPU nodes, one for prefill and one for decode.
  One former GLM node now runs the isolated Qwen36 deployment; the other is
  occupied by a `dpsk-v4-pro-standalone-b200-p1d1` pod.

Traffic/routing note:

- Default proxy ConfigMap `sglang-router-b200-proxy-config` currently has
  `PREFERRED_AWS_GLM51_BACKEND=b200`.
- The default external route can still prefer the GLM B200 backend. Treat the
  GLM deployment as potentially externally reachable unless this routing is
  changed or the service is explicitly isolated.

## Related S3 Evidence

Top-level evidence roots found under `s3://verdent.tech/test_training/`:

- `sft_opus_sonnet_swift/`
- `qwen36_megatron_bridge_overlay_sft/`
- `loss_alignment_full_repro/`
- `qwen36_cp_alignment/`
- `qwen36_cp2_qwen_text_sft/`
- `qwen36_packaged_qwen_text_sft/`
- `qwen36_cleanv5_lowdup_packed_sft/`
- `qwen36_cp_upgrade/`
- `qwen36_quality_swift500_sft/`
- `dolly_probe/`

Downloaded evidence files:

- `mb_comparison_report.md`
- `loss_alignment_update_20260603.md`
- `cp_alignment_report.md`
- `mask_snapshots_summary.json`
- `cleanv5_baseline_status_completed.json`
- `setup_nemo2604_overlay_runtime.sh`
- `run_qwen36_overlay_sft_job.sh`
- `prepare_qwen36_sft_dataset.py`
- `bridge_qwen36_train.py`
- `21-train-swift-job.yaml`

B300-specific evidence:

- S3 object:
  `s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/artifacts/source/k8s/21-train-swift-job.yaml`
- Local copy: `/tmp/test_training_evidence/21-train-swift-job.yaml`
- Node selector: `node.kubernetes.io/instance-type=ml.p6-b300.48xlarge`
- Image: `nvcr.io/nvidia/nemo:26.04.00`
- Resources: 8 GPUs, 16 EFA, 24Ti scratch, 128Gi `/dev/shm`
- Key flags: `TP=2`, `PP=1`, `CP=2`, `EP=8`,
  `SEQ_LENGTH=128000`, `CALCULATE_PER_TOKEN_LOSS=true`,
  `DDP_AVERAGE_IN_COLLECTIVE=false`

## TOS/Public Checkpoint Access

The path

`s3://verdent-tos-public/public/task_test_trajs/full_repro_20260603/checkpoints/checkpoint-854/`

could not be listed through normal AWS S3. The AWS endpoint reported
`NoSuchBucket`. Probing the BytePlus/TOS S3-compatible endpoint in
`ap-southeast-1` showed that the bucket exists, but anonymous list/read access
is denied. The loss-alignment evidence says the TOS package contains a
`checkpoint-854` category with 26 files and 70,234,626,365 bytes, but direct
listing requires signed TOS credentials.
