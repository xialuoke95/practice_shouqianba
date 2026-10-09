# Qwen36 Swift R13 Deployment Runbook

Last checked: 2026-06-09 UTC.

## Scope

This runbook is for the exported deployment candidate derived from the raw
Megatron checkpoint:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/training/qwen36_swift_all_full_r13/checkpoint_copy/`

The raw checkpoint is `torch_dist`/`.distcp` and must not be served directly by
SGLang. The exported HF/SGLang candidate is:

`s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/`

Current status:

- HF export is complete.
- `deploy_model.py` preflight reports `ready_for_isolated_sglang_deploy`.
- Smoke Job and isolated Deployment/Service manifests pass Kubernetes
  server-side dry-run.
- Live isolated SGLang smoke completed successfully on
  `hyperpod-i-060621000d6270d47`.
- Isolated Deployment/Service
  `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang` is applied and Ready.
- The isolated service is not wired to the existing GLM router or external
  proxy.
- GLM serving deployments remain intentionally scaled to `0/0`.
- The current GPU allocation snapshot is
  `docs/generated/gpu_allocation_20260609.md`.

## Generated Artifacts

- Preflight:
  `docs/generated/qwen36_swift_r13_hf_export_preflight.json`
- Isolated smoke Job:
  `docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml`
- Guarded smoke runner:
  `docs/generated/run_qwen36_sglang_smoke_with_restore.sh`
- Isolated Deployment/Service:
  `docs/generated/qwen36_swift_r13_sglang_deployment.yaml`

The smoke and deployment use model/cache name
`Qwen3.6-35B-A3B-Swift-R13`, separate from the existing `GLM-5.1-FP8` cache.

## Regenerate Manifests

```bash
python3 deploy_model.py \
  --checkpoint-s3-uri s3://verdent.tech/test_training/sft_opus_sonnet_swift/qwen36-swift-fullsft-20260527-205700/deployment_exports/qwen36_swift_all_full_r13_hf/ \
  --sglang-node hyperpod-i-060621000d6270d47 \
  --output-json docs/generated/qwen36_swift_r13_hf_export_preflight.json \
  --output-sglang-smoke-job-yaml docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml \
  --output-sglang-smoke-restore-script docs/generated/run_qwen36_sglang_smoke_with_restore.sh \
  --output-isolated-sglang-deployment-yaml docs/generated/qwen36_swift_r13_sglang_deployment.yaml
```

Expected preflight:

- `verdict=ready_for_isolated_sglang_deploy`
- `sglang_loadable=true`
- `blockers=[]`
- `object_count=34`
- `total_bytes=70,234,598,273`
- `has_complete_safetensors_index=true`

## Validate Without Mutation

```bash
python3 -m py_compile deploy_model.py
bash -n docs/generated/run_qwen36_sglang_smoke_with_restore.sh
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply --dry-run=server -f docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply --dry-run=server -f docs/generated/qwen36_swift_r13_sglang_deployment.yaml
```

The current dry-run results were:

- `job.batch/qwen36-swift-r13-sglang-smoke-20260608 created (server dry run)`
- `deployment.apps/qwen36-swift-r13-sglang unchanged (server dry run)`
- `service/qwen36-swift-r13-sglang unchanged (server dry run)`

## Live Smoke Result

The guarded runner is parameterized. It operated on the approved deployment
`glm51-pd-b200-p1d1/sglang-decode-anchor`, whose original replica count was
`0`.

```bash
bash docs/generated/run_qwen36_sglang_smoke_with_restore.sh
```

Observed result on 2026-06-09 UTC:

- Smoke pod: `qwen36-swift-r13-sglang-smoke-20260608-8prtb`
- Node: `hyperpod-i-060621000d6270d47`
- Model cache: `/opt/dlami/nvme/models/Qwen3.6-35B-A3B-Swift-R13`
- SGLang started with `--disable-flashinfer-autotune`.
- `GET /health` returned HTTP 200.
- `POST /v1/chat/completions` returned HTTP 200.
- The runner restored `sglang-decode-anchor` to `0` replicas.
- The completed temporary smoke Job was deleted after evidence was recorded.

## Apply Isolated Service

The isolated service was applied after smoke succeeded:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply -f docs/generated/qwen36_swift_r13_sglang_deployment.yaml
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 rollout status deploy/qwen36-swift-r13-sglang --timeout=1800s
```

The isolated service is `qwen36-swift-r13-sglang` in namespace
`glm51-pd-b200-p1d1`. It is a ClusterIP service and is not wired to the existing
GLM router or default proxy.

Current live state:

- Deployment: `1/1` Ready.
- Ready pod: `qwen36-swift-r13-sglang-76f4c76b45-4vm8d`
- Node: `hyperpod-i-060621000d6270d47`
- Pod IP: `10.90.17.128`
- Service ClusterIP: `172.20.17.137`
- Endpoint: `10.90.17.128:30000`

Verification from inside the live pod:

- `GET http://127.0.0.1:30000/health` returned HTTP `200`.
- ClusterIP `POST /v1/chat/completions` returned HTTP 200 JSON with
  `usage.total_tokens=17`.

## Output Verification

The deployment is serving Qwen36 under model name
`Qwen3.6-35B-A3B-Swift-R13`. The model was exported from the raw
`qwen36_swift_all_full_r13/checkpoint_copy/` Megatron checkpoint into the
HF/SGLang prefix documented above.

For behavioral checks, use a prompt with enough `max_tokens` and pass
`chat_template_kwargs.enable_thinking=false` when the expected product output
should be final-answer-only. Without that flag, this Qwen chat template can
emit a `Thinking Process` prefix before the answer, which proves generation is
working but is not the right output surface for normal product traffic.

Example in-cluster verification:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig exec -n glm51-pd-b200-p1d1 qwen36-swift-r13-sglang-76f4c76b45-4vm8d -- \
  curl -sS -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-Swift-R13","messages":[{"role":"system","content":"Reply with the final answer only. Do not explain."},{"role":"user","content":"What is 2+3?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  http://qwen36-swift-r13-sglang:30000/v1/chat/completions
```

Observed result:

- `content="5"`
- `finish_reason="stop"`
- `usage.completion_tokens=2`

Chinese smoke with the same flag returned `content="适合。"` for a simple
one-sentence prompt.

Smoke the isolated service from inside the cluster before any routing change:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 run qwen36-swift-r13-client --rm -i --restart=Never --image=curlimages/curl:8.10.1 -- \
  curl -sS http://qwen36-swift-r13-sglang:30000/health
```

## Rollback

If the smoke or isolated deployment fails:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 delete job qwen36-swift-r13-sglang-smoke-20260608 --ignore-not-found=true
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 scale deploy qwen36-swift-r13-sglang --replicas=0
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 delete deploy qwen36-swift-r13-sglang --ignore-not-found=true
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig -n glm51-pd-b200-p1d1 delete svc qwen36-swift-r13-sglang --ignore-not-found=true
```

If a serving deployment was scaled down for the smoke, restore it to the exact
replica count captured before the run. Do not assume the correct rollback value
is `1`; as of this check, the GLM serving deployments are intentionally `0/0`.

Do not route traffic to this model until an explicit route/config change is
approved and a rollback path to the old GLM backend is ready.
