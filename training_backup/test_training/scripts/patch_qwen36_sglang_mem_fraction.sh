#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-dry-run}"
MEM_FRACTION="${MEM_FRACTION:-0.60}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-false}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KUBECONFIG="${KUBECONFIG:-/tmp/eks-cluster-hypd-0324.kubeconfig}"
NAMESPACE="${NAMESPACE:-glm51-pd-b200-p1d1}"
DEPLOYMENT="${DEPLOYMENT:-qwen36-swift-r13-sglang}"
CONTAINER="${CONTAINER:-sglang}"

case "${ACTION}" in
  dry-run|apply|rollback|status) ;;
  *)
    echo "Usage: MEM_FRACTION=0.60 $0 [dry-run|apply|rollback|status]" >&2
    exit 2
    ;;
esac

kubectl_cmd=(env KUBECONFIG="${KUBECONFIG}" kubectl)

show_status() {
  "${kubectl_cmd[@]}" get deployment -n "${NAMESPACE}" "${DEPLOYMENT}" -o wide
  pod="$("${kubectl_cmd[@]}" get pod -n "${NAMESPACE}" \
    -l app.kubernetes.io/name="${DEPLOYMENT}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "${pod}" ]]; then
    "${kubectl_cmd[@]}" top pod -n "${NAMESPACE}" "${pod}" --containers || true
    "${kubectl_cmd[@]}" exec -n "${NAMESPACE}" "${pod}" -c "${CONTAINER}" -- \
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total \
      --format=csv,noheader,nounits || true
  fi
}

if [[ "${ACTION}" == "status" ]]; then
  show_status
  exit 0
fi

if [[ "${ACTION}" == "rollback" ]]; then
  MEM_FRACTION="0.82"
fi

current="$("${kubectl_cmd[@]}" get deployment -n "${NAMESPACE}" "${DEPLOYMENT}" \
  -o jsonpath='{.spec.template.spec.containers[0].args[0]}')"

if [[ "${current}" != *"--mem-fraction-static "* ]]; then
  echo "Could not find --mem-fraction-static in ${NAMESPACE}/${DEPLOYMENT}" >&2
  exit 1
fi

new="$(
  CURRENT_ARGS="${current}" \
  PATCH_MEM_FRACTION="${MEM_FRACTION}" \
  PATCH_DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH}" \
  PATCH_CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
  PATCH_KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
  python3 - <<'PY'
import os
import re

text = os.environ["CURRENT_ARGS"]


def insert_before_flag(source: str, anchor_flag: str, new_line: str) -> str:
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if anchor_flag in line:
            lines.insert(idx, new_line)
            return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    lines.append(new_line)
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def upsert_value_flag(source: str, flag: str, value: str, anchor_flag: str) -> str:
    if not value:
        return source
    pattern = re.compile(rf"(?m)^(\s*){re.escape(flag)}\s+\S+(\s*\\?)$")
    if pattern.search(source):
        return pattern.sub(lambda match: f"{match.group(1)}{flag} {value}{match.group(2)}", source)
    return insert_before_flag(source, anchor_flag, f"  {flag} {value} \\")


def ensure_bool_flag(source: str, flag: str, enabled: str, anchor_flag: str) -> str:
    if enabled.lower() not in {"1", "true", "yes", "on"}:
        return source
    pattern = re.compile(rf"(?m)^\s*{re.escape(flag)}(\s*\\?)$")
    if pattern.search(source):
        return source
    return insert_before_flag(source, anchor_flag, f"  {flag} \\")


text = upsert_value_flag(text, "--kv-cache-dtype", os.environ["PATCH_KV_CACHE_DTYPE"], "--enable-fused-qk-norm-rope")
text = ensure_bool_flag(text, "--disable-cuda-graph", os.environ["PATCH_DISABLE_CUDA_GRAPH"], "--enable-fused-qk-norm-rope")
text = upsert_value_flag(text, "--chunked-prefill-size", os.environ["PATCH_CHUNKED_PREFILL_SIZE"], "--enable-fused-qk-norm-rope")
text = upsert_value_flag(text, "--mem-fraction-static", os.environ["PATCH_MEM_FRACTION"], "--moe-runner-backend")

print(text, end="")
PY
)"

patch_file="$(mktemp)"
python3 - "${new}" >"${patch_file}" <<'PY'
import json
import sys

print(json.dumps([
    {
        "op": "replace",
        "path": "/spec/template/spec/containers/0/args/0",
        "value": sys.argv[1],
    }
]))
PY

if [[ "${ACTION}" == "dry-run" ]]; then
  "${kubectl_cmd[@]}" patch deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
    --type=json \
    --patch-file "${patch_file}" \
    --dry-run=server \
    -o yaml >/tmp/"${DEPLOYMENT}".mem-fraction-"${MEM_FRACTION}".dry-run.yaml
  echo "Dry-run rendered to /tmp/${DEPLOYMENT}.mem-fraction-${MEM_FRACTION}.dry-run.yaml"
  exit 0
fi

"${kubectl_cmd[@]}" patch deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
  --type=json \
  --patch-file "${patch_file}"
"${kubectl_cmd[@]}" rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=30m
show_status
