#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${KUBECONFIG_PATH:-/tmp/eks-cluster-hypd-0324.kubeconfig}"
JOB_NS="${JOB_NS:-glm51-pd-b200-p1d1}"
SCALE_NS="${SCALE_NS:-glm51-pd-b200-p1d1}"
DEPLOY="${DEPLOY:-sglang-decode-anchor}"
JOB="${JOB:-qwen36-swift-r13-sglang-smoke-20260608}"
MANIFEST="${MANIFEST:-docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml}"
SELECTOR="${SELECTOR:-app=sglang,component=decode,role=decode-anchor}"

k() {
  kubectl --kubeconfig "${KUBECONFIG_PATH}" "$@"
}

original_replicas="$(k -n "${SCALE_NS}" get deploy "${DEPLOY}" -o jsonpath='{.spec.replicas}')"
echo ">>> Original ${SCALE_NS}/${DEPLOY} replicas: ${original_replicas}"

restore() {
  status="$?"
  set +e
  echo ">>> Restoring ${SCALE_NS}/${DEPLOY} to ${original_replicas} replica(s)"
  k -n "${SCALE_NS}" scale deploy "${DEPLOY}" --replicas="${original_replicas}"
  if [ "${original_replicas}" != "0" ]; then
    k -n "${SCALE_NS}" rollout status deploy "${DEPLOY}" --timeout=900s
  fi
  exit "${status}"
}

trap restore EXIT INT TERM

echo ">>> Removing any previous smoke job"
k -n "${JOB_NS}" delete job "${JOB}" --ignore-not-found=true

echo ">>> Scaling ${SCALE_NS}/${DEPLOY} down to free the target node GPUs"
k -n "${SCALE_NS}" scale deploy "${DEPLOY}" --replicas=0

for attempt in $(seq 1 120); do
  pod_count="$(k -n "${SCALE_NS}" get pods -l "${SELECTOR}" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${pod_count}" = "0" ]; then
    break
  fi
  echo ">>> Waiting for ${DEPLOY} pods to terminate (${pod_count} remaining)"
  sleep 5
  if [ "${attempt}" = "120" ]; then
    echo "Timed out waiting for ${DEPLOY} pods to terminate" >&2
    exit 1
  fi
done

echo ">>> Applying isolated SGLang smoke job"
k apply -f "${MANIFEST}"

for attempt in $(seq 1 360); do
  status="$(k -n "${JOB_NS}" get job "${JOB}" -o jsonpath='{.status.succeeded} {.status.failed}' 2>/dev/null || true)"
  echo ">>> Smoke job status: ${status:-pending}"

  succeeded="${status%% *}"
  failed="${status##* }"

  if [ "${succeeded:-0}" != "" ] && [ "${succeeded:-0}" != "0" ]; then
    echo ">>> Smoke job completed"
    k -n "${JOB_NS}" logs "job/${JOB}" --tail=260 || true
    exit 0
  fi

  if [ "${failed:-0}" != "" ] && [ "${failed:-0}" != "0" ]; then
    echo ">>> Smoke job failed" >&2
    k -n "${JOB_NS}" describe job "${JOB}" || true
    k -n "${JOB_NS}" logs "job/${JOB}" --tail=260 || true
    exit 1
  fi

  if [ $((attempt % 10)) = "0" ]; then
    k -n "${JOB_NS}" get pods -l app.kubernetes.io/name=qwen36-swift-r13-sglang-smoke -o wide || true
    k -n "${JOB_NS}" logs "job/${JOB}" --tail=80 || true
  fi
  sleep 20
done

echo "Timed out waiting for smoke job completion" >&2
k -n "${JOB_NS}" describe job "${JOB}" || true
k -n "${JOB_NS}" logs "job/${JOB}" --tail=260 || true
exit 1
