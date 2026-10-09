#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-dry-run}"
KUBECONFIG="${KUBECONFIG:-/tmp/eks-cluster-hypd-0324.kubeconfig}"
DS_NAMESPACE="${DS_NAMESPACE:-kube-system}"
DS_NAME="${DS_NAME:-hyperpod-dependencies-nvidia-device-plugin}"
CONFIGMAP_PATH="${CONFIGMAP_PATH:-docs/generated/nvidia_device_plugin_time_slicing_configmap.yaml}"

case "${ACTION}" in
  dry-run|apply|rollback|status) ;;
  *)
    echo "Usage: $0 [dry-run|apply|rollback|status]" >&2
    exit 2
    ;;
esac

kubectl_cmd=(env KUBECONFIG="${KUBECONFIG}" kubectl)

show_status() {
  "${kubectl_cmd[@]}" get daemonset -n "${DS_NAMESPACE}" "${DS_NAME}" -o wide
  "${kubectl_cmd[@]}" get nodes \
    -L node.kubernetes.io/instance-type,nvidia.com/gpu.sharing-strategy,nvidia.com/mps.capable,nvidia.com/device-plugin.config
  "${kubectl_cmd[@]}" get node hyperpod-i-0da428942dd5a9cd9 \
    -o jsonpath='{.status.capacity.nvidia\.com/gpu}{" capacity\n"}{.status.allocatable.nvidia\.com/gpu}{" allocatable\n"}'
}

patch_payload='{
  "spec": {
    "template": {
      "metadata": {
        "annotations": {
          "verdent.ai/gpu-sharing-config": "time-slicing-replicas-2"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "nvidia-device-plugin-ctr",
            "env": [
              {
                "name": "CONFIG_FILE",
                "value": "/etc/nvidia-device-plugin/config.yaml"
              }
            ],
            "volumeMounts": [
              {
                "name": "nvidia-device-plugin-sharing-config",
                "mountPath": "/etc/nvidia-device-plugin",
                "readOnly": true
              }
            ]
          }
        ],
        "volumes": [
          {
            "name": "nvidia-device-plugin-sharing-config",
            "configMap": {
              "name": "nvidia-device-plugin-sharing-config"
            }
          }
        ]
      }
    }
  }
}'

if [[ "${ACTION}" == "status" ]]; then
  show_status
  exit 0
fi

if [[ "${ACTION}" == "rollback" ]]; then
  "${kubectl_cmd[@]}" set env daemonset/"${DS_NAME}" -n "${DS_NAMESPACE}" CONFIG_FILE-
  "${kubectl_cmd[@]}" rollout status daemonset/"${DS_NAME}" -n "${DS_NAMESPACE}" --timeout=10m
  show_status
  exit 0
fi

if [[ "${ACTION}" == "dry-run" ]]; then
  "${kubectl_cmd[@]}" apply --dry-run=server -f "${CONFIGMAP_PATH}"
  "${kubectl_cmd[@]}" patch daemonset "${DS_NAME}" -n "${DS_NAMESPACE}" \
    --type=strategic \
    --patch "${patch_payload}" \
    --dry-run=server \
    -o yaml >/tmp/"${DS_NAME}".sharing.dry-run.yaml
  echo "Dry-run rendered to /tmp/${DS_NAME}.sharing.dry-run.yaml"
  exit 0
fi

"${kubectl_cmd[@]}" apply -f "${CONFIGMAP_PATH}"
"${kubectl_cmd[@]}" patch daemonset "${DS_NAME}" -n "${DS_NAMESPACE}" \
  --type=strategic \
  --patch "${patch_payload}"
"${kubectl_cmd[@]}" rollout status daemonset/"${DS_NAME}" -n "${DS_NAMESPACE}" --timeout=10m
show_status
