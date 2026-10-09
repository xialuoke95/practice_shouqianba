#!/usr/bin/env python3
"""
Preflight and deployment guard for Qwen3.6 checkpoints on the GLM HyperPod fleet.

The requested checkpoint is expected to be on S3. This tool inspects the S3
layout and optional Kubernetes state before any live deployment action. It
intentionally refuses to mutate the GLM deployment when the checkpoint is not a
Hugging Face/SGLang-loadable model directory.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_S3_URI = (
    "s3://verdent.tech/test_training/sft_opus_sonnet_swift/"
    "qwen36-swift-fullsft-20260527-205700/training/"
    "qwen36_swift_all_full_r13/checkpoint_copy/"
)
DEFAULT_REGION = "us-west-2"
DEFAULT_KUBECONFIG = "/tmp/eks-cluster-hypd-0324.kubeconfig"
DEFAULT_NAMESPACE = "glm51-pd-b200-p1d1"
DEFAULT_MODEL_NAME = "Qwen3.6-35B-A3B-Swift-R13"
DEFAULT_PROXY_CONFIGMAP = "sglang-router-b200-proxy-config"
DEFAULT_HF_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_EXPORT_S3_URI = (
    "s3://verdent.tech/test_training/sft_opus_sonnet_swift/"
    "qwen36-swift-fullsft-20260527-205700/deployment_exports/"
    "qwen36_swift_all_full_r13_hf/"
)
DEFAULT_CONVERSION_IMAGE = (
    "763104351884.dkr.ecr.us-east-2.amazonaws.com/"
    "pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39-soci"
)
DEFAULT_CONVERSION_SERVICE_ACCOUNT = "sglang-s3-access"
DEFAULT_LOCAL_CONVERSION_ROOT = "/host-nvme/test_training/qwen36_swift_r13_hf_export"
DEFAULT_SGLANG_IMAGE = (
    "288761729453.dkr.ecr.us-east-2.amazonaws.com/"
    "sglang-kimik2-detector-patch:20260602085214"
)
DEFAULT_SGLANG_HFEXPORT_PVC = "s3-claim-verdent-qwen36-r13-hfexport"
DEFAULT_SGLANG_SMOKE_JOB_NAME = "qwen36-swift-r13-sglang-smoke-20260608"
DEFAULT_SGLANG_LOCAL_MODEL_ROOT = "/opt/dlami/nvme/models"
DEFAULT_SGLANG_DEPLOYMENT_NAME = "qwen36-swift-r13-sglang"
DEFAULT_SGLANG_SERVICE_NAME = "qwen36-swift-r13-sglang"
DEFAULT_SGLANG_RESTORE_NAMESPACE = DEFAULT_NAMESPACE
DEFAULT_SGLANG_RESTORE_DEPLOYMENT = "sglang-decode-anchor"
DEFAULT_SGLANG_RESTORE_SELECTOR = "app=sglang,component=decode,role=decode-anchor"
DEFAULT_BRIDGE_SOURCE_S3_URI = (
    "s3://verdent.tech/test_training/sft_opus_sonnet_swift/"
    "qwen36-swift-fullsft-20260527-205700/env/Megatron-Bridge-v0.4.1-source.tar.gz"
)
DEFAULT_BRIDGE_SETUP_S3_URI = (
    "s3://verdent.tech/test_training/sft_opus_sonnet_swift/"
    "qwen36-swift-fullsft-20260527-205700/artifacts/source/scripts/"
    "setup_aws_dlc_bridge_runtime.sh"
)

TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}


class ToolError(RuntimeError):
    pass


def run_cmd(args: list[str], *, check: bool = True) -> dict[str, Any]:
    proc = subprocess.run(args, text=True, capture_output=True)
    result = {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if check and proc.returncode != 0:
        rendered = " ".join(args)
        raise ToolError(
            f"command failed ({proc.returncode}): {rendered}\n{proc.stderr.strip()}"
        )
    return result


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    without_scheme = uri[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"missing S3 bucket in URI: {uri}")
    return bucket, prefix


def s3_uri_for_key(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def list_s3_objects(uri: str, region: str) -> list[dict[str, Any]]:
    bucket, prefix = parse_s3_uri(uri)
    objects: list[dict[str, Any]] = []
    continuation_token: str | None = None

    while True:
        cmd = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--region",
            region,
            "--output",
            "json",
        ]
        if continuation_token:
            cmd.extend(["--continuation-token", continuation_token])
        result = run_cmd(cmd)
        payload = json.loads(result["stdout"] or "{}")
        objects.extend(payload.get("Contents", []))
        if not payload.get("IsTruncated"):
            return objects
        continuation_token = payload.get("NextContinuationToken")
        if not continuation_token:
            raise ToolError("S3 response was truncated without NextContinuationToken")


def get_s3_object_text(uri: str, region: str) -> str | None:
    result = run_cmd(
        ["aws", "s3", "cp", uri, "-", "--region", region],
        check=False,
    )
    if result["returncode"] != 0:
        return None
    return result["stdout"]


def relative_keys(objects: list[dict[str, Any]], prefix: str) -> list[str]:
    rels: list[str] = []
    for obj in objects:
        key = obj.get("Key", "")
        if key.startswith(prefix):
            rels.append(key[len(prefix) :].lstrip("/"))
        else:
            rels.append(key)
    return rels


def classify_checkpoint(uri: str, region: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    bucket, prefix = parse_s3_uri(uri)
    rels = relative_keys(objects, prefix)
    total_bytes = sum(int(obj.get("Size", 0)) for obj in objects)

    metadata_text = None
    metadata = None
    metadata_key = next((obj["Key"] for obj in objects if obj["Key"].endswith("/metadata.json")), None)
    if metadata_key:
        metadata_text = get_s3_object_text(s3_uri_for_key(bucket, metadata_key), region)
        if metadata_text:
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = {"_parse_error": "metadata.json was not valid JSON"}

    latest_iter = None
    latest_key = next(
        (obj["Key"] for obj in objects if obj["Key"].endswith("latest_checkpointed_iteration.txt")),
        None,
    )
    if latest_key:
        latest_text = get_s3_object_text(s3_uri_for_key(bucket, latest_key), region)
        if latest_text is not None:
            latest_iter = latest_text.strip()

    existing_rel_keys = set(rels)
    has_config = "config.json" in rels
    has_tokenizer = any(rel in TOKENIZER_FILES for rel in rels)
    has_hf_weight = any(looks_like_hf_weight(rel) for rel in rels)
    has_safetensors_index = "model.safetensors.index.json" in rels
    safetensors_index_shards: list[str] = []
    missing_safetensors_index_shards: list[str] = []
    if has_safetensors_index:
        index_key = (
            f"{prefix.rstrip('/')}/model.safetensors.index.json"
            if prefix
            else "model.safetensors.index.json"
        )
        index_text = get_s3_object_text(s3_uri_for_key(bucket, index_key), region)
        if index_text:
            try:
                index_payload = json.loads(index_text)
                safetensors_index_shards = sorted(
                    set(index_payload.get("weight_map", {}).values())
                )
                missing_safetensors_index_shards = [
                    shard
                    for shard in safetensors_index_shards
                    if shard not in existing_rel_keys
                ]
            except json.JSONDecodeError:
                missing_safetensors_index_shards = ["<index JSON parse failed>"]
    has_complete_safetensors_index = (
        has_safetensors_index
        and bool(safetensors_index_shards)
        and not missing_safetensors_index_shards
    )
    has_distcp = any(rel.endswith(".distcp") for rel in rels)
    has_dist_metadata = any(rel.endswith("/.metadata") or rel == ".metadata" for rel in rels)
    sharded_backend = metadata.get("sharded_backend") if isinstance(metadata, dict) else None

    if has_distcp or has_dist_metadata or sharded_backend == "torch_dist":
        checkpoint_type = "megatron_torch_dist"
    elif has_config and has_tokenizer and (has_hf_weight or has_complete_safetensors_index):
        checkpoint_type = "hf_or_sglang_model_dir"
    else:
        checkpoint_type = "unknown"

    sglang_loadable = checkpoint_type == "hf_or_sglang_model_dir"
    blockers: list[str] = []
    if not objects:
        blockers.append("S3 prefix has no objects.")
    if checkpoint_type == "megatron_torch_dist":
        blockers.append(
            "Checkpoint is Megatron torch_dist/distcp, not a Hugging Face/SGLang "
            "model directory."
        )
    elif checkpoint_type == "unknown":
        blockers.append(
            "Checkpoint layout is unknown; required config/tokenizer/weight files "
            "were not found at the prefix root."
        )
    if has_safetensors_index and missing_safetensors_index_shards:
        blockers.append(
            "model.safetensors.index.json is present but referenced safetensors "
            "shards are missing from the same S3 prefix."
        )

    return {
        "uri": uri,
        "bucket": bucket,
        "prefix": prefix,
        "object_count": len(objects),
        "total_bytes": total_bytes,
        "sample_keys": rels[:25],
        "latest_checkpointed_iteration": latest_iter,
        "metadata": metadata,
        "features": {
            "has_config_json": has_config,
            "has_tokenizer_files": has_tokenizer,
            "has_hf_weight_files": has_hf_weight,
            "has_safetensors_index": has_safetensors_index,
            "has_complete_safetensors_index": has_complete_safetensors_index,
            "safetensors_index_shard_count": len(safetensors_index_shards),
            "missing_safetensors_index_shard_count": len(missing_safetensors_index_shards),
            "has_distcp_shards": has_distcp,
            "has_dist_checkpoint_metadata": has_dist_metadata,
            "sharded_backend": sharded_backend,
        },
        "checkpoint_type": checkpoint_type,
        "sglang_loadable": sglang_loadable,
        "blockers": blockers,
    }


def looks_like_hf_weight(relative_key: str) -> bool:
    filename = relative_key.rsplit("/", 1)[-1]
    if filename.endswith(".safetensors"):
        return True
    if filename.endswith(".bin"):
        return filename.startswith("pytorch_model") or filename.startswith("model")
    return False


def kubectl_base(kubeconfig: str | None) -> list[str]:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    return cmd


def kubectl_json(kubeconfig: str | None, args: list[str]) -> Any:
    result = run_cmd(kubectl_base(kubeconfig) + args + ["-o", "json"])
    return json.loads(result["stdout"] or "{}")


def collect_k8s_status(
    kubeconfig: str | None,
    namespace: str,
    proxy_configmap: str,
) -> dict[str, Any]:
    status: dict[str, Any] = {"namespace": namespace}
    errors: list[str] = []

    try:
        nodes = kubectl_json(kubeconfig, ["get", "nodes"])
        status["nodes"] = [
            {
                "name": item["metadata"]["name"],
                "internal_ip": next(
                    (
                        addr["address"]
                        for addr in item.get("status", {}).get("addresses", [])
                        if addr.get("type") == "InternalIP"
                    ),
                    None,
                ),
                "gpu": item.get("status", {})
                .get("allocatable", {})
                .get("nvidia.com/gpu", "0"),
                "efa": item.get("status", {})
                .get("allocatable", {})
                .get("vpc.amazonaws.com/efa", "0"),
                "cpu": item.get("status", {}).get("allocatable", {}).get("cpu"),
                "memory": item.get("status", {}).get("allocatable", {}).get("memory"),
            }
            for item in nodes.get("items", [])
        ]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"nodes: {exc}")

    try:
        pods = kubectl_json(kubeconfig, ["get", "pods", "-n", namespace])
        status["pods"] = [
            {
                "name": item["metadata"]["name"],
                "phase": item.get("status", {}).get("phase"),
                "node": item.get("spec", {}).get("nodeName"),
                "ready": summarize_ready(item),
            }
            for item in pods.get("items", [])
        ]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pods: {exc}")

    try:
        deployments = kubectl_json(kubeconfig, ["get", "deployments", "-n", namespace])
        status["deployments"] = [
            {
                "name": item["metadata"]["name"],
                "replicas": item.get("spec", {}).get("replicas", 0),
                "ready_replicas": item.get("status", {}).get("readyReplicas", 0),
                "available_replicas": item.get("status", {}).get("availableReplicas", 0),
            }
            for item in deployments.get("items", [])
        ]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"deployments: {exc}")

    try:
        configmap = kubectl_json(
            kubeconfig,
            ["get", "cm", "-n", "default", proxy_configmap],
        )
        status["default_proxy_configmap"] = {
            "name": proxy_configmap,
            "data": configmap.get("data", {}),
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"default proxy configmap: {exc}")

    status["errors"] = errors
    return status


def summarize_ready(pod: dict[str, Any]) -> str:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if not statuses:
        return "0/0"
    ready = sum(1 for item in statuses if item.get("ready"))
    return f"{ready}/{len(statuses)}"


def build_assessment(args: argparse.Namespace) -> dict[str, Any]:
    objects = list_s3_objects(args.checkpoint_s3_uri, args.region)
    checkpoint = classify_checkpoint(args.checkpoint_s3_uri, args.region, objects)

    k8s_status = None
    if not args.skip_k8s:
        k8s_status = collect_k8s_status(
            args.kubeconfig,
            args.namespace,
            args.proxy_configmap,
        )

    blockers = list(checkpoint["blockers"])
    warnings: list[str] = []

    if k8s_status:
        proxy_data = (
            k8s_status.get("default_proxy_configmap", {})
            .get("data", {})
        )
        preferred = proxy_data.get("PREFERRED_AWS_GLM51_BACKEND")
        if preferred == "b200":
            warnings.append(
                "Default GLM proxy currently prefers b200; mutating the existing "
                "GLM B200 deployment may affect the externally routed backend."
            )
        gpu_nodes = [
            node for node in k8s_status.get("nodes", []) if str(node.get("gpu")) not in {"", "0"}
        ]
        if gpu_nodes:
            scheduled_gpu_nodes = {
                pod.get("node")
                for pod in k8s_status.get("pods", [])
                if pod.get("node") in {node["name"] for node in gpu_nodes}
            }
            warnings.append(
                f"GPU nodes visible: {len(gpu_nodes)}; GLM namespace pods are "
                f"scheduled on {len(scheduled_gpu_nodes)} GPU nodes."
            )

    verdict = "ready_for_isolated_sglang_deploy" if not blockers else "blocked_before_live_deploy"

    conversion_plan = build_conversion_plan(args, checkpoint, k8s_status)

    isolated_deployment_ready = False
    if k8s_status:
        isolated_deployment_ready = any(
            item.get("name") == args.sglang_deployment_name
            and int(item.get("replicas") or 0) > 0
            and int(item.get("ready_replicas") or 0) >= int(item.get("replicas") or 0)
            for item in k8s_status.get("deployments", [])
        )

    next_actions = []
    if checkpoint["checkpoint_type"] == "megatron_torch_dist":
        next_actions = [
            "Export or convert the Megatron torch_dist checkpoint to a "
            "Hugging Face/SGLang-loadable model directory.",
            "Upload the exported directory to a new S3 prefix with a distinct model name.",
            "Clone the GLM SGLang deployment under a separate namespace/service before "
            "changing any externally routed backend.",
        ]
    elif checkpoint["sglang_loadable"] and isolated_deployment_ready:
        next_actions = [
            "Isolated SGLang Deployment/Service is already Ready; keep it isolated unless a route change is explicitly approved.",
            "Verify /health and /v1/chat/completions immediately before any routing change.",
            "Only wire the proxy route with a rollback path to the previous GLM backend.",
        ]
    elif checkpoint["sglang_loadable"]:
        next_actions = [
            "Run a one-node SGLang load smoke test using a new model name.",
            "Generate isolated SGLang smoke and Deployment/Service manifests; do not overwrite GLM-5.1-FP8 cache.",
            "Only wire the proxy route after readiness and rollback checks pass.",
        ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_model_name": args.model_name,
        "region": args.region,
        "checkpoint": checkpoint,
        "k8s": k8s_status,
        "warnings": warnings,
        "blockers": blockers,
        "verdict": verdict,
        "conversion_plan": conversion_plan,
        "next_actions": next_actions,
    }


def build_conversion_plan(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    k8s_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint["checkpoint_type"] != "megatron_torch_dist":
        return None

    local_checkpoint_dir = f"{args.local_conversion_root.rstrip('/')}/checkpoint_copy"
    local_hf_export_dir = f"{args.local_conversion_root.rstrip('/')}/hf_export"
    export_code = (
        "from megatron.bridge import AutoBridge; "
        f"b=AutoBridge.from_hf_pretrained({args.hf_model_id!r}, trust_remote_code=True); "
        f"b.export_ckpt({local_checkpoint_dir!r}, {local_hf_export_dir!r}, "
        "show_progress=True, strict=False)"
    )
    candidate_nodes = candidate_conversion_nodes(k8s_status)

    return {
        "required": True,
        "reason": "Requested checkpoint must be exported from Megatron torch_dist to a Hugging Face/SGLang-loadable directory before serving.",
        "autobridge_evidence": {
            "source": "Megatron-Bridge v0.4.1 source package from the target run S3 env directory",
            "api": "AutoBridge.from_hf_pretrained(...).export_ckpt(megatron_path, hf_path, show_progress=True, strict=False)",
            "notes": [
                "from_hf_pretrained loads a pretrained HF model, not just config/tokenizer metadata.",
                "export_ckpt performs its Megatron checkpoint load in a temporary CPU/gloo distributed context.",
            ],
        },
        "hf_model_id": args.hf_model_id,
        "checkpoint_s3_uri": args.checkpoint_s3_uri,
        "export_s3_uri": args.export_s3_uri,
        "local_checkpoint_dir": local_checkpoint_dir,
        "local_hf_export_dir": local_hf_export_dir,
        "conversion_image": args.conversion_image,
        "conversion_service_account": args.conversion_service_account,
        "bridge_source_s3_uri": args.bridge_source_s3_uri,
        "bridge_setup_s3_uri": args.bridge_setup_s3_uri,
        "candidate_glm_nodes": candidate_nodes,
        "risks": [
            "The HF base model must be accessible to the conversion job; metadata-only checkpoint-854 materials are not enough.",
            "The job needs enough local NVMe for the 70 GiB Megatron checkpoint and the exported HF/safetensors directory.",
            "The export path may use substantial CPU memory because AutoBridge export loads Megatron weights before saving HF files.",
            "The base AWS DLC image must run the archived setup_aws_dlc_bridge_runtime.sh before importing megatron.bridge.",
            "Generated conversion job YAML is intended for isolated export work and does not mutate the existing GLM SGLang deployment.",
        ],
        "commands": {
            "prepare_runtime": [
                "bash",
                "-lc",
                f"mkdir -p {args.local_conversion_root.rstrip('/')}/runtime_run/env "
                f"{args.local_conversion_root.rstrip('/')}/runtime_node0 && "
                f"aws s3 cp {args.bridge_source_s3_uri} "
                f"{args.local_conversion_root.rstrip('/')}/runtime_run/env/Megatron-Bridge-v0.4.1-source.tar.gz "
                f"--region {args.region} && "
                f"aws s3 cp {args.bridge_setup_s3_uri} "
                f"{args.local_conversion_root.rstrip('/')}/setup_aws_dlc_bridge_runtime.sh "
                f"--region {args.region} && "
                f"bash {args.local_conversion_root.rstrip('/')}/setup_aws_dlc_bridge_runtime.sh "
                f"{args.local_conversion_root.rstrip('/')}/runtime_run "
                f"{args.local_conversion_root.rstrip('/')}/runtime_node0",
            ],
            "sync_checkpoint": [
                "aws",
                "s3",
                "sync",
                args.checkpoint_s3_uri,
                local_checkpoint_dir,
                "--region",
                args.region,
            ],
            "export_hf": ["python", "-c", export_code],
            "upload_export": [
                "aws",
                "s3",
                "sync",
                local_hf_export_dir,
                args.export_s3_uri,
                "--region",
                args.region,
            ],
            "post_upload_preflight": [
                "python3",
                "deploy_model.py",
                "--checkpoint-s3-uri",
                args.export_s3_uri,
                "--region",
                args.region,
            ],
        },
    }


def candidate_conversion_nodes(k8s_status: dict[str, Any] | None) -> list[str]:
    if not k8s_status:
        return []
    pod_nodes = []
    for pod in k8s_status.get("pods", []):
        node = pod.get("node")
        if node and node not in pod_nodes:
            pod_nodes.append(node)
    if pod_nodes:
        return pod_nodes
    return [
        node["name"]
        for node in k8s_status.get("nodes", [])
        if str(node.get("gpu")) not in {"", "0"}
    ]


def render_conversion_job_yaml(args: argparse.Namespace, assessment: dict[str, Any]) -> str:
    plan = assessment.get("conversion_plan")
    if not plan:
        raise ToolError("conversion job YAML is only available for Megatron torch_dist checkpoints")

    node_name = args.conversion_node
    if not node_name:
        candidates = plan.get("candidate_glm_nodes") or []
        node_name = candidates[0] if candidates else "<set-glm-node-name>"

    local_checkpoint_dir = plan["local_checkpoint_dir"]
    local_hf_export_dir = plan["local_hf_export_dir"]
    export_code = (
        "from megatron.bridge import AutoBridge; "
        f"b=AutoBridge.from_hf_pretrained({args.hf_model_id!r}, trust_remote_code=True); "
        f"b.export_ckpt({local_checkpoint_dir!r}, {local_hf_export_dir!r}, "
        "show_progress=True, strict=False)"
    )
    shell_script = "\n".join(
        [
            "set -euxo pipefail",
            f"CHECKPOINT_S3={shlex.quote(args.checkpoint_s3_uri)}",
            f"EXPORT_S3={shlex.quote(args.export_s3_uri)}",
            f"LOCAL_CHECKPOINT={shlex.quote(local_checkpoint_dir)}",
            f"LOCAL_HF_EXPORT={shlex.quote(local_hf_export_dir)}",
            f"LOCAL_ROOT={shlex.quote(args.local_conversion_root.rstrip('/'))}",
            "RUNTIME_RUN_DIR=\"${LOCAL_ROOT}/runtime_run\"",
            "RUNTIME_NODE_ROOT=\"${LOCAL_ROOT}/runtime_node0\"",
            f"HF_MODEL_ID={shlex.quote(args.hf_model_id)}",
            "mkdir -p \"${LOCAL_CHECKPOINT}\" \"${LOCAL_HF_EXPORT}\" \"${RUNTIME_RUN_DIR}/env\" \"${RUNTIME_NODE_ROOT}\"",
            f"aws s3 cp {shlex.quote(args.bridge_source_s3_uri)} \"${{RUNTIME_RUN_DIR}}/env/Megatron-Bridge-v0.4.1-source.tar.gz\" --region \"${{AWS_REGION}}\"",
            f"aws s3 cp {shlex.quote(args.bridge_setup_s3_uri)} \"${{LOCAL_ROOT}}/setup_aws_dlc_bridge_runtime.sh\" --region \"${{AWS_REGION}}\"",
            "bash \"${LOCAL_ROOT}/setup_aws_dlc_bridge_runtime.sh\" \"${RUNTIME_RUN_DIR}\" \"${RUNTIME_NODE_ROOT}\"",
            "aws s3 sync \"${CHECKPOINT_S3}\" \"${LOCAL_CHECKPOINT}\" --region \"${AWS_REGION}\"",
            f"python -c {shlex.quote(export_code)}",
            "test -f \"${LOCAL_HF_EXPORT}/config.json\"",
            "test -f \"${LOCAL_HF_EXPORT}/model.safetensors.index.json\" || ls \"${LOCAL_HF_EXPORT}\"/*.safetensors",
            "aws s3 sync \"${LOCAL_HF_EXPORT}\" \"${EXPORT_S3}\" --region \"${AWS_REGION}\"",
        ]
    )
    indented_script = "\n".join(f"          {line}" for line in shell_script.splitlines())

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-swift-r13-hf-export
  namespace: {args.namespace}
  labels:
    app.kubernetes.io/name: qwen36-swift-r13-hf-export
    test-training.verdent.ai/purpose: megatron-to-hf-export
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app.kubernetes.io/name: qwen36-swift-r13-hf-export
    spec:
      restartPolicy: Never
      nodeName: {node_name}
      serviceAccountName: {args.conversion_service_account}
      containers:
      - name: export
        image: {args.conversion_image}
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-lc"]
        args:
        - |
{indented_script}
        env:
        - name: AWS_REGION
          value: {args.region}
        - name: AWS_DEFAULT_REGION
          value: {args.region}
        - name: AWS_EC2_METADATA_DISABLED
          value: "true"
        - name: AWS_STS_REGIONAL_ENDPOINTS
          value: regional
        - name: HF_HOME
          value: {args.local_conversion_root.rstrip('/')}/hf_cache
        - name: TRANSFORMERS_CACHE
          value: {args.local_conversion_root.rstrip('/')}/hf_cache
        resources:
          requests:
            cpu: "32"
            memory: "512Gi"
            ephemeral-storage: "200Gi"
          limits:
            cpu: "160"
            memory: "1600Gi"
        volumeMounts:
        - name: host-nvme
          mountPath: /host-nvme
      volumes:
      - name: host-nvme
        hostPath:
          path: /opt/dlami/nvme
          type: DirectoryOrCreate
"""


def candidate_sglang_node(k8s_status: dict[str, Any] | None) -> str:
    if not k8s_status:
        return "<set-glm-node-name>"

    for pod in k8s_status.get("pods", []):
        name = pod.get("name") or ""
        node = pod.get("node")
        if node and name.startswith("sglang-decode-anchor"):
            return node

    for pod in k8s_status.get("pods", []):
        name = pod.get("name") or ""
        node = pod.get("node")
        if node and name.startswith("sglang-prefill"):
            return node

    for node in k8s_status.get("nodes", []):
        if str(node.get("gpu")) not in {"", "0"}:
            return node["name"]

    return "<set-glm-node-name>"


def render_sglang_smoke_job_yaml(args: argparse.Namespace, assessment: dict[str, Any]) -> str:
    checkpoint = assessment["checkpoint"]
    if not checkpoint["sglang_loadable"]:
        raise ToolError("SGLang smoke job YAML is only available for SGLang-loadable model directories")

    node_name = args.sglang_node or candidate_sglang_node(assessment.get("k8s"))
    model_name = shlex.quote(args.model_name)
    local_model_root = args.sglang_local_model_root.rstrip("/")
    local_model_dir = f"{local_model_root}/${{MODEL_NAME}}"
    local_tmp_dir = f"{local_model_root}/${{MODEL_NAME}}.tmp"
    log_dir = f"/opt/dlami/nvme/test_training/{args.sglang_smoke_job_name}"
    server_log = f"{log_dir}/server.log"

    shell_script = "\n".join(
        [
            "set -euxo pipefail",
            f"MODEL_NAME={model_name}",
            f"MODEL_LOCAL={local_model_dir}",
            f"MODEL_TMP={local_tmp_dir}",
            f"SERVER_LOG={shlex.quote(server_log)}",
            f"mkdir -p {shlex.quote(local_model_root)} {shlex.quote(log_dir)}",
            "mkdir -p /opt/dlami/nvme/triton-cache /opt/dlami/nvme/deepgemm-cache /opt/dlami/nvme/tmp /opt/dlami/nvme/cuda-cache /opt/dlami/nvme/torch-extensions",
            "export TRITON_CACHE_DIR=/opt/dlami/nvme/triton-cache",
            "export SGLANG_DG_CACHE_DIR=/opt/dlami/nvme/deepgemm-cache",
            "export TMPDIR=/opt/dlami/nvme/tmp",
            "export CUDA_CACHE_PATH=/opt/dlami/nvme/cuda-cache",
            "export TORCH_EXTENSIONS_DIR=/opt/dlami/nvme/torch-extensions",
            "export NCCL_DEBUG=INFO",
            "export FI_PROVIDER=efa",
            "export FI_EFA_USE_DEVICE_RDMA=1",
            "export NVSHMEM_DISABLE_IBGDA=1",
            "export NVSHMEM_IB_ENABLE_IBGDA=0",
            "",
            "if [ ! -f \"${MODEL_LOCAL}/.copy_complete\" ]; then",
            "  rm -rf \"${MODEL_TMP}\"",
            "  mkdir -p \"${MODEL_TMP}\"",
            "  cp -r /s3export/. \"${MODEL_TMP}/\"",
            "  test -f \"${MODEL_TMP}/config.json\"",
            "  test -f \"${MODEL_TMP}/model.safetensors.index.json\"",
            "  rm -rf \"${MODEL_LOCAL}\"",
            "  mv \"${MODEL_TMP}\" \"${MODEL_LOCAL}\"",
            "  touch \"${MODEL_LOCAL}/.copy_complete\"",
            "  echo \">>> Model staged at ${MODEL_LOCAL}: $(du -sh \"${MODEL_LOCAL}\" | cut -f1)\"",
            "else",
            "  echo \">>> Reusing cached model at ${MODEL_LOCAL}: $(du -sh \"${MODEL_LOCAL}\" | cut -f1)\"",
            "fi",
            "",
            "sglang serve \\",
            "  --model-path \"${MODEL_LOCAL}\" \\",
            "  --served-model-name \"${MODEL_NAME}\" \\",
            "  --host 0.0.0.0 \\",
            "  --port 30000 \\",
            "  --tp-size 8 \\",
            "  --ep-size 8 \\",
            "  --trust-remote-code \\",
            "  --kv-cache-dtype fp8_e4m3 \\",
            "  --enable-fused-qk-norm-rope \\",
            "  --mem-fraction-static 0.82 \\",
            "  --moe-runner-backend flashinfer_trtllm \\",
            "  --moe-dense-tp-size 1 \\",
            "  --disable-flashinfer-autotune \\",
            "  --watchdog-timeout 1000000 \\",
            "  --enable-metrics \\",
            "  >\"${SERVER_LOG}\" 2>&1 &",
            "SERVER_PID=$!",
            "",
            "cleanup() {",
            "  if kill -0 \"${SERVER_PID}\" 2>/dev/null; then",
            "    kill \"${SERVER_PID}\" || true",
            "    wait \"${SERVER_PID}\" || true",
            "  fi",
            "}",
            "trap cleanup EXIT",
            "",
            "ready=0",
            "for _ in $(seq 1 540); do",
            "  if curl -sf http://127.0.0.1:30000/health >/dev/null; then",
            "    ready=1",
            "    break",
            "  fi",
            "  if ! kill -0 \"${SERVER_PID}\" 2>/dev/null; then",
            "    echo \"SGLang server exited before readiness\" >&2",
            "    tail -200 \"${SERVER_LOG}\" >&2 || true",
            "    exit 1",
            "  fi",
            "  sleep 10",
            "done",
            "if [ \"${ready}\" != \"1\" ]; then",
            "  echo \"SGLang server did not become healthy before timeout\" >&2",
            "  tail -200 \"${SERVER_LOG}\" >&2 || true",
            "  exit 1",
            "fi",
            "",
            "python - <<'PY'",
            "import json",
            "import urllib.request",
            "",
            "payload = {",
            f"    \"model\": {args.model_name!r},",
            "    \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}],",
            "    \"max_tokens\": 4,",
            "    \"temperature\": 0,",
            "}",
            "req = urllib.request.Request(",
            "    \"http://127.0.0.1:30000/v1/chat/completions\",",
            "    data=json.dumps(payload).encode(\"utf-8\"),",
            "    headers={\"Content-Type\": \"application/json\"},",
            "    method=\"POST\",",
            ")",
            "with urllib.request.urlopen(req, timeout=120) as resp:",
            "    body = resp.read().decode(\"utf-8\")",
            "    status = resp.status",
            "print(json.dumps({\"phase\": \"completion_smoke\", \"status\": status, \"body\": body[:1000]}, sort_keys=True))",
            "if status != 200:",
            "    raise SystemExit(f\"unexpected status: {status}\")",
            "PY",
            "",
            "echo \">>> SGLang smoke succeeded\"",
        ]
    )
    indented_script = "\n".join(f"          {line}" for line in shell_script.splitlines())

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {args.sglang_smoke_job_name}
  namespace: {args.namespace}
  labels:
    app.kubernetes.io/name: qwen36-swift-r13-sglang-smoke
    test-training.verdent.ai/purpose: isolated-sglang-load-smoke
spec:
  activeDeadlineSeconds: 7200
  backoffLimit: 0
  template:
    metadata:
      labels:
        app.kubernetes.io/name: qwen36-swift-r13-sglang-smoke
    spec:
      restartPolicy: Never
      nodeName: {node_name}
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: sglang-smoke
        image: {args.sglang_image}
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-lc"]
        args:
        - |
{indented_script}
        resources:
          requests:
            cpu: "120"
            memory: "1200Gi"
            ephemeral-storage: "50Gi"
            nvidia.com/gpu: "8"
            vpc.amazonaws.com/efa: "8"
          limits:
            cpu: "160"
            memory: "1600Gi"
            ephemeral-storage: "80Gi"
            nvidia.com/gpu: "8"
            vpc.amazonaws.com/efa: "8"
        ports:
        - name: http
          containerPort: 30000
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: host-nvme
          mountPath: /opt/dlami/nvme
        - name: hfexport
          mountPath: /s3export
          readOnly: true
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi
      - name: host-nvme
        hostPath:
          path: /opt/dlami/nvme
          type: DirectoryOrCreate
      - name: hfexport
        persistentVolumeClaim:
          claimName: {args.sglang_hfexport_pvc}
"""


def render_sglang_smoke_restore_script(args: argparse.Namespace) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG_PATH="${{KUBECONFIG_PATH:-{args.kubeconfig}}}"
JOB_NS="${{JOB_NS:-{args.namespace}}}"
SCALE_NS="${{SCALE_NS:-{args.sglang_restore_namespace}}}"
DEPLOY="${{DEPLOY:-{args.sglang_restore_deployment}}}"
JOB="${{JOB:-{args.sglang_smoke_job_name}}}"
MANIFEST="${{MANIFEST:-docs/generated/qwen36_swift_r13_sglang_smoke_job.yaml}}"
SELECTOR="${{SELECTOR:-{args.sglang_restore_selector}}}"

k() {{
  kubectl --kubeconfig "${{KUBECONFIG_PATH}}" "$@"
}}

original_replicas="$(k -n "${{SCALE_NS}}" get deploy "${{DEPLOY}}" -o jsonpath='{{.spec.replicas}}')"
echo ">>> Original ${{SCALE_NS}}/${{DEPLOY}} replicas: ${{original_replicas}}"

restore() {{
  status="$?"
  set +e
  echo ">>> Restoring ${{SCALE_NS}}/${{DEPLOY}} to ${{original_replicas}} replica(s)"
  k -n "${{SCALE_NS}}" scale deploy "${{DEPLOY}}" --replicas="${{original_replicas}}"
  if [ "${{original_replicas}}" != "0" ]; then
    k -n "${{SCALE_NS}}" rollout status deploy "${{DEPLOY}}" --timeout=900s
  fi
  exit "${{status}}"
}}

trap restore EXIT INT TERM

echo ">>> Removing any previous smoke job"
k -n "${{JOB_NS}}" delete job "${{JOB}}" --ignore-not-found=true

echo ">>> Scaling ${{SCALE_NS}}/${{DEPLOY}} down to free the target node GPUs"
k -n "${{SCALE_NS}}" scale deploy "${{DEPLOY}}" --replicas=0

for attempt in $(seq 1 120); do
  pod_count="$(k -n "${{SCALE_NS}}" get pods -l "${{SELECTOR}}" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${{pod_count}}" = "0" ]; then
    break
  fi
  echo ">>> Waiting for ${{DEPLOY}} pods to terminate (${{pod_count}} remaining)"
  sleep 5
  if [ "${{attempt}}" = "120" ]; then
    echo "Timed out waiting for ${{DEPLOY}} pods to terminate" >&2
    exit 1
  fi
done

echo ">>> Applying isolated SGLang smoke job"
k apply -f "${{MANIFEST}}"

for attempt in $(seq 1 360); do
  status="$(k -n "${{JOB_NS}}" get job "${{JOB}}" -o jsonpath='{{.status.succeeded}} {{.status.failed}}' 2>/dev/null || true)"
  echo ">>> Smoke job status: ${{status:-pending}}"

  succeeded="${{status%% *}}"
  failed="${{status##* }}"

  if [ "${{succeeded:-0}}" != "" ] && [ "${{succeeded:-0}}" != "0" ]; then
    echo ">>> Smoke job completed"
    k -n "${{JOB_NS}}" logs "job/${{JOB}}" --tail=260 || true
    exit 0
  fi

  if [ "${{failed:-0}}" != "" ] && [ "${{failed:-0}}" != "0" ]; then
    echo ">>> Smoke job failed" >&2
    k -n "${{JOB_NS}}" describe job "${{JOB}}" || true
    k -n "${{JOB_NS}}" logs "job/${{JOB}}" --tail=260 || true
    exit 1
  fi

  if [ $((attempt % 10)) = "0" ]; then
    k -n "${{JOB_NS}}" get pods -l app.kubernetes.io/name=qwen36-swift-r13-sglang-smoke -o wide || true
    k -n "${{JOB_NS}}" logs "job/${{JOB}}" --tail=80 || true
  fi
  sleep 20
done

echo "Timed out waiting for smoke job completion" >&2
k -n "${{JOB_NS}}" describe job "${{JOB}}" || true
k -n "${{JOB_NS}}" logs "job/${{JOB}}" --tail=260 || true
exit 1
"""


def render_isolated_sglang_deployment_yaml(args: argparse.Namespace, assessment: dict[str, Any]) -> str:
    checkpoint = assessment["checkpoint"]
    if not checkpoint["sglang_loadable"]:
        raise ToolError("isolated SGLang deployment YAML is only available for SGLang-loadable model directories")

    node_name = args.sglang_node or candidate_sglang_node(assessment.get("k8s"))
    local_model_root = args.sglang_local_model_root.rstrip("/")
    log_dir = f"/opt/dlami/nvme/test_training/{args.sglang_deployment_name}"

    shell_script = "\n".join(
        [
            "set -euxo pipefail",
            f"MODEL_NAME={shlex.quote(args.model_name)}",
            f"MODEL_LOCAL={local_model_root}/" + "${MODEL_NAME}",
            f"MODEL_TMP={local_model_root}/" + "${MODEL_NAME}.tmp",
            f"mkdir -p {shlex.quote(local_model_root)} {shlex.quote(log_dir)}",
            "mkdir -p /opt/dlami/nvme/triton-cache /opt/dlami/nvme/deepgemm-cache /opt/dlami/nvme/tmp /opt/dlami/nvme/cuda-cache /opt/dlami/nvme/torch-extensions",
            "export TRITON_CACHE_DIR=/opt/dlami/nvme/triton-cache",
            "export SGLANG_DG_CACHE_DIR=/opt/dlami/nvme/deepgemm-cache",
            "export TMPDIR=/opt/dlami/nvme/tmp",
            "export CUDA_CACHE_PATH=/opt/dlami/nvme/cuda-cache",
            "export TORCH_EXTENSIONS_DIR=/opt/dlami/nvme/torch-extensions",
            "export NCCL_DEBUG=INFO",
            "export FI_PROVIDER=efa",
            "export FI_EFA_USE_DEVICE_RDMA=1",
            "export NVSHMEM_DISABLE_IBGDA=1",
            "export NVSHMEM_IB_ENABLE_IBGDA=0",
            "",
            "if [ ! -f \"${MODEL_LOCAL}/.copy_complete\" ]; then",
            "  rm -rf \"${MODEL_TMP}\"",
            "  mkdir -p \"${MODEL_TMP}\"",
            "  cp -r /s3export/. \"${MODEL_TMP}/\"",
            "  test -f \"${MODEL_TMP}/config.json\"",
            "  test -f \"${MODEL_TMP}/model.safetensors.index.json\"",
            "  rm -rf \"${MODEL_LOCAL}\"",
            "  mv \"${MODEL_TMP}\" \"${MODEL_LOCAL}\"",
            "  touch \"${MODEL_LOCAL}/.copy_complete\"",
            "  echo \">>> Model staged at ${MODEL_LOCAL}: $(du -sh \"${MODEL_LOCAL}\" | cut -f1)\"",
            "else",
            "  echo \">>> Reusing cached model at ${MODEL_LOCAL}: $(du -sh \"${MODEL_LOCAL}\" | cut -f1)\"",
            "fi",
            "",
            "exec sglang serve \\",
            "  --model-path \"${MODEL_LOCAL}\" \\",
            "  --served-model-name \"${MODEL_NAME}\" \\",
            "  --host 0.0.0.0 \\",
            "  --port 30000 \\",
            "  --tp-size 8 \\",
            "  --ep-size 8 \\",
            "  --trust-remote-code \\",
            "  --kv-cache-dtype fp8_e4m3 \\",
            "  --enable-fused-qk-norm-rope \\",
            "  --mem-fraction-static 0.82 \\",
            "  --moe-runner-backend flashinfer_trtllm \\",
            "  --moe-dense-tp-size 1 \\",
            "  --disable-flashinfer-autotune \\",
            "  --watchdog-timeout 1000000 \\",
            "  --enable-metrics",
        ]
    )
    indented_script = "\n".join(f"          {line}" for line in shell_script.splitlines())

    labels_4 = f"""    app.kubernetes.io/name: {args.sglang_deployment_name}
    app.kubernetes.io/component: isolated-sglang
    test-training.verdent.ai/model: qwen36-swift-r13"""
    labels_8 = f"""        app.kubernetes.io/name: {args.sglang_deployment_name}
        app.kubernetes.io/component: isolated-sglang
        test-training.verdent.ai/model: qwen36-swift-r13"""

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {args.sglang_deployment_name}
  namespace: {args.namespace}
  labels:
{labels_4}
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: {args.sglang_deployment_name}
      app.kubernetes.io/component: isolated-sglang
  template:
    metadata:
      labels:
{labels_8}
    spec:
      nodeName: {node_name}
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: sglang
        image: {args.sglang_image}
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-lc"]
        args:
        - |
{indented_script}
        ports:
        - name: http
          containerPort: 30000
        startupProbe:
          httpGet:
            path: /health
            port: http
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 540
        readinessProbe:
          httpGet:
            path: /health
            port: http
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 3
        resources:
          requests:
            cpu: "120"
            memory: "1200Gi"
            ephemeral-storage: "50Gi"
            nvidia.com/gpu: "8"
            vpc.amazonaws.com/efa: "8"
          limits:
            cpu: "160"
            memory: "1600Gi"
            ephemeral-storage: "80Gi"
            nvidia.com/gpu: "8"
            vpc.amazonaws.com/efa: "8"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: host-nvme
          mountPath: /opt/dlami/nvme
        - name: hfexport
          mountPath: /s3export
          readOnly: true
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi
      - name: host-nvme
        hostPath:
          path: /opt/dlami/nvme
          type: DirectoryOrCreate
      - name: hfexport
        persistentVolumeClaim:
          claimName: {args.sglang_hfexport_pvc}
---
apiVersion: v1
kind: Service
metadata:
  name: {args.sglang_service_name}
  namespace: {args.namespace}
  labels:
{labels_4}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: {args.sglang_deployment_name}
    app.kubernetes.io/component: isolated-sglang
  ports:
  - name: http
    port: 30000
    targetPort: http
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a checkpoint and guard GLM/SGLang deployment.",
    )
    parser.add_argument(
        "--checkpoint-s3-uri",
        default=DEFAULT_CHECKPOINT_S3_URI,
        help="S3 prefix containing the checkpoint or exported model directory.",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--kubeconfig", default=DEFAULT_KUBECONFIG)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--proxy-configmap", default=DEFAULT_PROXY_CONFIGMAP)
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--export-s3-uri", default=DEFAULT_EXPORT_S3_URI)
    parser.add_argument("--conversion-image", default=DEFAULT_CONVERSION_IMAGE)
    parser.add_argument("--conversion-service-account", default=DEFAULT_CONVERSION_SERVICE_ACCOUNT)
    parser.add_argument("--local-conversion-root", default=DEFAULT_LOCAL_CONVERSION_ROOT)
    parser.add_argument("--sglang-image", default=DEFAULT_SGLANG_IMAGE)
    parser.add_argument("--sglang-hfexport-pvc", default=DEFAULT_SGLANG_HFEXPORT_PVC)
    parser.add_argument("--sglang-smoke-job-name", default=DEFAULT_SGLANG_SMOKE_JOB_NAME)
    parser.add_argument("--sglang-local-model-root", default=DEFAULT_SGLANG_LOCAL_MODEL_ROOT)
    parser.add_argument("--sglang-deployment-name", default=DEFAULT_SGLANG_DEPLOYMENT_NAME)
    parser.add_argument("--sglang-service-name", default=DEFAULT_SGLANG_SERVICE_NAME)
    parser.add_argument("--sglang-restore-namespace", default=DEFAULT_SGLANG_RESTORE_NAMESPACE)
    parser.add_argument("--sglang-restore-deployment", default=DEFAULT_SGLANG_RESTORE_DEPLOYMENT)
    parser.add_argument("--sglang-restore-selector", default=DEFAULT_SGLANG_RESTORE_SELECTOR)
    parser.add_argument(
        "--sglang-node",
        help="HyperPod node name for generated isolated SGLang smoke YAML. Defaults to the GLM decode node if visible.",
    )
    parser.add_argument("--bridge-source-s3-uri", default=DEFAULT_BRIDGE_SOURCE_S3_URI)
    parser.add_argument("--bridge-setup-s3-uri", default=DEFAULT_BRIDGE_SETUP_S3_URI)
    parser.add_argument(
        "--conversion-node",
        help="HyperPod node name for generated conversion job YAML. Defaults to a GLM pod node if visible.",
    )
    parser.add_argument("--skip-k8s", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the preflight assessment to a JSON file.",
    )
    parser.add_argument(
        "--output-conversion-job-yaml",
        type=Path,
        help="Write an isolated Megatron-to-HF conversion Job manifest.",
    )
    parser.add_argument(
        "--output-sglang-smoke-job-yaml",
        type=Path,
        help="Write an isolated one-node SGLang load-smoke Job manifest for an exported model directory.",
    )
    parser.add_argument(
        "--output-sglang-smoke-restore-script",
        type=Path,
        help="Write a guarded helper script that temporarily frees the GLM decode node, runs the smoke Job, and restores replicas.",
    )
    parser.add_argument(
        "--output-isolated-sglang-deployment-yaml",
        type=Path,
        help="Write an isolated SGLang Deployment and ClusterIP Service manifest for the exported model directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Attempt live deployment only if all preflight blockers are clear.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        assessment = build_assessment(args)
    except Exception as exc:  # noqa: BLE001
        print(f"deploy_model.py: preflight failed: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        write_json(args.output_json, assessment)

    if args.output_conversion_job_yaml:
        args.output_conversion_job_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.output_conversion_job_yaml.write_text(
            render_conversion_job_yaml(args, assessment),
            encoding="utf-8",
        )

    if args.output_sglang_smoke_job_yaml:
        args.output_sglang_smoke_job_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.output_sglang_smoke_job_yaml.write_text(
            render_sglang_smoke_job_yaml(args, assessment),
            encoding="utf-8",
        )

    if args.output_sglang_smoke_restore_script:
        args.output_sglang_smoke_restore_script.parent.mkdir(parents=True, exist_ok=True)
        args.output_sglang_smoke_restore_script.write_text(
            render_sglang_smoke_restore_script(args),
            encoding="utf-8",
        )

    if args.output_isolated_sglang_deployment_yaml:
        args.output_isolated_sglang_deployment_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.output_isolated_sglang_deployment_yaml.write_text(
            render_isolated_sglang_deployment_yaml(args, assessment),
            encoding="utf-8",
        )

    print(json.dumps(assessment, indent=2, sort_keys=True))

    if args.apply:
        if assessment["blockers"]:
            print(
                "deploy_model.py: refusing live deployment because preflight blockers "
                "are present.",
                file=sys.stderr,
            )
            return 2
        print(
            "deploy_model.py: preflight is clear, but live mutation is intentionally "
            "not implemented in this guard. Generate isolated SGLang smoke and "
            "Deployment/Service manifests first.",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
