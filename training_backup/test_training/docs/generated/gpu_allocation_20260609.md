# GPU Allocation Snapshot

Checked: 2026-06-09 UTC. Refreshed after the isolated Qwen36 deployment.

Command:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig get pods -A -o go-template='{{range .items}}{{ $node := .spec.nodeName }}{{ $ns := .metadata.namespace }}{{ $name := .metadata.name }}{{range .spec.containers}}{{with .resources.requests}}{{ $gpu := index . "nvidia.com/gpu" }}{{ $efa := index . "vpc.amazonaws.com/efa" }}{{if or $gpu $efa}}{{printf "%s\t%s\t%s\t%s\t%s\n" $node $ns $name $gpu $efa}}{{end}}{{end}}{{end}}{{end}}'
```

Current live GPU/EFA request table:

Accessible HyperPod/K8s nodes:

| Node | Instance type | Instance group | Health | Allocatable GPU | Allocatable EFA |
| --- | --- | --- | --- | ---: | ---: |
| `hyperpod-i-01e93a89cb27a5046` | `ml.p6-b200.48xlarge` | `b200-on-demand` | `Schedulable` | 8 | 8 |
| `hyperpod-i-060621000d6270d47` | `ml.p6-b200.48xlarge` | `b200-on-demand` | `Schedulable` | 8 | 8 |
| `hyperpod-i-0b7611eea4f8e7956` | `ml.p6-b200.48xlarge` | `b200-on-demand` | `Schedulable` | 8 | 8 |
| `hyperpod-i-0da428942dd5a9cd9` | `ml.p6-b200.48xlarge` | `b200-on-demand` | `Schedulable` | 8 | 8 |
| `hyperpod-i-0e1a42052af8e595e` | `ml.c5.2xlarge` | `c52xlarge` | `Schedulable` | 0 | 0 |

HyperPod instance-group check:

| Instance group | Instance type | Capacity | Current | Target | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `b200-on-demand` | `ml.p6-b200.48xlarge` | on-demand | 4 | 4 | all four nodes joined K8s |
| `b300-spot` | `ml.p6-b300.48xlarge` | spot | 0 | 0 | no node requested/joined |
| `b300-spot-1` | `ml.p6-b300.48xlarge` | spot | 0 | 0 | no node requested/joined |

| Node | Namespace | Pod | GPU | EFA |
| --- | --- | --- | ---: | ---: |
| `hyperpod-i-01e93a89cb27a5046` | `dpsk-v4-pro-standalone-b200-p1d1` | `sglang-standalone-56c69f8796-xc78p` | 8 | not requested |
| `hyperpod-i-060621000d6270d47` | `glm51-pd-b200-p1d1` | `qwen36-swift-r13-sglang-76f4c76b45-4vm8d` | 8 | 8 |
| `hyperpod-i-0da428942dd5a9cd9` | `kimi-k26-b200-p1d1` | `sglang-decode-anchor-7c56d5b957-5z5qd` | 8 | 8 |
| `hyperpod-i-0b7611eea4f8e7956` | `kimi-k26-b200-p1d1` | `sglang-prefill-676d4646cd-xxb4d` | 8 | 8 |

GLM status:

- `glm51-pd-b200-p1d1/qwen36-swift-r13-sglang`: `1/1`, isolated service
  only.
- `glm51-pd-b200-p1d1/sglang-decode-anchor`: `0/0`.
- `glm51-pd-b200-p1d1/sglang-prefill`: `0/0`.
- `glm51-pd-b200-p1d1/sglang-router`: `0/0`.
- `glm51-pd-b200-p1d1/sglang-monitor`: `0/0`.

Interpretation:

- The freed B200 node `hyperpod-i-060621000d6270d47` is now occupied by the
  isolated Qwen36 SGLang deployment.
- The isolated service is a ClusterIP service and is not wired to the existing
  GLM router or external proxy.
- The temporary smoke Job was deleted after evidence was recorded, so the live
  request table above contains only active serving pods.
- The expected extra spot node is still not visible in Kubernetes. The current
  Kubernetes node list shows the same four B200 nodes plus one CPU node.
