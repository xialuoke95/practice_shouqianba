# qwen36 B200 Adjusted Formal HPEA Preflight

Date: 2026-06-17 UTC

## Objective

Replace the current direct-command HyperPod run with the same B200-adjusted
training recipe submitted through a HPEA-compatible lifecycle wrapper, so the
HyperPod Training Operator and UI can observe/manage the training record.

## Current Live Job

Still running at last check:

- `HyperPodPyTorchJob`: `dolly-probe-training/qwen36-mb-swift-b200-adjusted-formal`
- Pod: `qwen36-mb-swift-b200-adjusted-formal-pod-79db45bfb6-j9ph6`
- Node: `hyperpod-i-0cd50aacaabb9612d`
- Latest observed phase: baseline eval, around `Evaluating iter 739/934`
- The pod still runs the old direct command:
  `/scripts/run_qwen36_overlay_sft_job.sh`

The old job still lacks a valid HPEA lifecycle endpoint and remains unsuitable
for Training History visibility.

## Prepared Replacement

Scaffold:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal_hpea`

Cluster resource prepared:

- `ConfigMap`: `dolly-probe-training/qwen36-mb-hpea-wrapper-r1`

The replacement keeps the same training recipe shape as the direct-command
formal run:

- `SEQ_LENGTH=128000`
- `GBS=16`
- `MBS=1`
- `TP=2`
- `PP=1`
- `CP=4`
- `EP=8`
- `ETP=1`
- `EPOCHS=2`
- dataset `sft_opus_sonnet_swift_mbjson_v2`
- B200 resources: 8 GPU, 8 EFA, `1800Gi/1900Gi`

## HPEA Wrapper Behavior

Wrapper:

`/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal_hpea/hpea_worker.py`

Endpoints:

- `/status`: reports `ready`, `running`, `completed`, `faulted`, `stopping`, or
  `stopped`.
- `/start`: launches `/scripts/run_qwen36_overlay_sft_job.sh`.
- `/stop`: terminates the training process group and returns HTTP 200.

Podmanager reverse-engineering showed the expected lowercase status flow:

- Initial state should be `ready`.
- Training state should be `running`.
- Successful completion should be `completed`.
- Failure should be `faulted`.

The earlier diagnostic probe returned uppercase `Running` too early, which kept
the operator in `Created` and caused repeated `/stop` calls.

## Entrypoint Validation

The replacement manifest no longer sets `container.command`. It passes only:

```yaml
args:
  - /hpea/hpea_worker.py
```

This was validated with a temporary no-GPU pod using the same image
`nvcr.io/nvidia/nemo:26.04.00`, the same wrapper ConfigMap, and no command
override:

- Pod `qwen36-hpea-args-entrypoint-test` reached `Running` and `Ready`.
- Logs showed the NeMo image default entrypoint banner followed by:
  `[hpea-wrapper] serving lifecycle API on 0.0.0.0:18082`
- In-pod `/status` returned:
  `{"status": "ready", "ready": "True", ...}`
- The temporary pod was deleted afterward.

This proves the replacement can preserve the image default entrypoint while
using the wrapper as the actual entry script.

## Validation

Completed locally:

- `python3 -m py_compile hpea_worker.py`
- `/home/ubuntu/.local/bin/hyp validate`
- Rendered template confirms:
  - `kind: HyperPodPyTorchJob`
  - `metadata.name: qwen36-mb-swift-b200-adjusted-formal-hpea`
  - no `container.command`
  - `args: ["/hpea/hpea_worker.py"]`

## Remaining Blocker

The old live job must be deleted before submitting the replacement to avoid
leaving the current B200 training running and to satisfy the requested flow.

Two `hyp delete` attempts were rejected by the automatic safety reviewer because
the transcript did not contain a normal user message explicitly approving
deletion of this specific live job:

`qwen36-mb-swift-b200-adjusted-formal`

Next safe step after explicit approval:

```bash
/home/ubuntu/.local/bin/hyp delete hyp-pytorch-job \
  --job-name qwen36-mb-swift-b200-adjusted-formal \
  -n dolly-probe-training
```

Then submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal_hpea
/home/ubuntu/.local/bin/hyp create
```

Post-submit verification should check:

- New pod starts with no `container.command` and `args=/hpea/hpea_worker.py`.
- Podmanager calls `/status` and `/start` successfully.
- `HyperPodPyTorchJob` condition reaches `PodsRunning=True`.
- `hyp list hyp-pytorch-job -n dolly-probe-training` moves beyond `Created`.
- Training History / management UI shows the submitted run.
