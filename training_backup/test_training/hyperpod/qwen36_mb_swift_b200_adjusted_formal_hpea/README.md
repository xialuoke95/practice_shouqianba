# qwen36 B200 Adjusted Formal HPEA HyperPod Run

This scaffold keeps the B200-adjusted formal training recipe from
`qwen36_mb_swift_b200_adjusted_formal`, but replaces the direct training command
with a small HPEA-compatible lifecycle wrapper.

The wrapper exposes port `8080` endpoints expected by the HyperPod podmanager:

- `/status`: starts as `ready`, then reports `running`, `completed`, `faulted`,
  `stopping`, or `stopped`.
- `/start`: starts `/scripts/run_qwen36_overlay_sft_job.sh`.
- `/stop`: terminates the training process group and returns HTTP 200.

Important behavior:

- The job does not set `container.command`; it passes the wrapper as `args` so
  the image's default entrypoint remains in control.
- Training starts only after podmanager calls `/start`.
- The original training script and recipe remain unchanged.
- The wrapper process stays alive after training completion so the operator can
  observe `completed` instead of hitting a Deployment restart loop.

Preserved recipe shape:

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
- image `nvcr.io/nvidia/nemo:26.04.00`
- B200 resources: 8 GPU, 8 EFA, `1800Gi/1900Gi`

Apply the wrapper ConfigMap:

```bash
kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig \
  create configmap qwen36-mb-hpea-wrapper-r1 \
  -n dolly-probe-training \
  --from-file=hpea_worker.py=/home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal_hpea/hpea_worker.py \
  --dry-run=client -o yaml \
| kubectl --kubeconfig /tmp/eks-cluster-hypd-0324.kubeconfig apply -f -
```

Submit:

```bash
cd /home/ubuntu/workspace/test_training/hyperpod/qwen36_mb_swift_b200_adjusted_formal_hpea
/home/ubuntu/.local/bin/hyp validate
/home/ubuntu/.local/bin/hyp create
```

Expected S3 root:

`s3://verdent.tech/test_training/qwen36_megatron_bridge_overlay_sft/qwen36-mb-overlay-aligned-fullsft-20260602-211523/training/qwen36_mb_swift_b200_adjusted_formal_hpea_20260617/`
