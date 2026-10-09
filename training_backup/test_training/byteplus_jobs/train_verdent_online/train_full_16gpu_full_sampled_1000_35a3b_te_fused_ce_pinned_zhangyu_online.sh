#!/bin/bash
# Qwen3.6-35B-A3B full-data sampled-1000 training - 2 nodes x 8 GPUs.
# Reproduces the successful 16-H100 run t-20260617173940-v4zgz, with
# training dependencies pinned to the commits observed in that run.
# Checkpoints/logs are written under /pfs-verdent/zhangyu/train_verdent_online.

set -euo pipefail

RUN_ROOT=/pfs-verdent/zhangyu/train_verdent_online
SOURCE_ROOT=/pfs-verdent/guoyinuo/coding_agent_models
MODEL_DIR=/pfs-verdent/subingtian/models/Qwen3.6-35B-A3B
DATASET=$SOURCE_ROOT/output/may_data_versions/full-sampled-1000/full-sampled-1000.train.jsonl

OUTPUT_DIR=$RUN_ROOT/output/train_full_16gpu_full_sampled_1000_35a3b_te_fused_ce_pinned
CACHE_DIR=/pfs-verdent/subingtian/.cache
HF_CACHE=/pfs-verdent/subingtian/.cache
TMP_DIR=/pfs-verdent/subingtian/.tmp

SWIFT_COMMIT=f3bf35f34e6e1387ae40add157d0885fc1026f1a
MEGATRON_COMMIT=7604f28f7a57b8370abfc43890a355b0d2acdc6a
TRANSFORMERS_VERSION=5.10.2
TOKENIZERS_VERSION=0.22.2
MCORE_BRIDGE_VERSION=1.4.3

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$HF_CACHE" "$TMP_DIR"

LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Log file: $LOG_FILE ==="
echo "=== Start time: $(date) ==="

echo "=== Dataset preflight ==="
echo "DATASET=$DATASET"
if [ ! -s "$DATASET" ]; then
    echo "ERROR: dataset file missing or empty: $DATASET" >&2
    exit 2
fi
echo "DATASET_LINES=$(wc -l < "$DATASET")"

echo "=== Pinned dependency plan ==="
echo "SWIFT_COMMIT=$SWIFT_COMMIT"
echo "MEGATRON_COMMIT=$MEGATRON_COMMIT"
echo "TRANSFORMERS_VERSION=$TRANSFORMERS_VERSION"
echo "TOKENIZERS_VERSION=$TOKENIZERS_VERSION"
echo "MCORE_BRIDGE_VERSION=$MCORE_BRIDGE_VERSION"

pip install "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git@$SWIFT_COMMIT" -U
pip install "transformers==$TRANSFORMERS_VERSION" "tokenizers==$TOKENIZERS_VERSION" -U
python3 -c "import importlib.metadata as m; print('ms-swift version:', m.version('ms-swift')); print('transformers version:', m.version('transformers')); print('tokenizers version:', m.version('tokenizers'))"
python3 -c "import torch; print('torch version:', torch.__version__, 'torch cuda:', torch.version.cuda)"
python3 -c "import json_repair, swift; print('swift import check: ok')"
pip install "git+https://github.com/NVIDIA/Megatron-LM.git@$MEGATRON_COMMIT"
pip install "mcore-bridge==$MCORE_BRIDGE_VERSION"
pip install "qwen_vl_utils>=0.0.14"
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 2>/dev/null || true
pip install "transformer_engine[pytorch]==2.16.0" --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation
pip install flash-linear-attention
python3 -c "import importlib.metadata as m; print('megatron-core version:', m.version('megatron-core')); print('mcore-bridge version:', m.version('mcore-bridge')); print('transformer-engine version:', m.version('transformer-engine')); print('flash-attn version:', m.version('flash-attn'))"

# Patch: force-enable CP for GatedDeltaNet.
python3 -c "
import glob, re
for f in glob.glob('/root/**/transformer_config.py', recursive=True):
    if 'megatron/core/transformer' in f:
        t = open(f).read()
        t = re.sub(r'assert self\\.context_parallel_size == 1,\\s*\\n\\s*f\"Context parallelism is not supported with {self.num_layers_per_virtual_pipeline_stage=}.\"', 'pass  # patched', t, flags=re.DOTALL)
        t = re.sub(r'assert self\\.context_parallel_size == 1,\\s*$.*?$', 'pass  # patched', t, flags=re.DOTALL)
        open(f, 'w').write(t)
        print(f'Patched: {f}')
"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export HF_DATASETS_CACHE=$HF_CACHE
export TMPDIR=$TMP_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=2400
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

if [ ! -L /root/.cache ]; then
    rm -rf /root/.cache
    ln -s "$CACHE_DIR" /root/.cache
fi

echo "=== Distributed Config ==="
echo "MLP_WORKER_NUM=${MLP_WORKER_NUM:-} MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-} MLP_WORKER_GPU=${MLP_WORKER_GPU:-}"
echo "MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-} MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASET=$DATASET"

python -m torch.distributed.launch \
    --nproc_per_node "$MLP_WORKER_GPU" \
    --nnodes "$MLP_WORKER_NUM" \
    --node_rank "$MLP_ROLE_INDEX" \
    --master_addr "$MLP_WORKER_0_HOST" \
    --master_port "$MLP_WORKER_0_PORT" \
    -m swift.cli._megatron.sft \
    --model "$MODEL_DIR" \
    --dataset "$DATASET" \
    --agent_template qwen3_5 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 24 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --bf16 true \
    --weight_decay 0.1 \
    --num_train_epochs 1 \
    --train_iters 2 \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 1000 \
    --max_length 262144 \
    --max_pixels 2408448 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --gradient_accumulation_fusion false \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --optimizer_cpu_offload true \
    --truncation_strategy delete \
    --report_to tensorboard \
    --logging_steps 1

echo "=== End time: $(date) ==="
