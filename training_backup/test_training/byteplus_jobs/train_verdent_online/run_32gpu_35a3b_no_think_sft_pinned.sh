#!/bin/bash
# Shared 32-H100 Qwen3.6-35B-A3B no-think SFT runner.
# The wrapper script must set DATASET and OUTPUT_SUBDIR.

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/pfs-verdent/zhangyu/train_verdent_online}
MODEL_DIR=${MODEL_DIR:-/pfs-verdent/subingtian/models/Qwen3.6-35B-A3B}
DATASET=${DATASET:?DATASET is required}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:?OUTPUT_SUBDIR is required}
DATASET_SUMMARY=${DATASET_SUMMARY:-$(dirname "$DATASET")/summary.json}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/output/$OUTPUT_SUBDIR}
CACHE_DIR=${CACHE_DIR:-/pfs-verdent/subingtian/.cache}
HF_CACHE=${HF_CACHE:-/pfs-verdent/subingtian/.cache}
TMP_DIR=${TMP_DIR:-/pfs-verdent/subingtian/.tmp}

SWIFT_COMMIT=${SWIFT_COMMIT:-1bac1fad8b70d22e8d46c2e4b5dd32a892393f91}
MEGATRON_COMMIT=${MEGATRON_COMMIT:-d1410e15e164923f23b8e1a3384bf22c29640fcc}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.10.4}
TOKENIZERS_VERSION=${TOKENIZERS_VERSION:-0.22.2}
MCORE_BRIDGE_VERSION=${MCORE_BRIDGE_VERSION:-1.4.3}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-72}
MAX_LENGTH=${MAX_LENGTH:-180000}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
TRAIN_ITERS=${TRAIN_ITERS:-80}
MIN_DATASET_LINES=${MIN_DATASET_LINES:-1}

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
DATASET_LINES=$(wc -l < "$DATASET")
echo "DATASET_LINES=$DATASET_LINES"
if [ "$DATASET_LINES" -lt "$MIN_DATASET_LINES" ]; then
    echo "ERROR: dataset has fewer rows than expected: $DATASET_LINES < $MIN_DATASET_LINES" >&2
    exit 3
fi
if [ -s "$DATASET_SUMMARY" ]; then
    echo "=== Dataset summary ==="
    cat "$DATASET_SUMMARY"
else
    echo "WARN: dataset summary missing: $DATASET_SUMMARY"
fi

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

if [ ! -L /root/.cache ]; then
    rm -rf /root/.cache
    ln -s "$CACHE_DIR" /root/.cache
fi

echo "=== Distributed Config ==="
echo "MLP_WORKER_NUM=${MLP_WORKER_NUM:-} MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-} MLP_WORKER_GPU=${MLP_WORKER_GPU:-}"
echo "MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-} MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASET=$DATASET"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE MAX_LENGTH=$MAX_LENGTH NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS TRAIN_ITERS=$TRAIN_ITERS"

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
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --bf16 true \
    --weight_decay 0.1 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --train_iters "$TRAIN_ITERS" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 1000 \
    --max_length "$MAX_LENGTH" \
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
    --logging_steps 5

echo "=== End time: $(date) ==="
