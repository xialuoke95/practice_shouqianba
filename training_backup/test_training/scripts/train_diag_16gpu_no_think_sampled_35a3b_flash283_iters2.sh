#!/bin/bash
# Qwen3.6-35B-A3B diagnostic retry - 2 nodes x 8 GPUs = 16xH100
# Purpose: validate the flash-attn pin and capture NVTE backend reasons if DPA fails.
# Parallel: TP2 x PP2 x CP2 -> decoder DP2; EP4 -> expert DP2; world = 16 GPUs.
# Context: keep max_length 256K to test the same long-context attention path as the failed job.

set -e

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
WORK_ROOT=/pfs-verdent/guoyinuo/coding_agent_models
MODEL_DIR=/pfs-verdent/subingtian/models/Qwen3.6-35B-A3B
DATASET=$WORK_ROOT/output/may_data_versions/no-think-sampled/no-think-sampled.train.jsonl

OUTPUT_DIR=$WORK_ROOT/output/train_diag_16gpu_no_think_sampled_35a3b_flash283_iters2
CACHE_DIR=/pfs-verdent/subingtian/.cache
HF_CACHE=/pfs-verdent/subingtian/.cache
TMP_DIR=/pfs-verdent/subingtian/.tmp

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$HF_CACHE" "$TMP_DIR"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Log file: $LOG_FILE ==="
echo "=== Start time: $(date) ==="

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
pip install "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git" -U
python3 -c "import importlib.metadata as m; print('ms-swift version:', m.version('ms-swift'))"
python3 -c "import torch; print('torch version:', torch.__version__, 'torch cuda:', torch.version.cuda)"
python3 -c "import json_repair, swift; print('swift import check: ok')"
pip install git+https://github.com/NVIDIA/Megatron-LM.git
pip install "mcore-bridge>=1.1.0" -U
pip install "qwen_vl_utils>=0.0.14"
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 2>/dev/null || true
pip install "transformer_engine[pytorch]==2.16.0" --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation
pip install flash-linear-attention
python3 -c "import importlib.metadata as m; print('transformer-engine version:', m.version('transformer-engine')); print('flash-attn version:', m.version('flash-attn'))"

# Patch: force-enable CP for GatedDeltaNet
python3 -c "
import glob, re
for f in glob.glob('/root/**/transformer_config.py', recursive=True):
    if 'megatron/core/transformer' in f:
        t = open(f).read()
        t = re.sub(r'assert self\.context_parallel_size == 1,\s*\\n\\s*f\"Context parallelism is not supported with {self.num_layers_per_virtual_pipeline_stage=}.\"', 'pass  # patched', t, flags=re.DOTALL)
        t = re.sub(r'assert self\.context_parallel_size == 1,\s*$.*?$', 'pass  # patched', t, flags=re.DOTALL)
        open(f, 'w').write(t)
        print(f'Patched: {f}')
"

# ---------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export HF_DATASETS_CACHE=$HF_CACHE
export TMPDIR=$TMP_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=2400
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# ---------------------------------------------------------------------------
# 3. Symlink /root/.cache to pfs
# ---------------------------------------------------------------------------
if [ ! -L /root/.cache ]; then
    rm -rf /root/.cache
    ln -s "$CACHE_DIR" /root/.cache
fi

# ---------------------------------------------------------------------------
# 4. Distributed config
# ---------------------------------------------------------------------------
echo "=== Distributed Config ==="
echo "MLP_WORKER_NUM=$MLP_WORKER_NUM MLP_ROLE_INDEX=$MLP_ROLE_INDEX MLP_WORKER_GPU=$MLP_WORKER_GPU"
echo "MLP_WORKER_0_HOST=$MLP_WORKER_0_HOST MLP_WORKER_0_PORT=$MLP_WORKER_0_PORT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASET=$DATASET"

# ---------------------------------------------------------------------------
# 5. Training - short diagnostic
# ---------------------------------------------------------------------------
python -m torch.distributed.launch \
    --nproc_per_node $MLP_WORKER_GPU \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
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
    --cross_entropy_fusion_impl native \
    --optimizer_cpu_offload true \
    --truncation_strategy delete \
    --report_to tensorboard \
    --logging_steps 1

echo "=== End time: $(date) ==="
