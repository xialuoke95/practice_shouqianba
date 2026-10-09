#!/usr/bin/env bash
# Qwen3.6-27B full SFT on the exact 47 renderer-approved SaaS/Frappe samples
# used by the completed Fireworks saas_sft_v0 run.

set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/pfs-verdent/guoyinuo/coding_agent_models}
MODEL_DIR=${MODEL_DIR:-$WORK_ROOT/models/Qwen3.6-27B}
DATA_ROOT=${DATA_ROOT:-/pfs-verdent/subingtian/coding_agent_models/output/saas_sft_v0_exact47_20260806}
DATASET=${DATASET:-$DATA_ROOT/saas_sft_v0_fireworks_exact47.train.jsonl}
DATASET_SUMMARY=${DATASET_SUMMARY:-$DATA_ROOT/summary.json}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK_ROOT/output/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2_20260806}

CACHE_DIR=${CACHE_DIR:-/pfs-verdent/subingtian/.cache}
HF_CACHE=${HF_CACHE:-/pfs-verdent/subingtian/.cache}
TMP_DIR=${TMP_DIR:-/pfs-verdent/subingtian/.tmp}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.org/simple}
MCORE_BRIDGE_WHEELHOUSE=${MCORE_BRIDGE_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/mcore_bridge_1_4_3}
TRANSFORMERS_WHEELHOUSE=${TRANSFORMERS_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/transformers_5_10_2}
TRANSFORMER_ENGINE_WHEELHOUSE=${TRANSFORMER_ENGINE_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/transformer_engine_2_15_0}
FLASH_ATTN_WHEELHOUSE=${FLASH_ATTN_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/flash_attn_2_8_3}
FLASH_LINEAR_ATTENTION_WHEELHOUSE=${FLASH_LINEAR_ATTENTION_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/flash_linear_attention_0_5_0}

EXPECTED_DATASET_LINES=${EXPECTED_DATASET_LINES:-47}
EXPECTED_DATASET_SHA256=${EXPECTED_DATASET_SHA256:-b8d936581953d8edf59d24906fa4b6a88911b0bec2da12183f2b8aa2ccaad139}
EXPECTED_TRAINABLE_MESSAGES=${EXPECTED_TRAINABLE_MESSAGES:-2621}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
SAVE_STEPS=${SAVE_STEPS:-24}
MAX_LENGTH=${MAX_LENGTH:-262144}

TP_SIZE=4
PP_SIZE=2
CP_SIZE=2
MICRO_BATCH_SIZE=1

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$HF_CACHE" "$TMP_DIR"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Log file: $LOG_FILE ==="
echo "=== Start time: $(date) ==="

echo "=== Dataset preflight ==="
echo "MODEL_DIR=$MODEL_DIR"
echo "DATASET=$DATASET"
echo "DATASET_SUMMARY=$DATASET_SUMMARY"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir missing: $MODEL_DIR" >&2
    exit 2
fi
if [ ! -s "$DATASET" ]; then
    echo "ERROR: dataset file missing or empty: $DATASET" >&2
    exit 3
fi

DATASET_LINES=$(wc -l < "$DATASET")
DATASET_SHA256=$(sha256sum "$DATASET" | awk '{print $1}')
echo "DATASET_LINES=$DATASET_LINES"
echo "DATASET_SHA256=$DATASET_SHA256"
if [ "$DATASET_LINES" -ne "$EXPECTED_DATASET_LINES" ]; then
    echo "ERROR: expected $EXPECTED_DATASET_LINES rows, got $DATASET_LINES" >&2
    exit 4
fi
if [ "$DATASET_SHA256" != "$EXPECTED_DATASET_SHA256" ]; then
    echo "ERROR: dataset SHA256 mismatch" >&2
    exit 5
fi

python3 - "$DATASET" "$EXPECTED_TRAINABLE_MESSAGES" <<'PY'
import json
import sys

path = sys.argv[1]
expected_trainable = int(sys.argv[2])
trace_ids = set()
trainable = 0
with open(path, "r", encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, 1):
        lowered = raw.lower()
        if "<think" in lowered or "reasoning_content" in raw:
            raise SystemExit(f"ERROR: thinking residue at line {line_number}")
        row = json.loads(raw)
        missing = [
            key
            for key in ("messages", "trace_id", "session_id", "model", "day")
            if key not in row
        ]
        if missing:
            raise SystemExit(f"ERROR: line {line_number} missing keys: {missing}")
        if row.get("thinking_type") != "none" or row.get("thinking_budget") is not None:
            raise SystemExit(f"ERROR: line {line_number} is not native no-think")
        if row["trace_id"] in trace_ids:
            raise SystemExit(f"ERROR: duplicate trace_id at line {line_number}")
        trace_ids.add(row["trace_id"])
        trainable += sum(
            1
            for message in row["messages"]
            if message.get("role") in {"assistant", "tool_call"}
            and message.get("loss", 1) == 1
        )
if trainable != expected_trainable:
    raise SystemExit(
        f"ERROR: expected {expected_trainable} trainable messages, got {trainable}"
    )
print(
    "dataset_json_check=ok",
    f"unique_trace_ids={len(trace_ids)}",
    f"trainable_messages={trainable}",
)
PY

if [ -s "$DATASET_SUMMARY" ]; then
    echo "=== Dataset summary ==="
    cat "$DATASET_SUMMARY"
else
    echo "WARN: dataset summary missing: $DATASET_SUMMARY"
fi

install_from_wheelhouse() {
    local spec="$1"
    local wheelhouse="$2"
    shift 2
    local extra_args=("$@")
    if [ -d "$wheelhouse" ] && find "$wheelhouse" -maxdepth 1 \( -name '*.whl' -o -name '*.tar.gz' \) -print -quit | grep -q .; then
        pip install "$spec" "${extra_args[@]}" --no-index --find-links "$wheelhouse" -U
    else
        pip install "$spec" "${extra_args[@]}" -U || \
            pip install "$spec" "${extra_args[@]}" -U --index-url "$PYPI_INDEX_URL"
    fi
}

install_mcore_bridge() {
    local spec="mcore-bridge==1.4.3"
    if compgen -G "$MCORE_BRIDGE_WHEELHOUSE/mcore_bridge-1.4.3-*.whl" >/dev/null; then
        pip install "$spec" --no-index --find-links "$MCORE_BRIDGE_WHEELHOUSE" -U
    else
        pip install "$spec" -U || pip install "$spec" -U --index-url "$PYPI_INDEX_URL"
    fi
}

install_transformers() {
    local spec="transformers==5.10.2"
    if compgen -G "$TRANSFORMERS_WHEELHOUSE/transformers-5.10.2-*.whl" >/dev/null; then
        pip install "$spec" --no-index --find-links "$TRANSFORMERS_WHEELHOUSE" -U
    else
        pip install "$spec" -U || pip install "$spec" -U --index-url "$PYPI_INDEX_URL"
    fi
}

export PIP_CACHE_DIR="$CACHE_DIR/pip"
pip install "ms-swift @ git+https://github.com/modelscope/ms-swift.git@85d98d60577d951da137fb7d1c97ef3dc5981c59" -U
pip install git+https://github.com/NVIDIA/Megatron-LM.git@930bb6f69f99fdde5daae59b4e8de9f348a1ed8a
install_mcore_bridge
install_transformers
pip install "qwen_vl_utils>=0.0.14"
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 2>/dev/null || true
install_from_wheelhouse "transformer_engine[pytorch]==2.15.0" "$TRANSFORMER_ENGINE_WHEELHOUSE" --no-build-isolation
install_from_wheelhouse "flash-attn==2.8.3" "$FLASH_ATTN_WHEELHOUSE" --no-build-isolation
install_from_wheelhouse "flash-linear-attention==0.5.0" "$FLASH_LINEAR_ATTENTION_WHEELHOUSE"

python3 -c "import importlib.metadata as m; print('ms-swift:', m.version('ms-swift')); print('mcore-bridge:', m.version('mcore-bridge')); print('transformers:', m.version('transformers')); print('transformer-engine:', m.version('transformer-engine')); print('flash-attn:', m.version('flash-attn'))"
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu_count:', torch.cuda.device_count())"

python3 - <<'PY'
import glob
import re

for path in glob.glob('/root/**/transformer_config.py', recursive=True):
    if 'megatron/core/transformer' not in path:
        continue
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    patched = re.sub(
        r'assert self\.context_parallel_size == 1,\s*\n\s*f"Context parallelism is not supported with {self.num_layers_per_virtual_pipeline_stage=}."',
        'pass  # Qwen3.6 GatedDeltaNet CP validated by prior BytePlus runs',
        text,
        flags=re.DOTALL,
    )
    patched = re.sub(
        r'assert self\.context_parallel_size == 1,\s*$.*?$',
        'pass  # Qwen3.6 GatedDeltaNet CP validated by prior BytePlus runs',
        patched,
        flags=re.DOTALL,
    )
    if patched != text:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(patched)
    print(f'CP patch checked: {path}')
PY

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export HF_DATASETS_CACHE="$HF_CACHE"
export TMPDIR="$TMP_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=2400
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

: "${MLP_WORKER_NUM:?MLP_WORKER_NUM is required}"
: "${MLP_ROLE_INDEX:?MLP_ROLE_INDEX is required}"
: "${MLP_WORKER_GPU:?MLP_WORKER_GPU is required}"
: "${MLP_WORKER_0_HOST:?MLP_WORKER_0_HOST is required}"
: "${MLP_WORKER_0_PORT:?MLP_WORKER_0_PORT is required}"

WORLD_SIZE=$((MLP_WORKER_NUM * MLP_WORKER_GPU))
if [ "$WORLD_SIZE" -ne "$EXPECTED_WORLD_SIZE" ]; then
    echo "ERROR: expected $EXPECTED_WORLD_SIZE GPUs, got $WORLD_SIZE" >&2
    exit 6
fi
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE * CP_SIZE)))
if [ "$((TP_SIZE * PP_SIZE * CP_SIZE * DP_SIZE))" -ne "$WORLD_SIZE" ]; then
    echo "ERROR: invalid parallel config for WORLD_SIZE=$WORLD_SIZE" >&2
    exit 7
fi
if [ "$((GLOBAL_BATCH_SIZE % (DP_SIZE * MICRO_BATCH_SIZE)))" -ne 0 ]; then
    echo "ERROR: GBS must be divisible by DP_SIZE * MICRO_BATCH_SIZE" >&2
    exit 8
fi
MICROBATCHES_PER_STEP=$((GLOBAL_BATCH_SIZE / (DP_SIZE * MICRO_BATCH_SIZE)))

echo "=== Distributed config ==="
echo "WORLD_SIZE=$WORLD_SIZE TP=$TP_SIZE PP=$PP_SIZE CP=$CP_SIZE DP=$DP_SIZE"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE MICROBATCHES_PER_STEP=$MICROBATCHES_PER_STEP"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS SAVE_STEPS=$SAVE_STEPS MAX_LENGTH=$MAX_LENGTH"

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
    --tensor_model_parallel_size "$TP_SIZE" \
    --pipeline_model_parallel_size "$PP_SIZE" \
    --context_parallel_size "$CP_SIZE" \
    --sequence_parallel true \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --optimizer_cpu_offload true \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style cosine \
    --bf16 true \
    --weight_decay 0.1 \
    --clip_grad 1.0 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --max_length "$MAX_LENGTH" \
    --max_pixels 2408448 \
    --dataloader_num_workers 2 \
    --no_save_optim true \
    --no_save_rng true \
    --gradient_accumulation_fusion false \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --calculate_per_token_loss true \
    --truncation_strategy delete \
    --report_to tensorboard \
    --logging_steps 1

echo "=== End time: $(date) ==="
