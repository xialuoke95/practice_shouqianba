#!/usr/bin/env bash
# Generalized Qwen3.6-27B full-SFT entrypoint for the three sequential
# Frappe/GLM dataset arms. It reuses the dependency/distributed/training tail
# from the exact-47 script that already completed successfully on BytePlus.

set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/pfs-verdent/guoyinuo/coding_agent_models}
MODEL_DIR=${MODEL_DIR:-$WORK_ROOT/models/Qwen3.6-27B}
CACHE_DIR=${CACHE_DIR:-/pfs-verdent/subingtian/.cache}
HF_CACHE=${HF_CACHE:-/pfs-verdent/subingtian/.cache}
TMP_DIR=${TMP_DIR:-/pfs-verdent/subingtian/.tmp}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.org/simple}
MCORE_BRIDGE_WHEELHOUSE=${MCORE_BRIDGE_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/mcore_bridge_1_4_3}
TRANSFORMERS_WHEELHOUSE=${TRANSFORMERS_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/transformers_5_10_2}
TRANSFORMER_ENGINE_WHEELHOUSE=${TRANSFORMER_ENGINE_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/transformer_engine_2_15_0}
FLASH_ATTN_WHEELHOUSE=${FLASH_ATTN_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/flash_attn_2_8_3}
FLASH_LINEAR_ATTENTION_WHEELHOUSE=${FLASH_LINEAR_ATTENTION_WHEELHOUSE:-$WORK_ROOT/output/wheelhouse/flash_linear_attention_0_5_0}

ARM_NAME=${ARM_NAME:?ARM_NAME is required}
DATASET=${DATASET:?DATASET is required}
DATASET_SUMMARY=${DATASET_SUMMARY:?DATASET_SUMMARY is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXPECTED_DATASET_LINES=${EXPECTED_DATASET_LINES:?EXPECTED_DATASET_LINES is required}
EXPECTED_DATASET_SHA256=${EXPECTED_DATASET_SHA256:?EXPECTED_DATASET_SHA256 is required}
EXPECTED_TRAINABLE_MESSAGES=${EXPECTED_TRAINABLE_MESSAGES:?EXPECTED_TRAINABLE_MESSAGES is required}
EXPECTED_UNIQUE_TRACE_IDS=${EXPECTED_UNIQUE_TRACE_IDS:?EXPECTED_UNIQUE_TRACE_IDS is required}
EXPECTED_SINGLETON_TRACE_IDS=${EXPECTED_SINGLETON_TRACE_IDS:?EXPECTED_SINGLETON_TRACE_IDS is required}
EXPECTED_DUPLICATED_TRACE_IDS=${EXPECTED_DUPLICATED_TRACE_IDS:?EXPECTED_DUPLICATED_TRACE_IDS is required}
EXPECTED_MAX_TRACE_MULTIPLICITY=${EXPECTED_MAX_TRACE_MULTIPLICITY:?EXPECTED_MAX_TRACE_MULTIPLICITY is required}
EXPECTED_THINKING_ENABLED=${EXPECTED_THINKING_ENABLED:?EXPECTED_THINKING_ENABLED is required}
EXPECTED_THINKING_NONE=${EXPECTED_THINKING_NONE:?EXPECTED_THINKING_NONE is required}
EXPECTED_TOTAL_STEPS=${EXPECTED_TOTAL_STEPS:?EXPECTED_TOTAL_STEPS is required}

EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
SAVE_STEPS=${SAVE_STEPS:?SAVE_STEPS is required}
MAX_LENGTH=${MAX_LENGTH:-262144}

TP_SIZE=4
PP_SIZE=2
CP_SIZE=2
MICRO_BATCH_SIZE=1

BASE_SCRIPT=${BASE_SCRIPT:-/pfs-verdent/subingtian/coding_agent_models/scripts/saas/train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2.sh}
BASE_SCRIPT_SHA256=${BASE_SCRIPT_SHA256:-ad6775a6aa84524589b2bc972c4f07d9d561717ac9c7edb6ad2c37566987b41e}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$HF_CACHE" "$TMP_DIR"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== ARM_NAME=$ARM_NAME ==="
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

python3 - \
    "$DATASET" \
    "$EXPECTED_TRAINABLE_MESSAGES" \
    "$EXPECTED_UNIQUE_TRACE_IDS" \
    "$EXPECTED_SINGLETON_TRACE_IDS" \
    "$EXPECTED_DUPLICATED_TRACE_IDS" \
    "$EXPECTED_MAX_TRACE_MULTIPLICITY" \
    "$EXPECTED_THINKING_ENABLED" \
    "$EXPECTED_THINKING_NONE" <<'PY'
import collections
import json
import sys

(
    path,
    expected_trainable,
    expected_unique,
    expected_singletons,
    expected_duplicates,
    expected_max_multiplicity,
    expected_thinking_enabled,
    expected_thinking_none,
) = sys.argv[1:]
expected_trainable = int(expected_trainable)
expected_unique = int(expected_unique)
expected_singletons = int(expected_singletons)
expected_duplicates = int(expected_duplicates)
expected_max_multiplicity = int(expected_max_multiplicity)
expected_thinking_enabled = int(expected_thinking_enabled)
expected_thinking_none = int(expected_thinking_none)

trace_counts = collections.Counter()
thinking = collections.Counter()
trainable = 0
rows = 0
with open(path, "r", encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, 1):
        row = json.loads(raw)
        missing = [
            key
            for key in ("messages", "trace_id", "session_id", "model", "day")
            if key not in row
        ]
        if missing:
            raise SystemExit(f"ERROR: line {line_number} missing keys: {missing}")
        if not row["messages"]:
            raise SystemExit(f"ERROR: line {line_number} has no messages")
        trace_counts[row["trace_id"]] += 1
        thinking[str(row.get("thinking_type"))] += 1
        rows += 1
        for message in row["messages"]:
            loss = message.get("loss", 1)
            if loss not in (0, 1):
                raise SystemExit(
                    f"ERROR: line {line_number} has invalid loss={loss!r}"
                )
            if message.get("role") in {"assistant", "tool_call"} and loss == 1:
                trainable += 1

singletons = sum(count == 1 for count in trace_counts.values())
duplicates = sum(count > 1 for count in trace_counts.values())
max_multiplicity = max(trace_counts.values(), default=0)
actual = {
    "rows": rows,
    "unique_trace_ids": len(trace_counts),
    "singleton_trace_ids": singletons,
    "duplicated_trace_ids": duplicates,
    "max_trace_multiplicity": max_multiplicity,
    "trainable_messages": trainable,
    "thinking_enabled": thinking.get("enabled", 0),
    "thinking_none": thinking.get("none", 0),
    "thinking_other": sum(
        count for key, count in thinking.items() if key not in {"enabled", "none"}
    ),
}
expected = {
    "unique_trace_ids": expected_unique,
    "singleton_trace_ids": expected_singletons,
    "duplicated_trace_ids": expected_duplicates,
    "max_trace_multiplicity": expected_max_multiplicity,
    "trainable_messages": expected_trainable,
    "thinking_enabled": expected_thinking_enabled,
    "thinking_none": expected_thinking_none,
    "thinking_other": 0,
}
for key, value in expected.items():
    if actual[key] != value:
        raise SystemExit(
            f"ERROR: {key} expected {value}, got {actual[key]}"
        )
print("dataset_json_check=ok", json.dumps(actual, sort_keys=True))
PY

if [ -s "$DATASET_SUMMARY" ]; then
    echo "=== Dataset summary ==="
    cat "$DATASET_SUMMARY"
else
    echo "ERROR: dataset summary missing: $DATASET_SUMMARY" >&2
    exit 6
fi

if [ ! -s "$BASE_SCRIPT" ]; then
    echo "ERROR: proven base script missing: $BASE_SCRIPT" >&2
    exit 7
fi
ACTUAL_BASE_SHA256=$(sha256sum "$BASE_SCRIPT" | awk '{print $1}')
if [ "$ACTUAL_BASE_SHA256" != "$BASE_SCRIPT_SHA256" ]; then
    echo "ERROR: proven base script SHA256 mismatch" >&2
    exit 8
fi
for REQUIRED_BASE_ARG in \
    "--optimizer_cpu_offload true" \
    "--logging_steps 1"; do
    if ! grep -Fq -- "$REQUIRED_BASE_ARG" "$BASE_SCRIPT"; then
        echo "ERROR: proven base script missing required arg: $REQUIRED_BASE_ARG" >&2
        exit 9
    fi
done

# Reuse the exact dependency pins, distributed launch, and training arguments
# from the already-successful exact-47 run. The prefix skipped here contains
# only its no-think-specific dataset preflight, which is replaced above.
source <(sed -n '/^install_from_wheelhouse()/,$p' "$BASE_SCRIPT")

python3 - "$OUTPUT_DIR" "$EXPECTED_TOTAL_STEPS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
runs = sorted(
    (path for path in output.iterdir() if path.is_dir() and path.name.startswith("v")),
    key=lambda path: path.stat().st_mtime,
)
if not runs:
    raise SystemExit(f"ERROR: no versioned run directory under {output}")
run = runs[-1]
logging_path = run / "logging.jsonl"
if not logging_path.is_file():
    raise SystemExit(f"ERROR: missing logging.jsonl: {logging_path}")
records = [json.loads(line) for line in logging_path.read_text().splitlines()]
loss_records = [record for record in records if "loss" in record]
if len(loss_records) != expected_steps:
    raise SystemExit(
        f"ERROR: expected {expected_steps} loss steps, got {len(loss_records)}"
    )
if loss_records[-1].get("iteration") != f"{expected_steps}/{expected_steps}":
    raise SystemExit(
        f"ERROR: bad final iteration: {loss_records[-1].get('iteration')}"
    )

checkpoint = run / f"checkpoint-{expected_steps}"
latest_path = checkpoint / "latest_checkpointed_iteration.txt"
if not latest_path.is_file() or latest_path.read_text().strip() != str(expected_steps):
    raise SystemExit(f"ERROR: bad latest checkpoint marker: {latest_path}")
index_path = checkpoint / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"ERROR: missing safetensors index: {index_path}")
index = json.loads(index_path.read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("ERROR: safetensors index references no shards")
missing = [name for name in shards if not (checkpoint / name).is_file()]
empty = [name for name in shards if (checkpoint / name).stat().st_size == 0]
if missing or empty:
    raise SystemExit(f"ERROR: checkpoint shard problem missing={missing} empty={empty}")
print(
    "TRAINING_POSTFLIGHT_OK",
    json.dumps(
        {
            "run_dir": str(run),
            "checkpoint": str(checkpoint),
            "steps": len(loss_records),
            "first_loss": loss_records[0]["loss"],
            "final_loss": loss_records[-1]["loss"],
            "peak_memory_gib": max(
                record.get("memory(GiB)", 0) for record in loss_records
            ),
            "safetensors_shards": len(shards),
            "indexed_tensors": len(index["weight_map"]),
            "indexed_weight_bytes": index.get("metadata", {}).get("total_size"),
        },
        sort_keys=True,
    ),
)
PY

echo "=== Verified end time: $(date) ==="
