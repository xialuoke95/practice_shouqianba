#!/bin/bash
# Validate the full non-sampled no-think dataset path from inside BytePlus MLP.

set -euo pipefail

DATASET=/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_drop_sonnet45_sonnet46_1m_20260617/day_01_29.train.no_think.drop_sonnet45_sonnet46_1m.clean.jsonl
DATASET_DIR=$(dirname "$DATASET")
SUMMARY=$DATASET_DIR/summary.json
OUTPUT_DIR=/pfs-verdent/guoyinuo/coding_agent_models/output/preflight_no_think_full_dataset_20260618

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/preflight_$(date +%Y%m%d_%H%M%S)_node${MLP_ROLE_INDEX:-0}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== No-think full dataset preflight ==="
echo "=== Start time: $(date) ==="
echo "LOG_FILE=$LOG_FILE"
echo "DATASET=$DATASET"
echo "DATASET_DIR=$DATASET_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: dataset directory does not exist: $DATASET_DIR" >&2
    exit 2
fi

echo "=== Dataset directory listing ==="
find "$DATASET_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort

if [ ! -s "$DATASET" ]; then
    echo "ERROR: dataset file missing or empty: $DATASET" >&2
    exit 3
fi

LINE_COUNT=$(wc -l < "$DATASET")
echo "LINE_COUNT=$LINE_COUNT"
if [ "$LINE_COUNT" -le 1000 ]; then
    echo "ERROR: expected full non-sampled dataset to contain more than 1000 rows" >&2
    exit 4
fi

if [ -s "$SUMMARY" ]; then
    echo "=== summary.json ==="
    cat "$SUMMARY"
else
    echo "WARN: summary.json not found at $SUMMARY"
fi

python3 - "$DATASET" <<'PY'
import json
import sys

path = sys.argv[1]
checked = 0
for line_no, line in enumerate(open(path, encoding="utf-8"), 1):
    if not line.strip():
        continue
    try:
        json.loads(line)
    except Exception as exc:
        raise SystemExit(f"ERROR: invalid JSON at line {line_no}: {exc}") from exc
    checked += 1
    if checked >= 20:
        break
print(f"JSON_HEAD_CHECKED={checked}")
if checked == 0:
    raise SystemExit("ERROR: no non-empty JSONL rows found")
PY

echo "=== End time: $(date) ==="
