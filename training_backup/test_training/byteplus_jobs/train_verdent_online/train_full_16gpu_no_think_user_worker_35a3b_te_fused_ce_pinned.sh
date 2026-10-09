#!/bin/bash
# 16-H100 pinned 35A3B SFT wrapper for the full no-think user_worker data.

set -euo pipefail

BASE_SCRIPT=/pfs-verdent/zhangyu/train_verdent_online/scripts/train_full_16gpu_full_sampled_1000_35a3b_te_fused_ce_pinned_zhangyu_online.sh
DATASET=/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_no_think_user_worker_drop_sonnet45_sonnet46_1m_20260620/day_01_29.train.no_think.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl
OUTPUT_DIR=/pfs-verdent/zhangyu/train_verdent_online/output/train_full_16gpu_no_think_user_worker_35a3b_te_fused_ce_pinned

TMP_SCRIPT=/tmp/train_full_16gpu_no_think_user_worker_35a3b_te_fused_ce_pinned.sh
sed \
  -e "s|^DATASET=.*|DATASET=$DATASET|" \
  -e "s|^OUTPUT_DIR=.*|OUTPUT_DIR=$OUTPUT_DIR|" \
  "$BASE_SCRIPT" > "$TMP_SCRIPT"
chmod +x "$TMP_SCRIPT"
exec bash "$TMP_SCRIPT"
