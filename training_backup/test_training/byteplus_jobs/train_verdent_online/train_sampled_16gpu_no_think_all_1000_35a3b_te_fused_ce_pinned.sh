#!/bin/bash
# 16-H100 pinned 35A3B SFT wrapper for the full no-think 1000-row sample.

set -euo pipefail

BASE_SCRIPT=/pfs-verdent/zhangyu/train_verdent_online/scripts/train_full_16gpu_full_sampled_1000_35a3b_te_fused_ce_pinned_zhangyu_online.sh
DATASET=/pfs-verdent/zhangyu/train_verdent_online/data/no_think_all_sampled_1000_20260621/no_think_all_sampled_1000.train.jsonl
OUTPUT_DIR=/pfs-verdent/zhangyu/train_verdent_online/output/train_sampled_16gpu_no_think_all_1000_35a3b_te_fused_ce_pinned

TMP_SCRIPT=/tmp/train_sampled_16gpu_no_think_all_1000_35a3b_te_fused_ce_pinned.sh
sed \
  -e "s|^DATASET=.*|DATASET=$DATASET|" \
  -e "s|^OUTPUT_DIR=.*|OUTPUT_DIR=$OUTPUT_DIR|" \
  "$BASE_SCRIPT" > "$TMP_SCRIPT"
chmod +x "$TMP_SCRIPT"
exec bash "$TMP_SCRIPT"
