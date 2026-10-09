#!/bin/bash
# 32-H100 Qwen3.6-35B-A3B user_worker SFT without no-think filtering.
# Checkpoints/logs are written under /pfs-verdent/zhangyu/train_verdent_online.

set -euo pipefail

export DATASET=/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_user_worker_drop_sonnet45_sonnet46_1m_20260622/day_01_29.train.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl
export DATASET_SUMMARY=/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_user_worker_drop_sonnet45_sonnet46_1m_20260622/manifest.json
export OUTPUT_SUBDIR=train_full_32gpu_user_worker_35a3b_te_fused_ce_gbs72_max180k
export MIN_DATASET_LINES=14000

exec /pfs-verdent/zhangyu/train_verdent_online/scripts/run_32gpu_35a3b_no_think_sft_pinned.sh
