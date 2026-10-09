#!/bin/bash
# 32-H100 Qwen3.6-35B-A3B no-think user_worker/no_persist/max180k SFT.
# Checkpoints/logs are written under /pfs-verdent/zhangyu/train_verdent_online.

set -euo pipefail

export DATASET=/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_no_think_user_worker_no_persist_drop_sonnet45_sonnet46_1m_max180k_20260620/day_01_29.train.no_think.user_worker.no_persist.drop_sonnet45_sonnet46_1m.max180k.clean.jsonl
export DATASET_SUMMARY=/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_no_think_user_worker_no_persist_drop_sonnet45_sonnet46_1m_max180k_20260620/manifest.json
export OUTPUT_SUBDIR=train_full_32gpu_no_think_user_worker_no_persist_35a3b_te_fused_ce_gbs72_max180k
export MIN_DATASET_LINES=1900

exec /pfs-verdent/zhangyu/train_verdent_online/scripts/run_32gpu_35a3b_no_think_sft_pinned.sh
