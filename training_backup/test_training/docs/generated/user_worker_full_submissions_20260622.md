# User Worker Full Submissions 2026-06-22

## Generated Data

Created a non-no-think `agent_source=user_worker` dataset under zhangyu PFS.

- Output:
  `/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_user_worker_drop_sonnet45_sonnet46_1m_20260622/day_01_29.train.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl`
- Source clean JSONL:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_drop_l4_l5_no_trunc_marker_drop_think_gt64k_drop_bad_toolcalls_20260616/day_01_29.train.drop_l4_l5.no_trunc_marker.drop_think_gt64k.drop_bad_toolcalls.clean.jsonl`
- Metadata source:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_drop_l4_l5_no_trunc_marker_drop_think_gt64k_20260616/day_01_29.train.drop_l4_l5.no_trunc_marker.drop_think_gt64k.with_metadata.jsonl`
- Rules:
  keep `agent_source=user_worker`; drop `claude-sonnet-4-5@20250929` and `claude-sonnet-4-6-1m`; do not apply no-think filtering; strip metadata fields.
- Rows: `14824`
- Size: `3674969788` bytes
- SHA256: `de58a09dbd3ae107453cd79b8b0109d25afb54c8077a33a929740b41b268f685`

## Submitted Tasks

| Task | Task ID | Cards | Initial observed status | Config |
| --- | --- | ---: | --- | --- |
| `train_full_sft_35a3b_user_worker_32gpu_te_ce_gbs72_max180k` | `t-20260622180732-m9hxx` | 32 | Running | `docs/generated/byteplus_user_worker_32gpu_te_fused_ce_gbs72_max180k_task.yaml` |
| `train_full_sft_35a3b_no_think_user_worker_16gpu_te_ce_pinned` | `t-20260622180745-h2mcv` | 16 | Queue | `docs/generated/byteplus_no_think_user_worker_full_16gpu_te_fused_ce_pinned_task.yaml` |

## Notes

- The 32-card task preflight logged `DATASET_LINES=14824` and read the generated manifest.
- The 16-card task uses the existing full no-think user_worker dataset:
  `/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_no_think_user_worker_drop_sonnet45_sonnet46_1m_20260620/day_01_29.train.no_think.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl`
- The no-think user_worker full dataset has `3354` rows and SHA256
  `2ea98f08f03ae7e7fdc778e86f8e6b6a274d68a052a5c92c592854f26f2892c7`.
