# No-Think Sampled 16-GPU Submissions 2026-06-21

## Sampling Method

All three new datasets were created with `scripts/create_jsonl_sample.py` copied to `/tmp/create_jsonl_sample.py` on `gpu_dev_huoshan`.

- Algorithm: deterministic reservoir sample over valid JSONL rows.
- Seed: `42`.
- Extra filtering: none beyond JSON decode validation.
- Destination root: `/pfs-verdent/zhangyu/train_verdent_online/data`.

## New Datasets

| Dataset | Source | Rows | Size | SHA256 |
| --- | --- | ---: | ---: | --- |
| `no_think_user_worker_sampled_1000` | `/pfs-verdent/zhangyu/train_verdent_online/data/may_day01_29_no_think_user_worker_drop_sonnet45_sonnet46_1m_20260620/day_01_29.train.no_think.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl` | 1000 / 3354 | 375M | `e11729461036adeba9f9ae2153ced56605290da406743f72ada7d82537fc93fc` |
| `no_think_all_sampled_1000` | `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_drop_sonnet45_sonnet46_1m_20260617/day_01_29.train.no_think.drop_sonnet45_sonnet46_1m.clean.jsonl` | 1000 / 25586 | 228M | `8f7bce715068abac00a6b86271f97dc0c7611c677b4a663a9cc4557281af0f57` |
| `no_think_all_sampled_2000` | `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_drop_sonnet45_sonnet46_1m_20260617/day_01_29.train.no_think.drop_sonnet45_sonnet46_1m.clean.jsonl` | 2000 / 25586 | 470M | `a0e1ee173a141ac91cf40a3d1d0ea4d5d5df7aa39bcb12f266474c1d5a6e0f1e` |

## 16-GPU Task Submissions

All three tasks reuse the pinned successful 16-H100 script by wrapper replacement of only `DATASET=` and `OUTPUT_DIR=`.

- Base script: `/pfs-verdent/zhangyu/train_verdent_online/scripts/train_full_16gpu_full_sampled_1000_35a3b_te_fused_ce_pinned_zhangyu_online.sh`
- Workers: `2 x ml.hpcpni3h.42xlarge` = 16 H100
- Queue: `q-20260128181205-z9l2h`
- Key inherited training config: `TP=2`, `PP=2`, `CP=2`, `EP=4`, `global_batch_size=24`, `max_length=262144`, `cross_entropy_loss_fusion=true`, `cross_entropy_fusion_impl=te`

| Task | Task ID | Initial Status |
| --- | --- | --- |
| `train_sampled_16gpu_no_think_user_worker_1000_35a3b_te_ce_pinned` | `t-20260621172930-j8pwh` | Queue |
| `train_sampled_16gpu_no_think_all_1000_35a3b_te_ce_pinned` | `t-20260621172935-qwk6h` | Queue |
| `train_sampled_16gpu_no_think_all_2000_35a3b_te_ce_pinned` | `t-20260621172940-rwxk9` | Queue |

## Observed Results

| Task ID | Status | Checkpoint |
| --- | --- | --- |
| `t-20260621172930-j8pwh` | Success | `/pfs-verdent/zhangyu/train_verdent_online/output/train_sampled_16gpu_no_think_user_worker_1000_35a3b_te_fused_ce_pinned/v0-20260621-101059/checkpoint-41` |
| `t-20260621172935-qwk6h` | Success | `/pfs-verdent/zhangyu/train_verdent_online/output/train_sampled_16gpu_no_think_all_1000_35a3b_te_fused_ce_pinned/v0-20260621-101059/checkpoint-41` |
| `t-20260621172940-rwxk9` | Success | `/pfs-verdent/zhangyu/train_verdent_online/output/train_sampled_16gpu_no_think_all_2000_35a3b_te_fused_ce_pinned/v0-20260621-115404/checkpoint-83` |

## Historical 1000-Row Samples

The successful zhangyu rerun `t-20260620040947-kxlcd` used:

- Dataset: `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/full-sampled-1000/full-sampled-1000.train.jsonl`
- Source: `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_drop_l4_l5_20260611/day_01_29.train.drop_l4_l5.clean.jsonl`
- Source rows: `60024`
- Sample rows: `1000`
- Seed: `42`
- Algorithm: deterministic reservoir sample over all valid JSONL rows; no no-think filtering.

Earlier `no-think-sampled` was different:

- Source: same 60024-row `drop_l4_l5` file.
- Native no-think eligible rows: `24594`.
- Sample rows: `1000`.
- Seed: `42`.
- Algorithm: filter native no-think first, then deterministic reservoir sample.
