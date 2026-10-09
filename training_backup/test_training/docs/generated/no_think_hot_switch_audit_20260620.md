# No-Think Data Hot-Switch Audit 2026-06-20

## Scope

Audited the no-think data chain used for the planned 32-GPU jobs:

- base no-think data:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_drop_sonnet45_sonnet46_1m_20260617/day_01_29.train.no_think.drop_sonnet45_sonnet46_1m.clean.jsonl`
- user_worker data:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_user_worker_drop_sonnet45_sonnet46_1m_20260617/day_01_29.train.no_think.user_worker.drop_sonnet45_sonnet46_1m.clean.jsonl`
- user_worker no_persist data:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_user_worker_no_persist_drop_sonnet45_sonnet46_1m_20260618/day_01_29.train.no_think.user_worker.no_persist.drop_sonnet45_sonnet46_1m.clean.jsonl`
- user_worker no_persist max180k data:
  `/pfs-verdent/guoyinuo/coding_agent_models/output/may_data_versions/may_day01_29_no_think_user_worker_no_persist_drop_sonnet45_sonnet46_1m_max180k_20260619/day_01_29.train.no_think.user_worker.no_persist.drop_sonnet45_sonnet46_1m.max180k.clean.jsonl`

## Result

The current JSONL artifacts do not prove robust handling of same-thread model hot-switching.

They preserve only sample-level `session_id`, `trace_id`, and `model`. Individual `messages` do not carry `source_model`, `source_run_id`, or `run_id`, so downstream loss masking cannot reliably distinguish which assistant/tool-call messages came from which model inside a mixed-model thread.

## Evidence

| Dataset | Rows | Sessions | Sessions With Multiple Sample-Level Models | Rows In Those Sessions | Message `source_model` | Message `source_run_id/run_id` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base no-think | 25586 | 22553 | 90 | 444 | 0 | 0 |
| user_worker | 3354 | 2136 | 50 | 255 | 0 | 0 |
| user_worker no_persist | 2059 | 1505 | 23 | 102 | 0 | 0 |
| user_worker no_persist max180k | 1990 | 1491 | 23 | 92 | 0 | 0 |

Examples of multi-model `session_id` values remaining in target data include:

- `886b8c66-c0fa-4422-97cb-249aad1e72a1`: `claude-opus-4-7`, `claude-sonnet-4-6`
- `7717bd6a-9ff8-4775-8806-1fe3ce59b9d0`: `claude-opus-4-7`, `claude-sonnet-4-6`
- `74f07d6e-061b-4af1-a48b-c68e2ed36d5a`: `claude-opus-4-6-1m`, `claude-opus-4-7-1m`

The cleaning summaries record these rules only:

- keep `agent_source=user_worker`
- drop samples containing non-system `<persisted-output>`
- cap estimated tokens at 180000

No summary records thread segmentation by model, dropping mixed-model sessions, or per-message source-model masking.

## Related Prototype

`scripts/generate_langsmith_real_visualization.py` and `scripts/generate_langsmith_simple_visualization.py` describe a thread-first RichMessage approach that would preserve `source_run_id`, `source_trace_id`, and `source_model`, then mask mismatched model outputs via `source_model_mismatch`.

Those scripts are analysis/prototype artifacts. The audited training JSONL files do not contain those RichMessage fields, so the current training data chain should be treated as not fully hot-switch-safe.
