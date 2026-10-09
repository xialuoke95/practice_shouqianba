#!/usr/bin/env python3
"""Generate an agent-readable spec for the LangSmith row-to-sample pipeline."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_langsmith_real_visualization import (  # noqa: E402
    PARQUETS,
    SELECTED_THREADS,
    SOURCE_URIS,
    read_rows,
)
from scripts.generate_langsmith_simple_visualization import build_summary  # noqa: E402


OUT = ROOT / "langsmith_pipeline_agent_readable.json"


def build_object_model() -> dict:
    return {
        "RawRun": {
            "purpose": "One LangSmith parquet row. This is the atomic observed execution span.",
            "important_fields": [
                "id",
                "trace_id",
                "thread_id",
                "parent_run_id",
                "dotted_order",
                "run_type",
                "inputs",
                "outputs",
                "extra.metadata.ls_model_name",
                "extra.metadata.thread_id",
            ],
            "training_status": "candidate_source_only",
        },
        "RedactedMessage": {
            "purpose": "Message-level summary used in these artifacts. It deliberately omits raw content.",
            "fields": [
                "role",
                "content_len",
                "content_hash",
                "block_types",
                "markers",
                "signature",
            ],
            "redaction_guarantee": "No raw content field is emitted.",
        },
        "RichMessage": {
            "purpose": "Proposed internal object before final JSONL emit.",
            "required_fields": [
                "role",
                "content_or_redacted_content",
                "source_run_id",
                "source_trace_id",
                "source_thread_id",
                "source_model",
                "source_kind",
                "window_id",
                "loss",
                "mask_reason",
            ],
            "why": "Keep provenance until selection, dedup, and model-specific masking are complete.",
        },
        "CandidateWindow": {
            "purpose": "A possible training window derived from one or more main LLM runs.",
            "not_final_sample": True,
            "typical_sources": [
                "one main LLM run as prompt/completion",
                "a strict prefix chain of main LLM runs",
                "a terminal or longest-prefix snapshot inside one trace",
            ],
        },
        "SelectedWindow": {
            "purpose": "A candidate window selected for training after branch/compact/fork logic.",
            "selection_examples": ["terminal_per_trace", "longest_prefix_per_trace"],
            "final_sample_count_should_be": "same order as trace/window count, not same order as raw run count",
        },
        "TrainingSample": {
            "purpose": "Final emitted JSONL row consumed by ms-swift.",
            "fields": ["messages", "tools", "session_id", "trace_id", "model", "masked_count", "trainable_asst_count"],
            "loss_semantics": {
                "missing_loss": "trainable when role/template strategy allows it",
                "loss_0": "force labels=-100 for that message span",
            },
        },
    }


def build_pipeline_specs() -> dict:
    return {
        "current_trace_first": {
            "summary": "Current implementation groups raw runs by trace_id and flattens one trace into one messages sample.",
            "steps": [
                {
                    "id": "group_by_trace",
                    "input": "RawRun[]",
                    "operation": "group rows by trace_id",
                    "output": "TraceRunGroup[]",
                },
                {
                    "id": "filter_main_llm",
                    "operation": "drop sidecar LLM runs by system-prompt signatures",
                    "output": "main_llm_runs",
                },
                {
                    "id": "choose_last_snapshot",
                    "operation": "first main LLM run defines history boundary; last main LLM run provides inputs + outputs",
                    "output": "TraceFirstSample",
                },
                {
                    "id": "flatten_messages",
                    "operation": "convert Anthropic blocks into target roles assistant/tool_call/tool_response",
                    "metadata_loss": ["source_run_id", "source_model", "source_trace_id"],
                },
                {
                    "id": "mask_compact_history",
                    "operation": "set loss=0 for history/compact generated spans",
                },
                {
                    "id": "global_postprocess",
                    "operation": "group emitted JSONL by session_id/thread_id and dedup by content signatures",
                    "limitation": "dedup sees flattened messages, not original run provenance",
                },
            ],
        },
        "proposed_thread_first_rich": {
            "summary": "Use thread_id as outer collection, preserve run provenance, then select and mask before flattening.",
            "steps": [
                {
                    "id": "group_by_thread",
                    "input": "RawRun[]",
                    "operation": "group rows by metadata.thread_id/session_id",
                    "output": "ThreadRunGroup[]",
                },
                {
                    "id": "build_rich_messages",
                    "operation": "convert inputs/outputs into RichMessage while preserving source_run_id/source_model/source_trace_id",
                    "output": "CandidateWindow[]",
                },
                {
                    "id": "classify_relationships",
                    "operation": "infer prefix_extend, branch_or_partial_replay, compact_or_reset, fork_or_side_branch",
                    "inputs": ["dotted_order", "parent_run_id", "tool_call_id", "message signatures", "compact markers"],
                },
                {
                    "id": "select_final_windows",
                    "operation": "choose terminal or longest-prefix windows; do not train every candidate",
                    "output": "SelectedWindow[]",
                },
                {
                    "id": "thread_dedup",
                    "operation": "mask duplicate generated turns within a thread before emit",
                    "dedup_key": "thread_id + role + normalized content signature",
                },
                {
                    "id": "model_mask",
                    "operation": "mask generated spans where source_model != target_model",
                    "requires": "source_model retained on RichMessage",
                },
                {
                    "id": "emit",
                    "operation": "drop provenance fields only after loss/mask decisions are final",
                    "output": "TrainingSample JSONL",
                },
            ],
        },
    }


def build_rulebook() -> dict:
    return {
        "selection_rules": {
            "terminal_per_trace": {
                "definition": "For each trace or branch, choose the latest main LLM window after sidecar filtering.",
                "good_for": "Training final actions while avoiding candidate explosion.",
            },
            "longest_prefix_per_trace": {
                "definition": "For each trace or branch, choose the candidate with the largest message count.",
                "good_for": "Training the richest available context snapshot.",
            },
            "not_recommended_as_final": {
                "all_candidates": "Training every main LLM run repeats cumulative prefixes and visually/compute-wise explodes.",
            },
        },
        "mask_rules_ordered": [
            {
                "id": "non_generated_context",
                "when": "role not in {assistant, tool_call}",
                "loss": 0,
                "reason": "prompt_or_context",
            },
            {
                "id": "compact_history",
                "when": "source_kind == compact_history",
                "loss": 0,
                "reason": "compact_history",
            },
            {
                "id": "source_model_mismatch",
                "when": "target_model != all and source_model != target_model",
                "loss": 0,
                "reason": "source_model_mismatch",
            },
            {
                "id": "thread_duplicate_turn",
                "when": "same generated message signature already trained in this thread",
                "loss": 0,
                "reason": "thread_duplicate_turn",
            },
            {
                "id": "train",
                "when": "generated message passes all masks",
                "loss": None,
                "reason": "train",
            },
        ],
        "framework_notes": {
            "ms_swift_default": "Framework masks system/user/tool context by role/template, but it cannot infer compact history, duplicates, or source_model mismatch.",
            "pipeline_responsibility": "Pipeline must write loss=0 for business/provenance masks before metadata is dropped.",
        },
    }


def build_decision_notes() -> dict:
    return {
        "why_not_raw_run_only": [
            "Raw run is a valid one-step prompt/completion training unit.",
            "However, run-only training repeats cumulative prefixes and cannot express thread-level dedup without grouping.",
            "Run-level is a good internal candidate representation; final sample selection still benefits from thread context.",
        ],
        "why_not_trace_only": [
            "Trace is a useful local call-tree boundary, but it can contain branches, sidecars, retries, and compact/reset effects.",
            "Flattening by trace too early drops source_model/source_run_id needed for target-model masking.",
        ],
        "recommended_default": [
            "Use thread_id as the outer collection key.",
            "Keep RawRun/RichMessage provenance through selection, dedup, and masking.",
            "Emit final JSONL only after loss decisions are complete.",
        ],
        "open_policy_choices": [
            "terminal_per_trace vs longest_prefix_per_trace selection",
            "whether to train only run.outputs or all same-window generated assistant/tool_call spans",
            "how to classify parallel subagent branches",
            "whether compact summaries should be context-only or partly trainable for specific models",
        ],
    }


def main() -> None:
    _, selected_rows = read_rows()
    summary = build_summary(selected_rows)
    doc = {
        "artifact_type": "langsmith_pipeline_agent_readable_spec",
        "version": "2026-06-24",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "redaction_policy": {
            "raw_message_content_included": False,
            "message_fields_kept": ["role", "content_len", "content_hash", "block_types", "markers", "signature"],
            "ids_kept": ["thread_id", "trace_id", "run_id"],
            "note": "This is intended for pipeline reasoning, not direct model training.",
        },
        "source": {
            "local_parquet_files": [str(p) for p in PARQUETS],
            "source_s3_uris": SOURCE_URIS,
            "selected_threads": SELECTED_THREADS,
            "row_count": len(selected_rows),
        },
        "object_model": build_object_model(),
        "pipelines": build_pipeline_specs(),
        "rulebook": build_rulebook(),
        "decision_notes": build_decision_notes(),
        "observed_summary": summary,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({"bytes": OUT.stat().st_size, "threads": summary["totals"]["threads"], "rows": summary["totals"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
