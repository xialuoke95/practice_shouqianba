#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from alignment_sft_lib import load_config, write_json


DATA_ALIGNMENT = Path(__file__).resolve().parent
DEFAULT_KIMI_RUN = Path(
    "/home/ubuntu/.remotelab/workspace/fireworks-runs/kimi-k27-adapter-validation-10-ready"
)
DEFAULT_GLM_RUN = Path(
    "/home/ubuntu/.remotelab/workspace/fireworks-runs/glm52-adapter-validation-10-ready"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(
    run_dir: Path,
    *,
    expected_images: int,
    expected_context: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    conversion = read_json(run_dir / "conversion_report.json")
    renderer = read_json(run_dir / "local_renderer_precheck_summary.json")
    dataloader = read_json(run_dir / "official_dataloader_precheck_summary.json")
    shape = read_json(run_dir / "training_shape_precheck_summary.json")
    model = read_json(run_dir / "base_model_precheck_summary.json")
    linkage = read_json(run_dir / "tool_linkage_audit.json")
    dry_run = read_json(run_dir / "dry_run_result.json")

    checks = {
        "rows_written": conversion.get("written_rows") == 10,
        "renderer_rows_ok": renderer.get("status_counts") == {"ok": 10},
        "single_datum_per_row": renderer.get("split_counts") == {"1": 10},
        "official_dataloader_rows": dataloader.get("dataset_rows") == 10,
        "official_dataloader_datums": dataloader.get("datum_count") == 10,
        "official_dataloader_in_process": dataloader.get("render_mode") == "in_process",
        "shape_resolved": shape.get("status") == "ok",
        "shape_context_exact": shape.get("max_supported_context_length") == expected_context,
        "base_model_ready": model.get("model_state") == "READY",
        "no_unresolved_tool_calls": linkage.get("unresolved_tool_call_ids") == 0,
        "no_orphan_tool_results": linkage.get("orphan_tool_results") == 0,
        "images_as_expected": conversion.get("image_rows_converted", 0) == expected_images,
        "no_image_mapping_errors": conversion.get("missing_images_for_placeholders", 0) == 0
        and conversion.get("unused_images", 0) == 0,
        "no_mixed_loss_groups": conversion.get("mixed_tool_call_loss_groups", 0) == 0,
        "no_remote_submission": dry_run.get("would_submit") is False,
        "full_parameter_sft": dry_run.get("lora_rank") == 0,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"{run_dir.name}: {name}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir),
        "dataset": dry_run.get("dataset"),
        "dataset_sha256": dry_run.get("dataset_sha256"),
        "renderer": renderer.get("renderer_name"),
        "training_shape_version": shape.get("training_shape_version"),
        "base_model": model.get("resolved_name"),
        "checks": checks,
        "terminal_unanswered_tool_call_ids": linkage.get(
            "terminal_unanswered_tool_call_ids", 0
        ),
    }, errors


def kimi_renderer_parity() -> dict[str, Any]:
    from renderer_adapters.kimi_k27_single_datum import (
        KimiK27CodeSingleDatumRenderer,
    )
    from training.renderer.kimi_k27_code import KimiK27CodeRenderer
    from training.utils.supervised import normalize_messages

    tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2.7-Code", trust_remote_code=True
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    messages = list(
        normalize_messages(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "<think>Need weather.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "tool_call_id": "call_1",
                    "content": "Sunny",
                },
                {"role": "user", "content": "Summarize."},
            ]
        )
    )
    official = KimiK27CodeRenderer(tokenizer)
    adapter = KimiK27CodeSingleDatumRenderer(tokenizer)
    official_messages = list(official.create_conversation_prefix_with_tools(tools)) + messages
    adapter_messages = list(adapter.create_conversation_prefix_with_tools(tools)) + messages
    official_tokens = list(
        official.build_generation_prompt(official_messages, role="assistant").to_ints()
    )
    adapter_tokens = list(
        adapter.build_generation_prompt(adapter_messages, role="assistant").to_ints()
    )
    return {
        "status": "PASS" if official_tokens == adapter_tokens else "FAIL",
        "token_exact_with_official_renderer": official_tokens == adapter_tokens,
        "token_count": len(adapter_tokens),
        "official_extension_property": bool(official.has_extension_property),
        "adapter_extension_property": bool(adapter.has_extension_property),
    }


def glm_hf_parity() -> dict[str, Any]:
    from renderer_adapters.glm52_alignment import (
        GLM52PreserveHistorySingleDatumRenderer,
    )
    from training.utils.supervised import normalize_messages

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.2", trust_remote_code=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "<think>Need weather.</think>",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_1",
            "content": "Sunny",
        },
        {"role": "user", "content": "Summarize."},
    ]
    renderer = GLM52PreserveHistorySingleDatumRenderer(tokenizer)
    normalized = list(normalize_messages(messages))
    system_prompt = normalized.pop(0)["content"]
    prefix = renderer.create_conversation_prefix_with_tools(
        [tools[0]["function"]], system_prompt=system_prompt
    )
    adapter_tokens = list(
        renderer.build_generation_prompt(
            list(prefix) + normalized, role="assistant"
        ).to_ints()
    )
    hf_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        clear_thinking=False,
    )
    hf_tokens = tokenizer.encode(hf_text, add_special_tokens=False)
    return {
        "status": "PASS" if adapter_tokens == hf_tokens else "FAIL",
        "token_exact_with_hf_clear_thinking_false": adapter_tokens == hf_tokens,
        "token_count": len(adapter_tokens),
        "adapter_extension_property": bool(renderer.has_extension_property),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Kimi 2.7 and GLM 5.2 adapters without submitting training."
    )
    parser.add_argument("--kimi-run", default=str(DEFAULT_KIMI_RUN))
    parser.add_argument("--glm-run", default=str(DEFAULT_GLM_RUN))
    parser.add_argument(
        "--output", default=str(DATA_ALIGNMENT / "model_adapter_readiness_report.json")
    )
    args = parser.parse_args()

    load_config(DATA_ALIGNMENT / "configs/kimi_k27_code_full_sft.yaml")
    load_config(DATA_ALIGNMENT / "configs/glm52_fp8_full_sft.yaml")
    kimi, kimi_errors = validate_run(
        Path(args.kimi_run), expected_images=3, expected_context=262144
    )
    glm, glm_errors = validate_run(
        Path(args.glm_run), expected_images=0, expected_context=204736
    )
    parity = {
        "kimi_official_renderer": kimi_renderer_parity(),
        "glm_hf_template": glm_hf_parity(),
    }
    errors = kimi_errors + glm_errors
    errors.extend(
        f"{name}: parity failed"
        for name, result in parity.items()
        if result["status"] != "PASS"
    )
    optional_warnings = []
    if not os.environ.get("HF_TOKEN"):
        optional_warnings.append("HF_TOKEN is unset; cached public tokenizers work but Hub rate limits are lower")
    if importlib.util.find_spec("sentencepiece") is None:
        optional_warnings.append("sentencepiece is absent; Kimi tokenizer validated via its TikToken fallback")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "ready_to_submit": not errors,
        "remote_training_submitted": False,
        "errors": errors,
        "models": {"kimi2.7": kimi, "glm5.2": glm},
        "renderer_parity": parity,
        "required_before_submission": [],
        "optional_environment_warnings": optional_warnings,
        "submission_boundary": (
            "Run run_alignment_sft.py without --dry-run; this is the only remaining "
            "step that creates billable trainer resources."
        ),
    }
    write_json(Path(args.output), report)
    print(json.dumps({"status": report["status"], "output": args.output}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
