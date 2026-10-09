from __future__ import annotations

import json
import os
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

import alignment_sft_lib


DATA_ALIGNMENT = Path(__file__).resolve().parent


def test_prepare_wandb_run_manifest_persists_dashboard_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.setenv("WANDB_APP_URL", "https://wandb.example/")

    manifest = alignment_sft_lib.prepare_wandb_run_manifest(
        entity="training-team",
        project="alignment-sft",
        run_name="kimi-smoke",
    )

    assert manifest["enabled"] is True
    assert manifest["run_id"] == os.environ["WANDB_RUN_ID"]
    assert manifest["run_name"] == "kimi-smoke"
    assert manifest["dashboard_url"] == (
        f"https://wandb.example/training-team/alignment-sft/runs/{manifest['run_id']}"
    )
    assert alignment_sft_lib.prepare_wandb_run_manifest(
        entity="training-team",
        project="alignment-sft",
        run_name="kimi-smoke",
    ) == manifest


def test_configure_wandb_pins_run_id_before_entity_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    observed: dict[str, str | None] = {}

    def detect_entity() -> str:
        observed["run_id"] = os.environ.get("WANDB_RUN_ID")
        observed["mode"] = os.environ.get("WANDB_MODE")
        observed["dir"] = os.environ.get("WANDB_DIR")
        return "training-team"

    monkeypatch.setattr(alignment_sft_lib, "detect_wandb_entity", detect_entity)

    entity, state = alignment_sft_lib.configure_wandb_env(tmp_path)

    assert entity == "training-team"
    assert observed["run_id"] == os.environ["WANDB_RUN_ID"]
    assert observed["mode"] == "online"
    assert observed["dir"] == str(tmp_path / "wandb")
    assert state["wandb_mode"] == "online"
    assert "WANDB_ENTITY" not in os.environ


def test_select_rows_honors_explicit_sample_ids() -> None:
    config = alignment_sft_lib.load_config(
        DATA_ALIGNMENT / "configs/qwen36_27b_preserve_thinking_official.yaml"
    )
    config["selection"]["sample_ids"] = [452, 321, 163]

    selected = alignment_sft_lib.select_core_rows(config, limit=3)

    assert [int(row["sample_id"]) for row in selected] == [452, 321, 163]


@pytest.mark.parametrize(
    ("config_name", "renderer_name", "adapter_module"),
    [
        (
            "kimi_k27_code_full_sft.yaml",
            "kimi_k27_code_single_datum",
            "renderer_adapters.kimi_k27_single_datum",
        ),
        (
            "glm52_fp8_full_sft.yaml",
            "glm52_preserve_history_single_datum",
            "renderer_adapters.glm52_alignment",
        ),
    ],
)
def test_model_config_loads_registered_adapter(
    config_name: str,
    renderer_name: str,
    adapter_module: str,
) -> None:
    config = alignment_sft_lib.load_config(DATA_ALIGNMENT / "configs" / config_name)

    assert config["model"]["renderer_name"] == renderer_name
    assert adapter_module in config["model"]["renderer_modules"]
    assert config["training"]["limit"] == 10
    assert config["training"]["render_workers"] == 1
    assert config["precheck"]["require_single_datum_per_row"] is True


def test_kimi_adapter_declares_extension_property() -> None:
    from renderer_adapters.kimi_k27_single_datum import (
        KimiK27CodeSingleDatumRenderer,
    )

    renderer = object.__new__(KimiK27CodeSingleDatumRenderer)
    assert renderer.has_extension_property is True


def test_glm_adapter_preserves_history_and_declares_tools() -> None:
    from renderer_adapters.glm52_alignment import (
        GLM52PreserveHistorySingleDatumRenderer,
    )

    renderer = object.__new__(GLM52PreserveHistorySingleDatumRenderer)
    assert renderer.has_extension_property is True
    assert (
        renderer._assistant_think_block(
            before_last_user=True,
            reasoning="historical reasoning",
        )
        == "<think>historical reasoning</think>"
    )

    messages = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        system_prompt="You are helpful.",
    )
    assert [message["role"] for message in messages] == ["system", "system"]
    assert "# Tools" in messages[0]["content"]
    assert json.dumps("get_weather") in messages[0]["content"]
    assert messages[1]["content"] == "You are helpful."


def test_glm_config_excludes_images_and_disables_qwen_tool_merge() -> None:
    config = alignment_sft_lib.load_config(
        DATA_ALIGNMENT / "configs/glm52_fp8_full_sft.yaml"
    )

    assert config["data"]["include_images"] is False
    assert config["selection"]["exclude_image_rows"] is True
    assert "--no-merge-consecutive-tool-responses" in config["data"]["converter_args"]


def test_project_renderer_rejects_spawn_workers() -> None:
    config = alignment_sft_lib.load_config(
        DATA_ALIGNMENT / "configs/kimi_k27_code_full_sft.yaml"
    )

    with pytest.raises(ValueError, match="render_workers <= 1"):
        alignment_sft_lib.validate_renderer_worker_mode(config, render_workers=2)

    alignment_sft_lib.validate_renderer_worker_mode(config, render_workers=1)


def test_model_configs_require_read_only_shape_resolution() -> None:
    for name in ["kimi_k27_code_full_sft.yaml", "glm52_fp8_full_sft.yaml"]:
        config = alignment_sft_lib.load_config(DATA_ALIGNMENT / "configs" / name)
        assert config["precheck"]["resolve_training_shape"] is True
        assert config["precheck"]["resolve_base_model"] is True


def test_kimi_overfit_probe_is_exactly_twenty_fixed_full_batch_steps() -> None:
    config = alignment_sft_lib.load_config(
        DATA_ALIGNMENT / "configs/kimi_k27_code_overfit_3x20.yaml"
    )

    assert config["selection"]["sample_ids"] == [236, 238, 602]
    assert config["data"]["include_images"] is True
    assert config["training"]["limit"] == 3
    assert config["training"]["batch_size"] == 3
    assert config["training"]["epochs"] == 20
    steps_per_epoch = (
        config["training"]["limit"] + config["training"]["batch_size"] - 1
    ) // config["training"]["batch_size"]
    assert steps_per_epoch * config["training"]["epochs"] == 20
    assert config["training"]["dcp_save_interval"] > config["training"]["epochs"]


def test_converter_links_tool_response_name_from_pending_call() -> None:
    converter_path = DATA_ALIGNMENT / "core/convert_core_aligned.fixed_0702.py"
    spec = spec_from_file_location("core_converter_for_test", converter_path)
    assert spec and spec.loader
    converter = module_from_spec(spec)
    spec.loader.exec_module(converter)
    converted = []
    stats = Counter()

    converter.append_tool_response(
        converted,
        role="tool",
        content="sunny",
        row_index=0,
        message_index=2,
        pending_tool_call_ids=[("call_1", "get_weather")],
        args=SimpleNamespace(
            tool_response_linking=True,
            tool_response_name_linking=True,
        ),
        stats=stats,
    )

    assert converted == [
        {
            "role": "tool",
            "content": "sunny",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "trainable": False,
        }
    ]


def test_tool_linkage_audit_understands_merged_qwen_tool_responses(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "merged.jsonl"
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "first\n</tool_response>\n<tool_response>\nsecond",
            },
        ]
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = alignment_sft_lib.audit_tool_linkage(dataset, tmp_path)

    assert result["unresolved_tool_call_ids"] == 0
    assert result["merged_tool_results_accounted"] == 1
