#!/usr/bin/env python3
from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml


DATA_ALIGNMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DATA_ALIGNMENT_DIR / "configs/qwen36_27b_preserve_thinking_official.yaml"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    for cookbook_root in cfg.get("paths", {}).get("cookbook_roots", []):
        root = Path(cookbook_root)
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    if str(DATA_ALIGNMENT_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_ALIGNMENT_DIR))
    for module_name in cfg.get("model", {}).get("renderer_modules", []):
        importlib.import_module(str(module_name))
    return cfg


def rel_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_bashrc_var(name: str) -> str | None:
    path = Path.home() / ".bashrc"
    if not path.exists():
        return None
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"=(.*)\s*$")
    for raw in path.read_text(errors="replace").splitlines():
        match = pattern.match(raw)
        if not match:
            continue
        value = match.group(1).strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def configure_fireworks_env(base_url: str) -> dict[str, Any]:
    bashrc_key = read_bashrc_var("FIREWORKS_API_KEY")
    if bashrc_key:
        os.environ["FIREWORKS_API_KEY"] = bashrc_key
    return {
        "fireworks_api_key_present": bool(os.environ.get("FIREWORKS_API_KEY")),
        "fireworks_base_url": base_url,
    }


def detect_wandb_entity() -> str | None:
    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        import wandb

        viewer = wandb.Api().viewer
        return getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    except Exception:
        return None


def ensure_wandb_run_id() -> str:
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        alphabet = string.ascii_lowercase + string.digits
        run_id = "".join(secrets.choice(alphabet) for _ in range(8))
        os.environ["WANDB_RUN_ID"] = run_id
    os.environ.setdefault("WANDB_RESUME", "allow")
    return run_id


def configure_wandb_env(run_dir: Path) -> tuple[str | None, dict[str, Any]]:
    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("W_and_B_API_KEY") or read_bashrc_var("W_and_B_API_KEY")
    if api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "online" if api_key else "offline"
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    ensure_wandb_run_id()
    entity = detect_wandb_entity()
    return entity, {
        "wandb_api_key_present": bool(api_key),
        "wandb_entity_present": bool(entity),
        "wandb_mode": os.environ.get("WANDB_MODE"),
    }


def prepare_wandb_run_manifest(
    *,
    entity: str | None,
    project: str,
    run_name: str,
) -> dict[str, Any]:
    """Pin the W&B run identity before Cookbook initializes and finishes it."""
    if not entity:
        return {
            "enabled": False,
            "run_id": None,
            "run_name": run_name,
            "project": project,
            "entity": None,
            "dashboard_url": None,
        }

    run_id = ensure_wandb_run_id()

    app_url = os.environ.get("WANDB_APP_URL", "https://wandb.ai").rstrip("/")
    dashboard_url = "/".join(
        [
            app_url,
            quote(entity, safe=""),
            quote(project, safe=""),
            "runs",
            quote(run_id, safe=""),
        ]
    )
    return {
        "enabled": True,
        "run_id": run_id,
        "run_name": run_name,
        "project": project,
        "entity": entity,
        "dashboard_url": dashboard_url,
    }


def _image_sample_ids(config: dict[str, Any]) -> set[int]:
    source_run = Path(config["paths"]["source_run"])
    sample_path = source_run / config["data"]["source_sample_jsonl"]
    return {
        sample_id
        for sample_id, row in enumerate(iter_jsonl(sample_path))
        if bool(row.get("images"))
    }


def select_core_rows(config: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data_alignment = Path(config["paths"]["data_alignment"])
    audit_path = rel_path(data_alignment, config["data"]["source_audit_jsonl"])
    selection_cfg = config.get("selection", {})
    explicit_sample_ids = [int(value) for value in selection_cfg.get("sample_ids", [])]
    excluded_image_ids = _image_sample_ids(config) if selection_cfg.get("exclude_image_rows", False) else set()
    candidates = []
    for rec in iter_jsonl(audit_path):
        if rec.get("status") != "ok":
            continue
        if int(rec.get("active_target_token_count") or 0) <= 0:
            continue
        if int(rec["sample_id"]) in excluded_image_ids:
            continue
        candidates.append(rec)
    if explicit_sample_ids:
        by_id = {int(rec["sample_id"]): rec for rec in candidates}
        missing = [sample_id for sample_id in explicit_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"configured sample_ids are unavailable after selection filters: {missing}")
        selected = [by_id[sample_id] for sample_id in explicit_sample_ids]
        if len(selected) != limit:
            raise ValueError(
                f"configured sample_ids count ({len(selected)}) must equal --limit ({limit})"
            )
        return selected
    candidates.sort(
        key=lambda r: (
            int(r.get("ms_swift_token_count") or r.get("default_fireworks_token_count") or 10**18),
            int(r.get("sample_id") or 0),
        )
    )
    selected = candidates[:limit]
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} eligible rows are available; requested {limit}")
    return selected


def select_shortest_core_rows(config: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Backward-compatible alias for older wrappers."""
    return select_core_rows(config, limit)


def materialize_selected_raw(config: dict[str, Any], selected: list[dict[str, Any]], run_dir: Path) -> tuple[Path, Path]:
    data_alignment = Path(config["paths"]["data_alignment"])
    source_run = Path(config["paths"]["source_run"])
    wanted = {int(r["sample_id"]) for r in selected}
    raw_out = run_dir / f"selected_raw_{len(selected)}.jsonl"
    manifest_out = run_dir / f"selected_manifest_{len(selected)}.jsonl"
    with raw_out.open("w", encoding="utf-8") as raw_dst:
        for sample_id, row in enumerate(iter_jsonl(source_run / config["data"]["source_sample_jsonl"])):
            if sample_id in wanted:
                raw_dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = rel_path(data_alignment, config["data"]["source_manifest_jsonl"])
    with manifest_out.open("w", encoding="utf-8") as manifest_dst:
        for row in iter_jsonl(manifest_path):
            if int(row["sample_id"]) in wanted:
                manifest_dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return raw_out, manifest_out


def converter_path(config: dict[str, Any]) -> Path:
    data_alignment = Path(config["paths"]["data_alignment"])
    primary = rel_path(data_alignment, config["data"]["converter"])
    fallback = rel_path(data_alignment, config["data"]["converter_fallback"])
    return primary if primary.exists() else fallback


def convert_selected(config: dict[str, Any], raw_path: Path, run_dir: Path) -> tuple[Path, Path]:
    converted = run_dir / config["data"]["converted_filename"]
    report = run_dir / config["data"]["conversion_report_filename"]
    converter = converter_path(config)
    cmd = [
        sys.executable,
        str(converter),
        "--input",
        str(raw_path),
        "--output",
        str(converted),
        "--report",
        str(report),
        "--mixed-tool-call-loss-policy",
        str(config["data"].get("mixed_tool_call_loss_policy", "report")),
    ]
    if config["data"].get("include_images", False):
        cmd.append("--include-images")
    else:
        cmd.append("--no-include-images")
    if config["data"].get("skip_image_rows", False):
        cmd.append("--skip-image-rows")
    cmd.extend(str(value) for value in config["data"].get("converter_args", []))
    subprocess.run(cmd, check=True)
    write_json(
        run_dir / "converter_used.json",
        {
            "converter": str(converter),
            "converter_sha256": sha256_file(converter),
            "fallback_used": converter != rel_path(Path(config["paths"]["data_alignment"]), config["data"]["converter"]),
        },
    )
    return converted, report


def validate_renderer_worker_mode(config: dict[str, Any], render_workers: int) -> None:
    renderer_modules = config.get("model", {}).get("renderer_modules", [])
    if renderer_modules and render_workers > 1:
        raise ValueError(
            "project-local renderer_modules require render_workers <= 1 because "
            "cookbook DataLoader spawn workers do not import modules from the YAML config"
        )


def audit_tool_linkage(dataset: Path, run_dir: Path) -> dict[str, Any]:
    terminal_calls: list[dict[str, Any]] = []
    unresolved_calls: list[dict[str, Any]] = []
    orphan_results: list[dict[str, Any]] = []
    merged_tool_results_accounted = 0
    for row_index, row in enumerate(iter_jsonl(dataset)):
        pending: dict[str, int] = {}
        messages = row.get("messages") or []
        for message_index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    call_id = tool_call.get("id")
                    if call_id:
                        pending[str(call_id)] = message_index
            elif message.get("role") == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if call_id in pending:
                    pending.pop(call_id)
                else:
                    orphan_results.append(
                        {"row_index": row_index, "message_index": message_index, "tool_call_id": call_id}
                    )
                content = message.get("content")
                merged_results = (
                    content.count("\n</tool_response>\n<tool_response>\n")
                    if isinstance(content, str)
                    else 0
                )
                for _ in range(merged_results):
                    if not pending:
                        break
                    pending.pop(next(iter(pending)))
                    merged_tool_results_accounted += 1
        terminal_ids = {
            str(tool_call.get("id"))
            for tool_call in ((messages[-1].get("tool_calls") or []) if messages else [])
            if tool_call.get("id")
        }
        for call_id, message_index in pending.items():
            record = {
                "row_index": row_index,
                "message_index": message_index,
                "tool_call_id": call_id,
            }
            if call_id in terminal_ids:
                terminal_calls.append(record)
            else:
                unresolved_calls.append(record)
    summary = {
        "terminal_unanswered_tool_call_ids": len(terminal_calls),
        "terminal_calls": terminal_calls,
        "unresolved_tool_call_ids": len(unresolved_calls),
        "unresolved_calls": unresolved_calls,
        "orphan_tool_results": len(orphan_results),
        "orphan_results": orphan_results,
        "merged_tool_results_accounted": merged_tool_results_accounted,
    }
    write_json(run_dir / "tool_linkage_audit.json", summary)
    if unresolved_calls or orphan_results:
        raise RuntimeError(
            "tool linkage audit failed: "
            f"unresolved={len(unresolved_calls)}, orphan_results={len(orphan_results)}"
        )
    return summary


def local_renderer_precheck(config: dict[str, Any], dataset: Path, run_dir: Path, limit: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from training.utils.supervised import build_renderer, render_messages_to_datums
    from tinker_cookbook.renderers.base import Renderer

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_model"], trust_remote_code=True)
    renderer = build_renderer(tokenizer, model_cfg["tokenizer_model"], model_cfg["renderer_name"])
    precheck_cfg = config.get("precheck", {})
    prefix_impl = getattr(type(renderer), "create_conversation_prefix_with_tools", None)
    renderer_declares_tools = prefix_impl is not Renderer.create_conversation_prefix_with_tools
    if precheck_cfg.get("require_extension_property", False) and not renderer.has_extension_property:
        raise RuntimeError(
            f"renderer {type(renderer).__name__} does not satisfy the extension property"
        )
    if precheck_cfg.get("require_tools_prefix", False) and not renderer_declares_tools:
        raise RuntimeError(f"renderer {type(renderer).__name__} does not declare top-level tools")
    rows = []
    status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    total_input_ids = 0
    total_active_targets = 0
    errors = []
    with (run_dir / "local_render_audit.jsonl").open("w", encoding="utf-8") as audit:
        for idx, row in enumerate(iter_jsonl(dataset)):
            if idx >= limit:
                break
            rec: dict[str, Any] = {"sample_id": idx, "status": "ok"}
            try:
                datums = render_messages_to_datums(
                    row.get("messages") or [],
                    renderer=renderer,
                    train_on_what=model_cfg["train_on_what"],
                    max_seq_len=int(model_cfg["max_seq_len"]),
                    include_loss_mask=False,
                    tools=row.get("tools"),
                )
                rec["split_count"] = len(datums)
                rec["input_len"] = sum(len(d.token_ids) for d in datums)
                rec["active_target_token_count"] = sum(sum(1 for w in d.token_weights if w > 0) for d in datums)
                rec["has_tools"] = bool(row.get("tools"))
                rec["has_images"] = any(
                    isinstance(m.get("content"), list)
                    for m in row.get("messages", [])
                    if isinstance(m, dict)
                )
                if precheck_cfg.get("forbid_images", False) and rec["has_images"]:
                    raise ValueError("image row selected for a text-only model")
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(rec)
            status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1
            if rec["status"] == "ok":
                split_counts[str(rec["split_count"])] = split_counts.get(str(rec["split_count"]), 0) + 1
                total_input_ids += int(rec["input_len"])
                total_active_targets += int(rec["active_target_token_count"])
            rows.append(rec)
            audit.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "renderer_name": model_cfg["renderer_name"],
        "renderer_class": type(renderer).__name__,
        "has_extension_property": bool(renderer.has_extension_property),
        "renderer_declares_tools": renderer_declares_tools,
        "count": len(rows),
        "status_counts": status_counts,
        "split_counts": split_counts,
        "total_input_ids": total_input_ids,
        "total_active_targets": total_active_targets,
        "errors": errors[:20],
    }
    write_json(run_dir / "local_renderer_precheck_summary.json", summary)
    if errors:
        raise RuntimeError(f"local renderer precheck has {len(errors)} errors; see local_renderer_precheck_summary.json")
    return summary


def official_dataloader_precheck(
    config: dict[str, Any],
    dataset: Path,
    run_dir: Path,
    *,
    limit: int,
    batch_size: int,
    render_workers: int,
) -> dict[str, Any]:
    import torch
    from training.recipes import sft_loop
    from training.utils.streaming import make_render_dataloader

    validate_renderer_worker_mode(config, render_workers)
    model_cfg = config["model"]
    sft_loop._init_render_worker(
        model_cfg["tokenizer_model"],
        model_cfg["renderer_name"],
        model_cfg["train_on_what"],
        int(model_cfg["max_seq_len"]),
    )
    dataset_obj = sft_loop.JsonlRenderDataset(
        str(dataset),
        sft_loop._render_one_worker,
        max_examples=limit,
    )
    effective_batch_size = max(1, min(batch_size, len(dataset_obj)))
    loader = make_render_dataloader(
        dataset_obj,
        batch_size=effective_batch_size,
        num_workers=render_workers,
        shuffle=False,
        generator=torch.Generator(),
        worker_init_fn=functools.partial(
            sft_loop._init_render_worker,
            model_cfg["tokenizer_model"],
            model_cfg["renderer_name"],
            model_cfg["train_on_what"],
            int(model_cfg["max_seq_len"]),
        ),
    )
    batch_count = 0
    datum_count = 0
    nested_datum_lists = 0
    for batch in loader:
        batch_count += 1
        for item in batch:
            if isinstance(item, list):
                nested_datum_lists += 1
                datum_count += len(item)
            else:
                datum_count += 1
    summary = {
        "status": "ok",
        "dataset_rows": len(dataset_obj),
        "datum_count": datum_count,
        "batch_count": batch_count,
        "effective_batch_size": effective_batch_size,
        "render_workers": render_workers,
        "render_mode": "in_process" if render_workers <= 1 else "spawn",
        "nested_datum_lists": nested_datum_lists,
    }
    write_json(run_dir / "official_dataloader_precheck_summary.json", summary)
    if datum_count != len(dataset_obj) or nested_datum_lists:
        raise RuntimeError(
            "official SFT DataLoader did not preserve one datum per row: "
            f"rows={len(dataset_obj)}, datums={datum_count}, nested={nested_datum_lists}"
        )
    return summary


def training_shape_precheck(
    config: dict[str, Any],
    run_dir: Path,
    *,
    base_url: str,
) -> dict[str, Any]:
    model_cfg = config["model"]
    shape_id = model_cfg["training_shape_id"]
    profile = trainer_mgr(base_url).resolve_training_profile(shape_id)
    summary = {
        "status": "ok",
        "training_shape_id": shape_id,
        "training_shape_version": profile.training_shape_version,
        "max_supported_context_length": profile.max_supported_context_length,
        "node_count": profile.node_count,
        "accelerator_type": profile.accelerator_type,
        "accelerator_count": profile.accelerator_count,
        "base_model_weight_precision": profile.base_model_weight_precision,
        "pipeline_parallelism": profile.pipeline_parallelism,
        "trainer_mode": profile.trainer_mode,
    }
    write_json(run_dir / "training_shape_precheck_summary.json", summary)
    requested_context = int(model_cfg["max_seq_len"])
    if requested_context > int(profile.max_supported_context_length):
        raise RuntimeError(
            f"configured max_seq_len={requested_context} exceeds shape capacity "
            f"{profile.max_supported_context_length}"
        )
    return summary


def base_model_precheck(
    config: dict[str, Any],
    run_dir: Path,
    *,
    base_url: str,
) -> dict[str, Any]:
    base_model = config["model"]["base_model"]
    response = trainer_mgr(base_url)._get(f"/v1/{base_model}")
    if not response.is_success:
        raise RuntimeError(
            f"base model precheck failed for {base_model}: HTTP {response.status_code}"
        )
    payload = response.json()
    summary = {
        "status": "ok",
        "base_model": base_model,
        "resolved_name": payload.get("name"),
        "model_state": payload.get("state"),
        "model_kind": payload.get("kind"),
    }
    write_json(run_dir / "base_model_precheck_summary.json", summary)
    if summary["resolved_name"] != base_model or summary["model_state"] != "READY":
        raise RuntimeError(f"base model is not ready: {summary}")
    return summary


def build_run_dir(config: dict[str, Any], run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg).resolve()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(config["paths"]["default_run_root"]) / f"{config['naming']['run_dir_prefix']}-{ts}"


def wandb_run_info() -> dict[str, Any]:
    try:
        import wandb

        run = wandb.run
        if run is None:
            return {"active": False}
        return {
            "active": True,
            "url": getattr(run, "url", None),
            "id": getattr(run, "id", None),
            "name": getattr(run, "name", None),
            "project": getattr(run, "project", None),
            "entity": getattr(run, "entity", None),
        }
    except Exception as exc:
        return {"active": False, "error": f"{type(exc).__name__}: {exc}"}


def build_sft_config(
    config: dict[str, Any],
    *,
    run_dir: Path,
    dataset: Path,
    job_id: str,
    output_model_id: str,
    args: Any,
    wandb_entity: str | None,
    wandb_project: str,
    wandb_run_name: str,
):
    from fireworks.training.sdk.training_spec import CosineSchedule
    from training.recipes.sft_loop import Config
    from training.utils import RunnerConfig, TrainerConfig, WandBConfig

    training_cfg = config["training"]
    model_cfg = config["model"]
    lr_schedule = CosineSchedule(warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio)
    return Config(
        log_path=str(run_dir / "sft_logs"),
        render_samples_file=str(run_dir / "render_samples.jsonl"),
        render_samples_limit=int(training_cfg.get("render_samples_limit", 10)),
        base_model=model_cfg["base_model"],
        dataset=str(dataset),
        tokenizer_model=model_cfg["tokenizer_model"],
        renderer_name=model_cfg["renderer_name"],
        train_on_what=model_cfg["train_on_what"],
        learning_rate=args.learning_rate,
        lr_scheduler=lr_schedule,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.limit,
        lora_rank=int(training_cfg.get("lora_rank", 0)),
        output_model_id=output_model_id,
        save_final_checkpoint=bool(training_cfg.get("save_final_checkpoint", True)),
        dcp_save_interval=args.dcp_save_interval,
        eval_auto_carveout=bool(training_cfg.get("eval_auto_carveout", False)),
        render_workers=args.render_workers,
        pipeline_depth=int(training_cfg.get("pipeline_depth", 1)),
        trainer=TrainerConfig(
            job_id=job_id,
            training_shape_id=model_cfg["training_shape_id"],
            timeout_s=args.timeout_s,
        ),
        wandb=WandBConfig(
            entity=wandb_entity,
            project=wandb_project,
            run_name=wandb_run_name,
        ),
        runner=RunnerConfig(
            status_file=str(run_dir / "status.json"),
            metadata_file=str(run_dir / "metadata.json"),
            metrics_file=str(run_dir / "metrics.jsonl"),
            output_model_path=str(run_dir / "output_model.json"),
        ),
    )


def sft_config_summary(cfg: Any, args: Any) -> dict[str, Any]:
    return {
        "base_model": cfg.base_model,
        "tokenizer_model": cfg.tokenizer_model,
        "renderer_name": cfg.renderer_name,
        "train_on_what": cfg.train_on_what,
        "training_shape_id": cfg.trainer.training_shape_id,
        "lora_rank": cfg.lora_rank,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler": {
            "type": "cosine",
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "max_examples": cfg.max_examples,
        "render_workers": args.render_workers,
        "dcp_save_interval": args.dcp_save_interval,
        "save_final_checkpoint": cfg.save_final_checkpoint,
        "eval_auto_carveout": cfg.eval_auto_carveout,
        "pipeline_depth": cfg.pipeline_depth,
        "dataset": cfg.dataset,
    }


def trainer_mgr(base_url: str):
    from fireworks.training.sdk import TrainerJobManager

    return TrainerJobManager(api_key=os.environ["FIREWORKS_API_KEY"], base_url=base_url)


def get_job_state(job_id: str, base_url: str) -> dict[str, Any]:
    try:
        return trainer_mgr(base_url).get(job_id)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def delete_job(job_id: str, base_url: str) -> dict[str, Any]:
    before = get_job_state(job_id, base_url)
    try:
        trainer_mgr(base_url).delete(job_id)
        time.sleep(3)
        after = get_job_state(job_id, base_url)
        return {"delete_attempted": True, "before": before, "after": after}
    except Exception as exc:
        return {
            "delete_attempted": True,
            "delete_error": f"{type(exc).__name__}: {exc}",
            "before": before,
            "after": get_job_state(job_id, base_url),
        }


def run_alignment_sft(config: dict[str, Any], args: Any) -> int:
    from training.recipes.sft_loop import main as run_sft

    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    run_dir = build_run_dir(config, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for source in [Path(__file__).resolve(), Path(config["_entrypoint"]).resolve() if config.get("_entrypoint") else None, Path(config["_config_path"]).resolve()]:
        if source and source.exists():
            shutil.copy2(source, run_dir / source.name)

    job_suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    job_id = f"{config['naming']['job_id_prefix']}-{job_suffix}"
    output_model_id = f"{config['naming']['output_model_id_prefix']}-{job_suffix}"

    os.environ["COOKBOOK_STATUS_FILE"] = str(run_dir / "status.json")
    os.environ["COOKBOOK_METADATA_FILE"] = str(run_dir / "metadata.json")
    os.environ["COOKBOOK_METRICS_FILE"] = str(run_dir / "metrics.jsonl")
    os.environ["COOKBOOK_OUTPUT_MODEL_PATH"] = str(run_dir / "output_model.json")

    fireworks_env = configure_fireworks_env(base_url)
    wandb_entity, wandb_env = configure_wandb_env(run_dir)
    wandb_project = os.environ.get("WANDB_PROJECT") or config.get("wandb", {}).get("project_default") or "sft-tinker"
    wandb_run_name = os.environ.get("WANDB_RUN_NAME") or run_dir.name
    wandb_manifest = prepare_wandb_run_manifest(
        entity=wandb_entity,
        project=wandb_project,
        run_name=wandb_run_name,
    )
    write_json(run_dir / "wandb_run.json", {**wandb_manifest, "status": "prepared"})

    selected = select_core_rows(config, args.limit)
    raw_path, manifest_path = materialize_selected_raw(config, selected, run_dir)
    dataset, conversion_report = convert_selected(config, raw_path, run_dir)
    tool_linkage = audit_tool_linkage(dataset, run_dir)
    precheck = local_renderer_precheck(config, dataset, run_dir, args.limit)
    dataloader_precheck = official_dataloader_precheck(
        config,
        dataset,
        run_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        render_workers=args.render_workers,
    )
    shape_precheck = None
    if config.get("precheck", {}).get("resolve_training_shape", False):
        shape_precheck = training_shape_precheck(config, run_dir, base_url=base_url)
    model_precheck = None
    if config.get("precheck", {}).get("resolve_base_model", False):
        model_precheck = base_model_precheck(config, run_dir, base_url=base_url)
    dataset_sha = sha256_file(dataset)

    selection_report = {
        "source": config.get("summary", {}).get("source", "alignment SFT conversion flow"),
        "render_contract": config.get("summary", {}).get("render_contract"),
        "source_run_dir": str(config["paths"]["source_run"]),
        "selection_rule": config.get("selection", {}).get(
            "rule",
            f"{args.limit} shortest rows from the configured audit with status=ok and active targets > 0",
        ),
        "selected": selected,
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_sha,
        "conversion_report": str(conversion_report),
        "local_renderer_precheck": precheck,
        "official_dataloader_precheck": dataloader_precheck,
        "tool_linkage_audit": tool_linkage,
        "training_shape_precheck": shape_precheck,
        "base_model_precheck": model_precheck,
    }
    write_json(run_dir / "selection_report.json", selection_report)

    if config.get("precheck", {}).get("require_single_datum_per_row", False) and precheck.get("split_counts") != {"1": args.limit}:
        raise RuntimeError(f"single-datum precheck failed: split_counts={precheck.get('split_counts')}")

    cfg = build_sft_config(
        config,
        run_dir=run_dir,
        dataset=dataset,
        job_id=job_id,
        output_model_id=output_model_id,
        args=args,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    generated_cfg = sft_config_summary(cfg, args)
    write_json(run_dir / "generated_sft_config.json", generated_cfg)

    summary = {
        "run_dir": str(run_dir),
        "job_id": job_id,
        "output_model_id": output_model_id,
        "base_url": base_url,
        "config_path": config["_config_path"],
        "entrypoint": config.get("_entrypoint"),
        "render_contract": config.get("summary", {}).get("render_contract"),
        "a_version_core_guarantees": config.get("summary", {}).get("guarantees", {}),
        **generated_cfg,
        "dataset_sha256": dataset_sha,
        "selected_sample_ids": [r["sample_id"] for r in selected],
        "local_renderer_precheck": precheck,
        "official_dataloader_precheck": dataloader_precheck,
        "tool_linkage_audit": tool_linkage,
        "training_shape_precheck": shape_precheck,
        "base_model_precheck": model_precheck,
        "fireworks_env": fireworks_env,
        "wandb_requested": bool(wandb_entity),
        "wandb_env": wandb_env,
        "wandb_entity": "<set>" if wandb_entity else None,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_run": wandb_manifest,
        "dry_run": bool(args.dry_run or args.precheck_only),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "run_config_summary.json", summary)

    if args.dry_run or args.precheck_only:
        dry = {**summary, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "would_submit": False}
        write_json(run_dir / "dry_run_result.json", dry)
        print("SFT_DRY_RUN_OK", json.dumps({"run_dir": str(run_dir), "split_counts": precheck["split_counts"]}, ensure_ascii=False), flush=True)
        return 0

    rc = 0
    try:
        write_json(
            run_dir / "wandb_run.json",
            {
                **wandb_manifest,
                "status": "launch_attempted",
                "launch_attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        metrics = run_sft(cfg)
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["result"] = metrics
        summary["wandb_run"] = {
            **wandb_manifest,
            "status": "finished",
            "post_training_state": wandb_run_info(),
        }
        write_json(run_dir / "wandb_run.json", summary["wandb_run"])
        summary["output_model"] = json.loads((run_dir / "output_model.json").read_text(encoding="utf-8")) if (run_dir / "output_model.json").exists() else None
        write_json(run_dir / "run_result.json", summary)
        print("SFT_OK", json.dumps(metrics, ensure_ascii=False, default=str), flush=True)
    except BaseException as exc:
        rc = 1
        failure = {
            **summary,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wandb_run": {
                **wandb_manifest,
                "status": "failed",
                "post_training_state": wandb_run_info(),
            },
        }
        write_json(run_dir / "wandb_run.json", failure["wandb_run"])
        write_json(run_dir / "run_failure.json", failure)
        print("SFT_ERR", type(exc).__name__, str(exc), flush=True)
        traceback.print_exc()
    finally:
        cleanup = delete_job(job_id, base_url)
        cleanup["cleanup_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(run_dir / "cleanup_result.json", cleanup)
        print("CLEANUP", json.dumps(cleanup, ensure_ascii=False, default=str)[:4000], flush=True)
    return rc
