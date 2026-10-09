#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


COOKBOOK_TRAINING = Path("/home/ubuntu/.remotelab/workspace/fw-ai-cookbook/training")
if str(COOKBOOK_TRAINING) not in sys.path:
    sys.path.insert(0, str(COOKBOOK_TRAINING))

from fireworks.training.sdk import TrainerJobManager
from transformers import AutoTokenizer

from training.recipes.sft_loop import Config, main as run_sft
from training.utils import RunnerConfig, TrainerConfig, WandBConfig
from training.utils.supervised import build_renderer, render_messages_to_datums


ROOT = Path("/home/ubuntu/workspace/fireworks")
DATA_ALIGNMENT = ROOT / "data_alignment"
SOURCE_RUN = ROOT / "runs/data-alignment-20260629T145032Z"
DEFAULT_RUN_ROOT = Path("/home/ubuntu/.remotelab/workspace/fireworks-runs")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

BASE_MODEL = "accounts/fireworks/models/qwen3p6-27b"
TOKENIZER_MODEL = "Qwen/Qwen3.6-27B"
RENDERER_NAME = "qwen3_6"
TRAINING_SHAPE_ID = "accounts/fireworks/trainingShapes/qwen3p6-27b-256k"
MAX_SEQ_LEN = 262144


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


def configure_fireworks_env() -> dict[str, Any]:
    bashrc_key = read_bashrc_var("FIREWORKS_API_KEY")
    if bashrc_key:
        os.environ["FIREWORKS_API_KEY"] = bashrc_key
    return {
        "fireworks_api_key_present": bool(os.environ.get("FIREWORKS_API_KEY")),
        "fireworks_base_url": BASE_URL,
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


def configure_wandb_env(run_dir: Path) -> tuple[str | None, dict[str, Any]]:
    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("W_and_B_API_KEY") or read_bashrc_var("W_and_B_API_KEY")
    if api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key
    entity = detect_wandb_entity()
    if entity and not os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = entity
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "online" if api_key else "offline"
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    return entity, {
        "wandb_api_key_present": bool(api_key),
        "wandb_entity_present": bool(entity),
        "wandb_mode": os.environ.get("WANDB_MODE"),
    }


def trainer_mgr() -> TrainerJobManager:
    return TrainerJobManager(api_key=os.environ["FIREWORKS_API_KEY"], base_url=BASE_URL)


def get_job_state(job_id: str) -> dict[str, Any]:
    try:
        return trainer_mgr().get(job_id)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def delete_job(job_id: str) -> dict[str, Any]:
    before = get_job_state(job_id)
    try:
        trainer_mgr().delete(job_id)
        time.sleep(3)
        after = get_job_state(job_id)
        return {"delete_attempted": True, "before": before, "after": after}
    except Exception as exc:
        return {
            "delete_attempted": True,
            "delete_error": f"{type(exc).__name__}: {exc}",
            "before": before,
            "after": get_job_state(job_id),
        }


def select_shortest_core_rows(limit: int) -> list[dict[str, Any]]:
    candidates = []
    for rec in iter_jsonl(DATA_ALIGNMENT / "core/fireworks_core_audit.jsonl"):
        if rec.get("status") != "ok":
            continue
        if int(rec.get("active_target_token_count") or 0) <= 0:
            continue
        candidates.append(rec)
    candidates.sort(key=lambda r: (int(r.get("ms_swift_token_count") or r.get("default_fireworks_token_count") or 10**18), int(r.get("sample_id") or 0)))
    return candidates[:limit]


def materialize_selected_raw(selected: list[dict[str, Any]], run_dir: Path) -> tuple[Path, Path]:
    wanted = {int(r["sample_id"]) for r in selected}
    raw_out = run_dir / "selected_raw_100.jsonl"
    manifest_out = run_dir / "selected_manifest_100.jsonl"
    with raw_out.open("w", encoding="utf-8") as raw_dst:
        for sample_id, row in enumerate(iter_jsonl(SOURCE_RUN / "sample_1000.jsonl")):
            if sample_id in wanted:
                raw_dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    with manifest_out.open("w", encoding="utf-8") as manifest_dst:
        for row in iter_jsonl(DATA_ALIGNMENT / "core/sample_manifest.jsonl"):
            if int(row["sample_id"]) in wanted:
                manifest_dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return raw_out, manifest_out


def convert_selected(raw_path: Path, run_dir: Path) -> tuple[Path, Path]:
    converted = run_dir / "qwen36_27b_a100_sft.jsonl"
    report = run_dir / "a100_conversion_report.json"
    cmd = [
        sys.executable,
        str(DATA_ALIGNMENT / "core/convert_core_aligned.py"),
        "--input",
        str(raw_path),
        "--output",
        str(converted),
        "--report",
        str(report),
        "--include-images",
        "--mixed-tool-call-loss-policy",
        "report",
    ]
    subprocess.run(cmd, check=True)
    return converted, report


def local_renderer_precheck(dataset: Path, run_dir: Path, limit: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    renderer = build_renderer(tokenizer, TOKENIZER_MODEL, RENDERER_NAME)
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
                    train_on_what="all_assistant_messages",
                    max_seq_len=MAX_SEQ_LEN,
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


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).resolve()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_RUN_ROOT / f"qwen36-27b-full-sft-a100-{ts}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3.6 27B full-param SFT on 100 A-aligned rows using Fireworks Training API.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--timeout-s", type=int, default=10800)
    args = parser.parse_args()

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, run_dir / Path(__file__).name)

    job_suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    job_id = f"training-api-service-qwen36-27b-a100-{job_suffix}"
    output_model_id = f"qwen36-27b-a100-sft-{job_suffix}"

    os.environ["COOKBOOK_STATUS_FILE"] = str(run_dir / "status.json")
    os.environ["COOKBOOK_METADATA_FILE"] = str(run_dir / "metadata.json")
    os.environ["COOKBOOK_METRICS_FILE"] = str(run_dir / "metrics.jsonl")
    os.environ["COOKBOOK_OUTPUT_MODEL_PATH"] = str(run_dir / "output_model.json")

    fireworks_env = configure_fireworks_env()
    wandb_entity, wandb_env = configure_wandb_env(run_dir)
    wandb_project = os.environ.get("WANDB_PROJECT") or "fireworks-qwen36-27b-a100-sft"
    wandb_run_name = os.environ.get("WANDB_RUN_NAME") or run_dir.name

    selected = select_shortest_core_rows(args.limit)
    raw_path, manifest_path = materialize_selected_raw(selected, run_dir)
    dataset, conversion_report = convert_selected(raw_path, run_dir)
    precheck = local_renderer_precheck(dataset, run_dir, args.limit)
    dataset_sha = sha256_file(dataset)

    selection_report = {
        "source": "A-version core aligned conversion flow",
        "source_run_dir": str(SOURCE_RUN),
        "selection_rule": f"{args.limit} shortest rows from the existing 1000-row A core audit with status=ok and active targets > 0",
        "selected": selected,
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_sha,
        "conversion_report": str(conversion_report),
        "local_renderer_precheck": precheck,
    }
    write_json(run_dir / "selection_report.json", selection_report)

    cfg = Config(
        log_path=str(run_dir / "sft_logs"),
        render_samples_file=str(run_dir / "render_samples.jsonl"),
        render_samples_limit=10,
        base_model=BASE_MODEL,
        dataset=str(dataset),
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        train_on_what="all_assistant_messages",
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.limit,
        lora_rank=0,
        output_model_id=output_model_id,
        save_final_checkpoint=True,
        dcp_save_interval=0,
        eval_auto_carveout=False,
        render_workers=0,
        pipeline_depth=1,
        trainer=TrainerConfig(
            job_id=job_id,
            training_shape_id=TRAINING_SHAPE_ID,
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

    summary = {
        "run_dir": str(run_dir),
        "job_id": job_id,
        "output_model_id": output_model_id,
        "base_url": BASE_URL,
        "base_model": cfg.base_model,
        "tokenizer_model": cfg.tokenizer_model,
        "renderer_name": cfg.renderer_name,
        "train_on_what": cfg.train_on_what,
        "training_shape_id": cfg.trainer.training_shape_id,
        "lora_rank": cfg.lora_rank,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "max_examples": cfg.max_examples,
        "dataset": cfg.dataset,
        "dataset_sha256": dataset_sha,
        "selected_sample_ids": [r["sample_id"] for r in selected],
        "local_renderer_precheck": precheck,
        "fireworks_env": fireworks_env,
        "wandb_requested": bool(wandb_entity),
        "wandb_env": wandb_env,
        "wandb_entity": "<set>" if wandb_entity else None,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "run_config_summary.json", summary)

    rc = 0
    try:
        metrics = run_sft(cfg)
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["result"] = metrics
        summary["wandb_run"] = wandb_run_info()
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
        }
        write_json(run_dir / "run_failure.json", failure)
        print("SFT_ERR", type(exc).__name__, str(exc), flush=True)
        traceback.print_exc()
    finally:
        cleanup = delete_job(job_id)
        cleanup["cleanup_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(run_dir / "cleanup_result.json", cleanup)
        print("CLEANUP", json.dumps(cleanup, ensure_ascii=False, default=str)[:4000], flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
