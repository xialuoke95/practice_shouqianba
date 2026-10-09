#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alignment_sft_lib import DEFAULT_CONFIG, load_config, run_alignment_sft


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    training = (config_defaults or {}).get("training", {})
    parser = argparse.ArgumentParser(description="Run alignment SFT from a YAML preset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--limit", type=int, default=int(training.get("limit", 100)))
    parser.add_argument("--batch-size", type=int, default=int(training.get("batch_size", 48)))
    parser.add_argument("--epochs", type=int, default=int(training.get("epochs", 1)))
    parser.add_argument("--learning-rate", type=float, default=float(training.get("learning_rate", 1e-5)))
    parser.add_argument("--warmup-ratio", type=float, default=float(training.get("warmup_ratio", 0.05)))
    parser.add_argument("--min-lr-ratio", type=float, default=float(training.get("min_lr_ratio", 0.1)))
    parser.add_argument("--weight-decay", type=float, default=float(training.get("weight_decay", 0.1)))
    parser.add_argument("--grad-clip-norm", type=float, default=float(training.get("grad_clip_norm", 1.0)))
    parser.add_argument("--render-workers", type=int, default=int(training.get("render_workers", 4)))
    parser.add_argument("--dcp-save-interval", type=int, default=int(training.get("dcp_save_interval", 200)))
    parser.add_argument("--timeout-s", type=int, default=int(training.get("timeout_s", 10800)))
    parser.add_argument("--dry-run", action="store_true", help="Run conversion and render precheck but do not submit Training API job.")
    parser.add_argument("--precheck-only", action="store_true", help="Alias for dry-run; kept explicit for automation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = initial.parse_known_args(argv)
    config = load_config(Path(known.config).resolve())
    parser = build_parser(config)
    args = parser.parse_args(argv)
    config = load_config(Path(args.config).resolve())
    config["_entrypoint"] = __file__
    return run_alignment_sft(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
