#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from run_alignment_sft import main as run_alignment_main


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/qwen36_27b_preserve_thinking_official.yaml"


def main(argv: list[str] | None = None) -> int:
    incoming = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in incoming:
        incoming = ["--config", str(DEFAULT_CONFIG), *incoming]
    return run_alignment_main(incoming)


if __name__ == "__main__":
    raise SystemExit(main())
