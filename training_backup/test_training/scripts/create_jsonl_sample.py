#!/usr/bin/env python3
"""Create a deterministic JSONL reservoir sample without content filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset-name", default="jsonl-sampled")
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{args.dataset_name}.train.jsonl"
    summary_json = args.output_dir / "summary.json"
    manifest_json = args.output_dir / "manifest.json"

    rng = random.Random(args.seed)
    reservoir: list[str] = []
    counters = {"total": 0, "decode_error": 0, "eligible": 0, "sampled": 0}

    with args.source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            counters["total"] += 1
            line = raw_line.rstrip("\n")
            try:
                json.loads(line)
            except json.JSONDecodeError:
                counters["decode_error"] += 1
                continue

            counters["eligible"] += 1
            seen = counters["eligible"]
            if len(reservoir) < args.sample_size:
                reservoir.append(line)
            else:
                replace_idx = rng.randrange(seen)
                if replace_idx < args.sample_size:
                    reservoir[replace_idx] = line

    if counters["eligible"] < args.sample_size:
        raise RuntimeError(
            f"Only found {counters['eligible']} valid rows, "
            f"less than requested sample size {args.sample_size}"
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for line in reservoir:
            handle.write(line)
            handle.write("\n")

    counters["sampled"] = len(reservoir)
    output_sha256 = sha256_file(output_jsonl)
    created_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "dataset_name": args.dataset_name,
        "created_at_utc": created_at,
        "source": str(args.source),
        "output_jsonl": str(output_jsonl),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selection": "deterministic reservoir sample over all valid JSONL rows; no no-think filtering",
        "counters": counters,
        "output_sha256": output_sha256,
    }
    manifest = {
        "name": args.dataset_name,
        "train_file": str(output_jsonl),
        "num_records": len(reservoir),
        "format": "jsonl",
        "source_file": str(args.source),
        "sha256": output_sha256,
        "created_at_utc": created_at,
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
