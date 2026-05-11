from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pii_audio_masking_pipeline.config import apply_performance_profile, load_config


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("--batch-sizes must contain positive integers")
    return sizes


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _read_jsonl_since(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < offset:
                continue
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _report_path(config_path: str, performance_profile: str | None) -> Path:
    config = load_config(config_path, apply_profile=performance_profile != "default")
    if performance_profile:
        config.runtime.performance_profile = performance_profile
        apply_performance_profile(config)
    return (
        Path(config.paths.work_dir)
        / "reports"
        / f"processing_results_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.jsonl"
    )


def _summarize(rows: list[dict[str, Any]], elapsed_sec: float, batch_size: int) -> dict[str, Any]:
    total_audio_sec = sum(float(row.get("duration_sec") or 0.0) for row in rows)
    success = sum(1 for row in rows if str(row.get("status", "")).startswith("success"))
    failed = sum(1 for row in rows if row.get("status") == "failed")
    skipped = sum(1 for row in rows if row.get("status") == "skipped_existing")
    peak_free_gpu_gb = None
    min_free_gpu_gb = None
    for row in rows:
        for sample in ((row.get("perf_metrics") or {}).get("cuda_memory") or []):
            if not sample.get("available"):
                continue
            free_gb = float(sample.get("free_gb", 0.0))
            peak_free_gpu_gb = free_gb if peak_free_gpu_gb is None else max(peak_free_gpu_gb, free_gb)
            min_free_gpu_gb = free_gb if min_free_gpu_gb is None else min(min_free_gpu_gb, free_gb)
    return {
        "batch_size": batch_size,
        "elapsed_sec": elapsed_sec,
        "rows": len(rows),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "audio_hours": total_audio_sec / 3600.0,
        "files_per_hour": (len(rows) / elapsed_sec * 3600.0) if elapsed_sec > 0 else 0.0,
        "audio_hours_per_hour": (total_audio_sec / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "max_free_gpu_gb": peak_free_gpu_gb,
        "min_free_gpu_gb": min_free_gpu_gb,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark file-batch sizes for the AMC PII masking pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=[1, 2, 4, 8])
    parser.add_argument("--performance-profile", choices=["default", "a10g_24gb"], default=None)
    parser.add_argument("--enable-asr-engines", default=None)
    parser.add_argument("--disable-asr-engines", default=None)
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed through to the pipeline command.")
    args = parser.parse_args()

    report_path = _report_path(args.config, args.performance_profile)
    summaries: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        before = _line_count(report_path)
        cmd = [
            sys.executable,
            "-m",
            "pii_audio_masking_pipeline.run",
            "--config",
            args.config,
            "--stage",
            "process",
            "--limit",
            str(args.limit),
            "--no-resume",
            "--file-batch-size",
            str(batch_size),
        ]
        if args.performance_profile:
            cmd.extend(["--performance-profile", args.performance_profile])
        if args.enable_asr_engines:
            cmd.extend(["--enable-asr-engines", args.enable_asr_engines])
        if args.disable_asr_engines:
            cmd.extend(["--disable-asr-engines", args.disable_asr_engines])
        cmd.extend(args.extra_arg)

        started = time.perf_counter()
        completed = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - started
        rows = _read_jsonl_since(report_path, before)
        summary = _summarize(rows, elapsed, batch_size)
        summary["exit_code"] = completed.returncode
        summaries.append(summary)

        print(
            "batch={batch_size} rows={rows} success={success} failed={failed} "
            "files_per_hour={files_per_hour:.2f} audio_hours_per_hour={audio_hours_per_hour:.2f} "
            "min_free_gpu_gb={min_free_gpu_gb}".format(**summary),
            flush=True,
        )

    out_path = report_path.parent / f"benchmark_matrix_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote benchmark summary: {out_path}")
    return 0 if all(row["exit_code"] == 0 for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
