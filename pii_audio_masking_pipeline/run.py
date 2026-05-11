from __future__ import annotations

from pathlib import Path
import argparse
import logging
import traceback

from .config import apply_performance_profile, load_config, save_config, validate_config
from .manifest import build_manifest, discover_audio_files
from .pipeline import PIIMaskingPipeline
from .state import SQLiteState
from .utils import append_jsonl, setup_logging, write_csv

logger = logging.getLogger(__name__)


def iter_target_files(config):
    files = discover_audio_files(config.paths.input_root, config.paths.input_glob, config=config)
    if config.runtime.shard_count > 1:
        files = [p for idx, p in enumerate(files) if idx % config.runtime.shard_count == config.runtime.shard_index]
    if config.runtime.limit is not None:
        files = files[: int(config.runtime.limit)]
    return files


def _is_success_status(status: str | None) -> bool:
    return str(status or "").startswith("success")


def run_process(config):
    files = iter_target_files(config)
    logger.info("Processing %d files", len(files))

    state = SQLiteState(Path(config.paths.work_dir) / f"run_state_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.sqlite")
    pipeline = PIIMaskingPipeline(config)

    report_path = Path(config.paths.work_dir) / "reports" / f"processing_results_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    recent_rows = []
    max_csv_rows = max(0, int(getattr(config.runtime, "max_csv_report_rows", 10000)))
    file_batch_size = max(1, int(getattr(config.runtime, "file_batch_size", 1)))
    current_file_batch_size = file_batch_size
    success_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0

    def record_result(result: dict) -> None:
        nonlocal success_count, failed_count, skipped_count, processed_count, recent_rows
        append_jsonl(report_path, [result])
        if max_csv_rows > 0:
            recent_rows.append(result)
            if len(recent_rows) > max_csv_rows:
                recent_rows.pop(0)

        input_path = str(result.get("input_path"))
        status = result.get("status", "unknown")
        if status == "failed":
            state.upsert(input_path, result.get("output_path"), "failed", error=result.get("error"))
            failed_count += 1
        else:
            state.upsert(
                input_path=input_path,
                output_path=result.get("output_path"),
                status=status,
                error=None,
                duration_sec=result.get("duration_sec"),
                num_words=result.get("num_words"),
                num_entities=result.get("num_entities"),
                num_spans=result.get("num_spans"),
            )
            if status == "skipped_existing":
                skipped_count += 1
            if _is_success_status(status) or status == "skipped_existing":
                success_count += 1
        processed_count += 1

    def make_failed_row(path, err: str) -> dict:
        return {"input_path": str(path), "status": "failed", "error": err}

    def delete_partial_outputs(path) -> None:
        if not getattr(config.runtime, "delete_failed_partial_outputs", True):
            return
        try:
            out = pipeline.output_path_for(path)
            sidecar = pipeline.sidecar_path_for(out)
            for partial in (out, sidecar):
                if Path(partial).exists():
                    Path(partial).unlink()
        except Exception:
            pass

    def process_one(path):
        try:
            logger.info("processing: %s", path)
            return pipeline.process_file(path)
        except Exception as e:
            if config.runtime.fail_fast:
                raise
            delete_partial_outputs(path)
            err = repr(e)
            logger.error("Failed: %s | %s", path, err)
            logger.debug(traceback.format_exc())
            return make_failed_row(path, err)

    def flush_batch(batch: list[Path]) -> None:
        nonlocal current_file_batch_size
        if not batch:
            return
        suggested_size = pipeline._suggest_file_batch_size(current_file_batch_size) if hasattr(pipeline, "_suggest_file_batch_size") else current_file_batch_size
        if suggested_size < current_file_batch_size:
            logger.info("Adaptive batching reduced file_batch_size from %d to %d", current_file_batch_size, suggested_size)
            current_file_batch_size = suggested_size
        if len(batch) > current_file_batch_size:
            for start in range(0, len(batch), current_file_batch_size):
                flush_batch(batch[start:start + current_file_batch_size])
            return
        if len(batch) == 1 or current_file_batch_size <= 1:
            for p in batch:
                record_result(process_one(p))
            return
        try:
            logger.info("processing micro-batch: size=%d first=%s", len(batch), batch[0])
            results = pipeline.process_files_batch(batch)
        except Exception as e:
            if config.runtime.fail_fast:
                raise
            logger.warning("Micro-batch failed, falling back to per-file processing: %s", e)
            results = [process_one(p) for p in batch]
        for result in results:
            record_result(result)
        if current_file_batch_size < file_batch_size and hasattr(pipeline, "_recover_file_batch_size"):
            recovered_size = pipeline._recover_file_batch_size(
                current_file_batch_size,
                configured_size=file_batch_size,
                free_gpu_mem_gb=pipeline._cuda_free_memory_gb(),
            )
            if recovered_size > current_file_batch_size:
                logger.info("Adaptive batching increased file_batch_size from %d to %d", current_file_batch_size, recovered_size)
                current_file_batch_size = recovered_size

    pending: list[Path] = []
    for i, path in enumerate(files, 1):
        path_str = str(path)
        existing = state.get(path_str)
        if config.runtime.resume and existing and _is_success_status(existing.get("status")):
            out = existing.get("output_path")
            if out:
                sidecar = str(out) + config.paths.sidecar_suffix
                if pipeline.can_resume_skip(path, out, sidecar):
                    skipped_count += 1
                    processed_count += 1
                    if i % max(1, int(config.runtime.log_every)) == 0:
                        logger.info("[%d/%d] resume skip: %s", i, len(files), path)
                    continue

        pending.append(path)
        if len(pending) >= current_file_batch_size:
            flush_batch(pending)
            pending = []

        if i % max(1, int(config.runtime.log_every)) == 0:
            logger.info(
                "Progress: seen=%d processed=%d success=%d failed=%d resume_skipped=%d micro_batch_size=%d",
                i,
                processed_count,
                success_count,
                failed_count,
                skipped_count,
                current_file_batch_size,
            )

    flush_batch(pending)

    if config.runtime.write_csv_reports and recent_rows:
        write_csv(Path(config.paths.work_dir) / "reports" / f"processing_results_latest_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.csv", recent_rows)
    state.close()


def main():
    parser = argparse.ArgumentParser(description="Fast speech PII masking pipeline for stereo 48 kHz Opus calls.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--stage", default="all", choices=["all", "manifest", "process", "write-default-config"])
    parser.add_argument("--limit", type=int, default=None, help="Override runtime.limit.")
    parser.add_argument("--input-root", default=None, help="Override paths.input_root.")
    parser.add_argument("--output-root", default=None, help="Override paths.output_root.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume skips for this run.")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--input-audio-strategy", choices=["single_decode", "ffmpeg_temp_wav"], default=None)
    parser.add_argument("--file-batch-size", type=int, default=None, help="Override runtime.file_batch_size for cross-file micro-batching.")
    parser.add_argument("--file-batch-max-decoded-audio-gb", type=float, default=None, help="Override runtime.file_batch_max_decoded_audio_gb.")
    parser.add_argument("--performance-profile", choices=["default", "a10g_24gb"], default=None, help="Apply a tuned runtime performance profile before other CLI overrides.")
    parser.add_argument("--enable-asr-engines", default=None, help="Comma-separated ASR engines to enable, for example: whisper,qwen,cohere,granite")
    parser.add_argument("--disable-asr-engines", default=None, help="Comma-separated ASR engines to disable")
    parser.add_argument("--asr-model-residency", choices=["keep_loaded", "unload_after_file"], default=None)
    parser.add_argument("--pii-device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    apply_config_profile = args.performance_profile is None
    config = load_config(args.config if args.stage != "write-default-config" else None, apply_profile=apply_config_profile)

    if args.limit is not None:
        config.runtime.limit = args.limit
    if args.input_root is not None:
        config.paths.input_root = args.input_root
    if args.output_root is not None:
        config.paths.output_root = args.output_root
    if args.no_resume:
        config.runtime.resume = False
    if args.shard_index is not None:
        config.runtime.shard_index = args.shard_index
    if args.shard_count is not None:
        config.runtime.shard_count = args.shard_count
    if args.input_audio_strategy is not None:
        config.asr.input_audio_strategy = args.input_audio_strategy
    if args.performance_profile is not None:
        config.runtime.performance_profile = args.performance_profile
        apply_performance_profile(config)
    if args.file_batch_size is not None:
        config.runtime.file_batch_size = args.file_batch_size
    if args.file_batch_max_decoded_audio_gb is not None:
        config.runtime.file_batch_max_decoded_audio_gb = args.file_batch_max_decoded_audio_gb
    if args.enable_asr_engines is not None:
        requested = {x.strip() for x in args.enable_asr_engines.split(",") if x.strip()}
        for name in requested:
            if name not in config.asr.engines:
                raise ValueError(f"Unknown ASR engine in --enable-asr-engines: {name}")
        for name, engine_cfg in config.asr.engines.items():
            engine_cfg["enabled"] = name in requested
    if args.disable_asr_engines is not None:
        for name in [x.strip() for x in args.disable_asr_engines.split(",") if x.strip()]:
            if name not in config.asr.engines:
                raise ValueError(f"Unknown ASR engine in --disable-asr-engines: {name}")
            config.asr.engines[name]["enabled"] = False
    if args.asr_model_residency is not None:
        config.asr.model_residency = args.asr_model_residency
    if args.pii_device is not None:
        config.pii.device = args.pii_device

    config = validate_config(config)

    Path(config.paths.work_dir).mkdir(parents=True, exist_ok=True)
    save_config(config, Path(config.paths.work_dir) / "effective_config.yaml")

    if args.stage == "write-default-config":
        save_config(config, args.config)
        logger.info("Wrote default config to %s", args.config)
        return

    if args.stage in {"all", "manifest"}:
        build_manifest(config)

    if args.stage in {"all", "process"}:
        run_process(config)


if __name__ == "__main__":
    main()
