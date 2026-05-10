from __future__ import annotations

from pathlib import Path
import argparse
import logging
import traceback

from .config import load_config, save_config, validate_config
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
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, path in enumerate(files, 1):
        path_str = str(path)
        existing = state.get(path_str)
        if config.runtime.resume and existing and _is_success_status(existing.get("status")):
            out = existing.get("output_path")
            if out:
                sidecar = str(out) + config.paths.sidecar_suffix
                if pipeline.can_resume_skip(path, out, sidecar):
                    skipped_count += 1
                    if i % max(1, int(config.runtime.log_every)) == 0:
                        logger.info("[%d/%d] resume skip: %s", i, len(files), path)
                    continue

        try:
            logger.info("[%d/%d] processing: %s", i, len(files), path)
            result = pipeline.process_file(path)
            append_jsonl(report_path, [result])
            if max_csv_rows > 0:
                recent_rows.append(result)
                if len(recent_rows) > max_csv_rows:
                    recent_rows.pop(0)
            state.upsert(
                input_path=path_str,
                output_path=result.get("output_path"),
                status=result.get("status", "unknown"),
                error=None,
                duration_sec=result.get("duration_sec"),
                num_words=result.get("num_words"),
                num_entities=result.get("num_entities"),
                num_spans=result.get("num_spans"),
            )
            if _is_success_status(result.get("status")) or result.get("status") == "skipped_existing":
                success_count += 1
        except Exception as e:
            if getattr(config.runtime, "delete_failed_partial_outputs", True):
                try:
                    out = pipeline.output_path_for(path)
                    sidecar = pipeline.sidecar_path_for(out)
                    for partial in (out, sidecar):
                        if Path(partial).exists():
                            Path(partial).unlink()
                except Exception:
                    pass
            err = repr(e)
            logger.error("Failed: %s | %s", path, err)
            logger.debug(traceback.format_exc())
            result = {"input_path": path_str, "status": "failed", "error": err}
            append_jsonl(report_path, [result])
            if max_csv_rows > 0:
                recent_rows.append(result)
                if len(recent_rows) > max_csv_rows:
                    recent_rows.pop(0)
            state.upsert(path_str, None, "failed", error=err)
            failed_count += 1
            if config.runtime.fail_fast:
                raise

        if i % max(1, int(config.runtime.log_every)) == 0:
            logger.info(
                "Progress: seen=%d success=%d failed=%d resume_skipped=%d",
                i,
                success_count,
                failed_count,
                skipped_count,
            )

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
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--input-audio-strategy", choices=["single_decode", "ffmpeg_temp_wav"], default=None)
    parser.add_argument("--enable-asr-engines", default=None, help="Comma-separated ASR engines to enable, for example: whisper,qwen,cohere,granite")
    parser.add_argument("--disable-asr-engines", default=None, help="Comma-separated ASR engines to disable")
    parser.add_argument("--asr-model-residency", choices=["keep_loaded", "unload_after_file"], default=None)
    parser.add_argument("--pii-device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config if args.stage != "write-default-config" else None)

    if args.limit is not None:
        config.runtime.limit = args.limit
    if args.input_root is not None:
        config.paths.input_root = args.input_root
    if args.output_root is not None:
        config.paths.output_root = args.output_root
    if args.shard_index is not None:
        config.runtime.shard_index = args.shard_index
    if args.shard_count is not None:
        config.runtime.shard_count = args.shard_count
    if args.input_audio_strategy is not None:
        config.asr.input_audio_strategy = args.input_audio_strategy
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
