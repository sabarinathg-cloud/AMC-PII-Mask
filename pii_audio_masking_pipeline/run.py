from __future__ import annotations

from pathlib import Path
import argparse
import copy
import logging
import traceback

from .asr import ASRResult, MultiASRTranscriber, clear_accelerator_cache
from .audio_io import decode_to_float32_stereo_48k, ffprobe_audio, make_asr_audio_inputs
from .config import apply_performance_profile, enabled_asr_engines, load_config, save_config, validate_config
from .manifest import build_manifest, discover_audio_files
from .pipeline import PIIMaskingPipeline
from .state import ASRResultCache, SQLiteState
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


def _state_path(config) -> Path:
    return Path(config.paths.work_dir) / f"run_state_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.sqlite"


def _asr_cache_path(config) -> Path:
    return Path(config.paths.work_dir) / f"asr_results_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.sqlite"


def _expected_asr_channels(config, meta: dict) -> list[int]:
    if config.asr.mode == "mono_mix":
        return [-1]
    n_input_channels = int(meta.get("channels") or 1)
    return list(range(min(2, max(1, n_input_channels))))


def _single_engine_asr_config(config, engine_name: str):
    asr_cfg = copy.deepcopy(config.asr)
    asr_cfg.engine_order = [engine_name]
    asr_cfg.model_residency = "keep_loaded"
    for name, engine_cfg in asr_cfg.engines.items():
        engine_cfg["enabled"] = name == engine_name
    return asr_cfg


def _safe_error(exc: BaseException | str) -> str:
    if isinstance(exc, BaseException):
        return type(exc).__name__
    return str(exc).split(":", 1)[0][:120] or "Error"


def _prepare_model_major_asr_inputs(config, input_path: Path, meta: dict, missing_channels: set[int]) -> list[dict]:
    duration = meta.get("duration_sec")
    if config.asr.max_audio_seconds is not None and duration is not None:
        if float(duration) > float(config.asr.max_audio_seconds):
            raise ValueError(f"Audio duration {duration:.2f}s exceeds max_audio_seconds={config.asr.max_audio_seconds}")

    audio_48k = decode_to_float32_stereo_48k(
        input_path,
        ffmpeg_path=config.runtime.ffmpeg_path,
        threads=config.runtime.ffmpeg_threads,
        sample_rate=config.masking.output_sample_rate,
        channels=config.masking.output_channels,
    )
    asr_inputs = make_asr_audio_inputs(
        audio_48k,
        mode=config.asr.mode,
        input_channels=max(1, len(_expected_asr_channels(config, meta))),
        source_sr=config.masking.output_sample_rate,
        target_sr=config.asr.channel_wav_sample_rate,
    )
    out = []
    for item in asr_inputs:
        channel = int(item["channel"])
        if channel not in missing_channels:
            continue
        item["file_id"] = str(input_path)
        out.append(item)
    return out


def _write_model_major_errors(cache: ASRResultCache, engine_name: str, asr_inputs: list[dict], error: str) -> None:
    rows = []
    for item in asr_inputs:
        file_id = str(item["file_id"])
        channel = int(item["channel"])
        result = ASRResult(
            file_id=file_id,
            channel=channel,
            transcript="",
            words=[],
            engine=engine_name,
            error=error,
        )
        rows.append((file_id, engine_name, channel, result))
    cache.upsert_results(rows)


def _flush_model_major_asr_batch(
    config,
    cache: ASRResultCache,
    transcriber: MultiASRTranscriber,
    engine_name: str,
    batch_asr_inputs: list[dict],
) -> None:
    if not batch_asr_inputs:
        return
    try:
        bundles_by_file = transcriber.transcribe_channel_batch(
            batch_asr_inputs,
            work_dir=Path(config.paths.work_dir),
            keep_temp=bool(config.runtime.keep_temp),
        )
    except Exception as exc:
        if config.runtime.fail_fast:
            raise
        logger.error("Model-major ASR batch failed for engine=%s: %s", engine_name, exc)
        _write_model_major_errors(cache, engine_name, batch_asr_inputs, _safe_error(exc))
        return

    rows = []
    for file_id, bundles in bundles_by_file.items():
        for bundle in bundles:
            for result in bundle.engine_results:
                if result.engine != engine_name:
                    continue
                result.file_id = file_id
                if result.error:
                    result.error = _safe_error(result.error)
                rows.append((file_id, engine_name, int(result.channel), result))
    cache.upsert_results(rows)


def _run_model_major_asr_pass(config, files: list[Path], cache: ASRResultCache, engine_name: str) -> None:
    logger.info("Model-major ASR pass started: engine=%s files=%d", engine_name, len(files))
    transcriber = MultiASRTranscriber(_single_engine_asr_config(config, engine_name))
    file_batch_size = max(1, int(getattr(config.runtime, "file_batch_size", 1)))
    batch_asr_inputs: list[dict] = []
    batch_file_count = 0
    try:
        for i, input_path in enumerate(files, 1):
            channels: list[int] = []
            try:
                meta = ffprobe_audio(input_path, ffprobe_path=config.runtime.ffprobe_path)
                channels = _expected_asr_channels(config, meta)
                missing_channels = {
                    channel
                    for channel in channels
                    if not cache.has_result(input_path, engine_name, channel)
                }
                if not missing_channels:
                    continue
                asr_inputs = _prepare_model_major_asr_inputs(config, input_path, meta, missing_channels)
            except Exception as exc:
                if config.runtime.fail_fast:
                    raise
                logger.error("Model-major ASR preparation failed: engine=%s file=%s | %s", engine_name, input_path, exc)
                logger.debug(traceback.format_exc())
                if channels:
                    error_inputs = [
                        {"file_id": str(input_path), "channel": channel}
                        for channel in channels
                        if not cache.has_result(input_path, engine_name, channel, include_errors=True)
                    ]
                    _write_model_major_errors(cache, engine_name, error_inputs, _safe_error(exc))
                continue

            if asr_inputs:
                batch_asr_inputs.extend(asr_inputs)
                batch_file_count += 1

            if batch_file_count >= file_batch_size:
                _flush_model_major_asr_batch(config, cache, transcriber, engine_name, batch_asr_inputs)
                batch_asr_inputs = []
                batch_file_count = 0

            if i % max(1, int(config.runtime.log_every)) == 0:
                logger.info("Model-major ASR pass progress: engine=%s seen=%d/%d", engine_name, i, len(files))

        _flush_model_major_asr_batch(config, cache, transcriber, engine_name, batch_asr_inputs)
    finally:
        transcriber.unload_all()
        clear_accelerator_cache()
        logger.info("Model-major ASR pass finished and unloaded: engine=%s", engine_name)


def run_process(config):
    if getattr(config.runtime, "pipeline_schedule", "file_major") == "model_major":
        run_process_model_major(config)
        return

    files = iter_target_files(config)
    logger.info("Processing %d files", len(files))

    state = SQLiteState(_state_path(config))
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


def run_process_model_major(config):
    files = iter_target_files(config)
    engines = enabled_asr_engines(config)
    logger.info("Processing %d files with model-major schedule across engines: %s", len(files), ", ".join(engines))

    state = SQLiteState(_state_path(config))
    cache = ASRResultCache(_asr_cache_path(config))
    try:
        cache_deleted = False
        for engine_name in engines:
            _run_model_major_asr_pass(config, files, cache, engine_name)

        pipeline = PIIMaskingPipeline(config)
        report_path = Path(config.paths.work_dir) / "reports" / f"processing_results_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        recent_rows = []
        max_csv_rows = max(0, int(getattr(config.runtime, "max_csv_report_rows", 10000)))
        success_count = 0
        failed_count = 0
        failsafe_count = 0
        skipped_count = 0
        processed_count = 0

        def record_result(result: dict) -> None:
            nonlocal success_count, failed_count, failsafe_count, skipped_count, processed_count, recent_rows
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
                if status == "success_unmapped_fallback":
                    failsafe_count += 1
                if _is_success_status(status) or status == "skipped_existing":
                    success_count += 1
            processed_count += 1

        for i, input_path in enumerate(files, 1):
            try:
                path_str = str(input_path)
                existing = state.get(path_str)
                if config.runtime.resume and existing and _is_success_status(existing.get("status")):
                    out = existing.get("output_path")
                    if out:
                        sidecar = str(out) + config.paths.sidecar_suffix
                        if pipeline.can_resume_skip(input_path, out, sidecar):
                            skipped_count += 1
                            processed_count += 1
                            continue

                meta = ffprobe_audio(input_path, ffprobe_path=config.runtime.ffprobe_path)
                channels = _expected_asr_channels(config, meta)
                if not cache.has_results_for_file(input_path, engines, channels, include_errors=True):
                    raise RuntimeError("Missing cached ASR results for one or more enabled engines/channels")
                result = pipeline.process_file_from_asr_results(input_path, cache.get_results_for_file(input_path), meta=meta)
            except Exception as exc:
                if config.runtime.fail_fast:
                    raise
                try:
                    pipeline._delete_partial_outputs_silent(input_path)
                except Exception:
                    pass
                logger.error("Model-major finalization failed: %s | %s", input_path, exc)
                logger.debug(traceback.format_exc())
                result = {"input_path": str(input_path), "status": "failed", "error": _safe_error(exc)}
            record_result(result)

            if i % max(1, int(config.runtime.log_every)) == 0:
                logger.info(
                    "Model-major finalization progress: seen=%d processed=%d success=%d failed=%d resume_skipped=%d",
                    i,
                    processed_count,
                    success_count,
                    failed_count,
                    skipped_count,
                )

        if config.runtime.write_csv_reports and recent_rows:
            write_csv(Path(config.paths.work_dir) / "reports" / f"processing_results_latest_shard{config.runtime.shard_index}_of_{config.runtime.shard_count}.csv", recent_rows)
        if bool(getattr(config.runtime, "delete_asr_cache_after_finalize", False)) and failed_count == 0 and failsafe_count == 0:
            cache.delete_cache_files()
            cache_deleted = True
        elif bool(getattr(config.runtime, "delete_asr_cache_after_finalize", False)) and failsafe_count > 0:
            logger.warning("Keeping ASR cache because %d files used unmapped/failsafe masking.", failsafe_count)
    finally:
        if not cache_deleted:
            cache.close()
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
    parser.add_argument("--pipeline-schedule", choices=["file_major", "model_major"], default=None, help="Choose file-major processing or model-major ASR passes with cached finalization.")
    parser.add_argument("--delete-asr-cache-after-finalize", action="store_true", help="Delete model-major ASR cache SQLite/WAL/SHM files after a failure-free finalization pass.")
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
    if args.pipeline_schedule is not None:
        config.runtime.pipeline_schedule = args.pipeline_schedule
    if args.delete_asr_cache_after_finalize:
        config.runtime.delete_asr_cache_after_finalize = True
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
