from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import json
import logging
import tempfile
import time

from .asr import ASRResult, ChannelASRBundle, MultiASRTranscriber, align_transcript_to_timed_words
from .audio_io import (
    apply_mask_spans,
    copy_audio_passthrough,
    decode_to_float32_stereo_48k,
    encode_float32_stereo_to_opus,
    extract_channel_wav,
    extract_mono_mix_wav,
    extract_stereo_channels_wav,
    ffprobe_audio,
    input_matches_required_opus,
    make_asr_audio_inputs,
    write_mono_wav_pcm16,
)
from .pii_detection import PIIDetector
from .timestamp_mapping import entities_to_spans, merge_spans
from .utils import ensure_dir, stable_relative_path, write_json
from .validation import validate_masked_file

logger = logging.getLogger(__name__)


class PIIMaskingPipeline:
    def __init__(self, config):
        self.config = config
        self.input_root = Path(config.paths.input_root)
        self.output_root = ensure_dir(config.paths.output_root)
        self.work_dir = ensure_dir(config.paths.work_dir)
        self.asr = MultiASRTranscriber(config.asr)
        self.detector = PIIDetector(config.pii)

    def _perf_enabled(self) -> bool:
        return bool(getattr(self.config.runtime, "write_perf_metrics", False))

    def _new_perf_metrics(self) -> Optional[dict]:
        if not self._perf_enabled():
            return None
        return {
            "stages_sec": {},
            "cuda_memory": [],
            "adaptive_batching": [],
        }

    @contextmanager
    def _time_perf_stage(self, metrics: Optional[dict], stage: str):
        if metrics is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            stages = metrics.setdefault("stages_sec", {})
            stages[stage] = float(stages.get(stage, 0.0) + (time.perf_counter() - started))

    def _record_cuda_memory(self, metrics: Optional[dict], label: str) -> None:
        if metrics is None:
            return
        snapshot: dict[str, Any] = {"label": label, "available": False}
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                snapshot.update({
                    "available": True,
                    "free_gb": free_bytes / (1024.0 ** 3),
                    "total_gb": total_bytes / (1024.0 ** 3),
                    "allocated_gb": torch.cuda.memory_allocated() / (1024.0 ** 3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024.0 ** 3),
                })
        except Exception as exc:
            snapshot["error"] = repr(exc)
        metrics.setdefault("cuda_memory", []).append(snapshot)

    def _cuda_free_memory_gb(self) -> Optional[float]:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / (1024.0 ** 3)
        except Exception:
            return None

    def _adapt_file_batch_size(self, current_size: int, free_gpu_mem_gb: Optional[float]) -> int:
        current_size = max(1, int(current_size))
        if not bool(getattr(self.config.runtime, "adaptive_file_batching", True)):
            return current_size
        if free_gpu_mem_gb is None:
            return current_size
        min_free = float(getattr(self.config.runtime, "min_free_gpu_mem_gb", 0.0))
        min_size = max(1, int(getattr(self.config.runtime, "adaptive_batch_min_size", 1)))
        if free_gpu_mem_gb >= min_free or current_size <= min_size:
            return current_size
        return max(min_size, current_size // 2)

    def _recover_file_batch_size(self, current_size: int, configured_size: int, free_gpu_mem_gb: Optional[float]) -> int:
        current_size = max(1, int(current_size))
        configured_size = max(current_size, int(configured_size))
        if not bool(getattr(self.config.runtime, "adaptive_file_batching", True)):
            return current_size
        if free_gpu_mem_gb is None or current_size >= configured_size:
            return current_size
        min_free = float(getattr(self.config.runtime, "min_free_gpu_mem_gb", 0.0))
        if free_gpu_mem_gb < min_free:
            return current_size
        return min(configured_size, max(current_size + 1, current_size * 2))

    def _suggest_file_batch_size(self, current_size: int, metrics: Optional[dict] = None) -> int:
        free_gb = self._cuda_free_memory_gb()
        next_size = self._adapt_file_batch_size(current_size, free_gb)
        if metrics is not None and next_size != current_size:
            metrics.setdefault("adaptive_batching", []).append({
                "from": int(current_size),
                "to": int(next_size),
                "free_gpu_mem_gb": free_gb,
                "min_free_gpu_mem_gb": float(getattr(self.config.runtime, "min_free_gpu_mem_gb", 0.0)),
            })
        return next_size

    def _merge_perf_metrics(self, target: Optional[dict], source: Optional[dict]) -> None:
        if target is None or source is None:
            return
        for stage, value in (source.get("stages_sec") or {}).items():
            stages = target.setdefault("stages_sec", {})
            stages[stage] = float(stages.get(stage, 0.0) + float(value))
        target.setdefault("cuda_memory", []).extend(source.get("cuda_memory") or [])
        target.setdefault("adaptive_batching", []).extend(source.get("adaptive_batching") or [])
        if source.get("asr_engine_sec"):
            engine_times = target.setdefault("asr_engine_sec", {})
            for engine, value in source["asr_engine_sec"].items():
                engine_times[engine] = float(engine_times.get(engine, 0.0) + float(value))

    def _merge_batch_perf_metrics(self, target: Optional[dict], source: Optional[dict]) -> None:
        if target is None or source is None:
            return
        for stage, value in (source.get("stages_sec") or {}).items():
            stages = target.setdefault("batch_stages_sec", {})
            stages[stage] = float(stages.get(stage, 0.0) + float(value))
        target.setdefault("cuda_memory", []).extend(source.get("cuda_memory") or [])
        target.setdefault("adaptive_batching", []).extend(source.get("adaptive_batching") or [])
        if source.get("asr_engine_sec"):
            engine_times = target.setdefault("asr_engine_sec", {})
            for engine, value in source["asr_engine_sec"].items():
                engine_times[engine] = float(engine_times.get(engine, 0.0) + float(value))
        if source.get("batch_size"):
            target["batch_size"] = int(source["batch_size"])
        target["batch_metrics_semantics"] = (
            "batch_stages_sec values are full micro-batch wall times shared by every file "
            "in the batch; do not sum batch_stages_sec across co-batch files."
        )

    def output_path_for(self, input_path: str | Path) -> Path:
        input_path = Path(input_path)
        if self.config.paths.preserve_relative_path:
            rel = stable_relative_path(input_path, self.input_root)
            out = self.output_root / rel
        else:
            out = self.output_root / input_path.name
        suffix = getattr(self.config.paths, "force_output_suffix", None)
        if suffix:
            out = out.with_suffix(str(suffix))
        try:
            if out.resolve() == input_path.resolve():
                raise ValueError(
                    f"Unsafe output path equals input path: {out}. "
                    "Use a separate paths.output_root to avoid overwriting original audio."
                )
        except FileNotFoundError:
            if str(out.absolute()) == str(input_path.absolute()):
                raise ValueError(
                    f"Unsafe output path equals input path: {out}. "
                    "Use a separate paths.output_root to avoid overwriting original audio."
                )
        return out

    def sidecar_path_for(self, output_path: str | Path) -> Path:
        return Path(str(output_path) + self.config.paths.sidecar_suffix)

    def process_file(self, input_path: str | Path, row: Optional[dict] = None) -> Dict[str, Any]:
        input_path = Path(input_path)
        started = time.time()
        perf_metrics = self._new_perf_metrics()
        self._record_cuda_memory(perf_metrics, "file_start")
        output_path = self.output_path_for(input_path)
        sidecar_path = self.sidecar_path_for(output_path)
        if input_path.resolve() == output_path.resolve():
            raise ValueError(
                f"Refusing to overwrite original audio. Configure paths.output_root outside the input file path: {input_path}"
            )

        with self._time_perf_stage(perf_metrics, "ffprobe"):
            meta = ffprobe_audio(input_path, ffprobe_path=self.config.runtime.ffprobe_path)

        if self._should_skip_existing(input_path, output_path, sidecar_path, meta):
            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "skipped_existing",
                "sidecar_path": str(sidecar_path),
                "valid": True,
            }

        duration = meta.get("duration_sec")
        if self.config.asr.max_audio_seconds is not None and duration is not None:
            if float(duration) > float(self.config.asr.max_audio_seconds):
                raise ValueError(f"Audio duration {duration:.2f}s exceeds max_audio_seconds={self.config.asr.max_audio_seconds}")

        n_input_channels = int(meta.get("channels") or 1)
        channel_count = min(2, max(1, n_input_channels))

        transcripts: list[dict] = []
        all_entities: list[dict] = []
        raw_spans: list[dict] = []
        unmapped_fallback_used = False
        no_pii_fast_copy_used = False
        audio_48k = None
        used_single_decode = False
        used_combined_channel_extract = False

        strategy = str(getattr(self.config.asr, "input_audio_strategy", "single_decode"))
        max_single_decode = getattr(self.config.asr, "single_decode_max_audio_seconds", None)
        if strategy == "single_decode" and max_single_decode is not None and duration is not None:
            if float(duration) > float(max_single_decode):
                logger.warning(
                    "Audio is %.2fs, above single_decode_max_audio_seconds=%.2fs. Using ffmpeg_temp_wav ASR path to avoid a RAM spike.",
                    float(duration),
                    float(max_single_decode),
                )
                strategy = "ffmpeg_temp_wav"

        if strategy == "single_decode":
            with self._time_perf_stage(perf_metrics, "decode"):
                audio_48k = decode_to_float32_stereo_48k(
                    input_path,
                    ffmpeg_path=self.config.runtime.ffmpeg_path,
                    threads=self.config.runtime.ffmpeg_threads,
                    sample_rate=self.config.masking.output_sample_rate,
                    channels=self.config.masking.output_channels,
                )
            used_single_decode = True
            with self._time_perf_stage(perf_metrics, "asr_input_prep"):
                asr_inputs = make_asr_audio_inputs(
                    audio_48k,
                    mode=self.config.asr.mode,
                    input_channels=channel_count,
                    source_sr=self.config.masking.output_sample_rate,
                    target_sr=self.config.asr.channel_wav_sample_rate,
                )
            with self._time_perf_stage(perf_metrics, "asr"):
                results = self._transcribe_asr_inputs(asr_inputs)
            if hasattr(self.asr, "last_engine_timings_sec") and perf_metrics is not None:
                perf_metrics["asr_engine_sec"] = dict(getattr(self.asr, "last_engine_timings_sec", {}) or {})
            self._record_cuda_memory(perf_metrics, "after_asr")
            with self._time_perf_stage(perf_metrics, "pii_detection_and_mapping"):
                unmapped_fallback_used = self._handle_asr_results(results, transcripts, all_entities, raw_spans, row, duration) or unmapped_fallback_used
        elif strategy == "ffmpeg_temp_wav":
            with self._time_perf_stage(perf_metrics, "ffmpeg_temp_wav_asr"):
                used_combined_channel_extract, temp_fallback_used = self._process_with_temp_wavs(
                    input_path=input_path,
                    channel_count=channel_count,
                    row=row,
                    duration=duration,
                    transcripts=transcripts,
                    all_entities=all_entities,
                    raw_spans=raw_spans,
                )
            unmapped_fallback_used = unmapped_fallback_used or temp_fallback_used
        else:
            raise ValueError(f"Unsupported asr.input_audio_strategy: {strategy}")

        return self._finalize_outputs(
            input_path=input_path,
            output_path=output_path,
            sidecar_path=sidecar_path,
            started=started,
            meta=meta,
            duration=duration,
            audio_48k=audio_48k,
            transcripts=transcripts,
            all_entities=all_entities,
            raw_spans=raw_spans,
            used_single_decode=used_single_decode,
            used_combined_channel_extract=used_combined_channel_extract,
            unmapped_fallback_used=unmapped_fallback_used,
            batch_optimized=False,
            perf_metrics=perf_metrics,
        )

    def process_files_batch(self, input_paths: Sequence[str | Path], rows: Optional[Sequence[Optional[dict]]] = None) -> list[Dict[str, Any]]:
        """Process a micro-batch of files with cross-file ASR and PII batching.

        This is the high-throughput path. It keeps the same per-file safety guarantees
        as process_file(), but batches transcript-only ASR engines and neural PII
        detectors across files. Whisper remains the timestamp anchor and uses
        faster-whisper's internal BatchedInferencePipeline per channel.
        """
        paths = [Path(p) for p in input_paths]
        if rows is None:
            row_list: list[Optional[dict]] = [None] * len(paths)
        else:
            row_list = list(rows)
            if len(row_list) != len(paths):
                raise ValueError("rows must be None or have the same length as input_paths")

        if len(paths) <= 1 or not hasattr(self.asr, "transcribe_channel_batch"):
            return [self.process_file(path, row=row) for path, row in zip(paths, row_list)]

        strategy = str(getattr(self.config.asr, "input_audio_strategy", "single_decode"))
        if strategy != "single_decode":
            return [self.process_file(path, row=row) for path, row in zip(paths, row_list)]

        batch_metrics = self._new_perf_metrics()
        adapted_size = self._suggest_file_batch_size(len(paths), batch_metrics)
        if adapted_size < len(paths):
            out: list[Dict[str, Any]] = []
            for start in range(0, len(paths), adapted_size):
                out.extend(self.process_files_batch(paths[start:start + adapted_size], rows=row_list[start:start + adapted_size]))
            return out

        result_slots: list[Optional[Dict[str, Any]]] = [None] * len(paths)
        perf_by_index = [self._new_perf_metrics() for _ in paths]
        metas_by_index: list[Optional[dict]] = [None] * len(paths)
        total_seconds = 0.0
        estimate_unknown = False

        for idx, input_path in enumerate(paths):
            try:
                with self._time_perf_stage(perf_by_index[idx], "ffprobe"):
                    meta = ffprobe_audio(input_path, ffprobe_path=self.config.runtime.ffprobe_path)
                metas_by_index[idx] = meta
                duration = meta.get("duration_sec")
                if duration is None:
                    estimate_unknown = True
                else:
                    total_seconds += float(duration)
            except Exception as e:
                if self.config.runtime.fail_fast:
                    raise
                self._delete_partial_outputs_silent(input_path)
                result_slots[idx] = self._failed_result(input_path, repr(e))

        if estimate_unknown:
            estimated_gb = None
        else:
            bytes_needed = total_seconds * float(self.config.masking.output_sample_rate) * float(self.config.masking.output_channels) * 4.0
            estimated_gb = bytes_needed / (1024.0 ** 3)
        max_gb = float(getattr(self.config.runtime, "file_batch_max_decoded_audio_gb", 2.0))
        if estimated_gb is not None and estimated_gb > max_gb:
            logger.warning(
                "Batch decoded-audio estimate %.2f GB exceeds runtime.file_batch_max_decoded_audio_gb=%.2f. Falling back to per-file processing.",
                estimated_gb,
                max_gb,
            )
            for idx, (path, row) in enumerate(zip(paths, row_list)):
                if result_slots[idx] is None:
                    result_slots[idx] = self.process_file(path, row=row)
            return [r for r in result_slots if r is not None]

        contexts: dict[str, dict] = {}
        batch_asr_inputs: list[dict] = []
        max_single_decode = getattr(self.config.asr, "single_decode_max_audio_seconds", None)

        for idx, (input_path, row) in enumerate(zip(paths, row_list)):
            if result_slots[idx] is not None:
                continue
            started = time.time()
            perf_metrics = perf_by_index[idx]
            self._merge_perf_metrics(perf_metrics, batch_metrics)
            self._record_cuda_memory(perf_metrics, "file_batch_prepare_start")
            try:
                output_path = self.output_path_for(input_path)
                sidecar_path = self.sidecar_path_for(output_path)
                if input_path.resolve() == output_path.resolve():
                    raise ValueError(
                        f"Refusing to overwrite original audio. Configure paths.output_root outside the input file path: {input_path}"
                    )

                meta = metas_by_index[idx]
                if meta is None:
                    raise RuntimeError("Missing cached ffprobe metadata for batch item")
                if self._should_skip_existing(input_path, output_path, sidecar_path, meta):
                    result_slots[idx] = {
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "status": "skipped_existing",
                        "sidecar_path": str(sidecar_path),
                        "valid": True,
                    }
                    continue

                duration = meta.get("duration_sec")
                if self.config.asr.max_audio_seconds is not None and duration is not None:
                    if float(duration) > float(self.config.asr.max_audio_seconds):
                        raise ValueError(f"Audio duration {duration:.2f}s exceeds max_audio_seconds={self.config.asr.max_audio_seconds}")

                if max_single_decode is not None and duration is not None and float(duration) > float(max_single_decode):
                    result_slots[idx] = self.process_file(input_path, row=row)
                    continue

                n_input_channels = int(meta.get("channels") or 1)
                channel_count = min(2, max(1, n_input_channels))
                with self._time_perf_stage(perf_metrics, "decode"):
                    audio_48k = decode_to_float32_stereo_48k(
                        input_path,
                        ffmpeg_path=self.config.runtime.ffmpeg_path,
                        threads=self.config.runtime.ffmpeg_threads,
                        sample_rate=self.config.masking.output_sample_rate,
                        channels=self.config.masking.output_channels,
                    )
                with self._time_perf_stage(perf_metrics, "asr_input_prep"):
                    asr_inputs = make_asr_audio_inputs(
                        audio_48k,
                        mode=self.config.asr.mode,
                        input_channels=channel_count,
                        source_sr=self.config.masking.output_sample_rate,
                        target_sr=self.config.asr.channel_wav_sample_rate,
                    )

                file_id = str(idx)
                for item in asr_inputs:
                    item["file_id"] = file_id
                    batch_asr_inputs.append(item)

                contexts[file_id] = {
                    "index": idx,
                    "input_path": input_path,
                    "row": row,
                    "started": started,
                    "output_path": output_path,
                    "sidecar_path": sidecar_path,
                    "meta": meta,
                    "duration": duration,
                    "audio_48k": audio_48k,
                    "transcripts": [],
                    "all_entities": [],
                    "raw_spans": [],
                    "unmapped_fallback_used": False,
                    "used_single_decode": True,
                    "used_combined_channel_extract": False,
                    "perf_metrics": perf_metrics,
                }
            except Exception as e:
                if self.config.runtime.fail_fast:
                    raise
                self._delete_partial_outputs_silent(input_path)
                result_slots[idx] = self._failed_result(input_path, repr(e))

        if batch_asr_inputs:
            try:
                asr_batch_metrics = self._new_perf_metrics()
                self._record_cuda_memory(asr_batch_metrics, "before_asr_batch")
                with self._time_perf_stage(asr_batch_metrics, "asr_batch"):
                    bundles_by_file = self.asr.transcribe_channel_batch(
                        batch_asr_inputs,
                        work_dir=Path(self.config.paths.work_dir),
                        keep_temp=bool(self.config.runtime.keep_temp),
                    )
                self._record_cuda_memory(asr_batch_metrics, "after_asr_batch")
                if hasattr(self.asr, "last_engine_timings_sec") and asr_batch_metrics is not None:
                    asr_batch_metrics["asr_engine_sec"] = dict(getattr(self.asr, "last_engine_timings_sec", {}) or {})
                if asr_batch_metrics is not None:
                    asr_batch_metrics["batch_size"] = len(contexts)
                for ctx in contexts.values():
                    self._merge_batch_perf_metrics(ctx.get("perf_metrics"), asr_batch_metrics)
            except Exception as e:
                logger.warning("ASR micro-batch failed; falling back to file-by-file processing. Error: %s", e)
                for file_id, ctx in contexts.items():
                    idx = int(ctx["index"])
                    if result_slots[idx] is not None:
                        continue
                    try:
                        result_slots[idx] = self.process_file(ctx["input_path"], row=ctx.get("row"))
                    except Exception as e2:
                        if self.config.runtime.fail_fast:
                            raise
                        self._delete_partial_outputs_silent(ctx["input_path"])
                        result_slots[idx] = self._failed_result(ctx["input_path"], repr(e2))
                return [r for r in result_slots if r is not None]

            detection_items: list[dict] = []
            for file_id, ctx in contexts.items():
                for bundle in bundles_by_file.get(file_id, []):
                    ctx["transcripts"].append(self._bundle_to_transcript_dict(bundle))
                    for item in self._detection_items_for_bundle(bundle):
                        item["file_id"] = file_id
                        detection_items.append(item)

            if detection_items:
                texts = [d["text"] for d in detection_items]
                pii_rows = [contexts[str(d["file_id"])].get("row") for d in detection_items]
                pii_batch_metrics = self._new_perf_metrics()
                self._record_cuda_memory(pii_batch_metrics, "before_pii_batch")
                with self._time_perf_stage(pii_batch_metrics, "pii_detection_batch"):
                    detected_batches = self.detector.detect_batch(texts, rows=pii_rows)
                self._record_cuda_memory(pii_batch_metrics, "after_pii_batch")
                if pii_batch_metrics is not None:
                    pii_batch_metrics["batch_size"] = len(contexts)
                for ctx in contexts.values():
                    self._merge_batch_perf_metrics(ctx.get("perf_metrics"), pii_batch_metrics)
                for item, entities in zip(detection_items, detected_batches):
                    file_id = str(item["file_id"])
                    ctx = contexts[file_id]
                    enriched = []
                    for e in entities:
                        ent = dict(e)
                        ent["channel"] = int(item["channel"])
                        ent["transcript_source"] = item["source"]
                        ent["asr_engine"] = item.get("engine")
                        enriched.append(ent)
                        ctx["all_entities"].append(ent)
                    spans, used = self._map_entities_to_spans_conservative(
                        enriched,
                        item.get("words") or [],
                        int(item["channel"]),
                        ctx.get("duration"),
                    )
                    ctx["raw_spans"].extend(spans)
                    ctx["unmapped_fallback_used"] = bool(ctx["unmapped_fallback_used"] or used)

            for file_id, ctx in contexts.items():
                idx = int(ctx["index"])
                if result_slots[idx] is not None:
                    continue
                try:
                    result_slots[idx] = self._finalize_outputs(
                        input_path=ctx["input_path"],
                        output_path=ctx["output_path"],
                        sidecar_path=ctx["sidecar_path"],
                        started=ctx["started"],
                        meta=ctx["meta"],
                        duration=ctx.get("duration"),
                        audio_48k=ctx["audio_48k"],
                        transcripts=ctx["transcripts"],
                        all_entities=ctx["all_entities"],
                        raw_spans=ctx["raw_spans"],
                        used_single_decode=bool(ctx.get("used_single_decode", True)),
                        used_combined_channel_extract=bool(ctx.get("used_combined_channel_extract", False)),
                        unmapped_fallback_used=bool(ctx.get("unmapped_fallback_used", False)),
                        batch_optimized=True,
                        perf_metrics=ctx.get("perf_metrics"),
                    )
                    ctx["audio_48k"] = None
                except Exception as e:
                    if self.config.runtime.fail_fast:
                        raise
                    self._delete_partial_outputs_silent(ctx["input_path"])
                    result_slots[idx] = self._failed_result(ctx["input_path"], repr(e))

        return [r if r is not None else self._failed_result(paths[i], "internal_error: missing batch result") for i, r in enumerate(result_slots)]

    def _finalize_outputs(
        self,
        input_path: Path,
        output_path: Path,
        sidecar_path: Path,
        started: float,
        meta: dict,
        duration: Optional[float],
        audio_48k,
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        used_single_decode: bool,
        used_combined_channel_extract: bool,
        unmapped_fallback_used: bool,
        batch_optimized: bool,
        perf_metrics: Optional[dict] = None,
    ) -> Dict[str, Any]:
        no_pii_fast_copy_used = False

        # Safety fail-safe: an empty ASR result is not proof that there is no PII.
        # Without this, an ASR outage could incorrectly take the no-PII fast-copy path.
        if not self._has_any_transcript(transcripts):
            if duration is None:
                raise RuntimeError("ASR produced no transcript and input duration is unavailable; refusing to copy unmasked audio")
            raw_spans = self._full_audio_spans(duration, channels=int(meta.get("channels") or self.config.masking.output_channels))
            all_entities = list(all_entities) + [{
                "text": "",
                "type": "EMPTY_TRANSCRIPT_FAILSAFE",
                "start": 0,
                "end": 0,
                "source": "pipeline",
                "score": 1.0,
                "reason": "ASR produced no transcript; full-audio masking applied to avoid unmasked PII leakage",
            }]
            unmapped_fallback_used = True

        if all_entities and not raw_spans:
            raw_spans, unmapped_fallback_used = self._apply_unmapped_entity_policy(all_entities, duration)

        merged_spans = merge_spans(
            raw_spans,
            merge_gap_sec=self.config.masking.merge_gap_sec,
            target_channels=self.config.masking.target_channels,
        )

        with self._time_perf_stage(perf_metrics, "mask_copy_encode"):
            if not merged_spans and self.config.runtime.copy_unmasked_when_no_pii and self.config.masking.copy_input_if_no_pii:
                if input_matches_required_opus(
                    meta,
                    sample_rate=self.config.masking.output_sample_rate,
                    channels=self.config.masking.output_channels,
                ):
                    copy_audio_passthrough(input_path, output_path, method=self.config.runtime.unmasked_copy_method)
                    no_pii_fast_copy_used = True
                else:
                    if audio_48k is None:
                        audio_48k = decode_to_float32_stereo_48k(
                            input_path,
                            ffmpeg_path=self.config.runtime.ffmpeg_path,
                            threads=self.config.runtime.ffmpeg_threads,
                            sample_rate=self.config.masking.output_sample_rate,
                            channels=self.config.masking.output_channels,
                        )
                    self._encode(audio_48k, output_path, meta)
                status = "success_no_pii_fast_copy" if no_pii_fast_copy_used else "success_no_pii_transcoded"
            else:
                if audio_48k is None:
                    audio_48k = decode_to_float32_stereo_48k(
                        input_path,
                        ffmpeg_path=self.config.runtime.ffmpeg_path,
                        threads=self.config.runtime.ffmpeg_threads,
                        sample_rate=self.config.masking.output_sample_rate,
                        channels=self.config.masking.output_channels,
                    )
                masked = apply_mask_spans(
                    audio_48k,
                    merged_spans,
                    sr=self.config.masking.output_sample_rate,
                    mode=self.config.masking.mode,
                    target_channels=self.config.masking.target_channels,
                    beep_freq_hz=self.config.masking.beep_freq_hz,
                    beep_gain=self.config.masking.beep_gain,
                    noise_gain=self.config.masking.noise_gain,
                    fade_ms=self.config.masking.fade_ms,
                    random_seed=self.config.runtime.random_seed,
                    inplace=True,
                )
                self._encode(masked, output_path, meta)
                status = "success_unmapped_fallback" if unmapped_fallback_used else "success"

        with self._time_perf_stage(perf_metrics, "validation"):
            if self.config.runtime.validate_outputs:
                validation = validate_masked_file(
                    input_path,
                    output_path,
                    ffprobe_path=self.config.runtime.ffprobe_path,
                    expected_sample_rate=self.config.masking.output_sample_rate,
                    expected_channels=self.config.masking.output_channels,
                    input_meta=meta,
                )
            else:
                validation = {"valid": True, "validation_skipped": True, "checks": {}}

        elapsed = time.time() - float(started)
        sidecar = self._build_sidecar(
            input_path=input_path,
            output_path=output_path,
            meta=meta,
            elapsed=elapsed,
            transcripts=transcripts,
            all_entities=all_entities,
            raw_spans=raw_spans,
            merged_spans=merged_spans,
            validation=validation,
            status=status,
            used_single_decode=used_single_decode,
            used_combined_channel_extract=used_combined_channel_extract,
            unmapped_fallback_used=unmapped_fallback_used,
            no_pii_fast_copy_used=no_pii_fast_copy_used,
            perf_metrics=perf_metrics,
        )
        if batch_optimized:
            sidecar["optimizations"]["asr_file_microbatching"] = True
            sidecar["optimizations"]["pii_cross_file_batching"] = True
            sidecar["runtime_file_batch_size"] = int(getattr(self.config.runtime, "file_batch_size", 1))
        with self._time_perf_stage(perf_metrics, "sidecar_write"):
            write_json(sidecar_path, sidecar)
        return self._result_row(input_path, output_path, sidecar_path, elapsed, duration, transcripts, all_entities, merged_spans, validation, status, perf_metrics=perf_metrics)

    def _estimate_decoded_batch_gb(self, paths: Sequence[Path]) -> Optional[float]:
        total_seconds = 0.0
        for input_path in paths:
            try:
                meta = ffprobe_audio(input_path, ffprobe_path=self.config.runtime.ffprobe_path)
                duration = meta.get("duration_sec")
                if duration is None:
                    return None
                total_seconds += float(duration)
            except Exception:
                return None
        bytes_needed = total_seconds * float(self.config.masking.output_sample_rate) * float(self.config.masking.output_channels) * 4.0
        return bytes_needed / (1024.0 ** 3)

    def _delete_partial_outputs_silent(self, input_path: str | Path) -> None:
        if not getattr(self.config.runtime, "delete_failed_partial_outputs", True):
            return
        try:
            out = self.output_path_for(input_path)
            sidecar = self.sidecar_path_for(out)
            for partial in (out, sidecar):
                if Path(partial).exists():
                    Path(partial).unlink()
        except Exception:
            pass

    def _failed_result(self, input_path: str | Path, error: str) -> Dict[str, Any]:
        return {"input_path": str(input_path), "status": "failed", "error": error}

    def can_resume_skip(self, input_path: str | Path, output_path: str | Path, sidecar_path: str | Path) -> bool:
        input_path = Path(input_path)
        output_path = Path(output_path)
        sidecar_path = Path(sidecar_path)
        try:
            input_meta = ffprobe_audio(input_path, ffprobe_path=self.config.runtime.ffprobe_path)
        except Exception:
            return False
        return self._should_skip_existing(input_path, output_path, sidecar_path, input_meta)

    def _should_skip_existing(self, input_path: Path, output_path: Path, sidecar_path: Path, input_meta: dict) -> bool:
        if not (self.config.runtime.resume and output_path.exists() and sidecar_path.exists()):
            return False
        if not getattr(self.config.runtime, "validate_existing_outputs", True) or not self.config.runtime.validate_outputs:
            logger.info("Skipping existing output: %s", output_path)
            return True
        validation = validate_masked_file(
            input_path,
            output_path,
            ffprobe_path=self.config.runtime.ffprobe_path,
            expected_sample_rate=self.config.masking.output_sample_rate,
            expected_channels=self.config.masking.output_channels,
            input_meta=input_meta,
        )
        if validation.get("valid"):
            logger.info("Skipping valid existing output: %s", output_path)
            return True
        logger.warning("Existing output failed validation and will be regenerated: %s", output_path)
        return False

    def _transcribe_asr_inputs(self, asr_inputs: Sequence[dict]) -> list[ASRResult] | list[ChannelASRBundle]:
        if hasattr(self.asr, "transcribe_channels"):
            return self.asr.transcribe_channels(
                asr_inputs,
                work_dir=Path(self.config.paths.work_dir),
                keep_temp=bool(self.config.runtime.keep_temp),
            )

        results: list[ASRResult] = []
        temp_ctx = None
        temp_dir: Optional[Path] = None
        try:
            for item in asr_inputs:
                channel = int(item["channel"])
                sample_rate = int(item["sample_rate"])
                audio = item["audio"]
                try:
                    results.append(self.asr.transcribe_audio(audio, channel=channel, sample_rate=sample_rate))
                except Exception as e:
                    if temp_ctx is None:
                        temp_parent = Path(self.config.paths.work_dir) / "tmp"
                        temp_parent.mkdir(parents=True, exist_ok=True)
                        temp_ctx = tempfile.TemporaryDirectory(prefix="pii_asr_numpy_fallback_", dir=str(temp_parent))
                        temp_dir = Path(temp_ctx.name)
                    assert temp_dir is not None
                    wav_path = temp_dir / f"channel_{channel}.wav"
                    logger.warning("In-memory ASR failed for channel=%s. Using temp WAV fallback. Error: %s", channel, e)
                    write_mono_wav_pcm16(audio, wav_path, sample_rate=sample_rate)
                    results.append(self.asr.transcribe_path(wav_path, channel=channel))
        finally:
            if temp_ctx is not None and not self.config.runtime.keep_temp:
                temp_ctx.cleanup()
        return results

    def _process_with_temp_wavs(
        self,
        input_path: Path,
        channel_count: int,
        row: Optional[dict],
        duration: Optional[float],
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
    ) -> tuple[bool, bool]:
        temp_parent = Path(self.config.paths.work_dir) / "tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        used_combined = False
        with tempfile.TemporaryDirectory(prefix="pii_mask_", dir=str(temp_parent)) as td:
            temp_dir = Path(td)
            results: list[ASRResult] = []
            if self.config.asr.mode == "mono_mix":
                wav_path = temp_dir / "mono_mix.wav"
                extract_mono_mix_wav(
                    input_path,
                    wav_path,
                    sample_rate=self.config.asr.channel_wav_sample_rate,
                    ffmpeg_path=self.config.runtime.ffmpeg_path,
                    threads=self.config.runtime.ffmpeg_threads,
                )
                if hasattr(self.asr, "transcribe_channels"):
                    import numpy as np
                    with __import__("wave").open(str(wav_path), "rb") as wf:
                        data = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    asr_inputs = [{"channel": -1, "audio": audio, "sample_rate": self.config.asr.channel_wav_sample_rate}]
                    bundles_or_results = self._transcribe_asr_inputs(asr_inputs)
                    fallback_used = self._handle_asr_results(bundles_or_results, transcripts, all_entities, raw_spans, row, duration)
                    return used_combined, bool(fallback_used)
                results.append(self.asr.transcribe_path(wav_path, channel=-1))
            elif self.config.asr.mode == "per_channel":
                channel_wavs: dict[int, Path] = {}
                if channel_count == 2:
                    ch0 = temp_dir / "ch0.wav"
                    ch1 = temp_dir / "ch1.wav"
                    try:
                        extract_stereo_channels_wav(
                            input_path,
                            ch0,
                            ch1,
                            sample_rate=self.config.asr.channel_wav_sample_rate,
                            ffmpeg_path=self.config.runtime.ffmpeg_path,
                            threads=self.config.runtime.ffmpeg_threads,
                        )
                        channel_wavs = {0: ch0, 1: ch1}
                        used_combined = True
                    except Exception as e:
                        logger.warning("Combined stereo channel extraction failed, using per-channel extraction: %s", e)
                for channel in range(channel_count):
                    wav_path = channel_wavs.get(channel)
                    if wav_path is None:
                        wav_path = temp_dir / f"ch{channel}.wav"
                        extract_channel_wav(
                            input_path,
                            wav_path,
                            channel=channel,
                            sample_rate=self.config.asr.channel_wav_sample_rate,
                            ffmpeg_path=self.config.runtime.ffmpeg_path,
                            threads=self.config.runtime.ffmpeg_threads,
                        )
                        channel_wavs[channel] = wav_path
                if hasattr(self.asr, "transcribe_channels"):
                    import numpy as np
                    asr_inputs = []
                    for channel, wav_path in sorted(channel_wavs.items()):
                        with __import__("wave").open(str(wav_path), "rb") as wf:
                            data = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        asr_inputs.append({"channel": int(channel), "audio": audio, "sample_rate": self.config.asr.channel_wav_sample_rate})
                    bundles_or_results = self._transcribe_asr_inputs(asr_inputs)
                    fallback_used = self._handle_asr_results(bundles_or_results, transcripts, all_entities, raw_spans, row, duration)
                    return used_combined, bool(fallback_used)
                for channel, wav_path in sorted(channel_wavs.items()):
                    results.append(self.asr.transcribe_path(wav_path, channel=channel))
            else:
                raise ValueError(f"Unsupported ASR mode: {self.config.asr.mode}")
            fallback_used = self._handle_asr_results(results, transcripts, all_entities, raw_spans, row, duration)
        return used_combined, bool(fallback_used)

    def _handle_asr_results(
        self,
        results: Sequence[ASRResult] | Sequence[ChannelASRBundle],
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        row: Optional[dict],
        duration: Optional[float],
    ) -> bool:
        if not results:
            return False
        first = results[0]
        if isinstance(first, ChannelASRBundle) or hasattr(first, "engine_results"):
            return self._handle_asr_bundles(results, transcripts, all_entities, raw_spans, row, duration)  # type: ignore[arg-type]

        fallback_used = False
        transcripts.extend(self._asr_result_to_dict(r) for r in results)  # type: ignore[arg-type]
        detected = self.detector.detect_batch([r.transcript for r in results], rows=[row for _ in results])  # type: ignore[attr-defined]
        for result, entities in zip(results, detected):  # type: ignore[assignment]
            enriched = []
            for e in entities:
                ent = dict(e)
                ent["channel"] = result.channel
                ent["transcript_source"] = result.engine
                ent["asr_engine"] = result.engine
                enriched.append(ent)
                all_entities.append(ent)
            spans, used = self._map_entities_to_spans_conservative(enriched, result.words, result.channel, duration)
            raw_spans.extend(spans)
            fallback_used = fallback_used or used
        return fallback_used

    def _handle_asr_bundles(
        self,
        bundles: Sequence[ChannelASRBundle],
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        row: Optional[dict],
        duration: Optional[float],
    ) -> bool:
        fallback_used = False
        detection_items: list[dict] = []

        for bundle in bundles:
            transcripts.append(self._bundle_to_transcript_dict(bundle))
            detection_items.extend(self._detection_items_for_bundle(bundle))

        texts = [d["text"] for d in detection_items]
        if not texts:
            return False
        detected_batches = self.detector.detect_batch(texts, rows=[row for _ in texts])
        for item, entities in zip(detection_items, detected_batches):
            enriched = []
            for e in entities:
                ent = dict(e)
                ent["channel"] = int(item["channel"])
                ent["transcript_source"] = item["source"]
                ent["asr_engine"] = item.get("engine")
                enriched.append(ent)
                all_entities.append(ent)
            spans, used = self._map_entities_to_spans_conservative(enriched, item.get("words") or [], int(item["channel"]), duration)
            raw_spans.extend(spans)
            fallback_used = fallback_used or used
        return fallback_used

    def _detection_items_for_bundle(self, bundle: ChannelASRBundle) -> list[dict]:
        scope = str(getattr(self.config.asr, "pii_detection_transcript_scope", "final_and_all_engines"))
        items: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(source: str, engine: Optional[str], text: str, words: list[dict]):
            text = str(text or "").strip()
            if not text:
                return
            key = (source, text.lower())
            if key in seen:
                return
            seen.add(key)
            items.append({"channel": bundle.channel, "source": source, "engine": engine, "text": text, "words": words})

        if scope in {"final_only", "final_and_all_engines"}:
            add("final_consensus", bundle.consensus.get("selected_engine"), bundle.final_transcript, bundle.final_words)

        if scope in {"all_engines_only", "final_and_all_engines"}:
            for result in bundle.engine_results:
                if not result.transcript or result.error:
                    continue
                if result.words:
                    words = result.words
                elif result.engine == bundle.anchor_engine:
                    words = bundle.anchor_words
                else:
                    words = align_transcript_to_timed_words(result.transcript, bundle.anchor_words, channel=bundle.channel)
                add(f"engine:{result.engine}", result.engine, result.transcript, words)
        return items

    def _map_entities_to_spans_conservative(
        self,
        entities: list[dict],
        words: list[dict],
        channel: int,
        duration: Optional[float],
    ) -> tuple[list[dict], bool]:
        spans: list[dict] = []
        fallback_used = False
        for ent in entities:
            mapped = entities_to_spans(
                [ent],
                words,
                channel=channel,
                pad_sec=self.config.masking.pad_sec,
                min_duration_sec=self.config.masking.min_duration_sec,
                audio_duration_sec=duration,
            )
            if mapped:
                spans.extend(mapped)
                continue
            fallback, used = self._apply_unmapped_entity_policy([ent], duration)
            spans.extend(fallback)
            fallback_used = fallback_used or used
        return spans, fallback_used

    def _has_any_transcript(self, transcripts: list[dict]) -> bool:
        for row in transcripts or []:
            if str(row.get("transcript") or row.get("final_transcript") or "").strip():
                return True
            engine_transcripts = row.get("engine_transcripts") or {}
            if isinstance(engine_transcripts, dict):
                for value in engine_transcripts.values():
                    if str(value or "").strip():
                        return True
        return False

    def _full_audio_spans(self, duration: float, channels: int = 2) -> list[dict]:
        n_channels = max(1, min(int(channels or 2), int(self.config.masking.output_channels or 2)))
        return [
            {
                "start": 0.0,
                "end": max(0.0, float(duration)),
                "duration": max(0.0, float(duration)),
                "channel": channel,
                "source": "empty_transcript_failsafe",
                "type": "EMPTY_TRANSCRIPT_FAILSAFE",
                "text": "",
            }
            for channel in range(n_channels)
        ]

    def _apply_unmapped_entity_policy(self, all_entities: list[dict], duration: Optional[float]) -> tuple[list[dict], bool]:
        policy = self.config.masking.unmapped_entity_policy
        if policy == "copy_original":
            logger.warning("PII entities detected but no timestamp spans mapped; copy_original policy leaves audio unchanged.")
            return [], False
        if policy == "fail":
            raise RuntimeError("PII entities were detected but none could be mapped to timestamps.")
        if policy != "mask_full_channel":
            raise ValueError(f"Unsupported unmapped_entity_policy: {policy}")
        if duration is None:
            raise RuntimeError("Cannot mask full channel for unmapped PII because input duration is unavailable.")

        spans: list[dict] = []
        channels = sorted({int(e.get("channel", -1)) for e in all_entities})
        for ch in channels:
            texts = [str(e.get("text", "")) for e in all_entities if int(e.get("channel", -1)) == ch]
            spans.append({
                "channel": ch,
                "start": 0.0,
                "end": float(duration),
                "duration": float(duration),
                "type": "UNMAPPED_PII_FALLBACK",
                "text": " | ".join(t for t in texts if t)[:500],
                "source": "unmapped_entity_policy",
                "score": 1.0,
            })
        return spans, True

    def _encode(self, audio, output_path: Path, input_meta: dict) -> None:
        bitrate = self.config.masking.opus_bitrate
        if self.config.masking.preserve_input_bitrate and input_meta.get("bit_rate"):
            br = int(input_meta["bit_rate"])
            if br > 0:
                bitrate = f"{max(16, round(br / 1000))}k"
        encode_float32_stereo_to_opus(
            audio,
            output_path,
            ffmpeg_path=self.config.runtime.ffmpeg_path,
            threads=self.config.runtime.ffmpeg_threads,
            sample_rate=self.config.masking.output_sample_rate,
            channels=self.config.masking.output_channels,
            bitrate=bitrate,
            application=self.config.masking.opus_application,
            vbr=self.config.masking.opus_vbr,
            compression_level=self.config.masking.opus_compression_level,
            frame_duration_ms=self.config.masking.opus_frame_duration_ms,
            atomic=self.config.runtime.atomic_output,
        )

    def _build_sidecar(
        self,
        input_path: Path,
        output_path: Path,
        meta: dict,
        elapsed: float,
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        merged_spans: list[dict],
        validation: dict,
        status: str,
        used_single_decode: bool,
        used_combined_channel_extract: bool,
        unmapped_fallback_used: bool,
        no_pii_fast_copy_used: bool,
        perf_metrics: Optional[dict] = None,
    ) -> dict:
        sidecar = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": status,
            "input_meta": meta,
            "elapsed_sec": elapsed,
            "asr_mode": self.config.asr.mode,
            "asr_input_audio_strategy": self.config.asr.input_audio_strategy,
            "asr_engines_enabled": [name for name in self.config.asr.engine_order if self.config.asr.engines.get(name, {}).get("enabled", True)],
            "asr_timestamp_anchor_engine": self.config.asr.timestamp_anchor_engine,
            "asr_pii_detection_transcript_scope": self.config.asr.pii_detection_transcript_scope,
            "asr_model_residency": self.config.asr.model_residency,
            "mask_mode": self.config.masking.mode,
            "target_channels": self.config.masking.target_channels,
            "optimizations": {
                "no_forced_alignment": True,
                "multi_asr_consensus": True,
                "detect_pii_on_final_and_engine_transcripts": self.config.asr.pii_detection_transcript_scope == "final_and_all_engines",
                "word_timestamp_anchor": self.config.asr.timestamp_anchor_engine,
                "word_timestamp_mapping": True,
                "single_decode_reused_for_asr_and_masking": used_single_decode,
                "combined_stereo_channel_extract": used_combined_channel_extract,
                "neural_pii_batching": True,
                "long_text_chunking": True,
                "no_pii_fast_copy": no_pii_fast_copy_used,
                "inplace_masking": True,
                "atomic_output": self.config.runtime.atomic_output,
            },
            "unmapped_entity_policy": self.config.masking.unmapped_entity_policy,
            "unmapped_fallback_used": unmapped_fallback_used,
            "transcripts": transcripts,
            "entities": all_entities,
            "raw_spans": raw_spans,
            "merged_spans": merged_spans,
            "validation": validation,
        }
        if perf_metrics is not None:
            sidecar["perf_metrics"] = perf_metrics
        return sidecar

    def _result_row(
        self,
        input_path: Path,
        output_path: Path,
        sidecar_path: Path,
        elapsed: float,
        duration: Optional[float],
        transcripts: list[dict],
        all_entities: list[dict],
        merged_spans: list[dict],
        validation: dict,
        status: str,
        perf_metrics: Optional[dict] = None,
    ) -> Dict[str, Any]:
        row = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "status": status,
            "elapsed_sec": elapsed,
            "duration_sec": duration,
            "num_words": sum(int(t.get("word_count", 0)) for t in transcripts),
            "num_entities": len(all_entities),
            "num_spans": len(merged_spans),
            "valid": validation.get("valid"),
        }
        if perf_metrics is not None:
            row["perf_metrics_json"] = json.dumps(perf_metrics, sort_keys=True)
        return row

    def _bundle_to_transcript_dict(self, bundle: ChannelASRBundle) -> Dict[str, Any]:
        engine_rows = [self._asr_result_to_dict(r) for r in bundle.engine_results]
        d: Dict[str, Any] = {
            "file_id": getattr(bundle, "file_id", None),
            "channel": bundle.channel,
            "engine": "consensus",
            "transcript": bundle.final_transcript,
            "final_transcript": bundle.final_transcript,
            "word_count": len(bundle.final_words),
            "anchor_engine": bundle.anchor_engine,
            "anchor_word_count": len(bundle.anchor_words),
            "consensus": bundle.consensus,
            "engine_transcripts": {r.engine: r.transcript for r in bundle.engine_results},
            "engine_errors": {r.engine: r.error for r in bundle.engine_results if r.error},
            "engine_results": engine_rows,
        }
        if self.config.runtime.sidecar_include_words:
            d["words"] = bundle.final_words
            d["anchor_words"] = bundle.anchor_words
        return d

    def _asr_result_to_dict(self, result: ASRResult) -> Dict[str, Any]:
        d = {
            "file_id": getattr(result, "file_id", None),
            "channel": result.channel,
            "engine": result.engine,
            "transcript": result.transcript,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "word_count": len(result.words),
            "timestamp_retry_used": result.timestamp_retry_used,
            "timestamp_suspicious": result.timestamp_suspicious,
            "error": result.error,
        }
        if self.config.runtime.sidecar_include_words:
            d["words"] = result.words
        return d
