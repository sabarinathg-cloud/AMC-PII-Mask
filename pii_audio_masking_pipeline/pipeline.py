from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import hashlib
import importlib.util
import json
import logging
import tempfile
import time
import wave

import numpy as np

from .asr import ASRResult, ChannelASRBundle, MultiASRTranscriber, align_transcript_to_timed_words, bundle_asr_results_by_file
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
from .forced_alignment import AlignmentResult, WhisperXForcedAligner
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
        if bool(getattr(config.runtime, "sidecar_include_words", False)):
            logger.warning(
                "runtime.sidecar_include_words=true stores word-level transcripts and confidence in sidecar JSON. "
                "Treat sidecars as sensitive PII/PHI artifacts."
            )
        self.asr = MultiASRTranscriber(config.asr)
        self.forced_aligner = self._create_forced_aligner()
        self.detector = PIIDetector(config.pii)

    def _create_forced_aligner(self):
        align_cfg = getattr(self.config, "alignment", None)
        if align_cfg is None or not bool(getattr(align_cfg, "enabled", False)):
            return None
        if str(getattr(align_cfg, "backend", "whisperx")) != "whisperx":
            raise ValueError(f"Unsupported alignment backend: {getattr(align_cfg, 'backend', None)}")
        if importlib.util.find_spec("whisperx") is None:
            raise ImportError(
                "alignment.enabled=true requires the optional whisperx package. "
                "Install it with `python3 -m pip install whisperx` or set alignment.enabled=false."
            )
        return WhisperXForcedAligner(
            device=str(getattr(align_cfg, "device", "auto")),
            compute_type=str(getattr(align_cfg, "compute_type", "float16")),
            batch_size=int(getattr(align_cfg, "batch_size", 16)),
            default_language=str(getattr(align_cfg, "default_language", "en")),
        )

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
                unmapped_fallback_used = self._handle_asr_results(
                    results,
                    transcripts,
                    all_entities,
                    raw_spans,
                    row,
                    duration,
                    alignment_audio_by_channel=self._alignment_audio_by_channel(asr_inputs),
                ) or unmapped_fallback_used
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

    def process_file_from_asr_results(
        self,
        input_path: str | Path,
        asr_results: Sequence[ASRResult],
        row: Optional[dict] = None,
        meta: Optional[dict] = None,
    ) -> Dict[str, Any]:
        input_path = Path(input_path)
        started = time.time()
        perf_metrics = self._new_perf_metrics()
        output_path = self.output_path_for(input_path)
        sidecar_path = self.sidecar_path_for(output_path)

        if meta is None:
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

        transcripts: list[dict] = []
        all_entities: list[dict] = []
        raw_spans: list[dict] = []
        bundles = self._bundles_from_cached_asr_results(asr_results)
        if not (meta or {}).get("channels"):
            inferred_channels = self._infer_channel_count_from_asr_results(asr_results)
            meta = dict(meta or {})
            meta["channels"] = inferred_channels
        alignment_audio_by_channel = self._decode_alignment_audio(input_path, meta)

        with self._time_perf_stage(perf_metrics, "pii_detection_and_mapping"):
            unmapped_fallback_used = self._handle_asr_bundles(
                bundles,
                transcripts,
                all_entities,
                raw_spans,
                row,
                duration,
                alignment_audio_by_channel=alignment_audio_by_channel,
            )

        return self._finalize_outputs(
            input_path=input_path,
            output_path=output_path,
            sidecar_path=sidecar_path,
            started=started,
            meta=meta,
            duration=duration,
            audio_48k=None,
            transcripts=transcripts,
            all_entities=all_entities,
            raw_spans=raw_spans,
            used_single_decode=False,
            used_combined_channel_extract=False,
            unmapped_fallback_used=bool(unmapped_fallback_used),
            batch_optimized=False,
            model_major_optimized=True,
            perf_metrics=perf_metrics,
        )

    def _bundles_from_cached_asr_results(self, asr_results: Sequence[ASRResult]) -> list[ChannelASRBundle]:
        if not asr_results:
            return []
        anchor_name = str(getattr(self.config.asr, "timestamp_anchor_engine", "whisper"))
        by_channel: dict[int, list[ASRResult]] = {}
        for result in asr_results:
            by_channel.setdefault(int(result.channel), []).append(result)

        for channel, rows in by_channel.items():
            anchor = next((r for r in rows if r.engine == anchor_name and r.words and not r.error), None)
            if anchor is None:
                logger.warning(
                    "Missing timestamp anchor words for cached ASR channel=%s. "
                    "Model-major finalization will use full-audio masking failsafe.",
                    channel,
                )
                return []

        asr_inputs = [
            {"file_id": str(result.file_id or self._cached_file_id(asr_results)), "channel": int(channel)}
            for channel, rows in sorted(by_channel.items())
            for result in rows[:1]
        ]
        grouped = bundle_asr_results_by_file(asr_results, asr_inputs, self.config.asr)
        file_id = str(asr_inputs[0]["file_id"]) if asr_inputs else self._cached_file_id(asr_results)
        return grouped.get(file_id, [])

    def _cached_file_id(self, asr_results: Sequence[ASRResult]) -> str:
        for result in asr_results:
            if result.file_id:
                return str(result.file_id)
        return "__cached__"

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
                    "alignment_audio_by_channel": self._alignment_audio_by_channel(asr_inputs),
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
                    transcript_row, bundle_items = self._prepare_bundle_for_detection(
                        bundle,
                        alignment_audio_by_channel=ctx.get("alignment_audio_by_channel"),
                    )
                    ctx["transcripts"].append(transcript_row)
                    for item in bundle_items:
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
                        ent = self._enrich_entity(
                            e,
                            channel=int(item["channel"]),
                            transcript_source=item["source"],
                            asr_engine=item.get("engine"),
                            existing_count=len(ctx["all_entities"]) + len(enriched),
                        )
                        ent["timestamp_source"] = item.get("timestamp_source")
                        ent["alignment_backend"] = item.get("alignment_backend")
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
        model_major_optimized: bool = False,
        perf_metrics: Optional[dict] = None,
    ) -> Dict[str, Any]:
        no_pii_fast_copy_used = False

        # Safety fail-safe: an empty ASR result is not proof that there is no PII.
        # Without this, an ASR outage could incorrectly take the no-PII fast-copy path.
        if not self._has_any_transcript(transcripts):
            if duration is None:
                raise RuntimeError("ASR produced no transcript and input duration is unavailable; refusing to copy unmasked audio")
            sentinel_id = f"ent_{len(all_entities) + 1:06d}"
            raw_spans = self._full_audio_spans(
                duration,
                channels=int(meta.get("channels") or self.config.masking.output_channels),
                entity_ids=[sentinel_id],
            )
            all_entities = list(all_entities) + [{
                "entity_id": sentinel_id,
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
        if model_major_optimized:
            sidecar["optimizations"]["model_major_schedule"] = True
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
                    audio = self._read_wav_float32(wav_path)
                    asr_inputs = [{"channel": -1, "audio": audio, "sample_rate": self.config.asr.channel_wav_sample_rate}]
                    bundles_or_results = self._transcribe_asr_inputs(asr_inputs)
                    fallback_used = self._handle_asr_results(
                        bundles_or_results,
                        transcripts,
                        all_entities,
                        raw_spans,
                        row,
                        duration,
                        alignment_audio_by_channel=self._alignment_audio_by_channel(asr_inputs),
                    )
                    return used_combined, bool(fallback_used)
                mono_audio = self._read_wav_float32(wav_path)
                alignment_audio_by_channel = (
                    {-1: {"audio": mono_audio, "sample_rate": self.config.asr.channel_wav_sample_rate}}
                    if self._alignment_enabled()
                    else {}
                )
                results.append(self.asr.transcribe_path(wav_path, channel=-1))
            elif self.config.asr.mode == "per_channel":
                channel_wavs: dict[int, Path] = {}
                alignment_audio_by_channel = {}
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
                    asr_inputs = []
                    for channel, wav_path in sorted(channel_wavs.items()):
                        audio = self._read_wav_float32(wav_path)
                        asr_inputs.append({"channel": int(channel), "audio": audio, "sample_rate": self.config.asr.channel_wav_sample_rate})
                    bundles_or_results = self._transcribe_asr_inputs(asr_inputs)
                    fallback_used = self._handle_asr_results(
                        bundles_or_results,
                        transcripts,
                        all_entities,
                        raw_spans,
                        row,
                        duration,
                        alignment_audio_by_channel=self._alignment_audio_by_channel(asr_inputs),
                    )
                    return used_combined, bool(fallback_used)
                for channel, wav_path in sorted(channel_wavs.items()):
                    if self._alignment_enabled():
                        alignment_audio_by_channel[int(channel)] = {
                            "audio": self._read_wav_float32(wav_path),
                            "sample_rate": self.config.asr.channel_wav_sample_rate,
                        }
                    results.append(self.asr.transcribe_path(wav_path, channel=channel))
            else:
                raise ValueError(f"Unsupported ASR mode: {self.config.asr.mode}")
            fallback_used = self._handle_asr_results(
                results,
                transcripts,
                all_entities,
                raw_spans,
                row,
                duration,
                alignment_audio_by_channel=alignment_audio_by_channel,
            )
        return used_combined, bool(fallback_used)

    @staticmethod
    def _read_wav_float32(wav_path: Path) -> np.ndarray:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getsampwidth() != 2:
                raise ValueError(f"Alignment WAV must be 16-bit PCM; got sample width {wf.getsampwidth()} bytes")
            data = wf.readframes(wf.getnframes())
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _infer_channel_count_from_asr_results(asr_results: Sequence[ASRResult]) -> int:
        channels = [int(result.channel) for result in asr_results or [] if int(result.channel) >= 0]
        if not channels:
            return 1
        return min(2, max(1, max(channels) + 1))

    def _alignment_enabled(self) -> bool:
        return bool(getattr(getattr(self.config, "alignment", None), "enabled", False)) and getattr(self, "forced_aligner", None) is not None

    @staticmethod
    def _alignment_audio_by_channel(asr_inputs: Sequence[dict]) -> dict[int, dict]:
        return {
            int(item["channel"]): {
                "audio": item["audio"],
                "sample_rate": int(item.get("sample_rate") or 16000),
            }
            for item in asr_inputs or []
            if "channel" in item and "audio" in item
        }

    def _decode_alignment_audio(self, input_path: Path, meta: Optional[dict]) -> dict[int, dict]:
        if not self._alignment_enabled():
            return {}
        try:
            n_input_channels = int((meta or {}).get("channels") or 1)
            channel_count = min(2, max(1, n_input_channels))
            audio_48k = decode_to_float32_stereo_48k(
                input_path,
                ffmpeg_path=self.config.runtime.ffmpeg_path,
                threads=self.config.runtime.ffmpeg_threads,
                sample_rate=self.config.masking.output_sample_rate,
                channels=self.config.masking.output_channels,
            )
            asr_inputs = make_asr_audio_inputs(
                audio_48k,
                mode=self.config.asr.mode,
                input_channels=channel_count,
                source_sr=self.config.masking.output_sample_rate,
                target_sr=self.config.asr.channel_wav_sample_rate,
            )
            return self._alignment_audio_by_channel(asr_inputs)
        except Exception as exc:
            logger.warning(
                "Could not prepare audio for forced alignment; using configured fallback policy. Error: %s",
                self._safe_alignment_error(exc),
            )
            return {}

    def _align_detection_item(self, item: dict, alignment_audio_by_channel: Optional[dict[int, dict]]) -> dict:
        if not self._alignment_enabled():
            metadata = self._alignment_metadata(status="disabled", result=None, error=None)
            item["alignment"] = metadata
            item["timestamp_source"] = "asr_words"
            return metadata

        audio_info = (alignment_audio_by_channel or {}).get(int(item.get("channel", -999)))
        if not audio_info:
            return self._apply_alignment_failure(item, status="missing_audio", result=None, error="alignment audio unavailable")

        try:
            result = self.forced_aligner.align(
                audio=audio_info["audio"],
                sample_rate=int(audio_info.get("sample_rate") or self.config.asr.channel_wav_sample_rate),
                transcript=str(item.get("text") or ""),
                language=item.get("language"),
                channel=int(item["channel"]),
            )
        except Exception as exc:
            return self._apply_alignment_failure(item, status="error", result=None, error=self._safe_alignment_error(exc))

        min_ratio = float(getattr(self.config.alignment, "min_aligned_words_ratio", 0.70))
        if result.words and float(result.coverage) >= min_ratio:
            item["words"] = result.words
            item["timestamp_source"] = "forced_alignment"
            item["alignment_backend"] = result.backend
            metadata = self._alignment_metadata(status="aligned", result=result, error=None)
            item["alignment"] = metadata
            return metadata
        status = "low_coverage" if float(result.coverage) < min_ratio else str(result.status or "unaligned")
        return self._apply_alignment_failure(item, status=status, result=result, error=result.error)

    def _apply_alignment_failure(self, item: dict, status: str, result: Optional[AlignmentResult], error: Optional[str]) -> dict:
        policy = str(getattr(self.config.alignment, "on_failure", "fallback_full_channel"))
        if policy == "fail":
            raise RuntimeError(f"Forced alignment failed for channel={item.get('channel')} source={item.get('source')}: {status}")
        if policy == "use_asr_words":
            if not item.get("words"):
                item["timestamp_source"] = "forced_alignment_fallback_full_channel"
                item["alignment_backend"] = getattr(self.forced_aligner, "backend", "whisperx")
                metadata = self._alignment_metadata(status=f"{status}_empty_asr_words_fallback_full_channel", result=result, error=error)
                item["alignment"] = metadata
                return metadata
            item["timestamp_source"] = "asr_words"
            item["alignment_backend"] = getattr(self.forced_aligner, "backend", "whisperx")
            metadata = self._alignment_metadata(status=f"{status}_used_asr_words", result=result, error=error)
            item["alignment"] = metadata
            return metadata

        item["words"] = []
        item["timestamp_source"] = "forced_alignment_fallback_full_channel"
        item["alignment_backend"] = getattr(self.forced_aligner, "backend", "whisperx")
        metadata = self._alignment_metadata(status=f"{status}_fallback_full_channel", result=result, error=error)
        item["alignment"] = metadata
        return metadata

    def _alignment_metadata(self, status: str, result: Optional[AlignmentResult], error: Optional[str]) -> dict:
        backend = getattr(getattr(self, "forced_aligner", None), "backend", getattr(getattr(self.config, "alignment", None), "backend", "whisperx"))
        return {
            "alignment_status": status,
            "alignment_backend": backend,
            "alignment_coverage": float(result.coverage) if result is not None else None,
            "alignment_word_count": int(result.aligned_word_count) if result is not None else 0,
            "alignment_transcript_word_count": int(result.transcript_word_count) if result is not None else None,
            "alignment_language": result.language if result is not None else None,
            "alignment_error": error,
        }

    def _apply_alignment_metadata_to_transcript_row(self, transcript_row: dict, item: dict) -> None:
        metadata = dict(item.get("alignment") or {})
        source = str(item.get("source") or "")
        words = item.get("words") or []
        metadata["timestamp_source"] = item.get("timestamp_source")
        if source == "final_consensus":
            transcript_row.update(metadata)
            if item.get("timestamp_source") == "forced_alignment":
                self._replace_transcript_words(transcript_row, words)
            return
        if source.startswith("engine:"):
            engine = source.split(":", 1)[1]
            for row in transcript_row.get("engine_results", []) or []:
                if row.get("engine") == engine:
                    row.update(metadata)
                    if item.get("timestamp_source") == "forced_alignment":
                        self._replace_transcript_words(row, words)
                    return
            logger.warning("Alignment metadata could not be attached to missing engine transcript row: %s", engine)

    def _replace_transcript_words(self, row: dict, words: list[dict]) -> None:
        row["word_count"] = len(words)
        if "words" in row:
            row["words"] = words
        row.update(self._confidence_summary(words))

    @staticmethod
    def _safe_alignment_error(exc: BaseException | str) -> str:
        if isinstance(exc, BaseException):
            return type(exc).__name__
        return str(exc).split(":", 1)[0][:120] or "AlignmentError"

    def _handle_asr_results(
        self,
        results: Sequence[ASRResult] | Sequence[ChannelASRBundle],
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        row: Optional[dict],
        duration: Optional[float],
        alignment_audio_by_channel: Optional[dict[int, dict]] = None,
    ) -> bool:
        if not results:
            return False
        first = results[0]
        if isinstance(first, ChannelASRBundle) or hasattr(first, "engine_results"):
            return self._handle_asr_bundles(  # type: ignore[arg-type]
                results,
                transcripts,
                all_entities,
                raw_spans,
                row,
                duration,
                alignment_audio_by_channel=alignment_audio_by_channel,
            )

        fallback_used = False
        items: list[dict] = []
        for result in results:  # type: ignore[assignment]
            item = {
                "channel": int(result.channel),
                "source": result.engine,
                "engine": result.engine,
                "text": result.transcript,
                "words": result.words,
                "language": result.language,
            }
            self._align_detection_item(item, alignment_audio_by_channel)
            transcripts.append(self._asr_result_to_dict(result))  # type: ignore[arg-type]
            transcripts[-1].update(item.get("alignment") or {})
            transcripts[-1]["timestamp_source"] = item.get("timestamp_source")
            if "words" in transcripts[-1] and item.get("timestamp_source") == "forced_alignment":
                transcripts[-1]["words"] = item.get("words") or []
            items.append(item)
        detected = self.detector.detect_batch([item["text"] for item in items], rows=[row for _ in items])  # type: ignore[attr-defined]
        for item, entities in zip(items, detected):
            enriched = []
            for e in entities:
                ent = self._enrich_entity(
                    e,
                    channel=int(item["channel"]),
                    transcript_source=item["source"],
                    asr_engine=item.get("engine"),
                    existing_count=len(all_entities) + len(enriched),
                )
                ent["timestamp_source"] = item.get("timestamp_source")
                ent["alignment_backend"] = item.get("alignment_backend")
                enriched.append(ent)
                all_entities.append(ent)
            spans, used = self._map_entities_to_spans_conservative(enriched, item.get("words") or [], int(item["channel"]), duration)
            raw_spans.extend(spans)
            fallback_used = fallback_used or used
        return fallback_used

    @staticmethod
    def _enrich_entity(
        entity: dict,
        channel: int,
        transcript_source: str,
        asr_engine: Optional[str],
        existing_count: int,
    ) -> dict:
        ent = dict(entity)
        ent["channel"] = int(channel)
        ent["transcript_source"] = transcript_source
        ent["asr_engine"] = asr_engine
        ent.setdefault("entity_id", f"ent_{int(existing_count) + 1:06d}")
        return ent

    def _handle_asr_bundles(
        self,
        bundles: Sequence[ChannelASRBundle],
        transcripts: list[dict],
        all_entities: list[dict],
        raw_spans: list[dict],
        row: Optional[dict],
        duration: Optional[float],
        alignment_audio_by_channel: Optional[dict[int, dict]] = None,
    ) -> bool:
        fallback_used = False
        detection_items: list[dict] = []

        for bundle in bundles:
            transcript_row, bundle_items = self._prepare_bundle_for_detection(
                bundle,
                alignment_audio_by_channel=alignment_audio_by_channel,
            )
            transcripts.append(transcript_row)
            detection_items.extend(bundle_items)

        texts = [d["text"] for d in detection_items]
        if not texts:
            return False
        detected_batches = self.detector.detect_batch(texts, rows=[row for _ in texts])
        for item, entities in zip(detection_items, detected_batches):
            enriched = []
            for e in entities:
                ent = self._enrich_entity(
                    e,
                    channel=int(item["channel"]),
                    transcript_source=item["source"],
                    asr_engine=item.get("engine"),
                    existing_count=len(all_entities) + len(enriched),
                )
                ent["timestamp_source"] = item.get("timestamp_source")
                ent["alignment_backend"] = item.get("alignment_backend")
                enriched.append(ent)
                all_entities.append(ent)
            spans, used = self._map_entities_to_spans_conservative(enriched, item.get("words") or [], int(item["channel"]), duration)
            raw_spans.extend(spans)
            fallback_used = fallback_used or used
        return fallback_used

    def _prepare_bundle_for_detection(
        self,
        bundle: ChannelASRBundle,
        alignment_audio_by_channel: Optional[dict[int, dict]] = None,
    ) -> tuple[dict, list[dict]]:
        transcript_row = self._bundle_to_transcript_dict(bundle)
        items = self._detection_items_for_bundle(bundle)
        for item in items:
            self._align_detection_item(item, alignment_audio_by_channel)
            self._apply_alignment_metadata_to_transcript_row(transcript_row, item)
        return transcript_row, items

    def _detection_items_for_bundle(self, bundle: ChannelASRBundle) -> list[dict]:
        scope = str(getattr(self.config.asr, "pii_detection_transcript_scope", "final_and_all_engines"))
        items: list[dict] = []
        seen: set[tuple[str, str]] = set()
        languages_by_engine = {r.engine: r.language for r in bundle.engine_results if r.language}

        def add(source: str, engine: Optional[str], text: str, words: list[dict], language: Optional[str]):
            text = str(text or "").strip()
            if not text:
                return
            key = (source, text.lower())
            if key in seen:
                return
            seen.add(key)
            items.append({
                "channel": bundle.channel,
                "source": source,
                "engine": engine,
                "text": text,
                "words": words,
                "language": language,
            })

        if scope in {"final_only", "final_and_all_engines"}:
            selected_engine = bundle.consensus.get("selected_engine")
            add("final_consensus", selected_engine, bundle.final_transcript, bundle.final_words, languages_by_engine.get(selected_engine))

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
                add(f"engine:{result.engine}", result.engine, result.transcript, words, result.language)
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

    def _full_audio_spans(self, duration: float, channels: int = 2, entity_ids: Optional[list[str]] = None) -> list[dict]:
        n_channels = max(1, min(int(channels or 2), int(self.config.masking.output_channels or 2)))
        ids = list(entity_ids or [])
        return [
            {
                "start": 0.0,
                "end": max(0.0, float(duration)),
                "duration": max(0.0, float(duration)),
                "channel": channel,
                "source": "empty_transcript_failsafe",
                "type": "EMPTY_TRANSCRIPT_FAILSAFE",
                "text": "",
                "entity_ids": ids,
                "entity_id": ids[0] if len(ids) == 1 else None,
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
            channel_entities = [e for e in all_entities if int(e.get("channel", -1)) == ch]
            texts = [str(e.get("text", "")) for e in channel_entities]
            entity_ids = sorted({str(e.get("entity_id")) for e in channel_entities if e.get("entity_id")})
            timestamp_sources = sorted({str(e.get("timestamp_source")) for e in channel_entities if e.get("timestamp_source")})
            alignment_backends = sorted({str(e.get("alignment_backend")) for e in channel_entities if e.get("alignment_backend")})
            spans.append({
                "channel": ch,
                "start": 0.0,
                "end": float(duration),
                "duration": float(duration),
                "type": "UNMAPPED_PII_FALLBACK",
                "text": " | ".join(t for t in texts if t)[:500],
                "source": "unmapped_entity_policy",
                "score": 1.0,
                "entity_ids": entity_ids,
                "entity_id": entity_ids[0] if len(entity_ids) == 1 else None,
                "timestamp_source": timestamp_sources[0] if len(timestamp_sources) == 1 else ("mixed_fallback" if timestamp_sources else "unmapped_entity_policy"),
                "alignment_backend": alignment_backends[0] if len(alignment_backends) == 1 else None,
            })
        return spans, True

    @staticmethod
    def _opus_bitrate_to_kbps(value) -> int:
        try:
            s = str(value).strip().lower()
        except Exception:
            return 64
        if not s:
            return 64
        if s.endswith("k"):
            try:
                return max(1, int(round(float(s[:-1]))))
            except ValueError:
                return 64
        try:
            n = float(s)
        except ValueError:
            return 64
        if n >= 1000:
            return max(1, int(round(n / 1000)))
        return max(1, int(round(n)))

    def _encode(self, audio, output_path: Path, input_meta: dict) -> None:
        floor_kbps = max(8, int(getattr(self.config.masking, "opus_min_bitrate_kbps", 24)))
        configured_kbps = self._opus_bitrate_to_kbps(self.config.masking.opus_bitrate)
        bitrate_kbps = configured_kbps
        if self.config.masking.preserve_input_bitrate and input_meta.get("bit_rate"):
            try:
                br = int(input_meta["bit_rate"])
            except (TypeError, ValueError):
                br = 0
            if br > 0:
                bitrate_kbps = round(br / 1000)
        bitrate_kbps = max(floor_kbps, bitrate_kbps)
        bitrate = f"{bitrate_kbps}k"
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

    @staticmethod
    def _confidence_summary(words: list[dict]) -> dict:
        probabilities: list[float] = []
        for word in words or []:
            value = word.get("probability")
            if value is None:
                continue
            try:
                probability = float(value)
            except (TypeError, ValueError):
                continue
            probabilities.append(max(0.0, min(1.0, probability)))

        if not probabilities:
            return {
                "transcript_confidence": None,
                "confidence_method": "unavailable",
                "confidence_summary": {
                    "avg_word_probability": None,
                    "min_word_probability": None,
                    "word_probability_count": 0,
                    "low_confidence_word_count": 0,
                },
            }

        avg_probability = float(sum(probabilities) / len(probabilities))
        min_probability = float(min(probabilities))
        return {
            "transcript_confidence": avg_probability,
            "confidence_method": "mean_word_probability",
            "confidence_summary": {
                "avg_word_probability": avg_probability,
                "min_word_probability": min_probability,
                "word_probability_count": len(probabilities),
                "low_confidence_word_count": sum(1 for p in probabilities if p < 0.50),
            },
        }

    def _pii_detectors_enabled(self) -> list[str]:
        detectors: list[str] = []
        cfg = self.config.pii
        if bool(getattr(cfg, "enable_regex", False)):
            detectors.append("regex")
        if bool(getattr(cfg, "enable_spoken_number_rules", False)):
            detectors.append("spoken_number_rule")
        detectors.append("rule_name")
        if bool(getattr(cfg, "enable_saved_pii_json", False)):
            detectors.append("saved_pii_json")
        if bool(getattr(cfg, "enable_gliner", False)):
            detectors.append("gliner")
        if bool(getattr(cfg, "enable_piiranha", False)):
            detectors.append("piiranha")
        if bool(getattr(cfg, "enable_spacy", False)):
            detectors.append("spacy")
        return detectors

    def _pii_detectors_loaded(self) -> dict:
        detector = getattr(self, "detector", None)
        return {
            "regex": bool(getattr(self.config.pii, "enable_regex", False)),
            "spoken_number_rule": bool(getattr(self.config.pii, "enable_spoken_number_rules", False)),
            "rule_name": True,
            "saved_pii_json": bool(getattr(self.config.pii, "enable_saved_pii_json", False)),
            "gliner": getattr(detector, "gliner_model", None) is not None,
            "piiranha": getattr(detector, "piiranha_pipe", None) is not None,
            "spacy": getattr(detector, "spacy_nlp", None) is not None,
        }

    def _pii_detection_items_from_transcripts(self, transcripts: list[dict]) -> list[dict]:
        scope = str(getattr(self.config.asr, "pii_detection_transcript_scope", "final_and_all_engines"))
        items: list[dict] = []
        seen: set[tuple[int, str, Optional[str], int]] = set()

        def add(channel: int, source: str, engine: Optional[str], text: str, word_count: int, alignment: Optional[dict] = None) -> None:
            text = str(text or "").strip()
            if not text:
                return
            text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
            key = (int(channel), source, engine, text_hash)
            if key in seen:
                return
            seen.add(key)
            items.append({
                "channel": int(channel),
                "source": source,
                "asr_engine": engine,
                "text_char_count": len(text),
                "word_count": int(word_count),
                **(alignment or {}),
            })

        for row in transcripts or []:
            channel = int(row.get("channel", -1))
            if row.get("engine") == "consensus":
                if scope in {"final_only", "final_and_all_engines"}:
                    consensus = row.get("consensus") or {}
                    add(
                        channel,
                        "final_consensus",
                        consensus.get("selected_engine"),
                        str(row.get("final_transcript") or row.get("transcript") or ""),
                        int(row.get("word_count") or 0),
                        self._transcript_alignment_summary(row),
                    )
                if scope in {"all_engines_only", "final_and_all_engines"}:
                    engine_transcripts = row.get("engine_transcripts") or {}
                    engine_results = {r.get("engine"): r for r in row.get("engine_results", []) if isinstance(r, dict)}
                    for engine, text in engine_transcripts.items():
                        result = engine_results.get(engine) or {}
                        add(channel, f"engine:{engine}", engine, str(text or ""), int(result.get("word_count") or 0), self._transcript_alignment_summary(result))
            else:
                engine = row.get("engine")
                add(channel, str(engine or "unknown"), engine, str(row.get("transcript") or ""), int(row.get("word_count") or 0), self._transcript_alignment_summary(row))
        return items

    @staticmethod
    def _transcript_alignment_summary(row: dict) -> dict:
        keys = [
            "alignment_status",
            "alignment_backend",
            "alignment_coverage",
            "alignment_word_count",
            "alignment_transcript_word_count",
            "timestamp_source",
        ]
        return {key: row.get(key) for key in keys if key in row}

    @staticmethod
    def _span_entity_ids(span: dict) -> set[str]:
        ids: set[str] = set()
        if span.get("entity_id"):
            ids.add(str(span["entity_id"]))
        for entity_id in span.get("entity_ids") or []:
            if entity_id:
                ids.add(str(entity_id))
        return ids

    def _masking_audit(self, all_entities: list[dict], raw_spans: list[dict], merged_spans: list[dict], unmapped_fallback_used: bool) -> dict:
        detected_ids = sorted({str(e.get("entity_id")) for e in all_entities if e.get("entity_id")})
        missing_id_count = sum(1 for e in all_entities or [] if not e.get("entity_id"))
        timestamp_ids: set[str] = set()
        fallback_ids: set[str] = set()
        for span in raw_spans or []:
            ids = self._span_entity_ids(span)
            if (
                str(span.get("type")) in {"UNMAPPED_PII_FALLBACK", "EMPTY_TRANSCRIPT_FAILSAFE"}
                or str(span.get("source")) in {"unmapped_entity_policy", "empty_transcript_failsafe"}
            ):
                fallback_ids.update(ids)
            elif ids:
                timestamp_ids.update(ids)

        covered = timestamp_ids | fallback_ids
        without_masking = sorted(set(detected_ids) - covered)
        return {
            "detected_entity_count": len(all_entities or []),
            "raw_span_count": len(raw_spans or []),
            "merged_span_count": len(merged_spans or []),
            "entity_ids_detected": detected_ids,
            "entity_ids_with_timestamp_spans": sorted(timestamp_ids),
            "entity_ids_with_full_channel_fallback": sorted(fallback_ids),
            "entity_ids_covered_by_masking": sorted(covered),
            "entity_ids_without_masking": without_masking,
            "entities_missing_id_field": missing_id_count,
            "all_detected_entities_masked": not without_masking and missing_id_count == 0,
            "unmapped_fallback_used": bool(unmapped_fallback_used),
        }

    def _alignment_audit(self, transcripts: list[dict], raw_spans: list[dict]) -> dict:
        align_cfg = getattr(self.config, "alignment", None)
        enabled = bool(getattr(align_cfg, "enabled", False))
        rows: list[dict] = []
        for row in transcripts or []:
            if "alignment_status" in row:
                rows.append(row)
            for engine_row in row.get("engine_results", []) or []:
                if isinstance(engine_row, dict) and "alignment_status" in engine_row:
                    rows.append(engine_row)

        statuses = sorted({str(row.get("alignment_status")) for row in rows if row.get("alignment_status")})
        fallback_span_count = sum(
            1
            for span in raw_spans or []
            if str(span.get("type")) in {"UNMAPPED_PII_FALLBACK", "EMPTY_TRANSCRIPT_FAILSAFE"}
            or str(span.get("source")) in {"unmapped_entity_policy", "empty_transcript_failsafe"}
        )
        forced_span_count = sum(1 for span in raw_spans or [] if str(span.get("timestamp_source")) == "forced_alignment")
        asr_words_span_count = sum(1 for span in raw_spans or [] if str(span.get("timestamp_source")) == "asr_words")
        alignment_fallback_span_count = sum(
            1
            for span in raw_spans or []
            if str(span.get("timestamp_source")) == "forced_alignment_fallback_full_channel"
        )
        if not enabled:
            status = "disabled"
        elif not rows:
            status = "fallback_used" if fallback_span_count else "not_run"
        elif statuses and all(value == "disabled" for value in statuses):
            status = "disabled"
        elif any("fallback_full_channel" in value for value in statuses) or fallback_span_count:
            status = "fallback_used"
        elif statuses == ["aligned"]:
            status = "aligned"
        elif any("error" in value or "low_coverage" in value or "unaligned" in value or "missing_audio" in value for value in statuses):
            status = "degraded"
        else:
            status = "mixed"

        coverage_values = []
        for row in rows:
            value = row.get("alignment_coverage")
            if value is None:
                continue
            try:
                coverage_values.append(float(value))
            except (TypeError, ValueError):
                continue

        return {
            "enabled": enabled,
            "backend": getattr(align_cfg, "backend", "whisperx"),
            "status": status,
            "statuses": statuses,
            "on_failure": getattr(align_cfg, "on_failure", "fallback_full_channel"),
            "min_aligned_words_ratio": getattr(align_cfg, "min_aligned_words_ratio", 0.70),
            "transcript_count": len(rows),
            "aligned_transcript_count": sum(1 for row in rows if row.get("alignment_status") == "aligned"),
            "forced_alignment_span_count": forced_span_count,
            "full_channel_fallback_span_count": fallback_span_count,
            "alignment_full_channel_fallback_span_count": alignment_fallback_span_count,
            "asr_words_span_count": asr_words_span_count,
            "avg_alignment_coverage": float(sum(coverage_values) / len(coverage_values)) if coverage_values else None,
            "timestamp_sources_in_spans": sorted({str(span.get("timestamp_source")) for span in raw_spans or [] if span.get("timestamp_source")}),
        }

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
            "alignment": self._alignment_audit(transcripts, raw_spans),
            "optimizations": {
                "forced_alignment": bool(getattr(getattr(self.config, "alignment", None), "enabled", False)),
                "no_forced_alignment": not bool(getattr(getattr(self.config, "alignment", None), "enabled", False)),
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
            "pii_detection": {
                "transcript_scope": self.config.asr.pii_detection_transcript_scope,
                "detectors_enabled": self._pii_detectors_enabled(),
                "detectors_loaded": self._pii_detectors_loaded(),
                "detection_items": self._pii_detection_items_from_transcripts(transcripts),
                "entity_sources": sorted({str(e.get("source")) for e in all_entities if e.get("source")}),
            },
            "masking_audit": self._masking_audit(all_entities, raw_spans, merged_spans, unmapped_fallback_used),
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
        confidence = self._confidence_summary(bundle.final_words)
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
            **confidence,
        }
        if self.config.runtime.sidecar_include_words:
            d["words"] = bundle.final_words
            d["anchor_words"] = bundle.anchor_words
        return d

    def _asr_result_to_dict(self, result: ASRResult) -> Dict[str, Any]:
        confidence = self._confidence_summary(result.words)
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
            **confidence,
        }
        if self.config.runtime.sidecar_include_words:
            d["words"] = result.words
        return d
