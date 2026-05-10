from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence
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
        output_path = self.output_path_for(input_path)
        sidecar_path = self.sidecar_path_for(output_path)
        if input_path.resolve() == output_path.resolve():
            raise ValueError(
                f"Refusing to overwrite original audio. Configure paths.output_root outside the input file path: {input_path}"
            )

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
            audio_48k = decode_to_float32_stereo_48k(
                input_path,
                ffmpeg_path=self.config.runtime.ffmpeg_path,
                threads=self.config.runtime.ffmpeg_threads,
                sample_rate=self.config.masking.output_sample_rate,
                channels=self.config.masking.output_channels,
            )
            used_single_decode = True
            asr_inputs = make_asr_audio_inputs(
                audio_48k,
                mode=self.config.asr.mode,
                input_channels=channel_count,
                source_sr=self.config.masking.output_sample_rate,
                target_sr=self.config.asr.channel_wav_sample_rate,
            )
            results = self._transcribe_asr_inputs(asr_inputs)
            unmapped_fallback_used = self._handle_asr_results(results, transcripts, all_entities, raw_spans, row, duration) or unmapped_fallback_used
        elif strategy == "ffmpeg_temp_wav":
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

        if all_entities and not raw_spans:
            raw_spans, unmapped_fallback_used = self._apply_unmapped_entity_policy(all_entities, duration)

        merged_spans = merge_spans(
            raw_spans,
            merge_gap_sec=self.config.masking.merge_gap_sec,
            target_channels=self.config.masking.target_channels,
        )

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
            validation = {
                "valid": True,
                "validation_skipped": True,
                "checks": {},
            }

        elapsed = time.time() - started
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
        )
        write_json(sidecar_path, sidecar)
        return self._result_row(input_path, output_path, sidecar_path, elapsed, duration, transcripts, all_entities, merged_spans, validation, status)

    def can_resume_skip(self, input_path: str | Path, output_path: str | Path, sidecar_path: str | Path) -> bool:
        """Return True only when an existing output is safe to reuse.

        The SQLite state alone is not trusted. The output file and sidecar must still exist,
        and by default the output is ffprobe-validated before skipping.
        """
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
        """Transcribe prepared ASR channel inputs.

        Production path uses MultiASRTranscriber: Whisper + optional Qwen/Cohere/Granite,
        with a consensus transcript per channel. Tests and older callers can still inject a
        single-model fake ASR exposing transcribe_audio/transcribe_path.
        """
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
                    # Use the already-created temp WAV as compatibility input for every enabled model.
                    from .audio_io import resample_mono_float32
                    with __import__("wave").open(str(wav_path), "rb") as wf:
                        data = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    asr_inputs = [{"channel": -1, "audio": resample_mono_float32(audio, self.config.asr.channel_wav_sample_rate, self.config.asr.channel_wav_sample_rate), "sample_rate": self.config.asr.channel_wav_sample_rate}]
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
                    if hasattr(self.asr, "transcribe_channels"):
                        # MultiASR optimized path normally uses single_decode. This fallback decodes the temp WAV
                        # into memory once per channel, then MultiASR reuses temp WAVs for transcript-only engines.
                        import numpy as np
                        with __import__("wave").open(str(wav_path), "rb") as wf:
                            data = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        results.append(ASRResult(channel=channel, transcript="", words=[], engine="placeholder"))
                    else:
                        results.append(self.asr.transcribe_path(wav_path, channel=channel))
                if hasattr(self.asr, "transcribe_channels"):
                    asr_inputs = []
                    import numpy as np
                    for channel, wav_path in sorted(channel_wavs.items() if channel_wavs else [(c, temp_dir / f"ch{c}.wav") for c in range(channel_count)]):
                        with __import__("wave").open(str(wav_path), "rb") as wf:
                            data = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        asr_inputs.append({"channel": int(channel), "audio": audio, "sample_rate": self.config.asr.channel_wav_sample_rate})
                    bundles_or_results = self._transcribe_asr_inputs(asr_inputs)
                    fallback_used = self._handle_asr_results(bundles_or_results, transcripts, all_entities, raw_spans, row, duration)
                    return used_combined, bool(fallback_used)
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
    ) -> dict:
        return {
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
    ) -> Dict[str, Any]:
        return {
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


    def _bundle_to_transcript_dict(self, bundle: ChannelASRBundle) -> Dict[str, Any]:
        engine_rows = [self._asr_result_to_dict(r) for r in bundle.engine_results]
        d: Dict[str, Any] = {
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
