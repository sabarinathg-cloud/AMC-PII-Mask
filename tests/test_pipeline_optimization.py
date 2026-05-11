import json
import os
from pathlib import Path
import wave

import numpy as np
import pytest

import pii_audio_masking_pipeline.audio_io as audio_io_module
import pii_audio_masking_pipeline.pipeline as pipeline_module
import pii_audio_masking_pipeline.utils as utils_module
from pii_audio_masking_pipeline.asr import ASRResult, ChannelASRBundle, build_consensus
from pii_audio_masking_pipeline.audio_io import AudioCommandError, encode_float32_stereo_to_opus
from pii_audio_masking_pipeline.config import load_config
from pii_audio_masking_pipeline.forced_alignment import AlignmentResult
from pii_audio_masking_pipeline.pipeline import PIIMaskingPipeline
from pii_audio_masking_pipeline.timestamp_mapping import merge_spans
from pii_audio_masking_pipeline.utils import write_json


def _pipe(tmp_path: Path) -> PIIMaskingPipeline:
    cfg = load_config(None)
    cfg.paths.input_root = str(tmp_path / "input")
    cfg.paths.output_root = str(tmp_path / "output")
    cfg.paths.work_dir = str(tmp_path / "work")
    cfg.runtime.resume = False
    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = Path(cfg.paths.input_root)
    pipe.output_root = Path(cfg.paths.output_root)
    pipe.work_dir = Path(cfg.paths.work_dir)
    return pipe


def test_perf_metrics_are_optional_and_written_to_sidecar_and_result(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.runtime.write_perf_metrics = True

    metrics = pipe._new_perf_metrics()
    with pipe._time_perf_stage(metrics, "decode"):
        pass
    pipe._record_cuda_memory(metrics, "after_decode")

    sidecar = pipe._build_sidecar(
        input_path=tmp_path / "input" / "audio.opus",
        output_path=tmp_path / "output" / "audio.opus",
        meta={"duration_sec": 1.0},
        elapsed=0.1,
        transcripts=[],
        all_entities=[],
        raw_spans=[],
        merged_spans=[],
        validation={"valid": True},
        status="success",
        used_single_decode=True,
        used_combined_channel_extract=False,
        unmapped_fallback_used=False,
        no_pii_fast_copy_used=False,
        perf_metrics=metrics,
    )
    row = pipe._result_row(
        input_path=tmp_path / "input" / "audio.opus",
        output_path=tmp_path / "output" / "audio.opus",
        sidecar_path=tmp_path / "output" / "audio.opus.pii_masking.json",
        elapsed=0.1,
        duration=1.0,
        transcripts=[],
        all_entities=[],
        merged_spans=[],
        validation={"valid": True},
        status="success",
        perf_metrics=metrics,
    )

    assert "perf_metrics" in sidecar
    assert "perf_metrics_json" in row
    assert json.loads(row["perf_metrics_json"])["stages_sec"]["decode"] >= 0.0
    assert sidecar["perf_metrics"]["stages_sec"]["decode"] >= 0.0
    assert sidecar["perf_metrics"]["cuda_memory"][-1]["label"] == "after_decode"


class BatchASR:
    def transcribe_channel_batch(self, asr_inputs, work_dir: Path, keep_temp: bool = False):
        grouped = {}
        for item in asr_inputs:
            file_id = str(item["file_id"])
            channel = int(item["channel"])
            words = [{"word": "clean", "start": 0.0, "end": 0.1, "channel": channel}]
            result = ASRResult(channel=channel, transcript="clean", words=words, engine="fake", file_id=file_id)
            grouped.setdefault(file_id, []).append(ChannelASRBundle(
                file_id=file_id,
                channel=channel,
                final_transcript="clean",
                final_words=words,
                engine_results=[result],
                anchor_engine="fake",
                anchor_words=words,
                consensus=build_consensus([result], {"min_agreement": 1, "fallback_priority": ["fake"]}),
            ))
        return grouped


class EmptyDetector:
    def detect_batch(self, texts, rows=None):
        return [[] for _ in texts]


class SingleEntityDetector:
    def detect_batch(self, texts, rows=None):
        out = []
        for text in texts:
            start = str(text).index("John")
            out.append([{
                "text": "John",
                "type": "PERSON_NAME",
                "start": start,
                "end": start + 4,
                "score": 0.99,
                "source": "test",
            }])
        return out


class StaticAligner:
    backend = "whisperx"

    def __init__(self, result: AlignmentResult):
        self.result = result

    def align(self, **kwargs):
        return self.result


class LegacyPathASR:
    def transcribe_path(self, wav_path: Path, channel: int):
        return ASRResult(channel=channel, transcript="hello John", words=[], engine="legacy", language="en")


def _write_silent_wav(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(sample_rate // 10, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _write_silent_float_wav(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(sample_rate // 10, dtype=np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_process_files_batch_reuses_ffprobe_metadata(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.asr = BatchASR()
    pipe.detector = EmptyDetector()

    paths = [pipe.input_root / "a" / "audio.opus", pipe.input_root / "b" / "audio.opus"]
    calls = []

    def fake_ffprobe(path, ffprobe_path="ffprobe"):
        calls.append(Path(path))
        return {"duration_sec": 2.0, "channels": 2, "codec_name": "opus", "sample_rate": 48000}

    monkeypatch.setattr(pipeline_module, "ffprobe_audio", fake_ffprobe)
    monkeypatch.setattr(
        pipeline_module,
        "decode_to_float32_stereo_48k",
        lambda *args, **kwargs: np.zeros((48000, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        pipeline_module,
        "make_asr_audio_inputs",
        lambda *args, **kwargs: [
            {"channel": 0, "audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
            {"channel": 1, "audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
        ],
    )
    monkeypatch.setattr(
        PIIMaskingPipeline,
        "_finalize_outputs",
        lambda self, **kwargs: {
            "input_path": str(kwargs["input_path"]),
            "status": "success",
            "duration_sec": kwargs["duration"],
        },
    )

    results = pipe.process_files_batch(paths)

    assert [r["status"] for r in results] == ["success", "success"]
    assert calls == paths


def test_temp_wav_transcribe_path_passes_audio_to_forced_alignment(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.asr = LegacyPathASR()
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="aligned",
        words=[],
        coverage=0.0,
        backend="whisperx",
        language="en",
        aligned_word_count=0,
        transcript_word_count=0,
    ))
    pipe.config.asr.mode = "per_channel"
    captured = {}

    def fake_extract_channel_wav(input_path, wav_path, **kwargs):
        _write_silent_wav(Path(wav_path), sample_rate=int(kwargs["sample_rate"]))

    def fake_handle(self, results, transcripts, all_entities, raw_spans, row, duration, alignment_audio_by_channel=None):
        captured["alignment_audio_by_channel"] = alignment_audio_by_channel
        return False

    monkeypatch.setattr(pipeline_module, "extract_channel_wav", fake_extract_channel_wav)
    monkeypatch.setattr(PIIMaskingPipeline, "_handle_asr_results", fake_handle)

    pipe._process_with_temp_wavs(
        input_path=tmp_path / "input" / "audio.opus",
        channel_count=1,
        row=None,
        duration=1.0,
        transcripts=[],
        all_entities=[],
        raw_spans=[],
    )

    alignment_audio = captured["alignment_audio_by_channel"][0]
    assert alignment_audio["sample_rate"] == 16000
    assert alignment_audio["audio"].dtype == np.float32
    assert alignment_audio["audio"].shape[0] > 0


def test_read_wav_float32_rejects_non_pcm16_alignment_audio(tmp_path: Path):
    wav_path = tmp_path / "float.wav"
    _write_silent_float_wav(wav_path)

    with pytest.raises(ValueError, match="16-bit PCM"):
        PIIMaskingPipeline._read_wav_float32(wav_path)


def test_handle_asr_bundles_uses_forced_aligned_words_for_pii_spans(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.detector = SingleEntityDetector()
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="aligned",
        words=[
            {"word": "hello", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": 0, "timestamp_source": "forced_alignment", "probability": 0.8},
            {"word": "John", "start": 1.0, "end": 1.3, "start_char": 6, "end_char": 10, "channel": 0, "timestamp_source": "forced_alignment", "probability": 0.6},
        ],
        coverage=1.0,
        backend="whisperx",
        language="en",
        aligned_word_count=2,
        transcript_word_count=2,
    ))
    asr_words = [
        {"word": "hello", "start": 9.0, "end": 9.2, "start_char": 0, "end_char": 5, "channel": 0},
        {"word": "John", "start": 9.3, "end": 9.5, "start_char": 6, "end_char": 10, "channel": 0},
    ]
    result = ASRResult(channel=0, transcript="hello John", words=asr_words, engine="whisper", language="en")
    bundle = ChannelASRBundle(
        channel=0,
        final_transcript="hello John",
        final_words=asr_words,
        engine_results=[result],
        anchor_engine="whisper",
        anchor_words=asr_words,
        consensus={"selected_engine": "whisper"},
    )
    transcripts: list[dict] = []
    all_entities: list[dict] = []
    raw_spans: list[dict] = []

    pipe._handle_asr_bundles(
        [bundle],
        transcripts,
        all_entities,
        raw_spans,
        row=None,
        duration=20.0,
        alignment_audio_by_channel={0: {"audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}},
    )

    assert raw_spans[0]["start"] == pytest.approx(0.88)
    assert raw_spans[0]["timestamp_source"] == "forced_alignment"
    assert raw_spans[0]["entity_id"] == "ent_000001"
    assert transcripts[0]["transcript_confidence"] == pytest.approx(0.7)
    assert transcripts[0]["confidence_method"] == "mean_word_probability"
    assert transcripts[0]["words"][1]["probability"] == 0.6


def test_low_forced_alignment_coverage_uses_full_channel_fallback(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.alignment.min_aligned_words_ratio = 0.75
    pipe.detector = SingleEntityDetector()
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="aligned",
        words=[],
        coverage=0.0,
        backend="whisperx",
        language="en",
        aligned_word_count=0,
        transcript_word_count=2,
    ))
    asr_words = [
        {"word": "hello", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": 0},
        {"word": "John", "start": 0.3, "end": 0.5, "start_char": 6, "end_char": 10, "channel": 0},
    ]
    result = ASRResult(channel=0, transcript="hello John", words=asr_words, engine="whisper", language="en")
    bundle = ChannelASRBundle(
        channel=0,
        final_transcript="hello John",
        final_words=asr_words,
        engine_results=[result],
        anchor_engine="whisper",
        anchor_words=asr_words,
        consensus={"selected_engine": "whisper"},
    )
    raw_spans: list[dict] = []

    transcripts: list[dict] = []
    all_entities: list[dict] = []
    used_fallback = pipe._handle_asr_bundles(
        [bundle],
        transcripts=transcripts,
        all_entities=all_entities,
        raw_spans=raw_spans,
        row=None,
        duration=5.0,
        alignment_audio_by_channel={0: {"audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}},
    )

    assert used_fallback is True
    assert raw_spans[0]["type"] == "UNMAPPED_PII_FALLBACK"
    assert raw_spans[0]["start"] == 0.0
    assert raw_spans[0]["end"] == 5.0
    assert raw_spans[0]["timestamp_source"] == "forced_alignment_fallback_full_channel"
    assert raw_spans[0]["alignment_backend"] == "whisperx"
    assert transcripts[0]["words"] == asr_words
    assert transcripts[0]["alignment_status"] == "low_coverage_fallback_full_channel"


def test_non_bundle_alignment_fallback_preserves_asr_words_in_sidecar(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.detector = SingleEntityDetector()
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="unaligned",
        words=[],
        coverage=0.0,
        backend="whisperx",
        language="en",
        aligned_word_count=0,
        transcript_word_count=2,
    ))
    asr_words = [
        {"word": "hello", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": 0},
        {"word": "John", "start": 0.3, "end": 0.5, "start_char": 6, "end_char": 10, "channel": 0},
    ]
    result = ASRResult(channel=0, transcript="hello John", words=asr_words, engine="legacy", language="en")
    transcripts: list[dict] = []
    all_entities: list[dict] = []
    raw_spans: list[dict] = []

    pipe._handle_asr_results(
        [result],
        transcripts=transcripts,
        all_entities=all_entities,
        raw_spans=raw_spans,
        row=None,
        duration=5.0,
        alignment_audio_by_channel={0: {"audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}},
    )

    assert transcripts[0]["words"] == asr_words
    assert transcripts[0]["alignment_status"] == "low_coverage_fallback_full_channel"
    assert raw_spans[0]["timestamp_source"] == "forced_alignment_fallback_full_channel"


def test_use_asr_words_with_empty_words_degrades_to_full_channel_fallback(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.alignment.on_failure = "use_asr_words"
    pipe.detector = SingleEntityDetector()
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="unaligned",
        words=[],
        coverage=0.0,
        backend="whisperx",
        language="en",
        aligned_word_count=0,
        transcript_word_count=2,
    ))
    result = ASRResult(channel=0, transcript="hello John", words=[], engine="legacy", language="en")
    raw_spans: list[dict] = []

    pipe._handle_asr_results(
        [result],
        transcripts=[],
        all_entities=[],
        raw_spans=raw_spans,
        row=None,
        duration=5.0,
        alignment_audio_by_channel={0: {"audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}},
    )

    assert raw_spans[0]["type"] == "UNMAPPED_PII_FALLBACK"
    assert raw_spans[0]["timestamp_source"] == "forced_alignment_fallback_full_channel"
    assert raw_spans[0]["alignment_backend"] == "whisperx"


def test_enabled_alignment_requires_whisperx_dependency(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.alignment.enabled = True
    monkeypatch.setattr(pipeline_module.importlib.util, "find_spec", lambda name: None if name == "whisperx" else object())

    with pytest.raises(ImportError, match="pip install whisperx"):
        pipe._create_forced_aligner()


def test_model_major_alignment_audio_infers_channels_from_cached_results(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.forced_aligner = StaticAligner(AlignmentResult(
        status="aligned",
        words=[],
        coverage=0.0,
        backend="whisperx",
        language="en",
        aligned_word_count=0,
        transcript_word_count=0,
    ))
    captured = {}

    def fake_decode(*args, **kwargs):
        return np.zeros((48000, 2), dtype=np.float32)

    def fake_make_asr_audio_inputs(audio, mode, input_channels, **kwargs):
        captured["input_channels"] = input_channels
        return [
            {"channel": 0, "audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
            {"channel": 1, "audio": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
        ][:input_channels]

    monkeypatch.setattr(pipeline_module, "decode_to_float32_stereo_48k", fake_decode)
    monkeypatch.setattr(pipeline_module, "make_asr_audio_inputs", fake_make_asr_audio_inputs)
    monkeypatch.setattr(PIIMaskingPipeline, "_should_skip_existing", lambda *args, **kwargs: False)
    monkeypatch.setattr(PIIMaskingPipeline, "_bundles_from_cached_asr_results", lambda *args, **kwargs: [])
    monkeypatch.setattr(PIIMaskingPipeline, "_handle_asr_bundles", lambda *args, **kwargs: False)
    monkeypatch.setattr(PIIMaskingPipeline, "_finalize_outputs", lambda self, **kwargs: {"status": "success"})

    pipe.process_file_from_asr_results(
        tmp_path / "input" / "audio.opus",
        asr_results=[
            ASRResult(channel=0, transcript="a", words=[], engine="whisper"),
            ASRResult(channel=1, transcript="b", words=[], engine="whisper"),
        ],
        meta={"duration_sec": 1.0},
    )

    assert captured["input_channels"] == 2


def test_adaptive_batch_size_shrinks_when_free_gpu_memory_is_low(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.runtime.min_free_gpu_mem_gb = 8.0
    pipe.config.runtime.adaptive_batch_min_size = 2

    assert pipe._adapt_file_batch_size(8, free_gpu_mem_gb=4.0) == 4
    assert pipe._adapt_file_batch_size(2, free_gpu_mem_gb=4.0) == 2
    assert pipe._adapt_file_batch_size(8, free_gpu_mem_gb=10.0) == 8


def test_adaptive_batching_can_be_disabled(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.runtime.adaptive_file_batching = False
    pipe.config.runtime.min_free_gpu_mem_gb = 8.0

    assert pipe._adapt_file_batch_size(8, free_gpu_mem_gb=1.0) == 8


def test_adaptive_batch_size_can_recover_after_memory_pressure(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.runtime.min_free_gpu_mem_gb = 8.0

    assert pipe._recover_file_batch_size(2, configured_size=8, free_gpu_mem_gb=12.0) == 4
    assert pipe._recover_file_batch_size(8, configured_size=8, free_gpu_mem_gb=12.0) == 8
    assert pipe._recover_file_batch_size(2, configured_size=8, free_gpu_mem_gb=7.0) == 2


def test_batch_metrics_are_kept_separate_from_per_file_stage_timings(tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.runtime.write_perf_metrics = True
    per_file = pipe._new_perf_metrics()
    batch = {
        "stages_sec": {"asr_batch": 8.0},
        "cuda_memory": [{"label": "after_asr_batch"}],
        "adaptive_batching": [{"from": 8, "to": 4}],
        "asr_engine_sec": {"whisper": 2.0},
        "batch_size": 4,
    }

    pipe._merge_batch_perf_metrics(per_file, batch)

    assert per_file["stages_sec"] == {}
    assert per_file["batch_stages_sec"]["asr_batch"] == 8.0
    assert per_file["asr_engine_sec"]["whisper"] == 2.0
    assert per_file["batch_size"] == 4
    assert "do not sum" in per_file["batch_metrics_semantics"]


def test_asr_result_dict_includes_word_and_transcript_confidence_by_default(tmp_path: Path):
    pipe = _pipe(tmp_path)

    result = ASRResult(
        channel=0,
        transcript="hello John",
        words=[
            {"word": "hello", "start": 0.0, "end": 0.2, "probability": 0.9, "start_char": 0, "end_char": 5},
            {"word": "John", "start": 0.3, "end": 0.5, "probability": 0.7, "start_char": 6, "end_char": 10},
        ],
        engine="whisper",
        language="en",
        language_probability=0.99,
    )

    row = pipe._asr_result_to_dict(result)

    assert "words" in row
    assert row["words"][0]["probability"] == 0.9
    assert row["transcript_confidence"] == pytest.approx(0.8)
    assert row["confidence_method"] == "mean_word_probability"
    assert row["confidence_summary"] == {
        "avg_word_probability": pytest.approx(0.8),
        "min_word_probability": pytest.approx(0.7),
        "word_probability_count": 2,
        "low_confidence_word_count": 0,
    }


def test_sidecar_includes_pii_detector_coverage_and_masking_audit(tmp_path: Path):
    pipe = _pipe(tmp_path)
    entities = [
        {
            "entity_id": "ent_000001",
            "text": "John Smith",
            "type": "PERSON_NAME",
            "start": 6,
            "end": 16,
            "score": 0.99,
            "source": "piiranha",
            "channel": 0,
            "asr_engine": "whisper",
            "transcript_source": "engine:whisper",
        },
        {
            "entity_id": "ent_000002",
            "text": "account number",
            "type": "ACCOUNT_NUMBER",
            "start": 20,
            "end": 34,
            "score": 0.92,
            "source": "gliner",
            "channel": 1,
            "asr_engine": "qwen",
            "transcript_source": "engine:qwen",
        },
    ]
    raw_spans = [
        {
            "entity_id": "ent_000001",
            "channel": 0,
            "start": 0.1,
            "end": 0.8,
            "duration": 0.7,
            "type": "PERSON_NAME",
            "source": "piiranha",
        },
        {
            "entity_ids": ["ent_000002"],
            "channel": 1,
            "start": 0.0,
            "end": 3.0,
            "duration": 3.0,
            "type": "UNMAPPED_PII_FALLBACK",
            "source": "unmapped_entity_policy",
        },
    ]

    sidecar = pipe._build_sidecar(
        input_path=tmp_path / "input" / "audio.opus",
        output_path=tmp_path / "output" / "audio.opus",
        meta={"duration_sec": 3.0},
        elapsed=0.1,
        transcripts=[],
        all_entities=entities,
        raw_spans=raw_spans,
        merged_spans=raw_spans,
        validation={"valid": True},
        status="success_unmapped_fallback",
        used_single_decode=True,
        used_combined_channel_extract=False,
        unmapped_fallback_used=True,
        no_pii_fast_copy_used=False,
    )

    assert sidecar["pii_detection"]["transcript_scope"] == "final_and_all_engines"
    assert "piiranha" in sidecar["pii_detection"]["detectors_enabled"]
    assert "gliner" in sidecar["pii_detection"]["detectors_enabled"]
    assert sidecar["pii_detection"]["entity_sources"] == ["gliner", "piiranha"]
    assert sidecar["masking_audit"]["detected_entity_count"] == 2
    assert sidecar["masking_audit"]["entity_ids_with_timestamp_spans"] == ["ent_000001"]
    assert sidecar["masking_audit"]["entity_ids_with_full_channel_fallback"] == ["ent_000002"]
    assert sidecar["masking_audit"]["entity_ids_without_masking"] == []
    assert sidecar["masking_audit"]["all_detected_entities_masked"] is True


def test_sidecar_records_forced_alignment_status_and_masking_provenance(tmp_path: Path):
    pipe = _pipe(tmp_path)
    transcripts = [{
        "channel": 0,
        "engine": "consensus",
        "transcript": "hello John",
        "final_transcript": "hello John",
        "word_count": 2,
        "alignment_status": "aligned",
        "alignment_backend": "whisperx",
        "alignment_coverage": 1.0,
        "alignment_word_count": 2,
        "timestamp_source": "forced_alignment",
    }]
    sidecar = pipe._build_sidecar(
        input_path=tmp_path / "input" / "audio.opus",
        output_path=tmp_path / "output" / "audio.opus",
        meta={"duration_sec": 2.0},
        elapsed=0.1,
        transcripts=transcripts,
        all_entities=[],
        raw_spans=[{"timestamp_source": "forced_alignment", "alignment_backend": "whisperx"}],
        merged_spans=[{"timestamp_source": "forced_alignment", "alignment_backend": "whisperx"}],
        validation={"valid": True},
        status="success",
        used_single_decode=True,
        used_combined_channel_extract=False,
        unmapped_fallback_used=False,
        no_pii_fast_copy_used=False,
    )

    assert sidecar["alignment"]["enabled"] is True
    assert sidecar["alignment"]["backend"] == "whisperx"
    assert sidecar["alignment"]["status"] == "aligned"
    assert sidecar["alignment"]["transcript_count"] == 1
    assert sidecar["alignment"]["forced_alignment_span_count"] == 1
    assert sidecar["alignment"]["full_channel_fallback_span_count"] == 0
    assert sidecar["alignment"]["asr_words_span_count"] == 0
    assert sidecar["optimizations"]["forced_alignment"] is True
    assert sidecar["transcripts"][0]["alignment_coverage"] == 1.0


def test_alignment_audit_classifies_missing_audio_as_degraded_for_asr_word_fallback(tmp_path: Path):
    pipe = _pipe(tmp_path)
    transcripts = [{
        "channel": 0,
        "engine": "consensus",
        "transcript": "hello John",
        "final_transcript": "hello John",
        "alignment_status": "missing_audio_used_asr_words",
        "alignment_backend": "whisperx",
        "timestamp_source": "asr_words",
    }]
    audit = pipe._alignment_audit(
        transcripts,
        raw_spans=[{"timestamp_source": "asr_words", "alignment_backend": "whisperx"}],
    )

    assert audit["status"] == "degraded"
    assert audit["asr_words_span_count"] == 1


def test_alignment_audit_reports_disabled_when_all_rows_are_disabled(tmp_path: Path):
    pipe = _pipe(tmp_path)
    transcripts = [{
        "channel": 0,
        "engine": "consensus",
        "transcript": "hello",
        "final_transcript": "hello",
        "alignment_status": "disabled",
        "timestamp_source": "asr_words",
    }]

    audit = pipe._alignment_audit(transcripts, raw_spans=[])

    assert audit["status"] == "disabled"


def test_write_json_creates_private_sidecar_file(tmp_path: Path):
    path = tmp_path / "audio.opus.pii_masking.json"

    write_json(path, {"transcripts": [{"transcript": "contains PHI"}]})

    assert path.stat().st_mode & 0o777 == 0o600


def test_write_json_creates_temp_file_private_from_first_byte(monkeypatch, tmp_path: Path):
    path = tmp_path / "audio.opus.pii_masking.json"
    real_open = os.open
    modes: list[int] = []

    def recording_open(file, flags, mode=0o777, *args, **kwargs):
        modes.append(mode)
        return real_open(file, flags, mode, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "open", recording_open)

    write_json(path, {"transcripts": [{"transcript": "contains PHI"}]})

    assert modes
    assert modes[0] == 0o600


def test_merge_spans_preserves_all_entity_ids_when_spans_overlap():
    merged = merge_spans([
        {"channel": 0, "start": 0.0, "end": 0.4, "type": "PHONE", "source": "regex", "text": "one", "entity_id": "ent_000001"},
        {"channel": 0, "start": 0.3, "end": 0.8, "type": "EMAIL", "source": "piiranha", "text": "two", "entity_id": "ent_000002"},
    ])

    assert len(merged) == 1
    assert merged[0]["entity_id"] is None
    assert merged[0]["entity_ids"] == ["ent_000001", "ent_000002"]


def test_confidence_summary_clamps_out_of_range_probabilities(tmp_path: Path):
    pipe = _pipe(tmp_path)
    result = ASRResult(
        channel=0,
        transcript="bad scale",
        words=[
            {"word": "bad", "probability": 1.5},
            {"word": "scale", "probability": -0.5},
        ],
        engine="whisper",
    )

    row = pipe._asr_result_to_dict(result)

    assert row["confidence_summary"]["avg_word_probability"] == pytest.approx(0.5)
    assert row["confidence_summary"]["min_word_probability"] == pytest.approx(0.0)
    assert row["transcript_confidence"] == pytest.approx(0.5)


def test_masking_audit_flags_entities_without_ids_as_not_fully_auditable(tmp_path: Path):
    pipe = _pipe(tmp_path)

    audit = pipe._masking_audit(
        all_entities=[{"text": "John Smith", "source": "piiranha"}],
        raw_spans=[],
        merged_spans=[],
        unmapped_fallback_used=False,
    )

    assert audit["entities_missing_id_field"] == 1
    assert audit["all_detected_entities_masked"] is False


def test_empty_transcript_failsafe_has_entity_id_and_audits_as_masked(tmp_path: Path):
    pipe = _pipe(tmp_path)
    sentinel_id = "ent_000001"
    spans = pipe._full_audio_spans(3.0, channels=2, entity_ids=[sentinel_id])
    audit = pipe._masking_audit(
        all_entities=[{"entity_id": sentinel_id, "type": "EMPTY_TRANSCRIPT_FAILSAFE", "source": "pipeline"}],
        raw_spans=spans,
        merged_spans=spans,
        unmapped_fallback_used=True,
    )

    assert all(span["entity_ids"] == [sentinel_id] for span in spans)
    assert audit["entities_missing_id_field"] == 0
    assert audit["entity_ids_with_timestamp_spans"] == []
    assert audit["entity_ids_with_full_channel_fallback"] == [sentinel_id]
    assert audit["all_detected_entities_masked"] is True


def test_encode_applies_minimum_bitrate_floor(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.masking.preserve_input_bitrate = True
    pipe.config.masking.opus_min_bitrate_kbps = 24

    captured = {}

    def fake_encode(*args, **kwargs):
        captured["bitrate"] = kwargs.get("bitrate")
        return Path(args[1] if len(args) > 1 else kwargs["output_path"])

    monkeypatch.setattr(pipeline_module, "encode_float32_stereo_to_opus", fake_encode)

    audio = np.zeros((48000, 2), dtype=np.float32)
    out = tmp_path / "out.opus"

    pipe._encode(audio, out, {"bit_rate": "15000"})
    assert captured["bitrate"] == "24k"

    pipe._encode(audio, out, {"bit_rate": "96000"})
    assert captured["bitrate"] == "96k"


def test_encode_floor_applies_when_no_bitrate_metadata(monkeypatch, tmp_path: Path):
    pipe = _pipe(tmp_path)
    pipe.config.masking.preserve_input_bitrate = True
    pipe.config.masking.opus_min_bitrate_kbps = 32
    pipe.config.masking.opus_bitrate = "16k"

    captured = {}

    def fake_encode(*args, **kwargs):
        captured["bitrate"] = kwargs.get("bitrate")
        return Path(kwargs["output_path"]) if "output_path" in kwargs else Path(args[1])

    monkeypatch.setattr(pipeline_module, "encode_float32_stereo_to_opus", fake_encode)

    pipe._encode(np.zeros((4800, 2), dtype=np.float32), tmp_path / "out.opus", {})
    assert captured["bitrate"] == "32k"


def test_encode_float32_stereo_to_opus_writes_pcm_to_tempfile_and_surfaces_stderr(monkeypatch, tmp_path: Path):
    """When ffmpeg fails the user must see the real stderr, not BrokenPipeError."""

    seen_cmds: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            seen_cmds.append(cmd)
            self.args = cmd
            self.returncode = 1
            self._stderr = (
                b"[libopus @ 0xdeadbeef] Error: bitrate 16000 not supported "
                b"for stereo voip application\n"
            )

        def communicate(self, input=None, timeout=None):
            return (b"", self._stderr)

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            pass

    monkeypatch.setattr(audio_io_module.subprocess, "Popen", FakeProc)

    audio = np.zeros((4800, 2), dtype=np.float32)
    with pytest.raises(AudioCommandError) as excinfo:
        encode_float32_stereo_to_opus(
            audio,
            tmp_path / "out.opus",
            ffmpeg_path="ffmpeg",
            bitrate="16k",
            atomic=False,
        )

    msg = str(excinfo.value)
    assert "bitrate 16000 not supported" in msg
    assert seen_cmds, "ffmpeg should have been invoked"
    cmd = seen_cmds[0]
    assert "pipe:0" not in cmd, "encoder must read PCM from a temp file, not stdin"
    pcm_inputs = [c for c in cmd if str(c).endswith(".f32le")]
    assert pcm_inputs, "encoder must pass a temp .f32le PCM file as input"


def test_encode_float32_stereo_to_opus_cleans_up_pcm_tempfile_on_success(monkeypatch, tmp_path: Path):
    written_paths: list[Path] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.args = cmd
            self.returncode = 0
            for token in cmd:
                p = Path(str(token))
                if p.suffix == ".f32le":
                    written_paths.append(p)

        def communicate(self, input=None, timeout=None):
            output_path = Path(str(self.args[-1]))
            output_path.write_bytes(b"FAKEOPUS")
            return (b"", b"")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(audio_io_module.subprocess, "Popen", FakeProc)

    out = tmp_path / "ok.opus"
    encode_float32_stereo_to_opus(
        np.zeros((4800, 2), dtype=np.float32),
        out,
        ffmpeg_path="ffmpeg",
        atomic=False,
    )
    assert out.exists()
    for p in written_paths:
        assert not p.exists(), f"temp PCM file {p} should be cleaned up"


def test_encode_float32_stereo_to_opus_normalizes_yaml_boolean_vbr(monkeypatch, tmp_path: Path):
    seen_cmds: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            seen_cmds.append(cmd)
            self.args = cmd
            self.returncode = 0

        def communicate(self, input=None, timeout=None):
            Path(str(self.args[-1])).write_bytes(b"FAKEOPUS")
            return (b"", b"")

        def poll(self):
            return self.returncode

        def kill(self):
            pass

    monkeypatch.setattr(audio_io_module.subprocess, "Popen", FakeProc)

    encode_float32_stereo_to_opus(
        np.zeros((4800, 2), dtype=np.float32),
        tmp_path / "out.opus",
        ffmpeg_path="ffmpeg",
        vbr=True,
        atomic=False,
    )

    cmd = seen_cmds[0]
    vbr_index = cmd.index("-vbr")
    assert cmd[vbr_index + 1] == "on"
