import json
import os
from pathlib import Path

import numpy as np
import pytest

import pii_audio_masking_pipeline.audio_io as audio_io_module
import pii_audio_masking_pipeline.pipeline as pipeline_module
import pii_audio_masking_pipeline.utils as utils_module
from pii_audio_masking_pipeline.asr import ASRResult, ChannelASRBundle, build_consensus
from pii_audio_masking_pipeline.audio_io import AudioCommandError, encode_float32_stereo_to_opus
from pii_audio_masking_pipeline.config import load_config
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
