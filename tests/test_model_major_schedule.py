from pathlib import Path

import pii_audio_masking_pipeline.pipeline as pipeline_module
from pii_audio_masking_pipeline.asr import ASRResult
from pii_audio_masking_pipeline.config import load_config
from pii_audio_masking_pipeline.pipeline import PIIMaskingPipeline
from pii_audio_masking_pipeline.state import ASRResultCache


def test_asr_result_cache_round_trips_engine_results(tmp_path: Path):
    cache = ASRResultCache(tmp_path / "asr.sqlite")
    result = ASRResult(
        file_id="call-a",
        channel=0,
        transcript="hello John Smith",
        words=[
            {"word": "hello", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5},
            {"word": "John", "start": 0.3, "end": 0.5, "start_char": 6, "end_char": 10},
        ],
        engine="whisper",
        language="en",
        duration=1.0,
    )

    cache.upsert_result("/input/call-a/audio.opus", "whisper", 0, result)

    assert cache.has_result("/input/call-a/audio.opus", "whisper", 0) is True
    restored = cache.get_results_for_file("/input/call-a/audio.opus")
    cache.close()

    assert len(restored) == 1
    assert restored[0].engine == "whisper"
    assert restored[0].transcript == "hello John Smith"
    assert restored[0].words[1]["word"] == "John"


def test_asr_result_cache_distinguishes_error_rows_from_clean_results(tmp_path: Path):
    cache = ASRResultCache(tmp_path / "asr.sqlite")
    result = ASRResult(file_id="call-a", channel=0, transcript="", words=[], engine="whisper", error="RuntimeError")

    cache.upsert_result("/input/call-a/audio.opus", "whisper", 0, result)

    assert cache.has_result("/input/call-a/audio.opus", "whisper", 0) is False
    assert cache.has_result("/input/call-a/audio.opus", "whisper", 0, include_errors=True) is True
    assert cache.has_results_for_file("/input/call-a/audio.opus", ["whisper"], [0]) is False
    assert cache.has_results_for_file("/input/call-a/audio.opus", ["whisper"], [0], include_errors=True) is True
    cache.close()


def test_asr_result_cache_requires_every_engine_channel_combination(tmp_path: Path):
    cache = ASRResultCache(tmp_path / "asr.sqlite")
    input_path = "/input/call-a/audio.opus"
    rows = [
        (input_path, "whisper", 0, ASRResult(file_id=input_path, channel=0, transcript="a", words=[{"word": "a"}], engine="whisper")),
        (input_path, "whisper", 1, ASRResult(file_id=input_path, channel=1, transcript="b", words=[{"word": "b"}], engine="whisper")),
        (input_path, "qwen", 0, ASRResult(file_id=input_path, channel=0, transcript="", words=[], engine="qwen", error="RuntimeError")),
    ]

    cache.upsert_results(rows)

    assert cache.has_results_for_file(input_path, ["whisper", "qwen"], [0, 1], include_errors=True) is False
    cache.upsert_result(input_path, "qwen", 1, ASRResult(file_id=input_path, channel=1, transcript="", words=[], engine="qwen", error="RuntimeError"))
    assert cache.has_results_for_file(input_path, ["whisper", "qwen"], [0, 1], include_errors=True) is True
    cache.close()


class DetectorWithName:
    def detect_batch(self, texts, rows=None):
        out = []
        for text in texts:
            if "John Smith" in text:
                out.append([{
                    "text": "John Smith",
                    "type": "PERSON_NAME",
                    "start": text.index("John Smith"),
                    "end": text.index("John Smith") + len("John Smith"),
                    "source": "test",
                    "score": 1.0,
                }])
            else:
                out.append([])
        return out


def test_pipeline_finalizes_from_cached_asr_results_without_transcribing(monkeypatch, tmp_path: Path):
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
    pipe.detector = DetectorWithName()

    input_path = pipe.input_root / "call-a" / "audio.opus"
    captured = {}

    monkeypatch.setattr(
        pipeline_module,
        "ffprobe_audio",
        lambda *args, **kwargs: {
            "duration_sec": 2.0,
            "channels": 2,
            "codec_name": "opus",
            "sample_rate": 48000,
        },
    )

    def fake_finalize(self, **kwargs):
        captured.update(kwargs)
        return {
            "input_path": str(kwargs["input_path"]),
            "status": "success",
            "num_entities": len(kwargs["all_entities"]),
            "num_spans": len(kwargs["raw_spans"]),
        }

    monkeypatch.setattr(PIIMaskingPipeline, "_finalize_outputs", fake_finalize)

    results = [
        ASRResult(
            file_id=str(input_path),
            channel=0,
            transcript="hello John Smith",
            words=[
                {"word": "hello", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": 0},
                {"word": "John", "start": 0.3, "end": 0.5, "start_char": 6, "end_char": 10, "channel": 0},
                {"word": "Smith", "start": 0.5, "end": 0.8, "start_char": 11, "end_char": 16, "channel": 0},
            ],
            engine="whisper",
        ),
        ASRResult(file_id=str(input_path), channel=0, transcript="hello John Smith", words=[], engine="qwen"),
        ASRResult(
            file_id=str(input_path),
            channel=1,
            transcript="clean channel",
            words=[{"word": "clean", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": 1}],
            engine="whisper",
        ),
    ]

    row = pipe.process_file_from_asr_results(input_path, results)

    assert row["status"] == "success"
    assert row["num_entities"] >= 1
    assert captured["raw_spans"][0]["channel"] == 0
    assert captured["used_single_decode"] is False


def test_cached_finalization_uses_full_audio_failsafe_when_anchor_words_are_missing(tmp_path: Path):
    cfg = load_config(None)
    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = tmp_path
    pipe.output_root = tmp_path / "out"
    pipe.work_dir = tmp_path / "work"

    assert pipe._bundles_from_cached_asr_results([
        ASRResult(file_id="f", channel=0, transcript="hello", words=[], engine="qwen")
    ]) == []
