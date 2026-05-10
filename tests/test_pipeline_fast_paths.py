from pathlib import Path

import numpy as np

from pii_audio_masking_pipeline.asr import ASRResult
from pii_audio_masking_pipeline.audio_io import encode_float32_stereo_to_opus, ffprobe_audio
from pii_audio_masking_pipeline.config import load_config
from pii_audio_masking_pipeline.pipeline import PIIMaskingPipeline


class FakeASR:
    def transcribe_audio(self, audio, channel: int, sample_rate: int = 16000):
        return ASRResult(
            channel=channel,
            transcript="hello this is a clean call",
            words=[],
            language="en",
            duration=len(audio) / sample_rate,
        )


class FakeDetector:
    def detect_batch(self, texts, rows=None):
        return [[] for _ in texts]


def test_no_pii_uses_fast_copy_and_preserves_required_output(tmp_path: Path):
    sr = 48000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    audio = np.stack([
        0.03 * np.sin(2 * np.pi * 440 * t),
        0.03 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    work_dir = tmp_path / "work"
    src = input_root / "2024" / "call-a" / "audio.opus"
    encode_float32_stereo_to_opus(audio, src)

    cfg = load_config(None)
    cfg.paths.input_root = str(input_root)
    cfg.paths.output_root = str(output_root)
    cfg.paths.work_dir = str(work_dir)
    cfg.runtime.unmasked_copy_method = "copy"
    cfg.runtime.resume = False

    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = input_root
    pipe.output_root = output_root
    pipe.work_dir = work_dir
    pipe.asr = FakeASR()
    pipe.detector = FakeDetector()

    result = pipe.process_file(src)
    out = Path(result["output_path"])
    assert result["status"] == "success_no_pii_fast_copy"
    assert out.exists()
    meta = ffprobe_audio(out)
    assert meta["codec_name"] == "opus"
    assert meta["sample_rate"] == 48000
    assert meta["channels"] == 2

class FakeASRWithPII:
    def transcribe_audio(self, audio, channel: int, sample_rate: int = 16000):
        if channel == 0:
            transcript = "hello John Smith"
            words = [
                {"word": "hello", "start": 0.05, "end": 0.20, "start_char": 0, "end_char": 5, "channel": 0, "segment_id": 0},
                {"word": "John", "start": 0.25, "end": 0.45, "start_char": 6, "end_char": 10, "channel": 0, "segment_id": 0},
                {"word": "Smith", "start": 0.50, "end": 0.70, "start_char": 11, "end_char": 16, "channel": 0, "segment_id": 0},
            ]
        else:
            transcript = "clean channel"
            words = [
                {"word": "clean", "start": 0.05, "end": 0.20, "start_char": 0, "end_char": 5, "channel": 1, "segment_id": 0},
                {"word": "channel", "start": 0.25, "end": 0.45, "start_char": 6, "end_char": 13, "channel": 1, "segment_id": 0},
            ]
        return ASRResult(channel=channel, transcript=transcript, words=words, language="en", duration=len(audio) / sample_rate)


class FakeDetectorWithPII:
    def detect_batch(self, texts, rows=None):
        out = []
        for text in texts:
            if "John Smith" in text:
                out.append([{"text": "John Smith", "type": "PERSON_NAME", "start": 6, "end": 16, "source": "test", "score": 1.0}])
            else:
                out.append([])
        return out


def test_pii_masking_encodes_masked_output(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.stack([
        0.05 * np.sin(2 * np.pi * 440 * t),
        0.05 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    work_dir = tmp_path / "work"
    src = input_root / "2024" / "call-b" / "audio.opus"
    encode_float32_stereo_to_opus(audio, src)

    cfg = load_config(None)
    cfg.paths.input_root = str(input_root)
    cfg.paths.output_root = str(output_root)
    cfg.paths.work_dir = str(work_dir)
    cfg.masking.mode = "silence"
    cfg.runtime.resume = False

    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = input_root
    pipe.output_root = output_root
    pipe.work_dir = work_dir
    pipe.asr = FakeASRWithPII()
    pipe.detector = FakeDetectorWithPII()

    result = pipe.process_file(src)
    out = Path(result["output_path"])
    assert result["status"] == "success"
    assert result["num_entities"] == 1
    assert result["num_spans"] == 1
    assert out.exists()
    meta = ffprobe_audio(out)
    assert meta["codec_name"] == "opus"
    assert meta["sample_rate"] == 48000
    assert meta["channels"] == 2


def test_cross_file_batch_no_pii_fast_copy(tmp_path: Path):
    sr = 48000
    t = np.arange(sr // 4, dtype=np.float32) / sr
    audio = np.stack([
        0.03 * np.sin(2 * np.pi * 440 * t),
        0.03 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)

    input_root = tmp_path / "input_batch"
    output_root = tmp_path / "output_batch"
    work_dir = tmp_path / "work_batch"
    src1 = input_root / "2024" / "call-a" / "audio.opus"
    src2 = input_root / "2024" / "call-b" / "audio.opus"
    encode_float32_stereo_to_opus(audio, src1)
    encode_float32_stereo_to_opus(audio, src2)

    cfg = load_config(None)
    cfg.paths.input_root = str(input_root)
    cfg.paths.output_root = str(output_root)
    cfg.paths.work_dir = str(work_dir)
    cfg.runtime.unmasked_copy_method = "copy"
    cfg.runtime.resume = False
    cfg.runtime.file_batch_size = 2

    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = input_root
    pipe.output_root = output_root
    pipe.work_dir = work_dir
    pipe.asr = FakeASR()
    pipe.detector = FakeDetector()

    results = pipe.process_files_batch([src1, src2])
    assert len(results) == 2
    assert {r["status"] for r in results} == {"success_no_pii_fast_copy"}
    for r in results:
        out = Path(r["output_path"])
        assert out.exists()
        meta = ffprobe_audio(out)
        assert meta["codec_name"] == "opus"
        assert meta["sample_rate"] == 48000
        assert meta["channels"] == 2
