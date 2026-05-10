from pathlib import Path

import numpy as np

from pii_audio_masking_pipeline.asr import ASRResult, ChannelASRBundle, build_consensus
from pii_audio_masking_pipeline.audio_io import encode_float32_stereo_to_opus, ffprobe_audio
from pii_audio_masking_pipeline.config import load_config
from pii_audio_masking_pipeline.pipeline import PIIMaskingPipeline


class BatchFakeASR:
    def __init__(self):
        self.batch_sizes = []

    def transcribe_channel_batch(self, asr_inputs, work_dir: Path, keep_temp: bool = False):
        asr_inputs = list(asr_inputs)
        self.batch_sizes.append(len(asr_inputs))
        grouped = {}
        for item in asr_inputs:
            file_id = str(item.get("file_id"))
            channel = int(item["channel"])
            transcript = f"clean channel {channel}"
            words = [
                {"word": "clean", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 5, "channel": channel, "segment_id": 0}
            ]
            result = ASRResult(channel=channel, transcript=transcript, words=words, engine="fake", file_id=file_id)
            grouped.setdefault(file_id, []).append(ChannelASRBundle(
                file_id=file_id,
                channel=channel,
                final_transcript=transcript,
                final_words=words,
                engine_results=[result],
                anchor_engine="fake",
                anchor_words=words,
                consensus=build_consensus([result], {"min_agreement": 1, "fallback_priority": ["fake"]}),
            ))
        return grouped


class RecordingDetector:
    def __init__(self):
        self.calls = []

    def detect_batch(self, texts, rows=None):
        texts = list(texts)
        self.calls.append(texts)
        return [[] for _ in texts]


def _make_opus(path: Path, seconds: float = 0.35) -> None:
    sr = 48000
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    audio = np.stack([
        0.03 * np.sin(2 * np.pi * 440 * t),
        0.03 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)
    encode_float32_stereo_to_opus(audio, path)


def test_process_files_batch_batches_pii_detection_and_preserves_output(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    work_dir = tmp_path / "work"
    src1 = input_root / "a" / "audio.opus"
    src2 = input_root / "b" / "audio.opus"
    _make_opus(src1)
    _make_opus(src2)

    cfg = load_config(None)
    cfg.paths.input_root = str(input_root)
    cfg.paths.output_root = str(output_root)
    cfg.paths.work_dir = str(work_dir)
    cfg.runtime.resume = False
    cfg.runtime.unmasked_copy_method = "copy"
    cfg.runtime.file_batch_size = 2
    cfg.runtime.file_batch_max_decoded_audio_gb = 1.0

    pipe = PIIMaskingPipeline.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = input_root
    pipe.output_root = output_root
    pipe.work_dir = work_dir
    asr = BatchFakeASR()
    pipe.asr = asr
    detector = RecordingDetector()
    pipe.detector = detector

    results = pipe.process_files_batch([src1, src2])
    assert len(results) == 2
    assert asr.batch_sizes == [4]  # two files times two channels in one ASR batch
    assert all(r["status"] == "success_no_pii_fast_copy" for r in results)
    assert len(detector.calls) == 1
    assert len(detector.calls[0]) == 8  # two files times two channels times final+engine transcript scope

    for result in results:
        meta = ffprobe_audio(result["output_path"])
        assert meta["codec_name"] == "opus"
        assert meta["sample_rate"] == 48000
        assert meta["channels"] == 2
