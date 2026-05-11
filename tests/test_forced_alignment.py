import numpy as np
import pytest

from pii_audio_masking_pipeline.forced_alignment import (
    WhisperXForcedAligner,
    build_canonical_aligned_words,
)


def test_build_canonical_aligned_words_preserves_transcript_char_spans():
    words = build_canonical_aligned_words(
        transcript="Call John Smith.",
        timed_words=[
            {"word": "Call", "start": 1.0, "end": 1.2, "score": 0.91},
            {"word": "John", "start": 1.25, "end": 1.45, "score": 0.82},
            {"word": "Smith", "start": 1.5, "end": 1.8, "score": 0.73},
        ],
        channel=1,
        backend="whisperx",
    )

    assert [row["word"] for row in words] == ["Call", "John", "Smith."]
    assert words[1]["start_char"] == 5
    assert words[1]["end_char"] == 9
    assert words[1]["start"] == pytest.approx(1.25)
    assert words[1]["probability"] == pytest.approx(0.82)
    assert words[1]["alignment_backend"] == "whisperx"
    assert words[1]["timestamp_source"] == "forced_alignment"
    assert words[1]["channel"] == 1


class FakeWhisperX:
    def __init__(self):
        self.load_calls = []

    def load_align_model(self, language_code, device):
        self.load_calls.append((language_code, device))
        return f"model-{language_code}", {"language": language_code}

    def align(self, segments, model, metadata, audio, device, **kwargs):
        assert kwargs["return_char_alignments"] is False
        assert kwargs["batch_size"] == 4
        assert model == "model-en"
        assert metadata == {"language": "en"}
        assert device == "cpu"
        assert np.asarray(audio).dtype == np.float32
        return {
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.3, "score": 0.88},
                        {"word": "John", "start": 0.35, "end": 0.6, "score": 0.77},
                    ]
                }
            ]
        }


class FakeWhisperXWithoutBatchSize(FakeWhisperX):
    def align(self, segments, model, metadata, audio, device, return_char_alignments=False):
        assert return_char_alignments is False
        assert model == "model-en"
        return {
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.3, "score": 0.88},
                        {"word": "John", "start": 0.35, "end": 0.6, "score": 0.77},
                    ]
                }
            ]
        }


def test_whisperx_forced_aligner_caches_models_by_language():
    fake = FakeWhisperX()
    aligner = WhisperXForcedAligner(
        device="cpu",
        compute_type="float32",
        batch_size=4,
        whisperx_module=fake,
    )

    first = aligner.align(
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        transcript="Hello John",
        language="en",
        channel=0,
    )
    second = aligner.align(
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        transcript="Hello John",
        language="en",
        channel=0,
    )

    assert fake.load_calls == [("en", "cpu")]
    assert first.status == "aligned"
    assert first.coverage == pytest.approx(1.0)
    assert first.words[1]["word"] == "John"
    assert second.words[0]["timestamp_source"] == "forced_alignment"


def test_whisperx_forced_aligner_supports_align_api_without_batch_size():
    aligner = WhisperXForcedAligner(
        device="cpu",
        compute_type="float32",
        batch_size=4,
        whisperx_module=FakeWhisperXWithoutBatchSize(),
    )

    result = aligner.align(
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        transcript="Hello John",
        language="en",
        channel=0,
    )

    assert result.status == "aligned"
    assert result.words[1]["word"] == "John"
