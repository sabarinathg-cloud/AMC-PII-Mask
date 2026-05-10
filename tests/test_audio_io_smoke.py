from pathlib import Path
import numpy as np

from pii_audio_masking_pipeline.audio_io import (
    apply_mask_spans,
    atomic_copy,
    decode_to_float32_stereo_48k,
    encode_float32_stereo_to_opus,
    ffprobe_audio,
    input_matches_required_output,
    make_asr_audio_inputs,
    resample_mono_float32,
)
from pii_audio_masking_pipeline.validation import validate_masked_file


def test_stereo_opus_roundtrip(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.stack([
        0.05 * np.sin(2 * np.pi * 440 * t),
        0.05 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)

    src = tmp_path / "src.opus"
    out = tmp_path / "masked.opus"
    encode_float32_stereo_to_opus(audio, src)
    decoded = decode_to_float32_stereo_48k(src)
    masked = apply_mask_spans(decoded, [{"start": 0.2, "end": 0.4, "channel": 0}], mode="silence")
    encode_float32_stereo_to_opus(masked, out)

    meta = ffprobe_audio(out)
    assert meta["codec_name"] == "opus"
    assert meta["sample_rate"] == 48000
    assert meta["channels"] == 2
    assert input_matches_required_output(meta) is True
    assert validate_masked_file(src, out)["valid"] is True


def test_single_decode_asr_inputs_and_copy(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.stack([
        0.05 * np.sin(2 * np.pi * 440 * t),
        0.05 * np.sin(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)

    asr_inputs = make_asr_audio_inputs(audio, mode="per_channel", input_channels=2)
    assert len(asr_inputs) == 2
    assert asr_inputs[0]["channel"] == 0
    assert asr_inputs[1]["channel"] == 1
    assert asr_inputs[0]["audio"].dtype == np.float32
    assert abs(len(asr_inputs[0]["audio"]) - 16000) <= 2

    mono = audio[:, 0]
    rs = resample_mono_float32(mono, 48000, 16000)
    assert rs.dtype == np.float32
    assert abs(len(rs) - 16000) <= 2

    src = tmp_path / "src.opus"
    dst = tmp_path / "nested" / "dst.opus"
    encode_float32_stereo_to_opus(audio, src)
    atomic_copy(src, dst)
    assert dst.exists()
    assert ffprobe_audio(dst)["codec_name"] == "opus"
