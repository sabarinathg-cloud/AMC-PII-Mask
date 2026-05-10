from pathlib import Path

import pytest

from pii_audio_masking_pipeline.config import load_config, validate_config, PipelineConfig
from pii_audio_masking_pipeline.manifest import discover_audio_files


def test_discovery_excludes_output_and_work_roots(tmp_path: Path):
    src = tmp_path / "input"
    out = src / "pii_masked_audio"
    work = src / "pii_masking_work"
    good = src / "2024" / "call-a" / "audio.opus"
    bad_out = out / "2024" / "call-a" / "audio.opus"
    bad_work = work / "tmp" / "audio.opus"
    for p in (good, bad_out, bad_work):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    cfg = PipelineConfig()
    cfg.paths.input_root = str(src)
    cfg.paths.output_root = str(out)
    cfg.paths.work_dir = str(work)

    files = discover_audio_files(src, "**/audio.opus", config=cfg)
    assert files == [good]


def test_config_rejects_unsafe_no_timestamp_setting():
    cfg = PipelineConfig()
    cfg.asr.word_timestamps = False
    with pytest.raises(ValueError, match="word_timestamps"):
        validate_config(cfg)


def test_default_config_loads_and_is_speed_safe():
    cfg = load_config(None)
    assert cfg.asr.input_audio_strategy == "single_decode"
    assert cfg.masking.copy_input_if_no_pii is True
    assert cfg.runtime.copy_unmasked_when_no_pii is True
    assert cfg.masking.output_sample_rate == 48000
    assert cfg.masking.output_channels == 2
