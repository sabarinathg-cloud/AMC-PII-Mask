from pathlib import Path

import pytest

from pii_audio_masking_pipeline.config import load_config
from pii_audio_masking_pipeline.pipeline import PIIMaskingPipeline


def test_config_example_exposes_pipeline_runtime_contract():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    assert cfg.asr.input_audio_strategy == "single_decode"
    assert cfg.masking.unmapped_entity_policy == "mask_full_channel"
    assert cfg.runtime.copy_unmasked_when_no_pii is True
    assert cfg.runtime.unmasked_copy_method == "hardlink_or_copy"
    assert cfg.runtime.atomic_output is True
    assert cfg.runtime.sidecar_include_words is False
    assert cfg.runtime.max_csv_report_rows == 10000
    assert cfg.runtime.file_batch_size == 2
    assert cfg.runtime.file_batch_max_decoded_audio_gb == 2.0


def test_output_path_refuses_same_input_path_configuration(tmp_path):
    cfg = load_config(None)
    cfg.paths.input_root = str(tmp_path)
    cfg.paths.output_root = str(tmp_path)
    cfg.paths.preserve_relative_path = True
    cfg.paths.force_output_suffix = ".opus"
    pipe = object.__new__(PIIMaskingPipeline)
    pipe.config = cfg
    pipe.input_root = tmp_path
    pipe.output_root = tmp_path
    input_path = tmp_path / "audio.opus"
    with pytest.raises(ValueError, match="Unsafe output path equals input path"):
        pipe.output_path_for(input_path)
