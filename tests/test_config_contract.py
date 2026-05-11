from pathlib import Path

import pytest

from pii_audio_masking_pipeline.config import apply_performance_profile, load_config, validate_config
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
    assert cfg.runtime.write_perf_metrics is False
    assert cfg.runtime.adaptive_file_batching is True
    assert cfg.runtime.adaptive_batch_min_size == 1
    assert cfg.runtime.min_free_gpu_mem_gb == 2.0
    assert cfg.runtime.performance_profile == "default"


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


def test_a10g_profile_sets_safe_throughput_defaults():
    cfg = load_config(None)
    cfg.runtime.performance_profile = "a10g_24gb"

    apply_performance_profile(cfg)

    assert cfg.asr.model_residency == "keep_loaded"
    assert cfg.runtime.file_batch_size == 4
    assert cfg.runtime.file_batch_max_decoded_audio_gb == 4.0
    assert cfg.runtime.write_perf_metrics is True
    assert cfg.runtime.adaptive_file_batching is True
    assert cfg.pii.batch_size == 32
    assert cfg.asr.engines["qwen"]["batch_size"] == 4
    assert cfg.asr.pii_detection_transcript_scope == "final_and_all_engines"
    assert cfg.masking.unmapped_entity_policy == "mask_full_channel"


def test_unknown_performance_profile_is_rejected():
    cfg = load_config(None)
    cfg.runtime.performance_profile = "mystery"
    with pytest.raises(ValueError, match="performance_profile"):
        validate_config(cfg)
