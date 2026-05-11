from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import yaml

VALID_PERFORMANCE_PROFILES = {"default", "a10g_24gb"}
VALID_PIPELINE_SCHEDULES = {"file_major", "model_major"}


@dataclass
class PathConfig:
    input_root: str = "/mnt/amc-data"
    input_glob: str = "**/audio.opus"
    output_root: str = "/mnt/amc-data/pii_masked_audio"
    work_dir: str = "/mnt/amc-data/pii_masking_work"
    preserve_relative_path: bool = True
    force_output_suffix: Optional[str] = ".opus"
    sidecar_suffix: str = ".pii_masking.json"
    exclude_output_root_from_discovery: bool = True
    exclude_work_dir_from_discovery: bool = True
    extra_exclude_globs: List[str] = field(default_factory=list)


def default_asr_engines() -> Dict[str, Dict[str, Any]]:
    return {
        "whisper": {
            "enabled": True,
            "kind": "faster_whisper",
            "model_dir": "/mnt/amc-data/pipeline/models/whisper-large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "use_batched_pipeline": True,
            "batch_size": 8,
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "vad_filter": True,
            "word_timestamps": True,
            "condition_on_previous_text": False,
        },
        "qwen": {
            "enabled": True,
            "kind": "qwen",
            "model_dir": "/mnt/amc-data/pipeline/models/qwen3-asr-1.7b",
            "device_map": "cuda:0",
            "dtype": "bfloat16",
            "batch_size": 2,
            "max_new_tokens": 256,
            "language": "English",
        },
        "cohere": {
            "enabled": True,
            "kind": "cohere",
            "model_dir": "/mnt/amc-data/pipeline/models/cohere-transcribe-03-2026",
            "device_map": "auto",
            "dtype": "float16",
            "batch_size": 2,
            "max_new_tokens": 256,
            "language": "en",
            "punctuation": True,
            "local_files_only": True,
        },
        "granite": {
            "enabled": True,
            "kind": "granite",
            "model_dir": "/mnt/amc-data/pipeline/models/granite-4.0-1b-speech",
            "device_map": "auto",
            "dtype": "bfloat16",
            "batch_size": 2,
            "max_new_tokens": 256,
            "local_files_only": True,
            "prompt": "<|audio|>can you transcribe the speech into a written format?",
        },
    }


def default_asr_consensus() -> Dict[str, Any]:
    return {
        "min_agreement": 2,
        "soft_similarity_threshold": 0.78,
        "fallback_priority": ["whisper", "qwen", "cohere", "granite"],
        "prefer_engine_on_tie": "whisper",
    }


@dataclass
class ASRConfig:
    # Backward-compatible top-level Whisper settings. The whisper engine config above is preferred.
    whisper_model_dir: str = "/mnt/amc-data/pipeline/models/whisper-large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    mode: str = "per_channel"  # per_channel or mono_mix
    input_audio_strategy: str = "single_decode"  # single_decode or ffmpeg_temp_wav

    # Whisper timestamp-anchor defaults retained for compatibility.
    use_batched_pipeline: bool = True
    batch_size: int = 8
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    vad_filter: bool = True
    vad_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    })
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    channel_wav_sample_rate: int = 16000
    max_audio_seconds: Optional[float] = None
    single_decode_max_audio_seconds: Optional[float] = 1800.0
    retry_base_on_bad_timestamps: bool = True
    timestamp_max_drift_sec: float = 1.0

    # Multi-ASR ensemble.
    engine_order: List[str] = field(default_factory=lambda: ["whisper", "qwen", "cohere", "granite"])
    timestamp_anchor_engine: str = "whisper"
    engines: Dict[str, Dict[str, Any]] = field(default_factory=default_asr_engines)
    consensus: Dict[str, Any] = field(default_factory=default_asr_consensus)

    # keep_loaded is fastest if the GPU has enough memory. unload_after_file is slower but safer on small GPUs.
    model_residency: str = "keep_loaded"  # keep_loaded or unload_after_file
    fail_on_engine_error: bool = False

    # PII scope. final_and_all_engines is safest: detect on consensus plus every enabled ASR transcript.
    pii_detection_transcript_scope: str = "final_and_all_engines"  # final_only, all_engines_only, final_and_all_engines


@dataclass
class PIIConfig:
    enable_regex: bool = True
    enable_spoken_number_rules: bool = True
    enable_gliner: bool = True
    enable_piiranha: bool = True
    enable_spacy: bool = False
    enable_saved_pii_json: bool = True
    gliner_model: str = "knowledgator/gliner-pii-large-v1.0"
    piiranha_model: str = "iiiorg/piiranha-v1-detect-personal-information"
    spacy_model: str = "en_core_web_sm"
    device: str = "auto"  # auto, cuda, cpu
    min_gpu_mem_gb_for_neural_pii: float = 12.0
    torch_float32_matmul_precision: Optional[str] = "high"
    batch_size: int = 16
    chunk_chars: int = 1800
    chunk_overlap_chars: int = 240
    gliner_threshold: float = 0.35
    piiranha_threshold: float = 0.45
    mask_clinical_phi: bool = False
    gliner_labels: List[str] = field(default_factory=lambda: [
        "person name",
        "age",
        "date of birth",
        "location address",
        "location street",
        "location city",
        "location state",
        "location zip",
        "phone number",
        "email address",
        "aadhaar number",
        "insurance id",
        "account number",
        "credit card number",
        "credit card expiration",
        "bank account",
        "routing number",
        "ssn",
        "doctor name",
        "institution name",
        "username",
        "password",
        "passport number",
        "driver license",
        "ip address",
        "url",
        "id number",
    ])
    clinical_phi_labels: List[str] = field(default_factory=lambda: [
        "medical condition",
        "lab result",
        "medication",
        "medical procedure",
        "diagnosis",
    ])


@dataclass
class MaskingConfig:
    mode: str = "silence"  # silence, beep, noise
    target_channels: str = "detected_channel"  # detected_channel or both
    pad_sec: float = 0.12
    min_duration_sec: float = 0.30
    merge_gap_sec: float = 0.05
    beep_freq_hz: float = 1000.0
    beep_gain: float = 0.35
    noise_gain: float = 0.03
    fade_ms: int = 8
    copy_input_if_no_pii: bool = True
    output_sample_rate: int = 48000
    output_channels: int = 2
    opus_bitrate: str = "64k"
    preserve_input_bitrate: bool = True
    # libopus stereo refuses very low rates (e.g. 16k) and FFmpeg's encoder pipe
    # closes with a BrokenPipe before any audio is consumed. We floor the resolved
    # bitrate to keep encodes valid for low-bitrate sources (e.g. 15 kb/s VoIP).
    opus_min_bitrate_kbps: int = 24
    opus_application: str = "voip"
    opus_vbr: str = "on"
    opus_compression_level: int = 5
    opus_frame_duration_ms: int = 20
    unmapped_entity_policy: str = "mask_full_channel"


@dataclass
class RuntimeConfig:
    resume: bool = True
    limit: Optional[int] = None
    shard_index: int = 0
    shard_count: int = 1
    performance_profile: str = "default"  # default or a10g_24gb
    pipeline_schedule: str = "file_major"  # file_major or model_major

    # True multi-file batching. This batches transcript-only ASR engines and neural PII
    # across several files while still using Whisper as the word-timestamp anchor.
    # Keep this conservative for long full-call audio because decoded 48 kHz stereo
    # buffers are held until the batch is finalized.
    file_batch_size: int = 2
    file_batch_max_decoded_audio_gb: float = 2.0
    adaptive_file_batching: bool = True
    adaptive_batch_min_size: int = 1
    min_free_gpu_mem_gb: float = 2.0
    write_perf_metrics: bool = False
    delete_asr_cache_after_finalize: bool = False
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ffmpeg_threads: int = 1
    num_manifest_workers: int = 8
    keep_temp: bool = False
    fail_fast: bool = False
    log_every: int = 10
    write_csv_reports: bool = True
    validate_outputs: bool = True
    validate_existing_outputs: bool = True
    sidecar_include_words: bool = True
    atomic_output: bool = True
    copy_unmasked_when_no_pii: bool = True
    unmasked_copy_method: str = "hardlink_or_copy"
    max_csv_report_rows: int = 10000
    delete_failed_partial_outputs: bool = True
    random_seed: int = 42


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _make_dataclass(cls, values: Dict[str, Any]):
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    clean = {k: v for k, v in values.items() if k in allowed}
    return cls(**clean)


def load_config(path: Optional[str | Path] = None, apply_profile: bool = True) -> PipelineConfig:
    default = _dataclass_to_dict(PipelineConfig())
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merged = _deep_update(default, loaded)
    else:
        merged = default

    config = PipelineConfig(
        paths=_make_dataclass(PathConfig, merged.get("paths", {})),
        asr=_make_dataclass(ASRConfig, merged.get("asr", {})),
        pii=_make_dataclass(PIIConfig, merged.get("pii", {})),
        masking=_make_dataclass(MaskingConfig, merged.get("masking", {})),
        runtime=_make_dataclass(RuntimeConfig, merged.get("runtime", {})),
    )
    if apply_profile:
        apply_performance_profile(config)
    return validate_config(config)


def apply_performance_profile(config: PipelineConfig) -> PipelineConfig:
    profile = str(getattr(config.runtime, "performance_profile", "default") or "default")
    if profile == "default":
        return config
    if profile not in VALID_PERFORMANCE_PROFILES:
        raise ValueError("runtime.performance_profile must be default or a10g_24gb")

    config.asr.model_residency = "keep_loaded"
    config.runtime.file_batch_size = 4
    config.runtime.file_batch_max_decoded_audio_gb = 4.0
    config.runtime.adaptive_file_batching = True
    config.runtime.adaptive_batch_min_size = 1
    config.runtime.min_free_gpu_mem_gb = 3.0
    config.runtime.write_perf_metrics = True
    config.pii.batch_size = 32

    whisper = config.asr.engines.get("whisper", {})
    whisper["batch_size"] = max(8, int(whisper.get("batch_size", 8)))
    whisper["use_batched_pipeline"] = True
    config.asr.engines["whisper"] = whisper

    for engine_name in ("qwen", "cohere", "granite"):
        if engine_name not in config.asr.engines:
            continue
        engine_cfg = config.asr.engines[engine_name]
        engine_cfg["batch_size"] = max(4, int(engine_cfg.get("batch_size", 2)))
        config.asr.engines[engine_name] = engine_cfg
    return config


def enabled_asr_engines(config: PipelineConfig) -> List[str]:
    engines = config.asr.engines or {}
    order = config.asr.engine_order or list(engines.keys())
    return [name for name in order if bool(engines.get(name, {}).get("enabled", True))]


def validate_config(config: PipelineConfig) -> PipelineConfig:
    if config.runtime.performance_profile not in VALID_PERFORMANCE_PROFILES:
        raise ValueError("runtime.performance_profile must be default or a10g_24gb")
    if config.runtime.pipeline_schedule not in VALID_PIPELINE_SCHEDULES:
        raise ValueError("runtime.pipeline_schedule must be file_major or model_major")
    if config.asr.mode not in {"per_channel", "mono_mix"}:
        raise ValueError("asr.mode must be 'per_channel' or 'mono_mix'")
    if config.asr.input_audio_strategy not in {"single_decode", "ffmpeg_temp_wav"}:
        raise ValueError("asr.input_audio_strategy must be 'single_decode' or 'ffmpeg_temp_wav'")
    if not config.asr.word_timestamps:
        raise ValueError("asr.word_timestamps must be true because audio masking depends on word-level timestamps")
    if int(config.asr.batch_size) < 1:
        raise ValueError("asr.batch_size must be >= 1")
    if int(config.asr.channel_wav_sample_rate) <= 0:
        raise ValueError("asr.channel_wav_sample_rate must be positive")
    if config.asr.model_residency not in {"keep_loaded", "unload_after_file"}:
        raise ValueError("asr.model_residency must be keep_loaded or unload_after_file")
    if config.asr.pii_detection_transcript_scope not in {"final_only", "all_engines_only", "final_and_all_engines"}:
        raise ValueError("asr.pii_detection_transcript_scope must be final_only, all_engines_only, or final_and_all_engines")

    known_kinds = {"faster_whisper", "whisper", "qwen", "cohere", "granite"}
    if not isinstance(config.asr.engines, dict) or not config.asr.engines:
        raise ValueError("asr.engines must define at least the whisper engine")
    for name in config.asr.engine_order:
        if name not in config.asr.engines:
            raise ValueError(f"asr.engine_order contains unknown engine: {name}")
    for name, engine_cfg in config.asr.engines.items():
        kind = str(engine_cfg.get("kind", name))
        if kind not in known_kinds:
            raise ValueError(f"Unsupported ASR engine kind for {name}: {kind}")
        if bool(engine_cfg.get("enabled", True)):
            model_dir = engine_cfg.get("model_dir") or (config.asr.whisper_model_dir if kind in {"faster_whisper", "whisper"} else None)
            if not model_dir:
                raise ValueError(f"Enabled ASR engine {name} requires model_dir")
            if int(engine_cfg.get("batch_size", config.asr.batch_size)) < 1:
                raise ValueError(f"ASR engine {name} batch_size must be >= 1")

    enabled = enabled_asr_engines(config)
    if not enabled:
        raise ValueError("At least one ASR engine must be enabled")
    if config.asr.timestamp_anchor_engine not in enabled:
        raise ValueError("asr.timestamp_anchor_engine must be enabled. Keep whisper enabled unless you implement another timestamp-capable engine.")
    anchor_cfg = config.asr.engines.get(config.asr.timestamp_anchor_engine, {})
    if str(anchor_cfg.get("kind", config.asr.timestamp_anchor_engine)) not in {"faster_whisper", "whisper"}:
        raise ValueError("The timestamp anchor must currently be a faster_whisper engine because PII masking needs word timestamps")
    if not bool(anchor_cfg.get("word_timestamps", config.asr.word_timestamps)):
        raise ValueError("The timestamp anchor engine must have word_timestamps=true")
    if int(anchor_cfg.get("batch_size", config.asr.batch_size)) < 1:
        raise ValueError("ASR engine batch_size must be >= 1")
    consensus = config.asr.consensus or {}
    if int(consensus.get("min_agreement", 2)) < 1:
        raise ValueError("asr.consensus.min_agreement must be >= 1")

    if config.masking.mode not in {"beep", "silence", "noise"}:
        raise ValueError("masking.mode must be one of: beep, silence, noise")
    if config.masking.target_channels not in {"detected_channel", "both"}:
        raise ValueError("masking.target_channels must be 'detected_channel' or 'both'")
    if int(config.masking.output_sample_rate) != 48000:
        raise ValueError("masking.output_sample_rate must be 48000 for the required output format")
    if int(config.masking.output_channels) != 2:
        raise ValueError("masking.output_channels must be 2 for the required stereo output format")
    if config.masking.unmapped_entity_policy not in {"mask_full_channel", "fail", "copy_original"}:
        raise ValueError("masking.unmapped_entity_policy must be mask_full_channel, fail, or copy_original")
    if config.masking.unmapped_entity_policy == "copy_original":
        raise ValueError("masking.unmapped_entity_policy='copy_original' is unsafe for de-identification")
    if float(config.masking.pad_sec) < 0 or float(config.masking.min_duration_sec) < 0:
        raise ValueError("masking.pad_sec and masking.min_duration_sec must be non-negative")
    if int(getattr(config.masking, "opus_min_bitrate_kbps", 24)) < 6:
        raise ValueError("masking.opus_min_bitrate_kbps must be >= 6 to keep libopus encodes valid")

    if int(config.pii.batch_size) < 1:
        raise ValueError("pii.batch_size must be >= 1")
    if int(config.pii.chunk_chars) < 256:
        raise ValueError("pii.chunk_chars must be >= 256")
    if int(config.pii.chunk_overlap_chars) < 0:
        raise ValueError("pii.chunk_overlap_chars must be >= 0")

    if int(config.runtime.shard_count) < 1:
        raise ValueError("runtime.shard_count must be >= 1")
    if not (0 <= int(config.runtime.shard_index) < int(config.runtime.shard_count)):
        raise ValueError("runtime.shard_index must satisfy 0 <= shard_index < shard_count")
    if int(getattr(config.runtime, "file_batch_size", 1)) < 1:
        raise ValueError("runtime.file_batch_size must be >= 1")
    if int(getattr(config.runtime, "adaptive_batch_min_size", 1)) < 1:
        raise ValueError("runtime.adaptive_batch_min_size must be >= 1")
    if float(getattr(config.runtime, "file_batch_max_decoded_audio_gb", 0.0)) <= 0:
        raise ValueError("runtime.file_batch_max_decoded_audio_gb must be > 0")
    if float(getattr(config.runtime, "min_free_gpu_mem_gb", 0.0)) < 0:
        raise ValueError("runtime.min_free_gpu_mem_gb must be >= 0")
    if config.runtime.unmasked_copy_method not in {"hardlink_or_copy", "copy"}:
        raise ValueError("runtime.unmasked_copy_method must be hardlink_or_copy or copy. Symlinks are not allowed for deliverable audio.")

    in_root = Path(config.paths.input_root).resolve()
    out_root = Path(config.paths.output_root).resolve()
    work_dir = Path(config.paths.work_dir).resolve()
    if out_root == in_root:
        raise ValueError("paths.output_root must not equal paths.input_root")
    if work_dir == in_root:
        raise ValueError("paths.work_dir must not equal paths.input_root")
    return config


def save_config(config: PipelineConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_dataclass_to_dict(config), f, sort_keys=False, allow_unicode=True)
