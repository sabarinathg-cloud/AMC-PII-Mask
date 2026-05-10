from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from .audio_io import ffprobe_audio


def validate_masked_file(
    input_path: str | Path,
    output_path: str | Path,
    ffprobe_path: str = "ffprobe",
    expected_sample_rate: int = 48000,
    expected_channels: int = 2,
    max_duration_delta_sec: float = 0.20,
    input_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    in_meta = input_meta if input_meta is not None else ffprobe_audio(input_path, ffprobe_path=ffprobe_path)
    out_meta = ffprobe_audio(output_path, ffprobe_path=ffprobe_path)
    duration_delta = None
    if in_meta.get("duration_sec") is not None and out_meta.get("duration_sec") is not None:
        duration_delta = abs(float(in_meta["duration_sec"]) - float(out_meta["duration_sec"]))

    checks = {
        "output_exists": Path(output_path).exists(),
        "codec_is_opus": out_meta.get("codec_name") == "opus",
        "sample_rate_ok": int(out_meta.get("sample_rate") or 0) == int(expected_sample_rate),
        "channels_ok": int(out_meta.get("channels") or 0) == int(expected_channels),
        "duration_ok": duration_delta is None or duration_delta <= max_duration_delta_sec,
    }
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_meta": in_meta,
        "output_meta": out_meta,
        "duration_delta_sec": duration_delta,
        "checks": checks,
        "valid": all(checks.values()),
    }
