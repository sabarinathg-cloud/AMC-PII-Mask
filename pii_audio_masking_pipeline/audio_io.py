from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, List
import json
import logging
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


class AudioCommandError(RuntimeError):
    pass


def run_cmd(cmd: list[str], input_bytes: bytes | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        raise AudioCommandError(
            "Command failed:\n"
            + " ".join(str(c) for c in cmd)
            + "\nSTDERR:\n"
            + proc.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    return proc


def ffprobe_audio(path: str | Path, ffprobe_path: str = "ffprobe") -> Dict[str, Any]:
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,channel_layout,bit_rate,duration:format=duration,bit_rate,format_name",
        "-of", "json",
        str(path),
    ]
    proc = run_cmd(cmd)
    obj = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    stream = (obj.get("streams") or [{}])[0]
    fmt = obj.get("format") or {}

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    duration = to_float(stream.get("duration")) or to_float(fmt.get("duration"))
    bit_rate = to_int(stream.get("bit_rate")) or to_int(fmt.get("bit_rate"))
    return {
        "path": str(path),
        "codec_name": stream.get("codec_name"),
        "sample_rate": to_int(stream.get("sample_rate")),
        "channels": to_int(stream.get("channels")),
        "channel_layout": stream.get("channel_layout"),
        "duration_sec": duration,
        "bit_rate": bit_rate,
        "format_name": fmt.get("format_name"),
    }


def input_matches_required_opus(meta: Dict[str, Any], sample_rate: int = 48000, channels: int = 2) -> bool:
    return (
        str(meta.get("codec_name") or "").lower() == "opus"
        and int(meta.get("sample_rate") or 0) == int(sample_rate)
        and int(meta.get("channels") or 0) == int(channels)
    )


# Compatibility alias used by tests and earlier package versions.
def input_matches_required_output(meta: Dict[str, Any], sample_rate: int = 48000, channels: int = 2) -> bool:
    return input_matches_required_opus(meta, sample_rate=sample_rate, channels=channels)


def atomic_copy(src: str | Path, dst: str | Path) -> Path:
    return copy_audio_passthrough(src, dst, method="copy")


def copy_audio_passthrough(input_path: str | Path, output_path: str | Path, method: str = "hardlink_or_copy") -> Path:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink(missing_ok=True)
        if method == "symlink":
            raise ValueError("Symlink outputs are not allowed for deliverable audio. Use hardlink_or_copy or copy.")
        elif method == "hardlink_or_copy":
            try:
                os.link(src, tmp)
            except Exception:
                shutil.copy2(src, tmp)
        elif method == "copy":
            shutil.copy2(src, tmp)
        else:
            raise ValueError(f"Unsupported unmasked_copy_method: {method}")
        os.replace(tmp, dst)
    finally:
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        except Exception:
            pass
    return dst


def decode_to_float32_stereo_48k(
    input_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    threads: int = 1,
    sample_rate: int = 48000,
    channels: int = 2,
) -> np.ndarray:
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-i", str(input_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "pipe:1",
    ]
    proc = run_cmd(cmd)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size % channels != 0:
        audio = audio[: audio.size - (audio.size % channels)]
    return audio.reshape(-1, channels).copy()


def resample_mono_float32(audio: np.ndarray, source_sr: int = 48000, target_sr: int = 16000) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(source_sr) == int(target_sr):
        return np.ascontiguousarray(x, dtype=np.float32)
    # Reduce ratio to keep polyphase filter efficient.
    import math
    g = math.gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(source_sr) // g
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return np.ascontiguousarray(y, dtype=np.float32)


def make_asr_audio_inputs(
    audio: np.ndarray,
    mode: str = "per_channel",
    input_channels: int | None = None,
    source_sr: int = 48000,
    target_sr: int = 16000,
) -> List[Dict[str, Any]]:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected audio as [samples, channels], got shape={x.shape}")
    n_channels = int(input_channels or x.shape[1])
    n_channels = min(max(n_channels, 1), x.shape[1], 2)

    if mode == "mono_mix":
        mono = np.mean(x[:, :n_channels], axis=1, dtype=np.float32)
        return [{"channel": -1, "audio": resample_mono_float32(mono, source_sr, target_sr), "sample_rate": int(target_sr)}]
    if mode != "per_channel":
        raise ValueError(f"Unsupported ASR mode: {mode}")

    rows: list[dict] = []
    for ch in range(n_channels):
        mono = x[:, ch]
        rows.append({"channel": ch, "audio": resample_mono_float32(mono, source_sr, target_sr), "sample_rate": int(target_sr)})
    return rows


def stereo_to_asr_inputs(audio: np.ndarray, source_sr: int = 48000, target_sr: int = 16000, mode: str = "per_channel"):
    return [(int(row["channel"]), row["audio"]) for row in make_asr_audio_inputs(audio, mode=mode, source_sr=source_sr, target_sr=target_sr)]


def write_mono_wav_pcm16(audio: np.ndarray, output_wav: str | Path, sample_rate: int = 16000) -> Path:
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return output_wav


def extract_channel_wav(
    input_path: str | Path,
    output_wav: str | Path,
    channel: int,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    threads: int = 1,
) -> Path:
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    af = f"pan=mono|c0=c{channel}"
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-i", str(input_path),
        "-af", af,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd)
    return output_wav


def extract_stereo_channels_wav(
    input_path: str | Path,
    output_ch0_wav: str | Path,
    output_ch1_wav: str | Path,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    threads: int = 1,
) -> tuple[Path, Path]:
    output_ch0_wav = Path(output_ch0_wav)
    output_ch1_wav = Path(output_ch1_wav)
    output_ch0_wav.parent.mkdir(parents=True, exist_ok=True)
    output_ch1_wav.parent.mkdir(parents=True, exist_ok=True)
    filt = "[0:a]asplit=2[a0][a1];[a0]pan=mono|c0=c0[ch0];[a1]pan=mono|c0=c1[ch1]"
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-i", str(input_path),
        "-filter_complex", filt,
        "-map", "[ch0]", "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(output_ch0_wav),
        "-map", "[ch1]", "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(output_ch1_wav),
    ]
    run_cmd(cmd)
    return output_ch0_wav, output_ch1_wav


def extract_mono_mix_wav(
    input_path: str | Path,
    output_wav: str | Path,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    threads: int = 1,
) -> Path:
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    run_cmd(cmd)
    return output_wav


def encode_float32_stereo_to_opus(
    audio: np.ndarray,
    output_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    threads: int = 1,
    sample_rate: int = 48000,
    channels: int = 2,
    bitrate: str = "64k",
    application: str = "voip",
    vbr: str = "on",
    compression_level: int | None = None,
    frame_duration_ms: int | None = None,
    atomic: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if x.shape[1] != channels:
        if x.shape[1] == 1 and channels == 2:
            x = np.repeat(x, 2, axis=1)
        else:
            raise ValueError(f"Expected {channels} channels, got shape={x.shape}")
    x = np.clip(x, -1.0, 1.0)
    x = np.ascontiguousarray(x, dtype=np.float32)

    final_path = output_path
    tmp_path: Path | None = None
    encode_path = final_path
    if atomic:
        fd, name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp.opus", dir=str(output_path.parent))
        os.close(fd)
        tmp_path = Path(name)
        encode_path = tmp_path

    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "pipe:0",
        "-vn",
        "-c:a", "libopus",
        "-application", application,
        "-b:a", str(bitrate),
        "-vbr", str(vbr),
    ]
    if compression_level is not None:
        cmd += ["-compression_level", str(int(compression_level))]
    if frame_duration_ms is not None:
        cmd += ["-frame_duration", str(int(frame_duration_ms))]
    cmd.append(str(encode_path))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    data = memoryview(x).cast("B")
    chunk = 16 * 1024 * 1024
    stderr = b""
    try:
        for offset in range(0, len(data), chunk):
            proc.stdin.write(data[offset : offset + chunk])
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stdout is not None:
            proc.stdout.read()
        code = proc.wait()
    except Exception:
        if proc.poll() is None:
            proc.kill()
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    if code != 0:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise AudioCommandError(
            "Opus encode failed:\n"
            + " ".join(str(c) for c in cmd)
            + "\nSTDERR:\n"
            + stderr.decode("utf-8", errors="replace")[-4000:]
        )
    if tmp_path is not None:
        os.replace(tmp_path, final_path)
    return final_path


def _fade_envelope(n: int, fade_samples: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    fade = min(fade_samples, max(1, n // 2))
    if fade > 1 and n > 2:
        env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return env


def apply_mask_spans(
    audio: np.ndarray,
    spans: Iterable[dict],
    sr: int = 48000,
    mode: str = "silence",
    target_channels: str = "detected_channel",
    beep_freq_hz: float = 1000.0,
    beep_gain: float = 0.35,
    noise_gain: float = 0.03,
    fade_ms: int = 8,
    random_seed: int = 42,
    inplace: bool = True,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if not inplace:
        x = x.copy()
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)

    n_samples, n_channels = x.shape
    fade_samples = max(1, int(sr * fade_ms / 1000.0))
    peak_val = float(np.max(np.abs(x))) if x.size else 0.0
    peak = peak_val if peak_val > 0 else 1.0
    beep_amp = beep_gain * peak
    noise_amp = noise_gain * peak
    rng = np.random.default_rng(random_seed)

    for span in spans:
        start = max(0, int(round(float(span["start"]) * sr)))
        end = min(n_samples, int(round(float(span["end"]) * sr)))
        if end <= start:
            continue
        if target_channels == "both" or span.get("channel") in (-1, "both", None):
            channel_indices = list(range(n_channels))
        else:
            ch = int(span.get("channel", 0))
            channel_indices = [ch] if 0 <= ch < n_channels else list(range(n_channels))
        n = end - start
        env = _fade_envelope(n, fade_samples)
        if mode == "silence":
            replacement = np.zeros(n, dtype=np.float32)
        elif mode == "noise":
            replacement = rng.normal(0.0, noise_amp, n).astype(np.float32) * env
        elif mode == "beep":
            t = np.arange(n, dtype=np.float32) / float(sr)
            replacement = (beep_amp * np.sin(2.0 * np.pi * beep_freq_hz * t)).astype(np.float32) * env
        else:
            raise ValueError(f"Unsupported mask mode: {mode}")
        for ch in channel_indices:
            x[start:end, ch] = replacement
    return x
