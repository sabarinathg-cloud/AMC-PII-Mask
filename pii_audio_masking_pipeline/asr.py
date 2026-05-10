from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import gc
import logging
import math
import re
import wave

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordRecord:
    word: str
    start: float
    end: float
    probability: Optional[float]
    start_char: int
    end_char: int
    segment_id: int
    channel: int


@dataclass
class ASRResult:
    channel: int
    transcript: str
    words: List[Dict[str, Any]]
    engine: str = "whisper"
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    timestamp_retry_used: bool = False
    timestamp_suspicious: bool = False
    error: Optional[str] = None
    file_id: Optional[str] = None


@dataclass
class ChannelASRBundle:
    channel: int
    final_transcript: str
    final_words: List[Dict[str, Any]]
    engine_results: List[ASRResult]
    anchor_engine: Optional[str]
    anchor_words: List[Dict[str, Any]]
    consensus: Dict[str, Any]
    file_id: Optional[str] = None


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def _engine_get(engine_cfg: Optional[Dict[str, Any]], name: str, default: Any = None) -> Any:
    if not engine_cfg:
        return default
    return engine_cfg.get(name, default)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def normalize_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def normalize_for_consensus(text: Any) -> str:
    text = str(text or "").strip().lower()
    if not text:
        return ""
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    # Keep digits and words. Remove punctuation because different ASR models punctuate differently.
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein_tokens(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    return float(_levenshtein_tokens(r, h) / max(1, len(r)))


def _cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    return float(_levenshtein_tokens(r, h) / max(1, len(r)))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def transcript_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(max(0.0, min(1.0, 0.45 * (1.0 - _wer(a, b)) + 0.35 * (1.0 - _cer(a, b)) + 0.20 * _jaccard(a, b))))


def tokenize_with_char_spans(text: str) -> List[Dict[str, Any]]:
    rows: list[dict] = []
    for m in re.finditer(r"[A-Za-z0-9@._%+\-']+", str(text or "")):
        tok = m.group(0)
        norm = normalize_token(tok)
        if not norm:
            continue
        rows.append({"word": tok, "norm": norm, "start_char": m.start(), "end_char": m.end()})
    return rows


def build_canonical_transcript(words: List[Dict[str, Any]], channel: int) -> tuple[str, List[Dict[str, Any]]]:
    text_parts: list[str] = []
    canonical_words: list[Dict[str, Any]] = []
    cursor = 0

    for w in words:
        raw = str(w.get("word", "")).strip()
        if not raw:
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start_char = cursor
        text_parts.append(raw)
        cursor += len(raw)
        end_char = cursor
        row = dict(w)
        row.update({
            "word": raw,
            "start_char": start_char,
            "end_char": end_char,
            "channel": channel,
        })
        canonical_words.append(row)

    return "".join(text_parts), canonical_words


def align_transcript_to_timed_words(
    transcript: str,
    timed_words: List[Dict[str, Any]],
    channel: int,
    conservative_replacements: bool = True,
) -> List[Dict[str, Any]]:
    """Project transcript character spans onto a timestamp-bearing word list.

    Non-Whisper engines usually do not provide word timestamps. This function aligns their
    tokens to the timestamp-anchor engine, normally faster-whisper. Only mapped tokens are
    returned. If an entity lands on unmapped tokens, the pipeline's fail-safe policy handles it.
    """
    if not transcript or not timed_words:
        return []

    txt_tokens = tokenize_with_char_spans(transcript)
    if not txt_tokens:
        return []

    anchor_rows: list[dict] = []
    for idx, w in enumerate(timed_words):
        raw = str(w.get("word", "")).strip()
        norm = normalize_token(raw)
        if not norm:
            continue
        row = dict(w)
        row["_anchor_index"] = idx
        row["_norm"] = norm
        anchor_rows.append(row)

    if not anchor_rows:
        return []

    txt_norms = [t["norm"] for t in txt_tokens]
    anchor_norms = [w["_norm"] for w in anchor_rows]
    mapping: Dict[int, int] = {}

    matcher = SequenceMatcher(None, txt_norms, anchor_norms, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ti, wi in zip(range(i1, i2), range(j1, j2)):
                mapping[ti] = wi
        elif tag == "replace" and conservative_replacements:
            # Allow pairwise projection for same-length replacements. This catches punctuation,
            # minor spelling, and number-format differences without hallucinating long spans.
            if (i2 - i1) == (j2 - j1):
                for ti, wi in zip(range(i1, i2), range(j1, j2)):
                    a = txt_norms[ti]
                    b = anchor_norms[wi]
                    if a == b or a.isdigit() or b.isdigit() or transcript_similarity(a, b) >= 0.55:
                        mapping[ti] = wi

    out: list[dict] = []
    for ti, wi in sorted(mapping.items()):
        tok = txt_tokens[ti]
        anchor = anchor_rows[wi]
        start = float(anchor.get("start", 0.0) or 0.0)
        end = float(anchor.get("end", start) or start)
        if end < start:
            end = start
        out.append({
            "word": tok["word"],
            "start": start,
            "end": end,
            "probability": anchor.get("probability"),
            "start_char": int(tok["start_char"]),
            "end_char": int(tok["end_char"]),
            "segment_id": int(anchor.get("segment_id", 0) or 0),
            "channel": channel,
            "anchor_word": anchor.get("word"),
            "anchor_engine": anchor.get("engine"),
        })
    return out


def build_consensus(results: Sequence[ASRResult], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or {}
    fallback_priority = list(cfg.get("fallback_priority", ["whisper", "qwen", "cohere", "granite"]))
    min_agreement = int(cfg.get("min_agreement", 2))
    soft_threshold = float(cfg.get("soft_similarity_threshold", 0.78))
    prefer_engine = str(cfg.get("prefer_engine_on_tie", "whisper"))

    texts_by_engine: Dict[str, str] = {}
    normalized: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for r in results:
        if r.error:
            errors[r.engine] = r.error
        text = str(r.transcript or "").strip()
        if text:
            texts_by_engine[r.engine] = text
            normalized[r.engine] = normalize_for_consensus(text)

    non_empty = {k: v for k, v in normalized.items() if v}
    if not non_empty:
        return {
            "final_transcript": "",
            "method": "all_empty",
            "selected_engine": None,
            "agreement_count": 0,
            "agreed_engines": [],
            "avg_similarity": 0.0,
            "normalized_by_engine": normalized,
            "errors": errors,
        }

    # Strict majority on normalized text.
    counts: Dict[str, List[str]] = {}
    for engine, text in non_empty.items():
        counts.setdefault(text, []).append(engine)
    majority_text, majority_engines = max(counts.items(), key=lambda kv: (len(kv[1]), prefer_engine in kv[1], len(kv[0])))
    if len(majority_engines) >= min_agreement:
        selected_engine = sorted(majority_engines, key=lambda e: (e != prefer_engine, fallback_priority.index(e) if e in fallback_priority else 999))[0]
        return {
            "final_transcript": texts_by_engine.get(selected_engine, majority_text),
            "method": "strict_majority",
            "selected_engine": selected_engine,
            "agreement_count": len(majority_engines),
            "agreed_engines": sorted(majority_engines),
            "avg_similarity": 1.0,
            "normalized_by_engine": normalized,
            "errors": errors,
        }

    # Soft center: pick the transcript most similar to the others.
    best: Optional[tuple] = None
    for engine, text in non_empty.items():
        sims = {m: transcript_similarity(text, other) for m, other in non_empty.items()}
        agreed = [m for m, score in sims.items() if score >= soft_threshold]
        avg_sim = float(np.mean(list(sims.values()))) if sims else 0.0
        priority = -(fallback_priority.index(engine) if engine in fallback_priority else 999)
        tie = 1 if engine == prefer_engine else 0
        score_tuple = (len(agreed), avg_sim, tie, priority, len(text))
        if best is None or score_tuple > best[0]:
            best = (score_tuple, engine, text, agreed, avg_sim)

    assert best is not None
    _, selected_engine, selected_norm, agreed, avg_sim = best
    if len(agreed) >= min_agreement:
        return {
            "final_transcript": texts_by_engine.get(selected_engine, selected_norm),
            "method": "soft_similarity",
            "selected_engine": selected_engine,
            "agreement_count": len(agreed),
            "agreed_engines": sorted(agreed),
            "avg_similarity": avg_sim,
            "normalized_by_engine": normalized,
            "errors": errors,
        }

    # PII masking cannot reject every disagreement the way an ASR training dataset can.
    # Pick the best available transcript, but still run PII detection over all enabled transcripts.
    for engine in fallback_priority:
        if engine in texts_by_engine:
            return {
                "final_transcript": texts_by_engine[engine],
                "method": "fallback_priority",
                "selected_engine": engine,
                "agreement_count": len(agreed),
                "agreed_engines": sorted(agreed),
                "avg_similarity": avg_sim,
                "normalized_by_engine": normalized,
                "errors": errors,
            }

    engine = next(iter(texts_by_engine))
    return {
        "final_transcript": texts_by_engine[engine],
        "method": "first_non_empty",
        "selected_engine": engine,
        "agreement_count": len(agreed),
        "agreed_engines": sorted(agreed),
        "avg_similarity": avg_sim,
        "normalized_by_engine": normalized,
        "errors": errors,
    }


def _read_wav_mono_float32(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV for ASR helper, got sample_width={sample_width}")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


class FasterWhisperASR:
    kind = "faster_whisper"
    supports_audio_input = True
    requires_timestamps = True

    def __init__(self, cfg: Any, engine_name: str = "whisper", engine_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.engine_name = engine_name
        self.engine_cfg = engine_cfg or {}
        from faster_whisper import WhisperModel

        device = _engine_get(self.engine_cfg, "device", _cfg_get(cfg, "device", "auto"))
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"

        compute_type = str(_engine_get(self.engine_cfg, "compute_type", _cfg_get(cfg, "compute_type", "float16")) or "float16")
        if device == "cpu" and compute_type in {"float16", "float32", "bfloat16"}:
            compute_type = "int8"

        model_dir = _engine_get(self.engine_cfg, "model_dir", _cfg_get(cfg, "whisper_model_dir", None))
        if not model_dir:
            raise ValueError("Whisper engine requires asr.engines.whisper.model_dir or asr.whisper_model_dir")

        self.device = device
        self.compute_type = compute_type
        logger.info("Loading faster-whisper engine=%s model=%s device=%s compute_type=%s", engine_name, model_dir, device, compute_type)
        model_kwargs = dict(_engine_get(self.engine_cfg, "model_kwargs", {}) or {})
        self.base_model = WhisperModel(str(model_dir), device=device, compute_type=compute_type, **model_kwargs)
        self.model = self.base_model
        self.using_batched_pipeline = False

        if bool(_engine_get(self.engine_cfg, "use_batched_pipeline", _cfg_get(cfg, "use_batched_pipeline", True))):
            try:
                from faster_whisper import BatchedInferencePipeline
                self.model = BatchedInferencePipeline(model=self.base_model)
                self.using_batched_pipeline = True
                logger.info("Using faster-whisper BatchedInferencePipeline for engine=%s batch_size=%s", engine_name, self._get_int("batch_size", 8))
            except Exception as e:
                logger.warning("Could not enable BatchedInferencePipeline. Falling back to WhisperModel. Error: %s", e)

    def _get(self, name: str, default: Any = None) -> Any:
        return _engine_get(self.engine_cfg, name, _cfg_get(self.cfg, name, default))

    def _get_int(self, name: str, default: int) -> int:
        return int(self._get(name, default))

    def _transcribe_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "language": self._get("language", _cfg_get(self.cfg, "language", "en")),
            "beam_size": self._get_int("beam_size", 1),
            "best_of": self._get_int("best_of", 1),
            "temperature": float(self._get("temperature", 0.0)),
            "vad_filter": bool(self._get("vad_filter", True)),
            "word_timestamps": bool(self._get("word_timestamps", True)),
            "condition_on_previous_text": bool(self._get("condition_on_previous_text", False)),
        }
        initial_prompt = self._get("initial_prompt", None)
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        vad_parameters = self._get("vad_parameters", None)
        if vad_parameters:
            kwargs["vad_parameters"] = vad_parameters
        return kwargs

    def transcribe_path(self, audio_path: str | Path, channel: int) -> ASRResult:
        return self._transcribe(audio_path, channel=channel, audio_duration_sec=None)

    def transcribe_audio(self, audio: np.ndarray, channel: int, sample_rate: int = 16000) -> ASRResult:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"Whisper numpy input must be mono 1-D, got shape={x.shape}")
        duration = float(x.size) / float(sample_rate) if sample_rate > 0 else None
        return self._transcribe(np.ascontiguousarray(x), channel=channel, audio_duration_sec=duration)

    def _transcribe(self, audio_input: str | Path | np.ndarray, channel: int, audio_duration_sec: Optional[float]) -> ASRResult:
        kwargs = self._transcribe_kwargs()
        result = self._run_transcribe(audio_input, channel, kwargs, audio_duration_sec, use_batched=self.using_batched_pipeline)

        if (
            result.timestamp_suspicious
            and self.using_batched_pipeline
            and bool(self._get("retry_base_on_bad_timestamps", True))
        ):
            logger.warning("Suspicious batched Whisper timestamps on channel=%s. Retrying with base WhisperModel.", channel)
            fallback = self._run_transcribe(audio_input, channel, kwargs, audio_duration_sec, use_batched=False)
            fallback.timestamp_retry_used = True
            return fallback
        return result

    def _run_transcribe(
        self,
        audio_input: str | Path | np.ndarray,
        channel: int,
        kwargs: Dict[str, Any],
        audio_duration_sec: Optional[float],
        use_batched: bool,
    ) -> ASRResult:
        call_kwargs = dict(kwargs)
        model = self.model if use_batched else self.base_model
        if use_batched:
            call_kwargs["batch_size"] = self._get_int("batch_size", 8)
        try:
            segments_iter, info = model.transcribe(audio_input if not isinstance(audio_input, Path) else str(audio_input), **call_kwargs)
        except TypeError as e:
            if use_batched:
                logger.warning("Batched transcribe signature failed, falling back to base WhisperModel: %s", e)
                call_kwargs.pop("batch_size", None)
                segments_iter, info = self.base_model.transcribe(audio_input if not isinstance(audio_input, Path) else str(audio_input), **call_kwargs)
            else:
                raise
        segments = list(segments_iter)
        return self._segments_to_result(segments, info, channel=channel, audio_duration_sec=audio_duration_sec)

    def _segments_to_result(self, segments: list[Any], info: Any, channel: int, audio_duration_sec: Optional[float]) -> ASRResult:
        words: list[Dict[str, Any]] = []
        segment_texts: list[str] = []
        info_duration = getattr(info, "duration", None)
        duration = audio_duration_sec
        if duration is None and info_duration is not None:
            try:
                duration = float(info_duration)
            except Exception:
                duration = None

        suspicious = False
        drift = float(self._get("timestamp_max_drift_sec", 1.0) or 1.0)
        for seg_idx, seg in enumerate(segments):
            seg_text = str(getattr(seg, "text", "")).strip()
            if seg_text:
                segment_texts.append(seg_text)
            seg_words = getattr(seg, "words", None)
            if seg_words:
                for w in seg_words:
                    word = str(getattr(w, "word", "")).strip()
                    if not word:
                        continue
                    start = float(getattr(w, "start", getattr(seg, "start", 0.0)) or 0.0)
                    end = float(getattr(w, "end", getattr(seg, "end", start)) or start)
                    if end < start:
                        suspicious = True
                        end = start
                    if duration is not None:
                        if start < -drift or end > duration + drift:
                            suspicious = True
                        start = min(max(0.0, start), max(0.0, duration))
                        end = min(max(start, end), max(0.0, duration))
                    probability = getattr(w, "probability", None)
                    words.append({
                        "word": word,
                        "start": start,
                        "end": end,
                        "probability": float(probability) if probability is not None else None,
                        "segment_id": seg_idx,
                        "engine": self.engine_name,
                    })
            elif seg_text:
                toks = seg_text.split()
                seg_start = float(getattr(seg, "start", 0.0) or 0.0)
                seg_end = float(getattr(seg, "end", seg_start) or seg_start)
                if duration is not None:
                    if seg_start < -drift or seg_end > duration + drift:
                        suspicious = True
                    seg_start = min(max(0.0, seg_start), max(0.0, duration))
                    seg_end = min(max(seg_start, seg_end), max(0.0, duration))
                dur = max(0.01, seg_end - seg_start)
                for i, tok in enumerate(toks):
                    words.append({
                        "word": tok,
                        "start": seg_start + dur * i / max(1, len(toks)),
                        "end": seg_start + dur * (i + 1) / max(1, len(toks)),
                        "probability": None,
                        "segment_id": seg_idx,
                        "engine": self.engine_name,
                    })

        transcript, canonical_words = build_canonical_transcript(words, channel=channel)
        if not transcript:
            fallback_words = []
            for seg_idx, txt in enumerate(segment_texts):
                for tok in txt.split():
                    fallback_words.append({"word": tok, "start": 0.0, "end": 0.0, "probability": None, "segment_id": seg_idx, "engine": self.engine_name})
            transcript, canonical_words = build_canonical_transcript(fallback_words, channel=channel)
        if not transcript:
            transcript = " ".join(segment_texts).strip()

        language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        return ASRResult(
            channel=channel,
            transcript=transcript,
            words=canonical_words,
            engine=self.engine_name,
            language=language,
            language_probability=float(language_probability) if language_probability is not None else None,
            duration=float(duration) if duration is not None else None,
            timestamp_suspicious=suspicious,
        )

    def unload(self) -> None:
        for attr in ("model", "base_model"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        clear_accelerator_cache()


class QwenASR:
    kind = "qwen"
    supports_audio_input = False
    requires_timestamps = False

    def __init__(self, cfg: Any, engine_name: str = "qwen", engine_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.engine_name = engine_name
        self.engine_cfg = engine_cfg or {}
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except Exception as e:
            raise ImportError("Qwen ASR is enabled but qwen_asr/Qwen3ASRModel is not importable. Disable asr.engines.qwen.enabled or install the local Qwen ASR package.") from e

        model_dir = _engine_get(self.engine_cfg, "model_dir", None)
        if not model_dir:
            raise ValueError("Qwen engine requires asr.engines.qwen.model_dir")
        dtype_name = str(_engine_get(self.engine_cfg, "dtype", "bfloat16" if _cuda_available() else "float32"))
        dtype = getattr(torch, dtype_name, torch.bfloat16 if _cuda_available() else torch.float32)
        device_map = _engine_get(self.engine_cfg, "device_map", "cuda:0" if _cuda_available() else "cpu")
        kwargs = dict(_engine_get(self.engine_cfg, "model_kwargs", {}) or {})
        kwargs.setdefault("dtype", dtype)
        kwargs.setdefault("device_map", device_map)
        kwargs.setdefault("max_inference_batch_size", int(_engine_get(self.engine_cfg, "batch_size", 2)))
        kwargs.setdefault("max_new_tokens", int(_engine_get(self.engine_cfg, "max_new_tokens", 256)))
        logger.info("Loading Qwen ASR engine=%s model=%s", engine_name, model_dir)
        self.model = Qwen3ASRModel.from_pretrained(str(model_dir), **kwargs)

    def _text(self, result: Any) -> str:
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        return str(getattr(result, "text", result) or "").strip()

    def transcribe_paths(self, paths: Sequence[str | Path], channels: Sequence[int]) -> List[ASRResult]:
        paths = [str(p) for p in paths]
        channels = [int(c) for c in channels]
        language = str(_engine_get(self.engine_cfg, "language", "English"))
        batch_size = max(1, int(_engine_get(self.engine_cfg, "batch_size", len(paths) or 1)))
        out: list[ASRResult] = []
        for start in range(0, len(paths), batch_size):
            chunk_paths = paths[start:start + batch_size]
            chunk_channels = channels[start:start + batch_size]
            try:
                raw = self.model.transcribe(audio=list(chunk_paths), language=language)
                if not isinstance(raw, list):
                    raw = [raw]
                texts = [self._text(r) for r in raw]
                if len(texts) != len(chunk_paths):
                    raise RuntimeError(f"Qwen returned {len(texts)} outputs for {len(chunk_paths)} inputs")
                out.extend(ASRResult(channel=ch, transcript=txt, words=[], engine=self.engine_name) for ch, txt in zip(chunk_channels, texts))
            except Exception as batch_error:
                logger.warning("Qwen batch transcription failed, falling back one-by-one: %s", batch_error)
                for p, ch in zip(chunk_paths, chunk_channels):
                    try:
                        out.append(ASRResult(channel=ch, transcript=self._text(self.model.transcribe(audio=p, language=language)), words=[], engine=self.engine_name))
                    except Exception as e:
                        out.append(ASRResult(channel=ch, transcript="", words=[], engine=self.engine_name, error=repr(e)))
        return out

    def unload(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        clear_accelerator_cache()


class CohereASR:
    kind = "cohere"
    supports_audio_input = False
    requires_timestamps = False

    def __init__(self, cfg: Any, engine_name: str = "cohere", engine_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.engine_name = engine_name
        self.engine_cfg = engine_cfg or {}
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import CohereAsrForConditionalGeneration as ModelCls
            except Exception:
                from transformers import AutoModelForSpeechSeq2Seq as ModelCls
        except Exception as e:
            raise ImportError("Cohere ASR is enabled but transformers/Cohere ASR classes are not importable. Disable asr.engines.cohere.enabled or install the required transformers build.") from e

        model_dir = _engine_get(self.engine_cfg, "model_dir", None)
        if not model_dir:
            raise ValueError("Cohere engine requires asr.engines.cohere.model_dir")
        logger.info("Loading Cohere ASR engine=%s model=%s", engine_name, model_dir)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=bool(_engine_get(self.engine_cfg, "local_files_only", True)))
        dtype_name = str(_engine_get(self.engine_cfg, "dtype", "float16" if _cuda_available() else "float32"))
        dtype = getattr(torch, dtype_name, torch.float16 if _cuda_available() else torch.float32)
        model_kwargs = dict(_engine_get(self.engine_cfg, "model_kwargs", {}) or {})
        model_kwargs.setdefault("torch_dtype", dtype)
        if _cuda_available():
            model_kwargs.setdefault("device_map", _engine_get(self.engine_cfg, "device_map", "auto"))
        model_kwargs.setdefault("local_files_only", bool(_engine_get(self.engine_cfg, "local_files_only", True)))
        self.model = ModelCls.from_pretrained(str(model_dir), **model_kwargs)
        self.model.eval()
        try:
            self.device = next(self.model.parameters()).device
        except Exception:
            self.device = "cuda" if _cuda_available() else "cpu"

    def _generate(self, paths: Sequence[str | Path]) -> List[str]:
        import torch
        audios = [_read_wav_mono_float32(p) for p in paths]
        language = _engine_get(self.engine_cfg, "language", "en")
        punctuation = bool(_engine_get(self.engine_cfg, "punctuation", True))
        try:
            inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt", padding=True, language=language, punctuation=punctuation)
        except TypeError:
            inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        max_new_tokens = int(_engine_get(self.engine_cfg, "max_new_tokens", 256))
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return [str(x).strip() for x in self.processor.batch_decode(outputs, skip_special_tokens=True)]

    def transcribe_paths(self, paths: Sequence[str | Path], channels: Sequence[int]) -> List[ASRResult]:
        paths = [str(p) for p in paths]
        channels = [int(c) for c in channels]
        batch_size = max(1, int(_engine_get(self.engine_cfg, "batch_size", len(paths) or 1)))
        out: list[ASRResult] = []
        for start in range(0, len(paths), batch_size):
            chunk_paths = paths[start:start + batch_size]
            chunk_channels = channels[start:start + batch_size]
            try:
                texts = self._generate(chunk_paths)
                if len(texts) != len(chunk_paths):
                    raise RuntimeError(f"Cohere returned {len(texts)} outputs for {len(chunk_paths)} inputs")
                out.extend(ASRResult(channel=ch, transcript=txt, words=[], engine=self.engine_name) for ch, txt in zip(chunk_channels, texts))
            except Exception as batch_error:
                logger.warning("Cohere batch transcription failed, falling back one-by-one: %s", batch_error)
                for p, ch in zip(chunk_paths, chunk_channels):
                    try:
                        text = self._generate([p])[0]
                        out.append(ASRResult(channel=ch, transcript=text, words=[], engine=self.engine_name))
                    except Exception as e:
                        out.append(ASRResult(channel=ch, transcript="", words=[], engine=self.engine_name, error=repr(e)))
        return out

    def unload(self) -> None:
        try:
            del self.model
            del self.processor
        except Exception:
            pass
        clear_accelerator_cache()


class GraniteASR:
    kind = "granite"
    supports_audio_input = False
    requires_timestamps = False

    def __init__(self, cfg: Any, engine_name: str = "granite", engine_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.engine_name = engine_name
        self.engine_cfg = engine_cfg or {}
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForSpeechSeq2Seq as ModelCls
            except Exception:
                from transformers import AutoModelForCausalLM as ModelCls
        except Exception as e:
            raise ImportError("Granite speech is enabled but transformers model classes are not importable. Disable asr.engines.granite.enabled or install the required transformers build.") from e

        model_dir = _engine_get(self.engine_cfg, "model_dir", None)
        if not model_dir:
            raise ValueError("Granite engine requires asr.engines.granite.model_dir")
        logger.info("Loading Granite speech engine=%s model=%s", engine_name, model_dir)
        local_only = bool(_engine_get(self.engine_cfg, "local_files_only", True))
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=local_only, trust_remote_code=True)
        dtype_name = str(_engine_get(self.engine_cfg, "dtype", "bfloat16" if _cuda_available() else "float32"))
        dtype = getattr(torch, dtype_name, torch.bfloat16 if _cuda_available() else torch.float32)
        model_kwargs = dict(_engine_get(self.engine_cfg, "model_kwargs", {}) or {})
        model_kwargs.setdefault("torch_dtype", dtype)
        if _cuda_available():
            model_kwargs.setdefault("device_map", _engine_get(self.engine_cfg, "device_map", "auto"))
        model_kwargs.setdefault("local_files_only", local_only)
        model_kwargs.setdefault("trust_remote_code", True)
        self.model = ModelCls.from_pretrained(str(model_dir), **model_kwargs)
        self.model.eval()
        try:
            self.device = next(self.model.parameters()).device
        except Exception:
            self.device = "cuda" if _cuda_available() else "cpu"
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def _prompt(self) -> str:
        user_prompt = str(_engine_get(self.engine_cfg, "prompt", "<|audio|>can you transcribe the speech into a written format?"))
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
        return user_prompt

    def _generate(self, paths: Sequence[str | Path]) -> List[str]:
        import torch
        wavs = [torch.from_numpy(_read_wav_mono_float32(p)) for p in paths]
        prompts = [self._prompt()] * len(wavs)
        try:
            inputs = self.processor(text=prompts, audio=wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        except TypeError:
            inputs = self.processor(text=prompts, audios=wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        max_new_tokens = int(_engine_get(self.engine_cfg, "max_new_tokens", 256))
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        if hasattr(self.processor, "batch_decode"):
            return [str(x).strip() for x in self.processor.batch_decode(outputs, skip_special_tokens=True)]
        if self.tokenizer is not None:
            return [str(x).strip() for x in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        return [str(x).strip() for x in outputs]

    def transcribe_paths(self, paths: Sequence[str | Path], channels: Sequence[int]) -> List[ASRResult]:
        paths = [str(p) for p in paths]
        channels = [int(c) for c in channels]
        batch_size = max(1, int(_engine_get(self.engine_cfg, "batch_size", len(paths) or 1)))
        out: list[ASRResult] = []
        for start in range(0, len(paths), batch_size):
            chunk_paths = paths[start:start + batch_size]
            chunk_channels = channels[start:start + batch_size]
            try:
                texts = self._generate(chunk_paths)
                if len(texts) != len(chunk_paths):
                    raise RuntimeError(f"Granite returned {len(texts)} outputs for {len(chunk_paths)} inputs")
                out.extend(ASRResult(channel=ch, transcript=txt, words=[], engine=self.engine_name) for ch, txt in zip(chunk_channels, texts))
            except Exception as batch_error:
                logger.warning("Granite batch transcription failed, falling back one-by-one: %s", batch_error)
                for p, ch in zip(chunk_paths, chunk_channels):
                    try:
                        text = self._generate([p])[0]
                        out.append(ASRResult(channel=ch, transcript=text, words=[], engine=self.engine_name))
                    except Exception as e:
                        out.append(ASRResult(channel=ch, transcript="", words=[], engine=self.engine_name, error=repr(e)))
        return out

    def unload(self) -> None:
        try:
            del self.model
            del self.processor
        except Exception:
            pass
        clear_accelerator_cache()


ENGINE_CLASSES = {
    "faster_whisper": FasterWhisperASR,
    "whisper": FasterWhisperASR,
    "qwen": QwenASR,
    "cohere": CohereASR,
    "granite": GraniteASR,
}


class MultiASRTranscriber:
    """Runs a configurable ASR ensemble and returns one consensus transcript per channel.

    Whisper remains the timestamp anchor because it provides word timestamps. Other ASR
    engines improve PII recall and transcript quality; their spans are projected onto the
    anchor word timeline.

    transcribe_channel_batch() is the high-throughput entry point. It accepts channels
    from multiple files at once. Whisper uses faster-whisper internal batching, while
    transcript-only engines get true cross-file batches.
    """

    _SINGLE_FILE_ID = "__single_file__"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.engines_cfg: Dict[str, Dict[str, Any]] = dict(_cfg_get(cfg, "engines", {}) or {})
        self.engine_order: List[str] = list(_cfg_get(cfg, "engine_order", []) or self.engines_cfg.keys() or ["whisper"])
        self.enabled_names = [name for name in self.engine_order if bool(self.engines_cfg.get(name, {}).get("enabled", True))]
        if not self.enabled_names:
            raise ValueError("No ASR engines are enabled. Enable at least asr.engines.whisper.enabled=true.")
        self._loaded: Dict[str, Any] = {}
        logger.info("Enabled ASR engines: %s", ", ".join(self.enabled_names))

    def _engine_kind(self, name: str) -> str:
        ecfg = self.engines_cfg.get(name, {})
        return str(ecfg.get("kind", name))

    def _get_engine(self, name: str):
        if name in self._loaded:
            return self._loaded[name]
        ecfg = self.engines_cfg.get(name, {})
        kind = self._engine_kind(name)
        cls = ENGINE_CLASSES.get(kind)
        if cls is None:
            raise ValueError(f"Unsupported ASR engine kind for {name}: {kind}")
        engine = cls(self.cfg, engine_name=name, engine_cfg=ecfg)
        self._loaded[name] = engine
        return engine

    def unload_all(self) -> None:
        for engine in list(self._loaded.values()):
            try:
                engine.unload()
            except Exception:
                pass
        self._loaded.clear()
        clear_accelerator_cache()

    def _should_unload_after_file(self) -> bool:
        return str(_cfg_get(self.cfg, "model_residency", "keep_loaded")) == "unload_after_file"

    def transcribe_channels(self, asr_inputs: Sequence[dict], work_dir: Path, keep_temp: bool = False) -> List[ChannelASRBundle]:
        rows = []
        for item in asr_inputs:
            row = dict(item)
            row["file_id"] = self._SINGLE_FILE_ID
            rows.append(row)
        grouped = self.transcribe_channel_batch(rows, work_dir=work_dir, keep_temp=keep_temp)
        return grouped.get(self._SINGLE_FILE_ID, [])

    def transcribe_channel_batch(self, asr_inputs: Sequence[dict], work_dir: Path, keep_temp: bool = False) -> Dict[str, List[ChannelASRBundle]]:
        from tempfile import TemporaryDirectory
        from .audio_io import write_mono_wav_pcm16

        items: list[dict] = []
        for idx, item in enumerate(asr_inputs):
            row = dict(item)
            row["file_id"] = str(row.get("file_id", self._SINGLE_FILE_ID))
            row["channel"] = int(row["channel"])
            row["sample_rate"] = int(row.get("sample_rate", 16000))
            row["_batch_index"] = idx
            items.append(row)

        if not items:
            return {}

        temp_ctx: Optional[TemporaryDirectory] = None
        temp_dir: Optional[Path] = None
        wav_paths: Dict[int, Path] = {}

        def ensure_wav(item: dict) -> Path:
            nonlocal temp_ctx, temp_dir
            batch_index = int(item["_batch_index"])
            if batch_index in wav_paths:
                return wav_paths[batch_index]
            if temp_ctx is None:
                temp_parent = Path(work_dir) / "tmp"
                temp_parent.mkdir(parents=True, exist_ok=True)
                temp_ctx = TemporaryDirectory(prefix="pii_multiasr_batch_", dir=str(temp_parent))
                temp_dir = Path(temp_ctx.name)
            assert temp_dir is not None
            channel = int(item["channel"])
            wav_path = temp_dir / f"item_{batch_index:06d}_ch{channel}.wav"
            write_mono_wav_pcm16(item["audio"], wav_path, sample_rate=int(item.get("sample_rate", 16000)))
            wav_paths[batch_index] = wav_path
            return wav_path

        all_results: list[ASRResult] = []
        fail_on_engine_error = bool(_cfg_get(self.cfg, "fail_on_engine_error", False))
        try:
            for name in self.enabled_names:
                engine = self._get_engine(name)
                try:
                    if getattr(engine, "supports_audio_input", False):
                        for item in items:
                            channel = int(item["channel"])
                            try:
                                r = engine.transcribe_audio(item["audio"], channel=channel, sample_rate=int(item.get("sample_rate", 16000)))
                            except Exception as e:
                                logger.warning(
                                    "ASR engine=%s in-memory input failed for file_id=%s channel=%s, using temp WAV: %s",
                                    name, item["file_id"], channel, e,
                                )
                                try:
                                    r = engine.transcribe_path(ensure_wav(item), channel=channel)
                                except Exception as e2:
                                    if fail_on_engine_error:
                                        raise
                                    r = ASRResult(channel=channel, transcript="", words=[], engine=name, error=repr(e2))
                            r.file_id = str(item["file_id"])
                            all_results.append(r)
                    else:
                        paths = [ensure_wav(item) for item in items]
                        channels = [int(item["channel"]) for item in items]
                        rows = engine.transcribe_paths(paths, channels)
                        if len(rows) != len(items):
                            raise RuntimeError(f"ASR engine={name} returned {len(rows)} rows for {len(items)} inputs")
                        for r, item in zip(rows, items):
                            r.file_id = str(item["file_id"])
                            all_results.append(r)
                except Exception as e:
                    if fail_on_engine_error:
                        raise
                    logger.warning("ASR engine=%s failed for this micro-batch: %s", name, e)
                    for item in items:
                        all_results.append(ASRResult(file_id=str(item["file_id"]), channel=int(item["channel"]), transcript="", words=[], engine=name, error=repr(e)))

            grouped = self._bundle_results_by_file(all_results, items)
            anchor_name = str(_cfg_get(self.cfg, "timestamp_anchor_engine", "whisper"))
            for bundles in grouped.values():
                for b in bundles:
                    if anchor_name in self.enabled_names and not b.anchor_words and any(r.transcript for r in b.engine_results):
                        logger.warning(
                            "Timestamp anchor engine '%s' produced no word timestamps for file_id=%s channel=%s. If PII is detected, unmapped-entity policy will be used.",
                            anchor_name, b.file_id, b.channel,
                        )
            return grouped
        finally:
            if temp_ctx is not None and not keep_temp:
                temp_ctx.cleanup()
            if self._should_unload_after_file():
                self.unload_all()

    def _bundle_results(self, results: Sequence[ASRResult], asr_inputs: Sequence[dict]) -> List[ChannelASRBundle]:
        rows = []
        for item in asr_inputs:
            row = dict(item)
            row["file_id"] = self._SINGLE_FILE_ID
            rows.append(row)
        grouped = self._bundle_results_by_file(results, rows)
        return grouped.get(self._SINGLE_FILE_ID, [])

    def _bundle_results_by_file(self, results: Sequence[ASRResult], asr_inputs: Sequence[dict]) -> Dict[str, List[ChannelASRBundle]]:
        file_channel_order: Dict[str, list[int]] = {}
        by_key: Dict[tuple[str, int], List[ASRResult]] = {}

        for item in asr_inputs:
            file_id = str(item.get("file_id", self._SINGLE_FILE_ID))
            channel = int(item["channel"])
            file_channel_order.setdefault(file_id, [])
            if channel not in file_channel_order[file_id]:
                file_channel_order[file_id].append(channel)
            by_key.setdefault((file_id, channel), [])

        for r in results:
            file_id = str(getattr(r, "file_id", None) or self._SINGLE_FILE_ID)
            channel = int(r.channel)
            by_key.setdefault((file_id, channel), []).append(r)
            file_channel_order.setdefault(file_id, [])
            if channel not in file_channel_order[file_id]:
                file_channel_order[file_id].append(channel)

        consensus_cfg = dict(_cfg_get(self.cfg, "consensus", {}) or {})
        anchor_name = str(_cfg_get(self.cfg, "timestamp_anchor_engine", "whisper"))
        grouped: Dict[str, List[ChannelASRBundle]] = {}

        for file_id, channels in file_channel_order.items():
            bundles: list[ChannelASRBundle] = []
            for channel in sorted(channels):
                rows = by_key.get((file_id, channel), [])
                consensus = build_consensus(rows, consensus_cfg)
                final_transcript = str(consensus.get("final_transcript") or "").strip()

                anchor = next((r for r in rows if r.engine == anchor_name and r.words), None)
                if anchor is None:
                    anchor = next((r for r in rows if r.words), None)
                anchor_words = [dict(w, engine=getattr(anchor, "engine", anchor_name)) for w in (anchor.words if anchor else [])]
                anchor_engine = anchor.engine if anchor else None

                selected_engine = consensus.get("selected_engine")
                selected = next((r for r in rows if r.engine == selected_engine), None)
                if selected is not None and selected.words and selected.transcript.strip() == final_transcript:
                    final_words = selected.words
                elif anchor is not None and anchor.transcript.strip() == final_transcript and anchor.words:
                    final_words = anchor.words
                else:
                    final_words = align_transcript_to_timed_words(final_transcript, anchor_words, channel=channel)

                bundles.append(ChannelASRBundle(
                    file_id=file_id,
                    channel=channel,
                    final_transcript=final_transcript,
                    final_words=final_words,
                    engine_results=rows,
                    anchor_engine=anchor_engine,
                    anchor_words=anchor_words,
                    consensus=consensus,
                ))
            grouped[file_id] = bundles
        return grouped
