from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .asr import align_transcript_to_timed_words, tokenize_with_char_spans


@dataclass
class AlignmentResult:
    status: str
    words: List[Dict[str, Any]]
    coverage: float
    backend: str
    language: str
    aligned_word_count: int
    transcript_word_count: int
    error: Optional[str] = None


@dataclass
class WhisperXForcedAligner:
    device: str = "auto"
    compute_type: str = "float16"
    batch_size: int = 16
    default_language: str = "en"
    whisperx_module: Any = None
    _model_cache: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = field(default_factory=dict, init=False)

    @property
    def backend(self) -> str:
        return "whisperx"

    def align(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
        language: Optional[str],
        channel: int,
    ) -> AlignmentResult:
        text = str(transcript or "").strip()
        transcript_word_count = len(tokenize_with_char_spans(text))
        if not text or transcript_word_count == 0:
            return AlignmentResult(
                status="empty_transcript",
                words=[],
                coverage=0.0,
                backend=self.backend,
                language=self._normalize_language(language),
                aligned_word_count=0,
                transcript_word_count=0,
            )

        resolved_language = self._normalize_language(language)
        resolved_device = self._resolve_device()
        model, metadata = self._load_model(resolved_language, resolved_device)
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_sec = float(waveform.shape[0]) / float(sample_rate) if sample_rate else 0.0
        segments = [{"text": text, "start": 0.0, "end": max(duration_sec, 0.0)}]
        result = self._call_align(
            segments,
            model,
            metadata,
            waveform,
            resolved_device,
        )
        aligned_words = build_canonical_aligned_words(
            transcript=text,
            timed_words=_extract_timed_words(result),
            channel=channel,
            backend=self.backend,
        )
        coverage = len(aligned_words) / transcript_word_count if transcript_word_count else 0.0
        return AlignmentResult(
            status="aligned" if aligned_words else "unaligned",
            words=aligned_words,
            coverage=coverage,
            backend=self.backend,
            language=resolved_language,
            aligned_word_count=len(aligned_words),
            transcript_word_count=transcript_word_count,
        )

    @property
    def _whisperx(self) -> Any:
        if self.whisperx_module is None:
            self.whisperx_module = importlib.import_module("whisperx")
        return self.whisperx_module

    def _load_model(self, language: str, device: str) -> Tuple[Any, Dict[str, Any]]:
        cache_key = (language.lower(), device)
        if cache_key not in self._model_cache:
            model, metadata = self._whisperx.load_align_model(language_code=language, device=device)
            self._model_cache[cache_key] = (model, metadata)
        return self._model_cache[cache_key]

    def _call_align(self, segments: list[dict], model: Any, metadata: dict, waveform: np.ndarray, device: str) -> Any:
        kwargs: dict[str, Any] = {
            "batch_size": int(self.batch_size),
            "return_char_alignments": False,
        }
        try:
            signature = inspect.signature(self._whisperx.align)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return self._whisperx.align(segments, model, metadata, waveform, device, **kwargs)

    def _normalize_language(self, language: Optional[str]) -> str:
        value = str(language or self.default_language or "en").strip().lower()
        if value.startswith("en"):
            return "en"
        return value.split("-")[0] or "en"

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return str(self.device)
        try:
            torch = importlib.import_module("torch")
            if bool(torch.cuda.is_available()):
                return "cuda"
        except Exception:
            pass
        return "cpu"


def build_canonical_aligned_words(
    *,
    transcript: str,
    timed_words: List[Dict[str, Any]],
    channel: int,
    backend: str,
) -> List[Dict[str, Any]]:
    timestamp_rows: list[dict[str, Any]] = []
    for row in timed_words:
        word = str(row.get("word", "")).strip()
        if not word or row.get("start") is None or row.get("end") is None:
            continue
        timestamp_rows.append({
            "word": word,
            "start": row.get("start"),
            "end": row.get("end"),
            "probability": _word_probability(row),
            "engine": backend,
            "segment_id": row.get("segment_id", 0),
        })

    words = align_transcript_to_timed_words(transcript, timestamp_rows, channel)
    for word in words:
        word["alignment_backend"] = backend
        word["timestamp_source"] = "forced_alignment"
    return words


def _extract_timed_words(result: Any) -> List[Dict[str, Any]]:
    words: list[dict[str, Any]] = []
    if isinstance(result, dict):
        segments = result.get("segments", [])
    else:
        segments = []
    for segment_id, segment in enumerate(segments or []):
        if not isinstance(segment, dict):
            continue
        for row in segment.get("words", []) or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item.setdefault("segment_id", segment_id)
            words.append(item)
    return words


def _word_probability(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("probability", row.get("score"))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
