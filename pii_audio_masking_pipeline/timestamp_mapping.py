from __future__ import annotations

from typing import Any, Dict, Iterable, List


def entities_to_spans(
    entities: Iterable[dict],
    words: List[Dict[str, Any]],
    channel: int,
    pad_sec: float = 0.12,
    min_duration_sec: float = 0.30,
    audio_duration_sec: float | None = None,
) -> List[Dict[str, Any]]:
    spans: list[dict] = []
    if not words:
        return spans

    for ent in entities:
        ent_start = int(ent.get("start", -1))
        ent_end = int(ent.get("end", -1))
        if ent_start < 0 or ent_end <= ent_start:
            continue

        matched = [
            w for w in words
            if int(w.get("end_char", -1)) > ent_start and int(w.get("start_char", -1)) < ent_end
        ]
        if not matched:
            continue

        start = min(float(w.get("start", 0.0)) for w in matched)
        end = max(float(w.get("end", start)) for w in matched)
        start = max(0.0, start - pad_sec)
        end = end + pad_sec

        if audio_duration_sec is not None:
            end = min(float(audio_duration_sec), end)

        if end - start < min_duration_sec:
            mid = (start + end) / 2.0
            start = max(0.0, mid - min_duration_sec / 2.0)
            end = mid + min_duration_sec / 2.0
            if audio_duration_sec is not None:
                end = min(float(audio_duration_sec), end)

        spans.append({
            "channel": channel,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "type": ent.get("type", "PII"),
            "text": ent.get("text", ""),
            "source": ent.get("source", "unknown"),
            "score": ent.get("score"),
            "entity_id": ent.get("entity_id"),
            "asr_engine": ent.get("asr_engine"),
            "transcript_source": ent.get("transcript_source"),
            "timestamp_source": ent.get("timestamp_source") or _timestamp_source_for_words(matched),
            "alignment_backend": ent.get("alignment_backend") or _alignment_backend_for_words(matched),
        })
    return spans


def _timestamp_source_for_words(words: List[Dict[str, Any]]) -> str:
    sources = {str(w.get("timestamp_source")) for w in words if w.get("timestamp_source")}
    if "forced_alignment" in sources:
        return "forced_alignment"
    if sources:
        return sorted(sources)[0]
    return "asr_words"


def _alignment_backend_for_words(words: List[Dict[str, Any]]) -> str | None:
    backends = {str(w.get("alignment_backend")) for w in words if w.get("alignment_backend")}
    if not backends:
        return None
    return sorted(backends)[0]


def merge_spans(spans: Iterable[dict], merge_gap_sec: float = 0.05, target_channels: str = "detected_channel") -> List[Dict[str, Any]]:
    def span_entity_ids(row: dict) -> set[str]:
        ids: set[str] = set()
        if row.get("entity_id"):
            ids.add(str(row["entity_id"]))
        for entity_id in row.get("entity_ids") or []:
            if entity_id:
                ids.add(str(entity_id))
        return ids

    def finalize(row: dict, entity_ids: set[str], types: set[str], sources: set[str], texts: list[str]) -> dict:
        row["types"] = sorted(types)
        row["sources"] = sorted(sources)
        row["texts"] = texts
        row["entity_ids"] = sorted(entity_ids)
        row["entity_id"] = row["entity_ids"][0] if len(row["entity_ids"]) == 1 else None
        return row

    rows = []
    for s in spans:
        row = dict(s)
        if target_channels == "both":
            row["channel"] = -1
        rows.append(row)

    rows = sorted(rows, key=lambda x: (int(x.get("channel", -1)), float(x.get("start", 0.0)), float(x.get("end", 0.0))))
    if not rows:
        return []

    merged: list[dict] = []
    cur = rows[0]
    cur_types = {str(cur.get("type", "PII"))}
    cur_sources = {str(cur.get("source", "unknown"))}
    cur_texts = [str(cur.get("text", ""))]
    cur_entity_ids = span_entity_ids(cur)

    for row in rows[1:]:
        same_channel = int(row.get("channel", -1)) == int(cur.get("channel", -1))
        close = float(row.get("start", 0.0)) <= float(cur.get("end", 0.0)) + merge_gap_sec
        if same_channel and close:
            cur["end"] = max(float(cur["end"]), float(row["end"]))
            cur["duration"] = float(cur["end"]) - float(cur["start"])
            cur_types.add(str(row.get("type", "PII")))
            cur_sources.add(str(row.get("source", "unknown")))
            cur_entity_ids.update(span_entity_ids(row))
            txt = str(row.get("text", ""))
            if txt:
                cur_texts.append(txt)
        else:
            merged.append(finalize(cur, cur_entity_ids, cur_types, cur_sources, cur_texts))
            cur = row
            cur_types = {str(cur.get("type", "PII"))}
            cur_sources = {str(cur.get("source", "unknown"))}
            cur_texts = [str(cur.get("text", ""))]
            cur_entity_ids = span_entity_ids(cur)

    merged.append(finalize(cur, cur_entity_ids, cur_types, cur_sources, cur_texts))
    return merged
