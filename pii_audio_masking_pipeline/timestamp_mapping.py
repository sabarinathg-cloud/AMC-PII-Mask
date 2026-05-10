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
        })
    return spans


def merge_spans(spans: Iterable[dict], merge_gap_sec: float = 0.05, target_channels: str = "detected_channel") -> List[Dict[str, Any]]:
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

    for row in rows[1:]:
        same_channel = int(row.get("channel", -1)) == int(cur.get("channel", -1))
        close = float(row.get("start", 0.0)) <= float(cur.get("end", 0.0)) + merge_gap_sec
        if same_channel and close:
            cur["end"] = max(float(cur["end"]), float(row["end"]))
            cur["duration"] = float(cur["end"]) - float(cur["start"])
            cur_types.add(str(row.get("type", "PII")))
            cur_sources.add(str(row.get("source", "unknown")))
            txt = str(row.get("text", ""))
            if txt:
                cur_texts.append(txt)
        else:
            cur["types"] = sorted(cur_types)
            cur["sources"] = sorted(cur_sources)
            cur["texts"] = cur_texts
            merged.append(cur)
            cur = row
            cur_types = {str(cur.get("type", "PII"))}
            cur_sources = {str(cur.get("source", "unknown"))}
            cur_texts = [str(cur.get("text", ""))]

    cur["types"] = sorted(cur_types)
    cur["sources"] = sorted(cur_sources)
    cur["texts"] = cur_texts
    merged.append(cur)
    return merged
