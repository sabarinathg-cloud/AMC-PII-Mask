from pii_audio_masking_pipeline.timestamp_mapping import entities_to_spans, merge_spans


def test_entity_to_word_span_and_merge():
    words = [
        {"word": "call", "start": 0.0, "end": 0.2, "start_char": 0, "end_char": 4},
        {"word": "John", "start": 0.3, "end": 0.6, "start_char": 5, "end_char": 9},
        {"word": "Smith", "start": 0.7, "end": 1.0, "start_char": 10, "end_char": 15},
    ]
    entities = [{"text": "John Smith", "type": "PERSON_NAME", "start": 5, "end": 15, "source": "test", "score": 1.0}]
    spans = entities_to_spans(entities, words, channel=1, pad_sec=0.1, min_duration_sec=0.3, audio_duration_sec=2.0)
    assert len(spans) == 1
    assert spans[0]["channel"] == 1
    assert spans[0]["start"] <= 0.3
    assert spans[0]["end"] >= 1.0

    merged = merge_spans(spans + [{**spans[0], "start": spans[0]["end"] + 0.01, "end": spans[0]["end"] + 0.2}], merge_gap_sec=0.05)
    assert len(merged) == 1
