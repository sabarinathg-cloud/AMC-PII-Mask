from pii_audio_masking_pipeline.asr import ASRResult, build_consensus, align_transcript_to_timed_words


def test_consensus_strict_majority_and_fallback_priority():
    rows = [
        ASRResult(channel=0, engine="whisper", transcript="hello john smith", words=[]),
        ASRResult(channel=0, engine="qwen", transcript="hello John Smith", words=[]),
        ASRResult(channel=0, engine="cohere", transcript="hello john smith", words=[]),
        ASRResult(channel=0, engine="granite", transcript="hello jane smith", words=[]),
    ]
    c = build_consensus(rows, {"min_agreement": 2, "fallback_priority": ["whisper", "qwen", "cohere", "granite"]})
    assert c["method"] == "strict_majority"
    assert c["selected_engine"] == "whisper"
    assert c["final_transcript"] == "hello john smith"


def test_align_transcript_to_anchor_word_timestamps():
    anchor_words = [
        {"word": "hello", "start": 0.0, "end": 0.2, "segment_id": 0},
        {"word": "john", "start": 0.3, "end": 0.5, "segment_id": 0},
        {"word": "smith", "start": 0.6, "end": 0.8, "segment_id": 0},
    ]
    mapped = align_transcript_to_timed_words("Hello John Smith", anchor_words, channel=1)
    assert len(mapped) == 3
    assert mapped[1]["word"] == "John"
    assert mapped[1]["start"] == 0.3
    assert mapped[1]["start_char"] == 6
    assert mapped[2]["end_char"] == 16
