from pii_audio_masking_pipeline.config import PIIConfig
from pii_audio_masking_pipeline.pii_detection import PIIDetector, PIIEntity, iter_text_chunks


def test_regex_and_spoken_number_rules_without_models():
    cfg = PIIConfig(enable_gliner=False, enable_piiranha=False, enable_spacy=False)
    detector = PIIDetector(cfg)
    text = "My name is John Smith. Call me at four one five five five five one two three four or email john@example.com."
    ents = detector.detect(text)
    types = {e["type"] for e in ents}
    assert "PERSON_NAME" in types
    assert "PHONE" in types
    assert "EMAIL" in types


def test_iter_text_chunks_terminates_and_overlaps():
    text = "word " * 1000
    chunks = list(iter_text_chunks(text, max_chars=500, overlap_chars=100))
    assert len(chunks) > 1
    assert chunks[0][0] == 0
    assert all(chunk for _, chunk in chunks)
    assert chunks[-1][0] < len(text)


def test_detect_batch_combines_rule_and_neural_detector_outputs(monkeypatch):
    cfg = PIIConfig(enable_gliner=False, enable_piiranha=False, enable_spacy=False)
    detector = PIIDetector(cfg)

    monkeypatch.setattr(
        detector,
        "_detect_gliner_many",
        lambda texts: [[PIIEntity("John Smith", "PERSON_NAME", 0, 10, 0.9, "gliner", "person name")] for _ in texts],
    )
    monkeypatch.setattr(
        detector,
        "_detect_piiranha_many",
        lambda texts: [[PIIEntity("acct1234", "ACCOUNT_NUMBER", 25, 33, 0.95, "piiranha", "ACCOUNTNUM")] for _ in texts],
    )
    monkeypatch.setattr(
        detector,
        "_detect_spacy_many",
        lambda texts: [[PIIEntity("Boston", "LOCATION", 34, 40, 0.6, "spacy", "GPE")] for _ in texts],
    )

    entities = detector.detect_batch(["John Smith 555-123-4567 acct1234 Boston"])[0]
    sources = {entity["source"] for entity in entities}

    assert {"regex", "gliner", "piiranha", "spacy"}.issubset(sources)
