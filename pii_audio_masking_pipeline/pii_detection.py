from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import contextlib
import json
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    text: str
    type: str
    start: int
    end: int
    score: float
    source: str
    raw_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "raw_label": self.raw_label,
        }


EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DOB_NUMERIC_RE = re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
CVV_CONTEXT_RE = re.compile(r"\b(?:cvv|cvc|security code)\s*(?:is|:|#)?\s*(\d{3,4})\b", re.IGNORECASE)
ROUTING_CONTEXT_RE = re.compile(r"\b(?:routing|aba)\s*(?:number|no\.?|#)?\s*(?:is|:)?\s*(\d{9})\b", re.IGNORECASE)
MONTH_DOB_RE = re.compile(
    r"\b(?:date of birth|dob|born|birthday)?\s*"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(?:\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty\s+first|twenty\s+second|twenty\s+third|twenty\s+fourth|twenty\s+fifth|twenty\s+sixth|twenty\s+seventh|twenty\s+eighth|twenty\s+ninth|thirtieth|thirty\s+first)"
    r"(?:,)?\s+(?:\d{2,4}|nineteen\s+\w+|twenty\s+\w+)",
    flags=re.IGNORECASE,
)
ID_CONTEXT_RE = re.compile(
    r"\b(?:member|account|policy|reference|case|claim|mrn|subscriber|insurance|booking|confirmation|patient|medical record|healthcare|authorization|auth|invoice|customer|passport|driver license|drivers license|bank account|routing)\s*"
    r"(?:number|no\.?|#|id|code)?\s*(?:is|:|\-)?\s*([A-Z0-9][A-Z0-9\-]{3,})\b",
    flags=re.IGNORECASE,
)

NAME_BLACKLIST = {
    "nurse", "assistant", "doctor", "agent", "representative", "coordinator", "service", "health",
    "calling", "message", "program", "care", "customer", "member", "patient", "blue", "cross", "shield",
    "voicemail", "appointment", "medication", "phone", "number", "my", "name", "is", "hello", "hi", "amc",
    "callback", "call", "department", "office", "team", "hospital", "clinic", "speaking", "with", "this",
    "from", "please", "thank", "you", "medical", "insurance", "pharmacy",
}

NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([a-z]+(?:\s+[a-z]+){0,2}?)(?=\s+(?:i am|i'm|calling|from|with|and|on behalf)\b|[.,;!?]|$)", re.I),
    re.compile(r"\bthis is\s+([a-z]+(?:\s+[a-z]+){0,2}?)(?=\s+(?:i am|i'm|calling|from|with|and|on behalf)\b|[.,;!?]|$)", re.I),
    re.compile(r"\bthis message is for\s+([a-z]+(?:\s+[a-z]+){0,2}?)(?=\s+(?:my name is|this is|i am|i'm|calling|from|with|and)\b|[.,;!?]|$)", re.I),
    re.compile(r"\bspeaking with\s+([a-z]+(?:\s+[a-z]+){0,2}?)(?=\s+(?:from|at|and)\b|[.,;!?]|$)", re.I),
    re.compile(r"\bpatient name is\s+([a-z]+(?:\s+[a-z]+){0,2}?)(?=\s+(?:date|dob|phone|member|and)\b|[.,;!?]|$)", re.I),
]

NUMBER_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
NUMBER_SEPARATORS = {"dash", "hyphen", "minus", "space", "spaces"}
NUMBER_REPEATERS = {"double": 2, "triple": 3}

PIIRANHA_LABEL_MAP = {
    "ACCOUNTNUM": "ACCOUNT_NUMBER",
    "BUILDINGNUM": "LOCATION",
    "CITY": "LOCATION",
    "CREDITCARDNUMBER": "CREDIT_CARD_NUMBER",
    "DATEOFBIRTH": "DOB",
    "DRIVERLICENSENUM": "DRIVER_LICENSE",
    "EMAIL": "EMAIL",
    "GIVENNAME": "PERSON_NAME",
    "IDCARDNUM": "ID_NUMBER",
    "PASSWORD": "PASSWORD",
    "SOCIALNUM": "SSN",
    "STREET": "LOCATION",
    "SURNAME": "PERSON_NAME",
    "TAXNUM": "ID_NUMBER",
    "TELEPHONENUM": "PHONE",
    "USERNAME": "USERNAME",
    "ZIPCODE": "ZIP",
}

SPACY_LABEL_MAP = {
    "PERSON": "PERSON_NAME",
    "ORG": "INSTITUTION_NAME",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
}

TYPE_PRIORITY = {
    "SSN": 1,
    "AADHAAR_NUMBER": 2,
    "CREDIT_CARD_NUMBER": 3,
    "CVV": 4,
    "PHONE": 5,
    "EMAIL": 6,
    "DOB": 7,
    "INSURANCE_ID": 8,
    "ACCOUNT_NUMBER": 9,
    "ROUTING_NUMBER": 10,
    "ID_NUMBER": 11,
    "ID": 12,
    "PASSPORT_NUMBER": 13,
    "DRIVER_LICENSE": 14,
    "PERSON_NAME": 15,
    "DOCTOR_NAME": 16,
    "USERNAME": 17,
    "PASSWORD": 18,
    "IP_ADDRESS": 19,
    "URL": 20,
    "LOCATION": 21,
    "INSTITUTION_NAME": 22,
    "ZIP": 23,
    "AGE": 24,
    "DATE": 25,
    "MEDICAL_CONDITION": 30,
    "MEDICATION": 31,
    "LAB_RESULT": 32,
}


def normalize_type(label: str) -> str:
    label = str(label or "PII").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "PHONE_NUMBER": "PHONE",
        "EMAIL_ADDRESS": "EMAIL",
        "DATE_OF_BIRTH": "DOB",
        "DOB": "DOB",
        "LOCATION_ADDRESS": "LOCATION",
        "LOCATION_STREET": "LOCATION",
        "LOCATION_CITY": "LOCATION",
        "LOCATION_STATE": "LOCATION",
        "LOCATION_COUNTRY": "LOCATION",
        "LOCATION_ZIP": "ZIP",
        "ZIPCODE": "ZIP",
        "AADHAAR": "AADHAAR_NUMBER",
        "INSURANCE": "INSURANCE_ID",
        "DOCTOR": "DOCTOR_NAME",
        "NAME_MEDICAL_PROFESSIONAL": "DOCTOR_NAME",
        "FIRST_NAME": "PERSON_NAME",
        "LAST_NAME": "PERSON_NAME",
        "PERSON": "PERSON_NAME",
        "NAME": "PERSON_NAME",
        "CREDIT_CARD": "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT": "ACCOUNT_NUMBER",
        "ROUTING_NUMBER": "ROUTING_NUMBER",
        "PASSPORT": "PASSPORT_NUMBER",
        "PASSPORT_NUMBER": "PASSPORT_NUMBER",
        "DRIVER_LICENSE": "DRIVER_LICENSE",
        "DRIVERS_LICENSE": "DRIVER_LICENSE",
        "IP_ADDRESS": "IP_ADDRESS",
        "MEDICAL_CODE": "ID_NUMBER",
        "HEALTHCARE_NUMBER": "ID_NUMBER",
        "ORGANIZATION_MEDICAL_FACILITY": "INSTITUTION_NAME",
        "ORGANIZATION": "INSTITUTION_NAME",
        "MEDICAL_FACILITY": "INSTITUTION_NAME",
        "CONDITION": "MEDICAL_CONDITION",
        "DIAGNOSIS": "MEDICAL_CONDITION",
        "DRUG": "MEDICATION",
        "DOSE": "MEDICATION",
        "MEDICAL_PROCESS": "MEDICAL_CONDITION",
    }
    return aliases.get(label, label)


def normalize_piiranha_label(label: str) -> str:
    label = str(label or "").upper().strip()
    return re.sub(r"^(B|I|E|S)-", "", label)


def token_spans(text: str) -> list[dict]:
    pattern = re.compile(r"[A-Za-z0-9@._%+\-']+")
    rows = []
    for m in pattern.finditer(str(text)):
        tok = m.group(0)
        norm = re.sub(r"[^a-z0-9]+", "", tok.lower())
        rows.append({"text": tok, "norm": norm, "start": m.start(), "end": m.end()})
    return rows


def likely_person_name(span_text: str) -> bool:
    toks = str(span_text).strip().lower().split()
    if not toks or len(toks) > 3:
        return False
    if any(len(t) < 2 for t in toks):
        return False
    if any(t in NAME_BLACKLIST for t in toks):
        return False
    if all(t.isdigit() for t in toks):
        return False
    return True


def iter_text_chunks(text: str, max_chars: int = 1800, overlap_chars: int = 240) -> Iterator[Tuple[int, str]]:
    text = str(text or "")
    n = len(text)
    max_chars = max(256, int(max_chars or 1800))
    overlap_chars = max(0, min(int(overlap_chars or 0), max_chars // 2))
    if n <= max_chars:
        yield 0, text
        return

    start = 0
    while start < n:
        hard_end = min(n, start + max_chars)
        end = hard_end
        if hard_end < n:
            window = text[start:hard_end]
            last_boundary = max(window.rfind(" "), window.rfind("."), window.rfind(","), window.rfind(";"))
            if last_boundary >= max_chars // 2:
                end = start + last_boundary + 1
        chunk = text[start:end]
        if chunk.strip():
            yield start, chunk
        if end >= n:
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        while next_start < n and next_start > 0 and text[next_start - 1].isalnum() and text[next_start].isalnum():
            next_start += 1
        start = next_start


def dedupe_entities(entities: Iterable[PIIEntity]) -> List[PIIEntity]:
    best: dict[tuple[int, int, str], PIIEntity] = {}
    for e in entities:
        if e.end <= e.start:
            continue
        key = (int(e.start), int(e.end), str(e.text).strip().lower())
        cur = best.get(key)
        if cur is None:
            best[key] = e
            continue
        new_key = (TYPE_PRIORITY.get(e.type, 999), -float(e.score))
        cur_key = (TYPE_PRIORITY.get(cur.type, 999), -float(cur.score))
        if new_key < cur_key:
            best[key] = e
    return sorted(best.values(), key=lambda x: (x.start, x.end))


def resolve_overlaps(entities: List[PIIEntity]) -> List[PIIEntity]:
    entities = dedupe_entities(entities)
    entities = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), TYPE_PRIORITY.get(e.type, 999), -e.score),
    )
    kept: list[PIIEntity] = []
    for ent in entities:
        replaced = False
        skip = False
        for i, cur in enumerate(kept):
            overlap = not (ent.end <= cur.start or ent.start >= cur.end)
            if not overlap:
                continue
            ent_key = (TYPE_PRIORITY.get(ent.type, 999), -ent.score, -(ent.end - ent.start))
            cur_key = (TYPE_PRIORITY.get(cur.type, 999), -cur.score, -(cur.end - cur.start))
            if ent_key < cur_key:
                kept[i] = ent
                replaced = True
            else:
                skip = True
            break
        if not replaced and not skip:
            kept.append(ent)
    return sorted(kept, key=lambda e: (e.start, e.end))


def resolve_torch_device(device: str, min_gpu_mem_gb: float = 12.0) -> str:
    requested = str(device or "auto").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        if requested == "cuda":
            return "cuda"
        props = torch.cuda.get_device_properties(0)
        total_gb = float(props.total_memory) / (1024 ** 3)
        if total_gb >= float(min_gpu_mem_gb):
            return "cuda"
        logger.info("Using CPU for neural PII because GPU memory is %.1f GB < %.1f GB", total_gb, float(min_gpu_mem_gb))
        return "cpu"
    except Exception:
        return "cpu"


class PIIDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gliner_model = None
        self.piiranha_pipe = None
        self.spacy_nlp = None
        # Avoid importing torch when only regex / spoken-number rules are active.
        # In large batch jobs this keeps startup cheap, and in some CI/runtime builds
        # torch finalization can keep worker processes alive longer than necessary.
        self._neural_pii_enabled = bool(getattr(cfg, "enable_gliner", False) or getattr(cfg, "enable_piiranha", False))
        self.device = (
            resolve_torch_device(
                getattr(cfg, "device", "auto"),
                getattr(cfg, "min_gpu_mem_gb_for_neural_pii", 12.0),
            )
            if self._neural_pii_enabled
            else "cpu"
        )
        self._torch = None
        if self._neural_pii_enabled:
            try:
                import torch
                self._torch = torch
                precision = getattr(cfg, "torch_float32_matmul_precision", None)
                if precision and hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision(str(precision))
            except Exception:
                self._torch = None

        if cfg.enable_gliner:
            try:
                from gliner import GLiNER
                try:
                    self.gliner_model = GLiNER.from_pretrained(cfg.gliner_model, device=self.device)
                except TypeError:
                    self.gliner_model = GLiNER.from_pretrained(cfg.gliner_model)
                    if hasattr(self.gliner_model, "to"):
                        self.gliner_model.to(self.device)
                logger.info("Loaded GLiNER model: %s on %s", cfg.gliner_model, self.device)
            except Exception as e:
                logger.warning("GLiNER unavailable. Continuing without it. Error: %s", e)

        if cfg.enable_piiranha:
            try:
                from transformers import pipeline
                device_idx = 0 if self.device == "cuda" else -1
                self.piiranha_pipe = pipeline(
                    "token-classification",
                    model=cfg.piiranha_model,
                    tokenizer=cfg.piiranha_model,
                    aggregation_strategy="simple",
                    device=device_idx,
                )
                logger.info("Loaded Piiranha model: %s on %s", cfg.piiranha_model, self.device)
            except Exception as e:
                logger.warning("Piiranha unavailable. Continuing without it. Error: %s", e)

        if cfg.enable_spacy:
            try:
                import spacy
                self.spacy_nlp = spacy.load(cfg.spacy_model)
                logger.info("Loaded spaCy model: %s", cfg.spacy_model)
            except Exception as e:
                logger.warning("spaCy unavailable. Continuing without it. Error: %s", e)

    def _inference_context(self):
        if self._torch is None:
            return contextlib.nullcontext()
        return self._torch.inference_mode()

    def detect(self, text: str, row: Optional[dict] = None) -> List[Dict[str, Any]]:
        return self.detect_batch([text], rows=[row])[0]

    def detect_many(self, texts: Iterable[str], rows: Optional[Iterable[Optional[dict]]] = None) -> List[List[Dict[str, Any]]]:
        return self.detect_batch(list(texts), rows=list(rows) if rows is not None else None)

    def detect_batch(self, texts: Iterable[str], rows: Optional[Iterable[Optional[dict]]] = None) -> List[List[Dict[str, Any]]]:
        text_list = [str(t or "") for t in texts]
        row_list = list(rows) if rows is not None else [None] * len(text_list)
        if len(row_list) != len(text_list):
            raise ValueError("rows must be None or have the same length as texts")

        buckets: list[list[PIIEntity]] = []
        for text, row in zip(text_list, row_list):
            if not text.strip():
                buckets.append([])
                continue
            entities: list[PIIEntity] = []
            if self.cfg.enable_regex:
                entities.extend(self._detect_regex(text))
            if self.cfg.enable_spoken_number_rules:
                entities.extend(self._detect_spoken_numbers(text))
            entities.extend(self._detect_rule_names(text))
            if self.cfg.enable_saved_pii_json and row is not None:
                entities.extend(self._detect_saved_json(text, row))
            buckets.append(entities)

        for i, entities in enumerate(self._detect_gliner_many(text_list)):
            buckets[i].extend(entities)
        for i, entities in enumerate(self._detect_piiranha_many(text_list)):
            buckets[i].extend(entities)
        for i, entities in enumerate(self._detect_spacy_many(text_list)):
            buckets[i].extend(entities)

        return [[e.to_dict() for e in resolve_overlaps(bucket)] for bucket in buckets]

    @staticmethod
    def _add_regex_matches(entities, text: str, regex, typ: str, score: float, group: int = 0, source: str = "regex"):
        for m in regex.finditer(text):
            entities.append(PIIEntity(m.group(group), typ, m.start(group), m.end(group), score, source))

    def _detect_regex(self, text: str) -> List[PIIEntity]:
        entities: list[PIIEntity] = []
        self._add_regex_matches(entities, text, EMAIL_RE, "EMAIL", 1.0)
        self._add_regex_matches(entities, text, URL_RE, "URL", 0.95)
        self._add_regex_matches(entities, text, IPV4_RE, "IP_ADDRESS", 0.95)
        self._add_regex_matches(entities, text, PHONE_RE, "PHONE", 1.0)
        self._add_regex_matches(entities, text, SSN_RE, "SSN", 1.0)
        self._add_regex_matches(entities, text, DOB_NUMERIC_RE, "DOB", 1.0)
        self._add_regex_matches(entities, text, MONTH_DOB_RE, "DOB", 0.95)
        self._add_regex_matches(entities, text, AADHAAR_RE, "AADHAAR_NUMBER", 1.0)
        self._add_regex_matches(entities, text, CVV_CONTEXT_RE, "CVV", 0.95, group=1, source="regex_context")
        self._add_regex_matches(entities, text, ROUTING_CONTEXT_RE, "ROUTING_NUMBER", 0.95, group=1, source="regex_context")
        for m in CREDIT_CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19:
                entities.append(PIIEntity(m.group(0), "CREDIT_CARD_NUMBER", m.start(), m.end(), 0.95, "regex"))
        for m in ZIP_RE.finditer(text):
            entities.append(PIIEntity(m.group(0), "ZIP", m.start(), m.end(), 0.75, "regex"))
        for m in ID_CONTEXT_RE.finditer(text):
            value = m.group(1)
            norm = re.sub(r"[^A-Za-z0-9]", "", value)
            if len(norm) >= 4 and any(c.isdigit() for c in norm):
                entities.append(PIIEntity(value, "ID", m.start(1), m.end(1), 0.95, "regex_context"))
        return entities

    def _detect_spoken_numbers(self, text: str) -> List[PIIEntity]:
        toks = token_spans(text)
        entities: list[PIIEntity] = []
        i = 0
        while i < len(toks):
            digits: list[str] = []
            start_i = i
            j = i
            consumed_any = False
            while j < len(toks):
                norm = toks[j]["norm"]
                if norm in NUMBER_SEPARATORS:
                    j += 1
                    continue
                if norm in NUMBER_REPEATERS and j + 1 < len(toks):
                    nxt = toks[j + 1]["norm"]
                    if nxt in NUMBER_WORDS:
                        digits.extend([NUMBER_WORDS[nxt]] * NUMBER_REPEATERS[norm])
                        j += 2
                        consumed_any = True
                        continue
                    if nxt.isdigit() and len(nxt) == 1:
                        digits.extend([nxt] * NUMBER_REPEATERS[norm])
                        j += 2
                        consumed_any = True
                        continue
                if norm in NUMBER_WORDS:
                    digits.append(NUMBER_WORDS[norm])
                    j += 1
                    consumed_any = True
                    continue
                if norm.isdigit():
                    digits.extend(list(norm))
                    j += 1
                    consumed_any = True
                    continue
                break
            count = len(digits)
            if consumed_any and count >= 7:
                start = toks[start_i]["start"]
                end = toks[j - 1]["end"]
                window_left = max(0, start - 100)
                window = text[window_left : min(len(text), end + 100)].lower()
                if count >= 10 or any(k in window for k in ["phone", "mobile", "number", "contact", "call me", "member", "account", "policy", "id", "mrn", "claim", "dob", "date of birth"]):
                    typ = "PHONE" if count >= 10 else "ID_NUMBER"
                    entities.append(PIIEntity(text[start:end], typ, start, end, 0.92, "spoken_number_rule"))
                i = max(j, i + 1)
            else:
                i += 1
        return entities

    def _detect_rule_names(self, text: str) -> List[PIIEntity]:
        entities = []
        for pat in NAME_PATTERNS:
            for m in pat.finditer(str(text)):
                span = re.sub(r"\s+", " ", m.group(1).strip())
                if likely_person_name(span):
                    entities.append(PIIEntity(span, "PERSON_NAME", m.start(1), m.end(1), 0.99, "rule_name"))
        return entities

    def _chunk_infos(self, texts: Sequence[str]) -> list[tuple[int, int, str]]:
        infos = []
        for text_idx, text in enumerate(texts):
            if not str(text).strip():
                continue
            for offset, chunk in iter_text_chunks(
                text,
                getattr(self.cfg, "chunk_chars", 1800),
                getattr(self.cfg, "chunk_overlap_chars", 240),
            ):
                if chunk.strip():
                    infos.append((text_idx, offset, chunk))
        return infos

    def _detect_gliner_many(self, texts: Sequence[str]) -> List[List[PIIEntity]]:
        if self.gliner_model is None:
            return [[] for _ in texts]
        labels = list(self.cfg.gliner_labels)
        if self.cfg.mask_clinical_phi:
            labels += list(self.cfg.clinical_phi_labels)
        infos = self._chunk_infos(texts)
        if not infos:
            return [[] for _ in texts]
        chunks = [chunk for _, _, chunk in infos]
        try:
            with self._inference_context():
                if hasattr(self.gliner_model, "batch_predict_entities"):
                    raw_all = self.gliner_model.batch_predict_entities(chunks, labels, threshold=self.cfg.gliner_threshold)
                else:
                    raw_all = [self.gliner_model.predict_entities(chunk, labels, threshold=self.cfg.gliner_threshold) for chunk in chunks]
        except Exception as e:
            logger.warning("GLiNER batch detection failed, retrying sequentially: %s", e)
            raw_all = []
            for chunk in chunks:
                try:
                    with self._inference_context():
                        raw_all.append(self.gliner_model.predict_entities(chunk, labels, threshold=self.cfg.gliner_threshold))
                except Exception as inner:
                    logger.warning("GLiNER detection failed: %s", inner)
                    raw_all.append([])

        out: list[list[PIIEntity]] = [[] for _ in texts]
        for (text_idx, offset, chunk), raw in zip(infos, raw_all):
            full_text = texts[text_idx]
            for e in raw or []:
                local_start = int(e.get("start", 0))
                local_end = int(e.get("end", 0))
                start = offset + local_start
                end = min(offset + local_end, len(full_text))
                if start < 0 or end <= start or start >= len(full_text):
                    continue
                typ = normalize_type(e.get("label"))
                out[text_idx].append(PIIEntity(
                    str(full_text[start:end]),
                    typ,
                    start,
                    end,
                    float(e.get("score", 0.0)),
                    "gliner",
                    raw_label=str(e.get("label", "")),
                ))
        return out

    def _detect_piiranha_many(self, texts: Sequence[str]) -> List[List[PIIEntity]]:
        if self.piiranha_pipe is None:
            return [[] for _ in texts]
        infos = self._chunk_infos(texts)
        if not infos:
            return [[] for _ in texts]
        chunks = [chunk for _, _, chunk in infos]
        try:
            with self._inference_context():
                raw_all = self.piiranha_pipe(chunks, batch_size=int(getattr(self.cfg, "batch_size", 16)))
        except Exception as e:
            logger.warning("Piiranha batch detection failed, retrying sequentially: %s", e)
            raw_all = []
            for chunk in chunks:
                try:
                    with self._inference_context():
                        raw_all.append(self.piiranha_pipe(chunk))
                except Exception as inner:
                    logger.warning("Piiranha detection failed: %s", inner)
                    raw_all.append([])
        if len(chunks) == 1 and raw_all and isinstance(raw_all[0], dict):
            raw_all = [raw_all]

        out: list[list[PIIEntity]] = [[] for _ in texts]
        for (text_idx, offset, chunk), raw in zip(infos, raw_all):
            full_text = texts[text_idx]
            for e in raw or []:
                raw_label = normalize_piiranha_label(e.get("entity_group", e.get("entity", "")))
                score = float(e.get("score", 0.0))
                if score < self.cfg.piiranha_threshold:
                    continue
                typ = PIIRANHA_LABEL_MAP.get(raw_label)
                if not typ:
                    continue
                local_start = int(e.get("start", -1))
                local_end = int(e.get("end", -1))
                start = offset + local_start
                end = min(offset + local_end, len(full_text))
                if start < 0 or end <= start or start >= len(full_text):
                    continue
                out[text_idx].append(PIIEntity(str(full_text[start:end]), typ, start, end, score, "piiranha", raw_label=raw_label))
        return out

    def _detect_spacy_many(self, texts: Sequence[str]) -> List[List[PIIEntity]]:
        if self.spacy_nlp is None:
            return [[] for _ in texts]
        infos = self._chunk_infos(texts)
        if not infos:
            return [[] for _ in texts]
        out: list[list[PIIEntity]] = [[] for _ in texts]
        try:
            docs = self.spacy_nlp.pipe([chunk for _, _, chunk in infos], batch_size=int(getattr(self.cfg, "batch_size", 16)))
            for (text_idx, offset, _), doc in zip(infos, docs):
                for ent in doc.ents:
                    typ = SPACY_LABEL_MAP.get(ent.label_)
                    if typ:
                        out[text_idx].append(PIIEntity(ent.text, typ, offset + int(ent.start_char), offset + int(ent.end_char), 0.60, "spacy", ent.label_))
        except Exception as e:
            logger.warning("spaCy detection failed: %s", e)
        return out

    def _detect_saved_json(self, text: str, row: dict) -> List[PIIEntity]:
        value = row.get("pii_phi_json") if isinstance(row, dict) else None
        if value is None or not str(value).strip():
            return []
        try:
            obj = json.loads(str(value))
        except Exception:
            return []
        entities = []
        for ent in obj.get("entities", []):
            if not isinstance(ent, dict):
                continue
            if str(ent.get("category", "")).upper() not in {"PII", "PHI"}:
                continue
            ent_text = str(ent.get("text", "")).strip()
            ent_type = normalize_type(ent.get("type", "PII"))
            if not ent_text:
                continue
            for m in re.finditer(re.escape(ent_text), str(text), flags=re.IGNORECASE):
                entities.append(PIIEntity(m.group(0), ent_type, m.start(), m.end(), 1.0, "saved_pii_json"))
        return entities
