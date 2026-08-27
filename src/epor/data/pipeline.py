"""Deterministic normalization, classification, redaction, and dedup helpers."""

from __future__ import annotations

import codecs
import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .models import Classification, DocumentRecord, Finding, SourceRegistration
from .store import STREAM_CHUNK_BYTES, sha256_bytes

PIPELINE_VERSION = "epor-data-v1"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{7,}\d)(?!\d)")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)
_AWS_KEY = re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")
_PREFIXED_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_|sk-)[A-Za-z0-9_-]{20,}")
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)"
    r"\s*[:=]\s*['\"]?[^\s'\";,]{8,}"
)
_BENCHMARK_MARKERS = re.compile(r"(?i)\b(?:HumanEval|MBPP|GSM8K|MMLU|ARC[- ]Challenge|HellaSwag)\b")
_UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("eicar_test_signature", re.compile(r"EICAR-STANDARD-ANTIVIRUS-TEST-FILE")),
    ("credential_theft_payload", re.compile(r"(?i)\b(?:mimikatz|lsass dump|credential dumping)\b")),
    ("malware_payload", re.compile(r"(?i)\b(?:meterpreter|msfvenom|powershell\s+-enc)\b")),
)


def normalize_text(path: Path) -> str:
    """Decode UTF-8 incrementally, normalize line endings, and remove controls."""

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pieces: list[str] = []
    pending_carriage_return = False
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(STREAM_CHUNK_BYTES), b""):
            text = decoder.decode(raw)
            if pending_carriage_return:
                text = "\r" + text
                pending_carriage_return = False
            if text.endswith("\r"):
                text = text[:-1]
                pending_carriage_return = True
            pieces.append(text.replace("\r\n", "\n").replace("\r", "\n"))
        tail = decoder.decode(b"", final=True)
        if pending_carriage_return:
            tail = "\n" + tail
        pieces.append(tail.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = unicodedata.normalize("NFC", "".join(pieces))
    return _CONTROL_CHARACTERS.sub("", normalized)


def _redact(
    text: str,
    patterns: Iterable[tuple[str, re.Pattern[str], str]],
) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    for kind, pattern, replacement in patterns:
        text, count = pattern.subn(replacement, text)
        if count:
            findings.append(Finding(kind=kind, count=count, action="redacted"))
    return text, findings


def redact_sensitive_text(text: str) -> tuple[str, list[Finding]]:
    """Remove common PII and credential forms without retaining matched values."""

    return _redact(
        text,
        (
            ("private_key", _PRIVATE_KEY, "[REDACTED:CREDENTIAL]"),
            ("aws_access_key", _AWS_KEY, "[REDACTED:CREDENTIAL]"),
            ("prefixed_api_token", _PREFIXED_TOKEN, "[REDACTED:CREDENTIAL]"),
            ("assigned_secret", _ASSIGNED_SECRET, "[REDACTED:CREDENTIAL]"),
            ("email_address", _EMAIL, "[REDACTED:EMAIL]"),
            ("phone_number", _PHONE, "[REDACTED:PHONE]"),
            ("ip_address", _IPV4, "[REDACTED:IP]"),
        ),
    )


def classify_language(text: str, hint: str | None = None) -> Classification:
    if hint:
        return Classification(
            label=hint.lower(), confidence=0.95, classifier="registration-hint-v1"
        )
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return Classification(label="unknown", confidence=0.0, classifier="unicode-script-v1")
    ascii_ratio = sum(ord(character) < 128 for character in letters) / len(letters)
    lower = f" {_WHITESPACE.sub(' ', text.lower())} "
    english_hits = sum(lower.count(word) for word in (" the ", " and ", " of ", " to ", " is "))
    if ascii_ratio > 0.92 and english_hits:
        confidence = min(0.99, 0.65 + english_hits / max(len(letters) / 50, 1) * 0.1)
        return Classification(label="en", confidence=confidence, classifier="unicode-script-v1")
    if ascii_ratio > 0.98:
        return Classification(
            label="latin-unknown", confidence=0.45, classifier="unicode-script-v1"
        )
    return Classification(label="multilingual", confidence=0.6, classifier="unicode-script-v1")


def classify_domain(
    text: str,
    media_type: str,
    hint: str | None = None,
) -> Classification:
    if hint:
        return Classification(
            label=hint.lower(), confidence=0.95, classifier="registration-hint-v1"
        )
    lower = text.lower()
    code_markers = sum(
        lower.count(item) for item in ("def ", "class ", "import ", "function ", "const ")
    )
    science_markers = sum(
        lower.count(item) for item in (" theorem ", " equation ", " experiment ", " hypothesis ")
    )
    technical_markers = sum(
        lower.count(item)
        for item in (" api ", " protocol ", " database ", " algorithm ", " system ")
    )
    if media_type in {"text/x-python", "text/x-source-code"} or code_markers >= 2:
        return Classification(label="code", confidence=0.9, classifier="domain-rules-v1")
    if science_markers >= 2:
        return Classification(label="math-science", confidence=0.8, classifier="domain-rules-v1")
    if technical_markers >= 2:
        return Classification(label="technical", confidence=0.75, classifier="domain-rules-v1")
    return Classification(label="general", confidence=0.55, classifier="domain-rules-v1")


def classify_quality(text: str) -> Classification:
    if not text:
        score = 0.0
    else:
        printable = sum(character.isprintable() or character in "\n\t" for character in text)
        printable_ratio = printable / len(text)
        tokens = _WHITESPACE.sub(" ", text).strip().split(" ")
        nonempty = [token for token in tokens if token]
        length_score = min(1.0, len(nonempty) / 100)
        unique_ratio = len(set(nonempty)) / max(len(nonempty), 1)
        score = round(0.5 * printable_ratio + 0.25 * length_score + 0.25 * unique_ratio, 6)
    label = "high" if score >= 0.8 else "medium" if score >= 0.55 else "low"
    return Classification(label=label, confidence=score, classifier="quality-rules-v1")


def unsafe_findings(raw_path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    with raw_path.open("rb") as stream:
        prefix = stream.read(4)
    if prefix.startswith((b"MZ", b"\x7fELF")):
        findings.append(Finding(kind="executable_binary", count=1, action="quarantined"))
    for kind, pattern in _UNSAFE_PATTERNS:
        count = len(pattern.findall(text))
        if count:
            findings.append(Finding(kind=kind, count=count, action="quarantined"))
    benchmark_count = len(_BENCHMARK_MARKERS.findall(text))
    if benchmark_count:
        findings.append(Finding(kind="benchmark_marker", count=benchmark_count, action="reported"))
    return findings


def make_document_record(
    registration: SourceRegistration,
    *,
    registration_sha256: str,
    raw_path: Path,
    raw_sha256: str,
    raw_size_bytes: int,
    relative_input_name: str,
    source_order: int,
) -> DocumentRecord:
    normalized = normalize_text(raw_path)
    redacted, findings = redact_sensitive_text(normalized)
    findings.extend(unsafe_findings(raw_path, redacted))
    language = classify_language(redacted, registration.language_hint)
    domain = classify_domain(redacted, registration.media_type, registration.domain_hint)
    quality = classify_quality(redacted)
    reasons: list[str] = []
    if "train" not in registration.allowed_uses:
        reasons.append("rights_not_training_eligible")
    if any(item.action == "quarantined" for item in findings):
        reasons.append("unsafe_content")
    if quality.confidence < 0.2:
        reasons.append("quality_below_floor")
    if not redacted.strip():
        reasons.append("empty_after_normalization")
    disposition = "quarantined" if reasons else "admitted"
    document_id = sha256_bytes(f"{registration.id}\0{relative_input_name}\0{raw_sha256}".encode())
    group_id = (
        registration.group_id
        or registration.repository_family
        or registration.benchmark_family
        or registration.id
    )
    return DocumentRecord(
        document_id=document_id,
        source_id=registration.id,
        relative_input_name=relative_input_name,
        source_order=source_order,
        media_type=registration.media_type,
        acquired_at=registration.acquired_at,
        raw_sha256=raw_sha256,
        raw_size_bytes=raw_size_bytes,
        registration_sha256=registration_sha256,
        normalized_sha256=sha256_bytes(redacted.encode("utf-8")),
        normalized_size_bytes=len(redacted.encode("utf-8")),
        text=redacted,
        language=language,
        domain=domain,
        quality=quality,
        group_id=group_id,
        repository_family=registration.repository_family,
        benchmark_family=registration.benchmark_family,
        findings=findings,
        disposition=disposition,
        reason_codes=sorted(set(reasons)),
    )


def canonical_dedup_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def exact_fingerprint(text: str) -> str:
    return sha256_bytes(canonical_dedup_text(text).encode("utf-8"))


def simhash64(text: str) -> int:
    tokens = canonical_dedup_text(text).split()
    if not tokens:
        return 0
    shingles = [" ".join(tokens[index : index + 4]) for index in range(max(1, len(tokens) - 3))]
    weights = [0] * 64
    for shingle, count in Counter(shingles).items():
        value = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += count if value & (1 << bit) else -count
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def split_for(group_id: str, salt: str, train: int, validation: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{salt}\0{group_id}".encode()).digest()[:8], "big")
    percentile = bucket % 10_000
    train_end = train * 100
    validation_end = train_end + validation * 100
    if percentile < train_end:
        return "train"
    if percentile < validation_end:
        return "validation"
    return "test"
