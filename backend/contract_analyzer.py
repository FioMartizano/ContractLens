import json
import math
import difflib
import unicodedata
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from standard_clause_library import STANDARD_CLAUSES
from vector_store import ensure_standard_clauses, retrieve_standard_clauses


logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DISCLAIMER = (
    "AI-generated information for general review only; it is not legal advice. "
    "Have a qualified lawyer review important decisions."
)


class AnalysisUnavailable(Exception):
    pass


class AnalysisFailed(Exception):
    pass


class ExtractedClause(BaseModel):
    page_number: int = Field(ge=1)
    clause_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    party_obligated: str = Field(min_length=1)
    deadlines: list[str] = Field(default_factory=list)
    money_amounts: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    clauses: list[ExtractedClause]


class RiskFinding(BaseModel):
    clause_id: str
    risk_score: int = Field(ge=0, le=100)
    explanation: str = Field(min_length=1)
    deviation: str = Field(min_length=1)
    suggested_rewrite: str = Field(min_length=1)


class RiskResult(BaseModel):
    findings: list[RiskFinding]


def require_groq_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise AnalysisUnavailable(
            "Set GROQ_API_KEY in backend/.env to enable contract analysis."
        )
    return api_key


def _retry_delay_seconds(exc: Exception, attempt: int) -> int:
    """Use Groq's suggested wait when present, with a small safety buffer."""
    message = str(exc)
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", message, re.IGNORECASE)
    exponential_delay = 2 ** (attempt + 1)
    if not match:
        return exponential_delay

    provider_delay = math.ceil(float(match.group(1))) + 2
    return max(exponential_delay, provider_delay)


def _request_json(
    client: Groq,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = 5
    last_error = None

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = response.choices[0].message.content
            if not content:
                raise AnalysisFailed("The AI returned an empty response.")

            result = json.loads(content)
            if not isinstance(result, dict):
                raise AnalysisFailed("The AI returned an invalid JSON object.")
            return result

        except AnalysisFailed:
            raise
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            message = str(exc).lower()
            error_name = exc.__class__.__name__.lower()

            is_auth_error = status_code in (401, 403)
            is_transient = (
                status_code in (408, 429, 500, 502, 503, 504)
                or "timeout" in error_name
                or "timeout" in message
                or "connection" in error_name
                or "network" in message
                or "temporarily unavailable" in message
            )

            if is_transient and attempt < max_attempts - 1:
                delay = _retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Groq request failed on attempt %s/%s. Retrying in %s seconds. Error: %s",
                    attempt + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            logger.exception(
                "Groq request or response parsing failed after %s attempt(s)",
                attempt + 1,
            )

            if is_auth_error or "api key" in message or "authentication" in message:
                raise AnalysisUnavailable(
                    "Groq authentication failed. Check GROQ_API_KEY in backend/.env."
                ) from exc

            if status_code == 429 or "rate limit" in message or "quota" in message:
                raise AnalysisUnavailable(
                    "Groq rate limit reached. Please wait before trying again."
                ) from exc

            if status_code in (500, 502, 503, 504) or "temporarily unavailable" in message:
                raise AnalysisUnavailable(
                    "Groq is temporarily unavailable. Please retry shortly."
                ) from exc

            if "timeout" in error_name or "timeout" in message:
                raise AnalysisUnavailable(
                    "The Groq request timed out. Try again in a few moments."
                ) from exc

            if "connection" in error_name or "network" in message:
                raise AnalysisUnavailable(
                    "ContractLens could not connect to Groq."
                ) from exc

            raise AnalysisFailed("The Groq request failed.") from exc

    raise AnalysisUnavailable("The Groq request failed after retries.") from last_error


def _normalized(text: str) -> str:
    """Normalize whitespace and Unicode while preserving readable clause text."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _verification_tokens(text: str) -> list[str]:
    """Create comparison tokens tolerant of OCR punctuation and casing differences."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def _quote_is_verifiable(quote: str, source_text: str) -> bool:
    """
    Verify a AI quotation against extracted page text.

    Exact matching is preferred. For longer quotations, token-level fuzzy matching
    tolerates small OCR substitutions while still requiring a close, contiguous match.
    Short quotations remain exact-match only to avoid accepting generic wording.
    """
    normalized_quote = _normalized(quote)
    normalized_source = _normalized(source_text)
    if not normalized_quote:
        return False

    if normalized_quote.casefold() in normalized_source.casefold():
        return True

    quote_tokens = _verification_tokens(quote)
    source_tokens = _verification_tokens(source_text)

    if len(quote_tokens) < 8 or not source_tokens:
        return False

    quote_length = len(quote_tokens)
    min_window = max(1, round(quote_length * 0.88))
    max_window = min(len(source_tokens), round(quote_length * 1.12))
    matcher = difflib.SequenceMatcher(autojunk=False)

    # Search windows close to the quotation's token length. A 0.90 threshold
    # allows a few OCR character/word errors but rejects materially different text.
    matcher.set_seq2(quote_tokens)
    for start in range(len(source_tokens)):
        remaining = len(source_tokens) - start
        if remaining < min_window:
            break
        upper = min(max_window, remaining)
        for window_length in range(min_window, upper + 1):
            matcher.set_seq1(source_tokens[start:start + window_length])
            if matcher.ratio() >= 0.90:
                return True

    return False

def _page_chunks(text: str, chunk_size: int = 4000, overlap: int = 250) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _extract_clauses(client: Groq, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = (
        "Extract legally meaningful contract clauses from the supplied page text. "
        "Return JSON only with a top-level 'clauses' array. For each clause provide "
        "page_number, clause_type, text, party_obligated, deadlines, and money_amounts. "
        "The text field must be an exact, contiguous quotation copied verbatim from "
        "the supplied page, including its original wording. Do not invent, combine, "
        "or paraphrase clause text. Use empty arrays when there are no deadlines or "
        "amounts, and use 'Unclear' for an unidentified obligated party. Extract "
        "operative obligations, rights, restrictions, and remedies; omit headings "
        "without operative text."
    )
    extracted = []
    seen = set()

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]
        for chunk in _page_chunks(page_text):
            raw = _request_json(
                client,
                system_prompt,
                {"page_number": page_number, "page_text": chunk},
            )
            try:
                result = ExtractionResult.model_validate(raw)
            except ValidationError as exc:
                raise AnalysisFailed("The AI returned invalid clause data.") from exc

            for clause in result.clauses:
                if clause.page_number != page_number:
                    raise AnalysisFailed(
                        f"The AI returned an invalid page reference for page {page_number}."
                    )
                normalized_quote = _normalized(clause.text)
                if not _quote_is_verifiable(clause.text, page_text):
                    logger.warning(
                        "Could not verify extracted clause quote on page %s: %s",
                        page_number,
                        clause.text[:180],
                    )
                    raise AnalysisFailed(
                        f"An AI-generated clause quote could not be verified on page {page_number}. "
                        "Try a clearer scan or a higher-resolution image."
                    )
                identity = (page_number, normalized_quote.casefold())
                if identity in seen:
                    continue
                seen.add(identity)
                extracted.append(
                    {
                        "id": str(uuid.uuid4()),
                        **clause.model_dump(),
                    }
                )

    return extracted


def _risk_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _assess_clauses(
    client: Groq,
    clauses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ensure_standard_clauses(STANDARD_CLAUSES)
    references = {
        clause["id"]: retrieve_standard_clauses(clause["text"], top_k=1)
        for clause in clauses
    }

    system_prompt = (
        "Assess each supplied contract clause against its retrieved standard-language "
        "reference. Return JSON only with a top-level 'findings' array. For each "
        "clause_id return risk_score (integer 0-100), explanation, deviation, and "
        "suggested_rewrite. Score only the text provided: 0 is balanced/low exposure; "
        "100 is unusually broad or severe exposure. Do not infer facts or applicable "
        "law that are not in the text. Explain uncertainty and distinguish the "
        "reference template from a legal requirement. The rewrite must preserve the "
        "commercial intent where possible and must not add unsupported facts. This "
        "is general information, not legal advice."
    )
    assessed = []
    batch_size = 3

    for start in range(0, len(clauses), batch_size):
        batch = clauses[start : start + batch_size]
        payload_clauses = [
            {
                "clause_id": clause["id"],
                "clause_type": clause["clause_type"],
                "text": clause["text"],
                "party_obligated": clause["party_obligated"],
                "deadlines": clause["deadlines"],
                "money_amounts": clause["money_amounts"],
                "retrieved_references": references[clause["id"]],
            }
            for clause in batch
        ]
        raw = _request_json(
            client,
            system_prompt,
            {"clauses": payload_clauses},
        )
        try:
            findings = RiskResult.model_validate(raw).findings
        except ValidationError as exc:
            raise AnalysisFailed("The AI returned invalid risk assessment data.") from exc

        by_id = {finding.clause_id: finding for finding in findings}
        expected_ids = {clause["id"] for clause in batch}
        if len(findings) != len(batch) or set(by_id) != expected_ids:
            raise AnalysisFailed("The AI returned incomplete or mismatched clause assessments.")

        for clause in batch:
            finding = by_id[clause["id"]]
            reference = references[clause["id"]]
            assessed.append(
                {
                    **clause,
                    "risk_score": finding.risk_score,
                    "risk_level": _risk_level(finding.risk_score),
                    "explanation": finding.explanation,
                    "deviation": finding.deviation,
                    "suggested_rewrite": finding.suggested_rewrite,
                    "standard_reference": reference[0] if reference else None,
                }
            )

    return assessed


def analyze_contract(
    pages: list[dict[str, Any]],
    document_id: str,
) -> dict[str, Any]:
    client = Groq(api_key=require_groq_key(), timeout=60.0, max_retries=0)
    clauses = _extract_clauses(client, pages)
    if not clauses:
        raise AnalysisFailed("No contract clauses could be extracted from this document.")

    assessed = _assess_clauses(client, clauses)
    risk_counts = {
        level: sum(clause["risk_level"] == level for clause in assessed)
        for level in ("high", "medium", "low")
    }
    average_risk = sum(clause["risk_score"] for clause in assessed) / len(assessed)

    return {
        "document_id": document_id,
        "overall_score": round(100 - average_risk),
        "risk_counts": risk_counts,
        "clauses": assessed,
        "disclaimer": DISCLAIMER,
    }


def compare_analyses(
    client: Groq,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "Compare two analyzed contract clause lists. Match clauses by meaning, not "
        "only title. Return JSON only with a top-level 'changes' array. Each entry "
        "must contain first_clause_id (string or null), second_clause_id (string or "
        "null), risk_change (increased, decreased, unchanged, added, or removed), "
        "and explanation. Use only IDs present in the input; never invent clause IDs. "
        "Include materially changed clauses and newly added or removed obligations."
    )
    raw = _request_json(
        client,
        system_prompt,
        {
            "first_clauses": first["clauses"],
            "second_clauses": second["clauses"],
        },
    )
    changes = raw.get("changes")
    allowed_changes = {"increased", "decreased", "unchanged", "added", "removed"}
    if not isinstance(changes, list):
        raise AnalysisFailed("The AI returned invalid comparison data.")

    first_clauses = {clause["id"]: clause for clause in first["clauses"]}
    second_clauses = {clause["id"]: clause for clause in second["clauses"]}
    verified = []
    for change in changes:
        if not isinstance(change, dict):
            raise AnalysisFailed("The AI returned an invalid comparison result.")
        first_id = change.get("first_clause_id")
        second_id = change.get("second_clause_id")
        risk_change = change.get("risk_change")
        explanation = change.get("explanation")
        if (
            (first_id is not None and (not isinstance(first_id, str) or first_id not in first_clauses))
            or (
                second_id is not None
                and (not isinstance(second_id, str) or second_id not in second_clauses)
            )
            or (first_id is None and second_id is None)
            or not isinstance(risk_change, str)
            or risk_change not in allowed_changes
            or not isinstance(explanation, str)
            or not explanation.strip()
            or (risk_change == "added" and (first_id is not None or second_id is None))
            or (risk_change == "removed" and (first_id is None or second_id is not None))
        ):
            raise AnalysisFailed("The AI returned an invalid comparison result.")
        verified.append(
            {
                "first_clause": first_clauses.get(first_id),
                "second_clause": second_clauses.get(second_id),
                "risk_change": risk_change,
                "explanation": explanation,
            }
        )

    return {"changes": verified, "disclaimer": DISCLAIMER}


def compare_documents(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    client = Groq(api_key=require_groq_key(), timeout=60.0, max_retries=0)
    return compare_analyses(client, first, second)
