import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from standard_clause_library import STANDARD_CLAUSES
from vector_store import ensure_standard_clauses, retrieve_standard_clauses


logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
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


def require_gemini_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AnalysisUnavailable(
            "Set GEMINI_API_KEY in backend/.env to enable contract analysis."
        )
    return api_key


def _request_json(
    client: genai.Client,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    last_error = None

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=json.dumps(user_payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            content = response.text
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
            status_code = getattr(exc, "status_code", getattr(exc, "code", None))
            message = str(exc).lower()
            error_name = exc.__class__.__name__.lower()
            logger.exception("Gemini request or response parsing failed")

            is_transient = status_code in (408, 429, 500, 502, 503, 504)
            if is_transient and attempt < 3:
                time.sleep(2 ** attempt)
                continue

            if status_code in (401, 403) or "api key" in message or "permission" in message:
                raise AnalysisFailed(
                    "Gemini authentication or access failed. Replace GEMINI_API_KEY in "
                    "backend/.env and check the key and project permissions."
                ) from exc
            if status_code == 429 or "quota" in message or "rate" in message:
                raise AnalysisFailed(
                    "Gemini is rate-limited or the project has no remaining quota."
                ) from exc
            if status_code == 503 or "unavailable" in message or "high demand" in message:
                raise AnalysisFailed(
                    "Gemini is temporarily unavailable due to high demand. Please retry shortly."
                ) from exc
            if "timeout" in error_name or "timeout" in message:
                raise AnalysisFailed("The Gemini request timed out. Try again in a few moments.") from exc
            if "connection" in error_name or "network" in message:
                raise AnalysisFailed("ContractLens could not connect to Gemini.") from exc
            raise AnalysisFailed("The Gemini request failed.") from exc

    raise AnalysisFailed("The Gemini request failed after retries.") from last_error


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_chunks(text: str, chunk_size: int = 7000, overlap: int = 400) -> list[str]:
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


def _extract_clauses(client: genai.Client, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                if not normalized_quote or normalized_quote not in _normalized(page_text):
                    raise AnalysisFailed(
                        f"An AI-generated clause quote could not be verified on page {page_number}."
                    )
                identity = (page_number, normalized_quote)
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
    client: genai.Client,
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
    batch_size = 8

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
    client = genai.Client(api_key=require_gemini_key())
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
    client: genai.Client,
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
    client = genai.Client(api_key=require_gemini_key())
    return compare_analyses(client, first, second)
