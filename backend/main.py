
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from document_parser import extract_pdf_text, extract_image_text
from contract_analyzer import (
    AnalysisFailed,
    AnalysisUnavailable,
    analyze_contract,
    compare_documents,
    require_gemini_key,
)
from vector_store import (
    get_analysis,
    index_document,
    list_analyses,
    save_analysis,
    search_documents,
)


logger = logging.getLogger(__name__)


app = FastAPI(
    title="ContractLens API",
    description="AI-Powered Contract Risk Reviewer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {".pdf", ".jpg", ".jpeg", ".png"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = None


class CompareRequest(BaseModel):
    first_document_id: str = Field(min_length=1)
    second_document_id: str = Field(min_length=1)


async def _extract_upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PDF, JPG, JPEG, and PNG.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB.")

    if extension == ".pdf":
        return file.filename, "PDF", extract_pdf_text(file_bytes)
    return file.filename, "Image", extract_image_text(file_bytes)


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "application": "ContractLens API"
    }


@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    try:
        filename, file_type, pages = await _extract_upload(file)
        total_characters = sum(len(page["text"]) for page in pages)

        return {
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(pages),
            "total_characters": total_characters,
            "pages": pages
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("Document processing failed")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        )

    finally:
        await file.close()


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    try:
        require_gemini_key()
        filename, file_type, pages = await _extract_upload(file)
        total_characters = sum(len(page["text"]) for page in pages)
        if total_characters == 0:
            raise HTTPException(
                status_code=422,
                detail="No readable text was found in the document.",
            )

        indexed = await run_in_threadpool(
            index_document,
            filename,
            file_type,
            pages,
        )
        analysis = await run_in_threadpool(
            analyze_contract,
            pages,
            indexed["document_id"],
        )
        analysis.update({
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(pages),
            "total_characters": total_characters,
            "chunks_indexed": indexed["chunks_indexed"],
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })
        await run_in_threadpool(
            save_analysis,
            indexed["document_id"],
            filename,
            analysis,
        )
        return analysis
    except AnalysisUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AnalysisFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Document extraction, analysis, or ChromaDB indexing failed.",
        ) from exc
    finally:
        await file.close()


@app.get("/api/documents")
def list_documents():
    try:
        return {"documents": list_analyses()}
    except Exception as exc:
        logger.exception("Document listing failed")
        raise HTTPException(
            status_code=500,
            detail="Could not load analyzed documents from ChromaDB.",
        ) from exc


@app.post("/api/compare")
def compare_contracts(request: CompareRequest):
    if request.first_document_id == request.second_document_id:
        raise HTTPException(
            status_code=400,
            detail="Choose two different contracts to compare.",
        )

    try:
        first = get_analysis(request.first_document_id)
        second = get_analysis(request.second_document_id)
        if first is None or second is None:
            raise HTTPException(
                status_code=404,
                detail="One or both analyzed contracts could not be found.",
            )
        comparison = compare_documents(first, second)
        return {
            "first_document": {
                "document_id": request.first_document_id,
                "filename": first["filename"],
                "overall_score": first["overall_score"],
            },
            "second_document": {
                "document_id": request.second_document_id,
                "filename": second["filename"],
                "overall_score": second["overall_score"],
            },
            **comparison,
        }
    except AnalysisUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AnalysisFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Contract comparison failed")
        raise HTTPException(
            status_code=500,
            detail="Contract comparison failed.",
        ) from exc


@app.post("/api/search")
def search_contracts(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query cannot be empty.")

    try:
        results = search_documents(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        return {"results": results}
    except Exception as exc:
        logger.exception("Document search failed")
        raise HTTPException(
            status_code=500,
            detail="Document search failed.",
        ) from exc
