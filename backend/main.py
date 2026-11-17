
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from document_parser import extract_pdf_text, extract_image_text


app = FastAPI(
    title="ContractLens API",
    description="AI-Powered Contract Risk Reviewer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {".pdf", ".jpg", ".jpeg", ".png"}


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "application": "ContractLens API"
    }


@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PDF, JPG, JPEG, and PNG."
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Maximum file size is 10 MB."
            )

        if extension == ".pdf":
            pages = extract_pdf_text(file_bytes)
            file_type = "PDF"
        else:
            pages = extract_image_text(file_bytes)
            file_type = "Image"

        total_characters = sum(
            len(page["text"]) for page in pages
        )

        return {
            "filename": file.filename,
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

    except Exception as exc:
        print(
            f"DOCUMENT PROCESSING ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {type(exc).__name__}: {exc}"
        )

    finally:
        await file.close()
