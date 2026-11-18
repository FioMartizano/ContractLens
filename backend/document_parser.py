
import io
import os
import shutil
from pathlib import Path

import pdfplumber
import pytesseract

from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


# Load local environment variables from the backend folder
load_dotenv(Path(__file__).resolve().parent / ".env")

# Resolve Tesseract on this computer instead of reusing another developer's path.
def _resolve_tesseract() -> str | None:
    configured = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        configured,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract") or "",
    ]

    for candidate in candidates:
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                return str(candidate_path)
    return None


TESSERACT_EXE = _resolve_tesseract()
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def extract_pdf_text(pdf_bytes: bytes):
    """
    Extract text from a PDF document.
    Preserves page numbers for clause referencing.
    """
    extracted_pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    return extracted_pages


def extract_image_text(image_bytes: bytes):
    """
    Extract text from an image using Tesseract OCR.
    Returns the same page-based structure as PDF extraction.
    """
    if not TESSERACT_EXE:
        raise ValueError(
            "Tesseract OCR is not installed or could not be found on this backend "
            "computer. Install Tesseract, then set TESSERACT_CMD to tesseract.exe "
            "and restart the backend."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

            # Small screenshots often contain text that is too tiny for reliable OCR.
            # Upscale first, then normalize contrast and sharpen character edges.
            working = image.convert("RGB")
            max_dimension = 2600
            if max(working.size) < max_dimension:
                scale = min(3.0, max_dimension / max(working.size))
                working = working.resize(
                    (round(working.width * scale), round(working.height * scale)),
                    Image.Resampling.LANCZOS,
                )

            processed = ImageOps.grayscale(working)
            processed = ImageOps.autocontrast(processed)
            processed = ImageEnhance.Contrast(processed).enhance(1.35)
            processed = processed.filter(ImageFilter.SHARPEN)

            # Automatic page segmentation works better for mixed paragraphs and tables
            # than assuming the entire image is one uniform text block.
            text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config="--oem 3 --psm 3",
            )

        return [{
            "page": 1,
            "text": text
        }]

    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise ValueError(
            "Tesseract OCR is not installed or could not be found on this backend "
            "computer."
        ) from exc
    except OSError as exc:
        raise ValueError(
            "Tesseract OCR could not start. Check TESSERACT_CMD and restart the backend."
        ) from exc
