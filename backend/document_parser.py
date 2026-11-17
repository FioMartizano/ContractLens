
import io
import os
from pathlib import Path

import pdfplumber
import pytesseract

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError


# Load local environment variables from the backend folder
load_dotenv(Path(__file__).resolve().parent / ".env")

# Get the Tesseract executable path from the developer's .env file
TESSERACT_EXE = os.getenv("TESSERACT_CMD")

# Configure Tesseract if a custom path was provided
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
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

            text = pytesseract.image_to_string(
                image.convert("RGB"),
                lang="eng"
            )

        return [{
            "page": 1,
            "text": text
        }]

    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
