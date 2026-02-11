from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

from pypdf import PdfReader
from docx import Document


FileType = Literal["pdf", "docx", "txt", "unknown"]


@dataclass
class ExtractResult:
    file_type: FileType
    text: str
    warnings: list[str]


def detect_file_type(path: Path) -> FileType:
    suffix = path.suffix.lower().strip(".")
    if suffix in ("pdf", "docx", "txt"):
        return suffix  # type: ignore
    return "unknown"


def extract_text_from_pdf(path: Path) -> Tuple[str, list[str]]:
    warnings: list[str] = []
    text_parts: list[str] = []

    reader = PdfReader(str(path))
    if not reader.pages:
        return "", ["PDF has 0 pages."]

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            warnings.append(f"Failed extracting page {i + 1}: {e}")
            page_text = ""
        if page_text.strip():
            text_parts.append(page_text)

    text = "\n\n".join(text_parts).strip()
    if not text:
        warnings.append("No extractable text found in PDF (may be image-based).")

    return text, warnings


def extract_text_from_docx(path: Path) -> Tuple[str, list[str]]:
    warnings: list[str] = []
    doc = Document(str(path))

    parts: list[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    text = "\n".join(parts).strip()
    if not text:
        warnings.append("DOCX contains no extractable paragraph text.")

    return text, warnings


def extract_text(path: Path) -> ExtractResult:
    ft = detect_file_type(path)

    if ft == "pdf":
        text, warnings = extract_text_from_pdf(path)
        return ExtractResult(file_type=ft, text=text, warnings=warnings)

    if ft == "docx":
        text, warnings = extract_text_from_docx(path)
        return ExtractResult(file_type=ft, text=text, warnings=warnings)

    if ft == "txt":
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            warnings: list[str] = []
            if not text:
                warnings.append("TXT file is empty.")
            return ExtractResult(file_type=ft, text=text, warnings=warnings)
        except Exception as e:
            return ExtractResult(file_type="txt", text="", warnings=[f"Failed reading TXT: {e}"])

    return ExtractResult(file_type="unknown", text="", warnings=["Unsupported file type. Upload PDF or DOCX."])
