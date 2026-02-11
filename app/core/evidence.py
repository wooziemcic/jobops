from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


BULLET_CHARS = ("•", "·", "‣", "▪", "●", "○", "–", "-", "—", "*", ">")


SECTION_HEADER_RE = re.compile(
    r"^(experience|professional experience|work experience|education|skills|projects|certifications|summary|profile|technical skills)\b[:\s]*$",
    re.IGNORECASE,
)


DATE_HINT_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b|\b(19|20)\d{2}\b",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    # Normalize whitespace but preserve meaningful punctuation
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _is_probable_header(line: str) -> bool:
    if not line:
        return False
    if SECTION_HEADER_RE.match(line.strip()):
        return True

    # Heuristic: short, title-case-ish, no bullet, no date, not too long
    if len(line) <= 40 and not line.lstrip().startswith(BULLET_CHARS) and not DATE_HINT_RE.search(line):
        # many headers are words with no punctuation
        if sum(ch.isalpha() for ch in line) / max(len(line), 1) > 0.6 and line.count(".") == 0:
            # avoid classifying normal sentences as headers
            if line.lower() == line or line.isupper() or line.istitle():
                return True
    return False


def _strip_bullet_prefix(line: str) -> str:
    s = line.lstrip()
    for b in BULLET_CHARS:
        if s.startswith(b):
            s = s[len(b):].lstrip()
            break
    return s


def build_evidence_map(resume_text: str) -> Dict[str, Any]:
    """
    Deterministically convert resume text into an evidence map.

    Output format:
    {
      "evidence": [
        {
          "id": "E0001",
          "section": "Experience" | "Skills" | "Education" | "Projects" | "Other",
          "source_text": "...",
          "location": {"type": "line", "index": <int>}
        }
      ],
      "meta": {
        "total_lines": <int>,
        "total_evidence": <int>
      }
    }
    """
    raw_lines = resume_text.splitlines()
    lines = [_clean_line(l) for l in raw_lines]
    lines = [l for l in lines if l]  # remove empties

    current_section = "Other"
    evidence_items: List[Dict[str, Any]] = []

    def normalize_section(header: str) -> str:
        h = header.strip().lower()
        if "experience" in h or "work" in h:
            return "Experience"
        if "education" in h:
            return "Education"
        if "skill" in h:
            return "Skills"
        if "project" in h:
            return "Projects"
        if "cert" in h:
            return "Certifications"
        if "summary" in h or "profile" in h:
            return "Summary"
        return "Other"

    eid = 1

    for idx, line in enumerate(lines):
        # Detect section headers
        if _is_probable_header(line):
            current_section = normalize_section(line)
            continue

        # Evidence line rules:
        # 1) Bullet lines become evidence
        # 2) Non-bullet but meaningful lines can become evidence in Experience/Education/Projects
        is_bullet = line.lstrip().startswith(BULLET_CHARS)

        if is_bullet:
            text = _strip_bullet_prefix(line)
            if len(text) < 3:
                continue
            evidence_items.append({
                "id": f"E{eid:04d}",
                "section": current_section,
                "source_text": text,
                "location": {"type": "line", "index": idx},
            })
            eid += 1
            continue

        # For non-bullet lines:
        # Keep them if they look like role/title/company lines or degree lines or project headings,
        # but avoid capturing random single words.
        if current_section in ("Experience", "Education", "Projects", "Certifications", "Summary"):
            if len(line) >= 8:
                evidence_items.append({
                    "id": f"E{eid:04d}",
                    "section": current_section,
                    "source_text": line,
                    "location": {"type": "line", "index": idx},
                })
                eid += 1

    return {
        "evidence": evidence_items,
        "meta": {
            "total_lines": len(lines),
            "total_evidence": len(evidence_items),
        },
    }
