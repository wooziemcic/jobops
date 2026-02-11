from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class JobPosting(BaseModel):
    job_id: str = Field(..., description="Stable internal id (hash).")
    source: str = Field(..., description="Connector name, e.g., 'rss:<feedname>' or 'ats:<name>'.")
    source_job_id: Optional[str] = Field(default=None, description="Provider's job id if available.")

    title: str
    company: str
    location: Optional[str] = None
    remote: Optional[bool] = None

    apply_url: str
    job_url: Optional[str] = None

    description: Optional[str] = None

    posted_at: Optional[str] = None     # ISO string if available
    first_seen_at: str                  # ISO string (always set by our system)

    role_bucket: Optional[str] = None   # DA/DS/BA/Quant/ML/AI/Other
    tags: List[str] = Field(default_factory=list)

    raw: Dict[str, Any] = Field(default_factory=dict)
