from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    resume: Path
    jobs: Path
    history: Path


def get_paths() -> AppPaths:
    # repo_root/jobops/app/core/config.py -> repo_root/jobops
    root = Path(__file__).resolve().parents[2]
    data = root / "data"
    resume = data / "resume"
    jobs = data / "jobs"
    history = data / "history"

    for p in [data, resume, jobs, history]:
        p.mkdir(parents=True, exist_ok=True)

    return AppPaths(root=root, data=data, resume=resume, jobs=jobs, history=history)


def get_openai_api_key() -> str | None:
    # Local: optional .env usage if you want later; Streamlit Cloud: secrets/env var.
    # We don't force it now (Resume Intake step doesn't need OpenAI).
    return os.getenv("OPENAI_API_KEY")
