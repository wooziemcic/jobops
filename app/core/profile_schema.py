from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, ValidationError


class SkillItem(BaseModel):
    name: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SkillsBlock(BaseModel):
    analytics_visualization: List[SkillItem] = Field(default_factory=list)
    programming_data: List[SkillItem] = Field(default_factory=list)
    statistical_analysis: List[SkillItem] = Field(default_factory=list)
    cloud_etl: List[SkillItem] = Field(default_factory=list)
    workflow_collaboration: List[SkillItem] = Field(default_factory=list)
    other: List[SkillItem] = Field(default_factory=list)


class ExperienceBullet(BaseModel):
    text: str
    evidence: List[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    company: str
    title: str
    start: Optional[str] = None   # "YYYY-MM" or "YYYY"
    end: Optional[str] = None     # "YYYY-MM" or "Present"
    location: Optional[str] = None
    bullets: List[ExperienceBullet] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str
    degree: Optional[str] = None
    graduation: Optional[str] = None
    gpa: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str
    bullets: List[ExperienceBullet] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class SummaryBlock(BaseModel):
    headline: str
    seniority: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    summary: SummaryBlock
    target_roles: List[str] = Field(default_factory=list)
    skills: SkillsBlock
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


def validate_profile_strict(profile_dict: Dict[str, Any], evidence_ids: Set[str]) -> CandidateProfile:
    """
    Strict validation:
    - Pydantic schema validation
    - Every evidence ID referenced must exist in evidence_map
    - Every SkillItem must have >= 1 evidence ID
    - Every bullet must have >= 1 evidence ID
    """
    try:
        profile = CandidateProfile.model_validate(profile_dict)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed:\n{e}")

    # Collect all referenced evidence ids
    referenced: List[str] = []

    referenced.extend(profile.summary.evidence)

    for s_list in [
        profile.skills.analytics_visualization,
        profile.skills.programming_data,
        profile.skills.statistical_analysis,
        profile.skills.cloud_etl,
        profile.skills.workflow_collaboration,
        profile.skills.other,
    ]:
        for s in s_list:
            if not s.evidence:
                raise ValueError(f"Skill '{s.name}' has no evidence IDs. Strict mode requires evidence.")
            referenced.extend(s.evidence)

    for exp in profile.experience:
        for b in exp.bullets:
            if not b.evidence:
                raise ValueError(f"Experience bullet has no evidence IDs: '{b.text[:80]}'")
            referenced.extend(b.evidence)

    for edu in profile.education:
        referenced.extend(edu.evidence)

    for proj in profile.projects:
        referenced.extend(proj.evidence)
        for b in proj.bullets:
            if not b.evidence:
                raise ValueError(f"Project bullet has no evidence IDs: '{b.text[:80]}'")
            referenced.extend(b.evidence)

    missing = sorted({eid for eid in referenced if eid not in evidence_ids})
    if missing:
        raise ValueError(f"Profile references unknown evidence IDs: {missing}")

    return profile
