from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


EDUCATION_LEVELS = {
    "高中": 1,
    "中专": 1,
    "大专": 2,
    "专科": 2,
    "本科": 3,
    "硕士": 4,
    "研究生": 4,
    "博士": 5,
}

TOTAL_SCORE_WEIGHTS = {
    "skill_score": 0.28,
    "tfidf_score": 0.20,
    "word2vec_score": 0.15,
    "education_score": 0.12,
    "experience_score": 0.10,
    "city_score": 0.05,
    "certificate_score": 0.05,
    "salary_score": 0.05,
}


@dataclass
class RuleScoreResult:
    skill_score: float
    education_score: float
    experience_score: float
    city_score: float
    certificate_score: float
    salary_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    matched_certificates: list[str]


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if pd.isna(value):
        return []
    return [item for item in str(value).split(";") if item]


def _score_level(value: str) -> int:
    text = str(value)
    for name, level in EDUCATION_LEVELS.items():
        if name in text:
            return level
    return 0


def skill_match_score(resume_skills: Iterable[str], required_skills: Iterable[str]) -> tuple[float, list[str], list[str]]:
    resume_set = set(resume_skills)
    required = list(dict.fromkeys(required_skills))
    if not required:
        return 100.0, [], []
    matched = [skill for skill in required if skill in resume_set]
    missing = [skill for skill in required if skill not in resume_set]
    return round(len(matched) / len(required) * 100, 2), matched, missing


def education_score(resume_education: str, required_education: str) -> float:
    resume_level = _score_level(resume_education)
    required_level = _score_level(required_education)
    if required_level == 0:
        return 100.0
    if resume_level >= required_level:
        return 100.0
    if required_level - resume_level == 1:
        return 70.0
    return 40.0


def experience_score(resume_years: float, required_years: float) -> float:
    resume_years = float(resume_years or 0)
    required_years = float(required_years or 0)
    if required_years <= 0 or resume_years >= required_years:
        return 100.0
    gap = required_years - resume_years
    if gap <= 1:
        return 75.0
    return max(0.0, 60.0 - gap * 20.0)


def city_score(expected_city: str, job_city: str) -> float:
    expected = str(expected_city).strip()
    job = str(job_city).strip()
    if not expected or expected in {"不限", "全国"}:
        return 85.0
    if job in {"不限", "全国", "远程"}:
        return 90.0
    return 100.0 if expected == job else 60.0


def certificate_score(resume_certs: Iterable[str], preferred_certs: Iterable[str]) -> tuple[float, list[str]]:
    preferred = [cert for cert in preferred_certs if cert]
    if not preferred:
        return 100.0, []
    resume_set = set(resume_certs)
    matched = [cert for cert in preferred if cert in resume_set]
    if not matched:
        return 60.0, []
    return round(len(matched) / len(preferred) * 100, 2), matched


def salary_score(expected_salary: float, job_salary: float) -> float:
    expected = float(expected_salary or 0)
    salary = float(job_salary or 0)
    if expected <= 0:
        return 80.0
    if salary >= expected:
        return 100.0
    if salary >= expected * 0.8:
        return 75.0
    return 50.0


def calculate_rule_scores(resume: pd.Series, job: pd.Series) -> RuleScoreResult:
    skill_score_value, matched_skills, missing_skills = skill_match_score(
        _as_list(resume.get("normalized_skills")),
        _as_list(job.get("normalized_required_skills")),
    )
    certificate_score_value, matched_certificates = certificate_score(
        _as_list(resume.get("normalized_certificates")),
        _as_list(job.get("normalized_preferred_certificates")),
    )
    return RuleScoreResult(
        skill_score=skill_score_value,
        education_score=education_score(str(resume.get("education", "")), str(job.get("required_education", ""))),
        experience_score=experience_score(resume.get("experience_years", 0), job.get("min_experience_years", 0)),
        city_score=city_score(str(resume.get("city", "")), str(job.get("city", ""))),
        certificate_score=certificate_score_value,
        salary_score=salary_score(resume.get("expected_salary", 0), job.get("salary", 0)),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_certificates=matched_certificates,
    )


def total_score(scores: dict[str, float]) -> float:
    return round(sum(scores[name] * weight for name, weight in TOTAL_SCORE_WEIGHTS.items()), 2)


def semantic_level(score: float) -> str:
    if score >= 75:
        return "文本语义相似度较高"
    if score >= 50:
        return "文本语义相似度中等"
    return "文本语义相似度偏低"


def build_reason(
    resume: pd.Series,
    job: pd.Series,
    rule_scores: RuleScoreResult,
    semantic_score: float,
) -> str:
    parts = []
    if rule_scores.matched_skills:
        parts.append(f"共同技能包括 {', '.join(rule_scores.matched_skills)}")
    else:
        parts.append("岗位核心技能重合较少")

    if rule_scores.missing_skills:
        parts.append(f"缺少 {', '.join(rule_scores.missing_skills)}，建议补充相关练习或项目经验")
    else:
        parts.append("岗位要求技能基本覆盖")

    parts.append("学历满足岗位要求" if rule_scores.education_score >= 100 else "学历与岗位最低要求存在差距")
    parts.append("经验满足岗位要求" if rule_scores.experience_score >= 100 else "经验年限略低于岗位要求")

    if rule_scores.city_score >= 100:
        parts.append("期望城市与岗位城市一致")
    elif rule_scores.city_score >= 85:
        parts.append("城市要求较灵活")
    else:
        parts.append("期望城市与岗位城市不一致")

    if rule_scores.matched_certificates:
        parts.append(f"证书匹配 {', '.join(rule_scores.matched_certificates)}")
    elif _as_list(job.get("normalized_preferred_certificates")):
        parts.append("岗位偏好证书匹配不足")

    parts.append(semantic_level(semantic_score))
    return "；".join(parts) + "。"
