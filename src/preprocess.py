from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import jieba
except ImportError:  # pragma: no cover - used only when jieba is not installed
    jieba = None


RESUME_REQUIRED_FIELDS = [
    "resume_id",
    "name",
    "education",
    "major",
    "skills",
    "experience_years",
    "city",
    "expected_salary",
    "certificates",
    "project_experience",
    "self_description",
]

JOB_REQUIRED_FIELDS = [
    "job_id",
    "job_title",
    "company",
    "required_education",
    "required_skills",
    "min_experience_years",
    "city",
    "salary",
    "preferred_certificates",
    "job_description",
]

TEXT_SPLIT_RE = re.compile(r"[;；、,，/|]+")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


def load_stopwords(path: str | Path) -> set[str]:
    stopword_path = Path(path)
    if not stopword_path.exists():
        return set()
    return {
        line.strip().lower()
        for line in stopword_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_skill_alias(path: str | Path) -> dict[str, str]:
    alias_path = Path(path)
    if not alias_path.exists():
        return {}
    raw = json.loads(alias_path.read_text(encoding="utf-8"))
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}


def normalize_skill(skill: str, alias_map: dict[str, str]) -> str:
    cleaned = str(skill).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return alias_map.get(cleaned, cleaned)


def split_items(value: object, alias_map: dict[str, str] | None = None) -> list[str]:
    if pd.isna(value):
        return []
    alias_map = alias_map or {}
    items = []
    for item in TEXT_SPLIT_RE.split(str(value)):
        normalized = normalize_skill(item, alias_map)
        if normalized and normalized not in items:
            items.append(normalized)
    return items


def register_jieba_words(words: Iterable[str]) -> None:
    if jieba is None:
        return
    for word in words:
        if word and re.search(r"[\u4e00-\u9fff]", word):
            jieba.add_word(word)


def tokenize_text(text: str, stopwords: set[str], alias_map: dict[str, str]) -> list[str]:
    normalized = str(text).lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff+#.]+", " ", normalized)

    if jieba is not None:
        tokens = jieba.lcut(normalized)
    else:
        tokens = TOKEN_RE.findall(normalized)

    result: list[str] = []
    for token in tokens:
        token = normalize_skill(token, alias_map)
        if not token or token in stopwords or token.isspace():
            continue
        if len(token) == 1 and not re.match(r"[a-z0-9]", token):
            continue
        result.append(token)
    return result


def build_resume_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("major", "")),
            str(row.get("skills", "")),
            str(row.get("certificates", "")),
            str(row.get("project_experience", "")),
            str(row.get("self_description", "")),
        ]
    )


def build_job_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("job_title", "")),
            str(row.get("required_skills", "")),
            str(row.get("preferred_certificates", "")),
            str(row.get("job_description", "")),
        ]
    )


def check_required_fields(df: pd.DataFrame, required_fields: list[str], dataset_name: str) -> None:
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 缺少必要字段: {', '.join(missing)}")


def load_csv_data(resume_path: str | Path, job_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    resume_path = Path(resume_path)
    job_path = Path(job_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"简历文件不存在: {resume_path}")
    if not job_path.exists():
        raise FileNotFoundError(f"岗位文件不存在: {job_path}")

    resumes = pd.read_csv(resume_path)
    jobs = pd.read_csv(job_path)
    check_required_fields(resumes, RESUME_REQUIRED_FIELDS, "resumes.csv")
    check_required_fields(jobs, JOB_REQUIRED_FIELDS, "jobs.csv")
    return resumes.fillna(""), jobs.fillna("")


def preprocess_resumes(
    resumes: pd.DataFrame,
    stopwords: set[str],
    alias_map: dict[str, str],
) -> pd.DataFrame:
    resumes = resumes.copy()
    resumes["experience_years"] = pd.to_numeric(resumes["experience_years"], errors="coerce").fillna(0)
    resumes["expected_salary"] = pd.to_numeric(resumes["expected_salary"], errors="coerce").fillna(0)
    resumes["normalized_skills"] = resumes["skills"].apply(lambda value: split_items(value, alias_map))
    resumes["normalized_certificates"] = resumes["certificates"].apply(lambda value: split_items(value, {}))
    register_jieba_words({skill for skills in resumes["normalized_skills"] for skill in skills})
    resumes["raw_text"] = resumes.apply(build_resume_text, axis=1)
    resumes["tokens"] = resumes["raw_text"].apply(lambda text: tokenize_text(text, stopwords, alias_map))
    resumes["clean_text"] = resumes["tokens"].apply(lambda tokens: " ".join(tokens))
    return resumes


def preprocess_jobs(
    jobs: pd.DataFrame,
    stopwords: set[str],
    alias_map: dict[str, str],
) -> pd.DataFrame:
    jobs = jobs.copy()
    jobs["min_experience_years"] = pd.to_numeric(jobs["min_experience_years"], errors="coerce").fillna(0)
    jobs["salary"] = pd.to_numeric(jobs["salary"], errors="coerce").fillna(0)
    jobs["normalized_required_skills"] = jobs["required_skills"].apply(lambda value: split_items(value, alias_map))
    jobs["normalized_preferred_certificates"] = jobs["preferred_certificates"].apply(lambda value: split_items(value, {}))
    register_jieba_words({skill for skills in jobs["normalized_required_skills"] for skill in skills})
    jobs["raw_text"] = jobs.apply(build_job_text, axis=1)
    jobs["tokens"] = jobs["raw_text"].apply(lambda text: tokenize_text(text, stopwords, alias_map))
    jobs["clean_text"] = jobs["tokens"].apply(lambda tokens: " ".join(tokens))
    return jobs
