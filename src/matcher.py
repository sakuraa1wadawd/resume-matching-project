from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    load_csv_data,
    load_skill_alias,
    load_stopwords,
    preprocess_jobs,
    preprocess_resumes,
)
from src.scoring import TOTAL_SCORE_WEIGHTS, build_reason, calculate_rule_scores, total_score
from src.similarity import compute_semantic_scores, compute_tfidf_scores, compute_word2vec_scores

DEFAULT_RESUME_PATH = PROJECT_ROOT / "data" / "resumes.csv"
DEFAULT_JOB_PATH = PROJECT_ROOT / "data" / "jobs.csv"
DEFAULT_STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "data" / "skill_alias.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def _join(values: list[str]) -> str:
    return ";".join(values)


def build_match_results(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    tfidf_scores,
    word2vec_scores,
    semantic_scores,
) -> pd.DataFrame:
    rows = []
    for resume_index, resume in resumes.iterrows():
        for job_index, job in jobs.iterrows():
            rule = calculate_rule_scores(resume, job)
            score_values = {
                "skill_score": rule.skill_score,
                "tfidf_score": float(tfidf_scores[resume_index, job_index]),
                "word2vec_score": float(word2vec_scores[resume_index, job_index]),
                "education_score": rule.education_score,
                "experience_score": rule.experience_score,
                "city_score": rule.city_score,
                "certificate_score": rule.certificate_score,
                "salary_score": rule.salary_score,
            }
            semantic_score = float(semantic_scores[resume_index, job_index])
            rows.append(
                {
                    "resume_id": resume["resume_id"],
                    "name": resume["name"],
                    "education": resume["education"],
                    "major": resume["major"],
                    "resume_city": resume["city"],
                    "job_id": job["job_id"],
                    "job_title": job["job_title"],
                    "company": job["company"],
                    "job_city": job["city"],
                    "total_score": total_score(score_values),
                    "tfidf_score": score_values["tfidf_score"],
                    "word2vec_score": score_values["word2vec_score"],
                    "semantic_score": semantic_score,
                    "skill_score": score_values["skill_score"],
                    "education_score": score_values["education_score"],
                    "experience_score": score_values["experience_score"],
                    "city_score": score_values["city_score"],
                    "certificate_score": score_values["certificate_score"],
                    "salary_score": score_values["salary_score"],
                    "matched_skills": _join(rule.matched_skills),
                    "missing_skills": _join(rule.missing_skills),
                    "matched_certificates": _join(rule.matched_certificates),
                    "reason": build_reason(resume, job, rule, semantic_score),
                }
            )
    return pd.DataFrame(rows).sort_values(["resume_id", "total_score"], ascending=[True, False])


def run_matching(
    resume_path: str | Path = DEFAULT_RESUME_PATH,
    job_path: str | Path = DEFAULT_JOB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 5,
) -> dict[str, Path | str | pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(DEFAULT_STOPWORDS_PATH)
    alias_map = load_skill_alias(DEFAULT_ALIAS_PATH)
    resumes_raw, jobs_raw = load_csv_data(resume_path, job_path)
    resumes = preprocess_resumes(resumes_raw, stopwords, alias_map).reset_index(drop=True)
    jobs = preprocess_jobs(jobs_raw, stopwords, alias_map).reset_index(drop=True)

    tfidf_matrix = compute_tfidf_scores(resumes["clean_text"].tolist(), jobs["clean_text"].tolist())
    word2vec_matrix, word2vec_method = compute_word2vec_scores(resumes["tokens"].tolist(), jobs["tokens"].tolist())
    semantic_matrix = compute_semantic_scores(tfidf_matrix, word2vec_matrix)
    result = build_match_results(resumes, jobs, tfidf_matrix, word2vec_matrix, semantic_matrix)

    all_result_path = output_dir / "explainable_match_result.csv"
    top_resume_path = output_dir / "top_matches.csv"
    top_job_path = output_dir / "job_candidate_matches.csv"

    top_by_resume = result.groupby("resume_id", group_keys=False).head(top_n)
    top_by_job = (
        result.sort_values(["job_id", "total_score"], ascending=[True, False])
        .groupby("job_id", group_keys=False)
        .head(top_n)
    )

    result.to_csv(all_result_path, index=False, encoding="utf-8-sig")
    top_by_resume.to_csv(top_resume_path, index=False, encoding="utf-8-sig")
    top_by_job.to_csv(top_job_path, index=False, encoding="utf-8-sig")

    return {
        "all_result_path": all_result_path,
        "top_resume_path": top_resume_path,
        "top_job_path": top_job_path,
        "word2vec_method": word2vec_method,
        "weights": str(TOTAL_SCORE_WEIGHTS),
        "result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简历-岗位人才匹配系统本地批处理入口")
    parser.add_argument("--resumes", default=str(DEFAULT_RESUME_PATH), help="简历 CSV 路径")
    parser.add_argument("--jobs", default=str(DEFAULT_JOB_PATH), help="岗位 CSV 路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--top-n", type=int, default=5, help="每个学生/岗位保留的 Top-N 数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = run_matching(args.resumes, args.jobs, args.output, args.top_n)
    print("匹配计算完成")
    print(f"Word2Vec 实现: {info['word2vec_method']}")
    print(f"全量结果: {info['all_result_path']}")
    print(f"学生视角 Top-N: {info['top_resume_path']}")
    print(f"岗位视角 Top-N: {info['top_job_path']}")


if __name__ == "__main__":
    main()
