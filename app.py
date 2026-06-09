from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.matcher import DEFAULT_OUTPUT_DIR, run_matching


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_PATH = DEFAULT_OUTPUT_DIR / "explainable_match_result.csv"


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    if not RESULT_PATH.exists():
        run_matching()
    return pd.read_csv(RESULT_PATH)


def score_columns() -> list[str]:
    return [
        "skill_score",
        "tfidf_score",
        "word2vec_score",
        "education_score",
        "experience_score",
        "city_score",
        "certificate_score",
        "salary_score",
    ]


def show_match_rows(rows: pd.DataFrame, title_col: str, subtitle_cols: list[str]) -> None:
    for _, row in rows.iterrows():
        with st.container(border=True):
            st.subheader(f"{row[title_col]}  ·  {row['total_score']:.1f} 分")
            st.caption(" / ".join(str(row[col]) for col in subtitle_cols if col in row and pd.notna(row[col])))
            cols = st.columns(4)
            cols[0].metric("技能分", f"{row['skill_score']:.1f}")
            cols[1].metric("TF-IDF", f"{row['tfidf_score']:.1f}")
            cols[2].metric("Word2Vec", f"{row['word2vec_score']:.1f}")
            cols[3].metric("综合语义", f"{row['semantic_score']:.1f}")
            st.write(row["reason"])
            detail = row[score_columns()].astype(float)
            st.bar_chart(detail)
            if isinstance(row.get("missing_skills"), str) and row["missing_skills"]:
                st.warning(f"缺少技能：{row['missing_skills']}")


def main() -> None:
    st.set_page_config(page_title="简历-岗位人才匹配系统", layout="wide")
    st.title("基于大数据+AI的简历-岗位人才匹配系统")

    with st.sidebar:
        st.header("控制台")
        top_n = st.slider("Top-N", min_value=1, max_value=10, value=5)
        min_score = st.slider("最低综合分", min_value=0, max_value=100, value=0)
        if st.button("重新计算匹配结果", use_container_width=True):
            run_matching(top_n=top_n)
            st.cache_data.clear()
            st.success("已重新计算")

    data = load_results()
    filtered = data[data["total_score"] >= min_score].copy()
    if filtered.empty:
        st.info("当前筛选条件下没有匹配结果，请降低最低综合分。")
        return

    tab_student, tab_job, tab_data = st.tabs(["学生视角", "岗位视角", "结果数据"])

    with tab_student:
        names = filtered[["resume_id", "name"]].drop_duplicates()
        selected_name = st.selectbox("选择学生", names["name"].tolist())
        resume_id = names.loc[names["name"] == selected_name, "resume_id"].iloc[0]
        rows = (
            filtered[filtered["resume_id"] == resume_id]
            .sort_values("total_score", ascending=False)
            .head(top_n)
        )
        show_match_rows(rows, "job_title", ["company", "job_city"])

    with tab_job:
        jobs = filtered[["job_id", "job_title", "company"]].drop_duplicates()
        selected_job = st.selectbox("选择岗位", jobs["job_title"].tolist())
        job_id = jobs.loc[jobs["job_title"] == selected_job, "job_id"].iloc[0]
        rows = (
            filtered[filtered["job_id"] == job_id]
            .sort_values("total_score", ascending=False)
            .head(top_n)
        )
        show_match_rows(rows, "name", ["education", "major", "resume_city"])

    with tab_data:
        st.dataframe(filtered.sort_values("total_score", ascending=False), use_container_width=True)
        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载匹配结果 CSV",
            data=csv,
            file_name="explainable_match_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
