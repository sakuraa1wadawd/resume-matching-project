from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace


def clean_text_column(column_name: str):
    return regexp_replace(lower(col(column_name)), r"[^0-9a-zA-Z\u4e00-\u9fff+#.]+", " ")


def run_spark_clean_job(input_base: str, output_base: str) -> None:
    spark = (
        SparkSession.builder.appName("ResumeJobMatchingCleanJob")
        .config("spark.sql.session.timeZone", "Asia/Shanghai")
        .getOrCreate()
    )

    resumes = spark.read.option("header", True).option("inferSchema", True).csv(f"{input_base}/resumes.csv")
    jobs = spark.read.option("header", True).option("inferSchema", True).csv(f"{input_base}/jobs.csv")

    resumes_cleaned = (
        resumes.dropDuplicates(["resume_id"])
        .fillna("")
        .withColumn(
            "resume_full_text",
            concat_ws(" ", "major", "skills", "certificates", "project_experience", "self_description"),
        )
        .withColumn("resume_clean_text", clean_text_column("resume_full_text"))
    )

    jobs_cleaned = (
        jobs.dropDuplicates(["job_id"])
        .fillna("")
        .withColumn(
            "job_full_text",
            concat_ws(" ", "job_title", "required_skills", "preferred_certificates", "job_description"),
        )
        .withColumn("job_clean_text", clean_text_column("job_full_text"))
    )

    resumes_cleaned.write.mode("overwrite").option("header", True).csv(f"{output_base}/cleaned_resumes")
    jobs_cleaned.write.mode("overwrite").option("header", True).csv(f"{output_base}/cleaned_jobs")

    print("PySpark 清洗完成")
    print(f"cleaned resumes: {output_base}/cleaned_resumes")
    print(f"cleaned jobs: {output_base}/cleaned_jobs")
    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 HDFS 读取 CSV，使用 PySpark 清洗后写回 HDFS")
    parser.add_argument("--input-base", default="hdfs:///resume_matching/raw_data")
    parser.add_argument("--output-base", default="hdfs:///resume_matching/cleaned_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_spark_clean_job(args.input_base, args.output_base)


if __name__ == "__main__":
    main()
