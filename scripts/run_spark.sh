#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
spark-submit src/spark_job.py \
  --input-base hdfs:///resume_matching/raw_data \
  --output-base hdfs:///resume_matching/cleaned_data
