#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
