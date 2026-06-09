from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


HDFS_ROOT = "/resume_matching"


def run_hdfs_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    command = ["hdfs", "dfs", *args]
    print("$ " + " ".join(command))
    return subprocess.run(command, check=True, text=True, capture_output=True)


def mkdir(path: str) -> None:
    run_hdfs_command(["-mkdir", "-p", path])


def put(local_path: str | Path, hdfs_path: str) -> None:
    run_hdfs_command(["-put", "-f", str(local_path), hdfs_path])


def ls(path: str) -> str:
    result = run_hdfs_command(["-ls", path])
    return result.stdout


def prepare_hdfs(project_root: str | Path = ".") -> None:
    project_root = Path(project_root).resolve()
    mkdir(f"{HDFS_ROOT}/raw_data")
    mkdir(f"{HDFS_ROOT}/cleaned_data")
    mkdir(f"{HDFS_ROOT}/results")
    put(project_root / "data" / "resumes.csv", f"{HDFS_ROOT}/raw_data/")
    put(project_root / "data" / "jobs.csv", f"{HDFS_ROOT}/raw_data/")
    print(ls(f"{HDFS_ROOT}/raw_data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HDFS 数据目录初始化与上传脚本")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_hdfs(args.project_root)


if __name__ == "__main__":
    main()
