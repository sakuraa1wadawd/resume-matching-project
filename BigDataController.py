from __future__ import annotations

import shutil
import subprocess


def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return result.returncode, (result.stdout + result.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def check_command(name: str, command: list[str]) -> None:
    print(f"\n[{name}]")
    if shutil.which(command[0]) is None:
        print(f"未找到命令: {command[0]}")
        return
    code, output = run(command)
    print(output if output else f"退出码: {code}")


def main() -> None:
    print("BigDataController 环境检查")
    check_command("Java", ["java", "-version"])
    check_command("Hadoop", ["hadoop", "version"])
    check_command("Spark", ["spark-submit", "--version"])
    check_command("HDFS 根目录", ["hdfs", "dfs", "-ls", "/"])
    check_command("JPS 进程", ["jps"])
    print("\n如果 NameNode、DataNode、ResourceManager、NodeManager 都存在，说明 Hadoop/YARN 基本可用。")


if __name__ == "__main__":
    main()
