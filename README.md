# 基于大数据+AI的简历-岗位人才匹配系统

本项目实现了任务书要求的最小闭环和若干扩展功能：读取简历与岗位 CSV，完成中文分词、停用词过滤、技能词标准化，使用 TF-IDF、Word2Vec 和余弦相似度计算文本相似度，再结合技能、学历、经验、城市、证书、薪资等规则分生成综合匹配分与推荐理由，最后通过 Streamlit 展示学生视角和岗位视角的 Top-N 推荐结果。

## 目录结构

```text
resume_matching_project/
├── data/
│   ├── resumes.csv
│   ├── jobs.csv
│   ├── stopwords.txt
│   └── skill_alias.json
├── output/
├── src/
│   ├── preprocess.py
│   ├── similarity.py
│   ├── scoring.py
│   ├── matcher.py
│   ├── hdfs_utils.py
│   └── spark_job.py
├── scripts/
│   ├── run_local.sh
│   ├── run_streamlit.sh
│   ├── hdfs_upload.sh
│   └── run_spark.sh
├── docs/
├── app.py
├── BigDataController.py
├── requirements.txt
└── README.md
```

## Ubuntu 虚拟机运行步骤

推荐在 Ubuntu 24 + Hadoop 3.3.6 + Spark 3.5.1 环境中运行。

```bash
cd ~/resume_matching_project
python3 -m venv ~/ai_env
source ~/ai_env/bin/activate
pip install -r requirements.txt
```

先检查 Hadoop、Spark、HDFS 是否可用：

```bash
python BigDataController.py
jps
```

本地版匹配计算：

```bash
bash scripts/run_local.sh
```

运行后会生成：

```text
output/explainable_match_result.csv
output/top_matches.csv
output/job_candidate_matches.csv
```

启动 Streamlit 页面：

```bash
bash scripts/run_streamlit.sh
```

浏览器访问：

```text
http://localhost:8501
```

## HDFS 与 PySpark

启动 Hadoop/YARN 后执行：

```bash
bash scripts/hdfs_upload.sh
hdfs dfs -ls /resume_matching/raw_data
```

执行 PySpark 清洗任务：

```bash
bash scripts/run_spark.sh
hdfs dfs -ls /resume_matching/cleaned_data
```

该 Spark 作业会从 `hdfs:///resume_matching/raw_data` 读取 `resumes.csv` 和 `jobs.csv`，完成去重、空值填充、文本拼接、简单清洗，并写回 `hdfs:///resume_matching/cleaned_data`。

## 综合评分公式

综合分权重如下，所有单项分均映射到 0-100：

```text
综合匹配分 =
技能分 × 0.28
+ TF-IDF 分 × 0.20
+ Word2Vec 分 × 0.15
+ 学历分 × 0.12
+ 经验分 × 0.10
+ 城市分 × 0.05
+ 证书分 × 0.05
+ 薪资分 × 0.05
```

权重设计思路：岗位匹配首先看技能覆盖，所以技能分最高；文本相似度用于衡量简历描述和岗位描述的整体接近程度；学历、经验是招聘硬性条件；城市、证书、薪资作为补充因素。

## 推荐理由生成

每条匹配结果都会输出可解释原因，包括：

- 共同技能；
- 缺少技能；
- 学历是否满足；
- 经验是否满足；
- 城市是否匹配；
- 证书是否匹配；
- 文本语义相似度高、中、低；
- 提升建议。

## 答辩截图清单

建议按顺序保存以下截图：

- `jps` 进程截图；
- HDFS 9870 页面截图；
- YARN 8088 页面截图；
- `hdfs dfs -ls /resume_matching/raw_data` 截图；
- `spark-submit src/spark_job.py` 运行截图；
- `hdfs dfs -ls /resume_matching/cleaned_data` 截图；
- `python -m src.matcher` 输出截图；
- Streamlit 学生视角、岗位视角、结果数据页截图。

## 常见问题

如果 `gensim` 安装失败，可以先运行本地匹配流程，代码会自动使用离线备用词向量保证演示可跑；正式答辩建议安装 `gensim`，并说明本项目 Word2Vec 使用平均词向量计算简历与岗位语义相似度。

如果 Streamlit 打不开，先确认虚拟机端口：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
