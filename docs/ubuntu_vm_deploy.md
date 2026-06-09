# Ubuntu 虚拟机部署说明

以下步骤假设 Hadoop 3.3.6、Spark 3.5.1 和 JDK 8 已经按课程要求安装到本机目录。不同同学安装路径可能不同，按自己的路径调整环境变量即可。

## 1. 环境变量示例

编辑 `~/.bashrc`：

```bash
export JAVA_HOME=/usr/local/jdk1.8.0_202
export HADOOP_HOME=/usr/local/hadoop-3.3.6
export SPARK_HOME=/usr/local/spark-3.5.1-bin-hadoop3
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=$HOME/ai_env/bin/python
```

使配置生效：

```bash
source ~/.bashrc
```

## 2. Python 虚拟环境

```bash
cd ~/resume_matching_project
python3 -m venv ~/ai_env
source ~/ai_env/bin/activate
pip install -r requirements.txt
```

如果安装 `gensim` 或 `scikit-learn` 太慢，可以换国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 启动 Hadoop 与 YARN

首次格式化 NameNode 只做一次：

```bash
hdfs namenode -format
```

启动服务：

```bash
start-dfs.sh
start-yarn.sh
jps
```

`jps` 中建议能看到：

```text
NameNode
DataNode
SecondaryNameNode
ResourceManager
NodeManager
```

浏览器检查：

```text
HDFS: http://localhost:9870
YARN: http://localhost:8088
```

## 4. 运行项目

本地算法：

```bash
bash scripts/run_local.sh
```

HDFS 上传：

```bash
bash scripts/hdfs_upload.sh
hdfs dfs -ls /resume_matching/raw_data
```

PySpark 清洗：

```bash
bash scripts/run_spark.sh
hdfs dfs -ls /resume_matching/cleaned_data
```

Streamlit 页面：

```bash
bash scripts/run_streamlit.sh
```

## 5. 验收命令顺序

答辩现场建议按这个顺序演示：

```bash
source ~/ai_env/bin/activate
python BigDataController.py
bash scripts/run_local.sh
bash scripts/hdfs_upload.sh
bash scripts/run_spark.sh
bash scripts/run_streamlit.sh
```

然后打开 Streamlit 页面展示学生视角、岗位视角和结果数据下载。
