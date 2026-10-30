# 基于查询特征分类与自适应检索策略的 RAG 问答系统

## 项目目标
使用传统模式识别算法对查询进行分类，并根据分类结果选择 BM25、TF-IDF 向量检索或混合检索。项目优先完成可复现的分类与检索实验；大模型生成模块为可选项。

## 环境
推荐 Python 3.10 或 3.11。Windows PowerShell 命令见课程设计计划。

## 快速开始
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_experiments.py
```

输出：
- `results/classification_metrics.csv`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/retrieval_metrics.csv`
- `results/retrieval_comparison.png`

## 数据说明
`data/queries.csv` 是用于演示流程的课程项目小型标注数据；`data/documents.csv` 是配套知识库。该数据是为课程原型手工整理的 60 条查询与 20 篇短文档，仅适合验证代码流程，不是标准公开基准，也不足以支撑强泛化结论。正式提交前应扩充问题数量，并人工检查类别与相关文档标注。

查询类别：
- `fact`：精确事实或术语查询 -> BM25
- `explain`：概念解释或原因问题 -> TF-IDF 检索
- `compare`：比较、综合问题 -> BM25 与 TF-IDF 融合

## 可选：调用大模型生成答案
复制 `.env.example` 为 `.env`，填入兼容 OpenAI Chat Completions API 的地址、密钥和模型名，然后运行：
```powershell
python answer_demo.py
```
请勿将 `.env` 或 API 密钥提交到 Git。
