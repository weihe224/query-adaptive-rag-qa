from pathlib import Path
import json
import pandas as pd

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "benchmark" / "t2retrieval"
OUTPUT.mkdir(parents=True, exist_ok=True)

REPO = "mteb/T2Retrieval"


def download_data(revision):
    print("正在加载文档库...")
    corpus = load_dataset(
        REPO, name="corpus",
        split="dev", revision=revision
    ).to_pandas()

    print("正在加载查询...")
    queries = load_dataset(
        REPO, name="queries",
        split="dev", revision=revision
    ).to_pandas()

    print("正在加载相关性标注...")
    qrels = load_dataset(
        REPO, name="default",
        split="dev", revision=revision
    ).to_pandas()

    return corpus, queries, qrels


def prepare_data(corpus, queries, qrels):
    corpus = corpus.rename(
        columns={"_id": "doc_id"}
    )

    queries = queries.rename(
        columns={
            "_id": "query_id",
            "text": "query"
        }
    )

    qrels = qrels.rename(
        columns={
            "query-id": "query_id",
            "corpus-id": "doc_id",
            "score": "relevance"
        }
    )

    # 统一 ID 类型，避免数字和字符串匹配失败
    corpus["doc_id"] = corpus["doc_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)

    return corpus, queries, qrels


def check_data(corpus, queries, qrels):
    print("\n========== 数据统计 ==========")
    print("文档数量：", len(corpus))
    print("查询数量：", len(queries))
    print("相关性标注数量：", len(qrels))

    print("\n========== 字段 ==========")
    print("corpus:", corpus.columns.tolist())
    print("queries:", queries.columns.tolist())
    print("qrels:", qrels.columns.tolist())

    print("\n========== 数据质量 ==========")
    print("重复文档 ID：", corpus["doc_id"].duplicated().sum())
    print("重复查询 ID：", queries["query_id"].duplicated().sum())
    print("空查询：", (
        queries["query"].isna()
        | queries["query"].astype(str).str.strip().eq("")
    ).sum())

    missing_queries = ~qrels["query_id"].isin(
        queries["query_id"]
    )
    missing_docs = ~qrels["doc_id"].isin(
        corpus["doc_id"]
    )

    print("找不到查询的 qrels：", missing_queries.sum())
    print("找不到文档的 qrels：", missing_docs.sum())

    print("\n========== 相关性等级 ==========")
    print(qrels["relevance"].value_counts().sort_index())

    relevant = qrels[qrels["relevance"] > 0]
    counts = relevant.groupby("query_id")["doc_id"].nunique()

    print("\n每条有标注查询的相关文档数量：")
    print(counts.describe())

    print("\n========== 样例 ==========")
    print("第一条查询：")
    print(queries.iloc[0].to_dict())

    print("\n第一篇文档（正文只显示前 200 字）：")
    doc = corpus.iloc[0]
    print({
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "text": str(doc["text"])[:200]
    })

    print("\n前三条相关性标注：")
    print(qrels.head(3).to_string(index=False))

    assert not missing_queries.any(), "qrels 存在未知查询 ID"
    assert not missing_docs.any(), "qrels 存在未知文档 ID"
    assert not corpus["doc_id"].duplicated().any()
    assert not queries["query_id"].duplicated().any()


def main():
    revision = HfApi().dataset_info(REPO).sha

    print("数据集：", REPO)
    print("数据版本：", revision)

    corpus, queries, qrels = download_data(revision)
    corpus, queries, qrels = prepare_data(
        corpus, queries, qrels
    )

    check_data(corpus, queries, qrels)

    corpus.to_parquet(
        OUTPUT / "corpus.parquet", index=False
    )
    queries.to_parquet(
        OUTPUT / "queries.parquet", index=False
    )
    qrels.to_parquet(
        OUTPUT / "qrels.parquet", index=False
    )

    metadata = {
        "repository": REPO,
        "revision": revision,
        "source_split": "dev",
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "qrels_count": len(qrels)
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n数据准备完成！")
    print("保存位置：", OUTPUT)


if __name__ == "__main__":
    main()
