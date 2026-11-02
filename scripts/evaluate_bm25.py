
from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_retrieval.bm25 import BM25Retriever


EVAL_DIR = (
    ROOT / "data" / "benchmark" / "t2retrieval"
    / "processed" / "evaluation"
)

SPLIT_DIR = EVAL_DIR / "splits"

INDEX_DIR = ROOT / "data" / "indexes" / "bm25_jieba"

RESULT_DIR = ROOT / "results" / "bm25"

TOP_K = 10


def calculate_metrics(retrieved_ids, relevant_ids, k=10):

    relevant_ids = set(relevant_ids)
    retrieved_ids = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0, 0.0, 0.0

    hits = [
        doc_id in relevant_ids
        for doc_id in retrieved_ids
    ]

    # Recall@K
    recall = sum(hits) / len(relevant_ids)

    # MRR@K
    mrr = 0.0

    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    # NDCG@K，当前数据使用二元相关性
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
        if hit
    )

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, mrr, ndcg


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制查询数量，仅用于流程测试"
    )

    args = parser.parse_args()

    print("加载 BM25 检索器...")

    retriever = BM25Retriever(
        index_dir=INDEX_DIR,
        k1=1.5,
        b=0.75
    )

    print("加载查询和 Qrels...")

    queries = pd.read_parquet(
        SPLIT_DIR / f"{args.split}_queries.parquet"
    )

    qrels = pd.read_parquet(
        SPLIT_DIR / f"{args.split}_qrels.parquet"
    )

    queries["query_id"] = queries["query_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)

    # 每条查询对应的相关文档集合
    relevant_map = {}

    for query_id, group in qrels.groupby("query_id"):

        relevant_map[str(query_id)] = set(
            group.loc[
                group["relevance"] > 0, "doc_id"
            ].astype(str)
        )

    if args.limit is not None:
        queries = queries.head(args.limit)

    print("\n========== 开始检索 ==========")
    print("数据集：", args.split)
    print("查询数量：", len(queries))
    print("Top-K：", TOP_K)

    metric_rows = []
    result_rows = []

    for index, row in enumerate(
        queries.itertuples(index=False),
        start=1
    ):

        query_id = str(row.query_id)
        query_text = str(row.query)

        results = retriever.search(
            query_text,
            top_k=TOP_K
        )

        retrieved_ids = [
            item["doc_id"] for item in results
        ]

        retrieved_scores = [
            item["score"] for item in results
        ]

        relevant_ids = relevant_map.get(
            query_id, set()
        )

        recall, mrr, ndcg = calculate_metrics(
            retrieved_ids,
            relevant_ids,
            TOP_K
        )

        metric_rows.append({
            "query_id": query_id,
            "query": query_text,
            "relevant_count": len(relevant_ids),
            "hits_at_10": sum(
                doc_id in relevant_ids
                for doc_id in retrieved_ids
            ),
            "recall_at_10": recall,
            "mrr_at_10": mrr,
            "ndcg_at_10": ndcg
        })

        for rank, item in enumerate(results, start=1):

            result_rows.append({
                "query_id": query_id,
                "query": query_text,
                "rank": rank,
                "doc_id": item["doc_id"],
                "score": item["score"],
                "is_relevant": (
                    item["doc_id"] in relevant_ids
                )
            })

        if index % 500 == 0:
            print(
                f"已完成：{index}/{len(queries)}"
            )

    metrics_df = pd.DataFrame(metric_rows)
    results_df = pd.DataFrame(
        result_rows,
        columns=[
            "query_id", "query", "rank",
            "doc_id", "score", "is_relevant"
        ]
    )

    output_dir = RESULT_DIR / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存逐查询指标
    metrics_df.to_csv(
        output_dir / "per_query_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 保存 Top-10 检索结果
    results_df.to_parquet(
        output_dir / "top10_results.parquet",
        index=False
    )

    summary = {
        "model": "BM25",
        "tokenizer": "Jieba",
        "split": args.split,
        "query_count": len(metrics_df),
        "document_count": retriever.n_docs,
        "top_k": TOP_K,
        "k1": retriever.k1,
        "b": retriever.b,
        "recall_at_10": float(
            metrics_df["recall_at_10"].mean()
        ),
        "mrr_at_10": float(
            metrics_df["mrr_at_10"].mean()
        ),
        "ndcg_at_10": float(
            metrics_df["ndcg_at_10"].mean()
        )
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== BM25 评价结果 ==========")

    print("Recall@10：", round(
        summary["recall_at_10"], 4
    ))

    print("MRR@10：", round(
        summary["mrr_at_10"], 4
    ))

    print("NDCG@10：", round(
        summary["ndcg_at_10"], 4
    ))

    print("\n结果保存位置：", output_dir)


if __name__ == "__main__":
    main()
