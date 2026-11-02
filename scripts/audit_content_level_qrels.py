
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROCESSED_DIR / "duplicate_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def content_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def main():

    corpus = pd.read_parquet(
        PROCESSED_DIR / "corpus_clean.parquet"
    )

    qrels = pd.read_parquet(
        DATA_DIR / "qrels.parquet"
    )

    queries = pd.read_parquet(
        DATA_DIR / "queries.parquet"
    )

    for df, col in [
        (corpus, "doc_id"),
        (qrels, "doc_id"),
        (qrels, "query_id"),
        (queries, "query_id")
    ]:
        df[col] = df[col].astype(str)

    # 建立文档 ID 到内容 ID 的映射
    corpus["content_id"] = (
        corpus["text"].fillna("").map(content_hash)
    )

    nonempty = corpus[
        corpus["text"].fillna("").str.strip().ne("")
    ].copy()

    doc_to_content = dict(zip(
        nonempty["doc_id"],
        nonempty["content_id"]
    ))

    # 原始 Qrels 中涉及空文档的记录
    empty_doc_ids = set(
        corpus[
            corpus["text"].fillna("").str.strip().eq("")
        ]["doc_id"]
    )

    empty_qrels = qrels[
        qrels["doc_id"].isin(empty_doc_ids)
    ].copy()

    # 只分析可以进行文本检索的相关文档
    searchable_qrels = qrels[
        qrels["doc_id"].isin(doc_to_content)
    ].copy()

    searchable_qrels["content_id"] = (
        searchable_qrels["doc_id"].map(doc_to_content)
    )

    # Document-level 相关性对
    document_pairs = searchable_qrels.drop_duplicates(
        ["query_id", "doc_id"]
    )

    # Content-level 相关性对
    content_pairs = searchable_qrels.drop_duplicates(
        ["query_id", "content_id"]
    )

    # 每条查询的相关文档数量对比
    doc_counts = (
        document_pairs.groupby("query_id")["doc_id"]
        .nunique()
        .rename("document_relevant_count")
    )

    content_counts = (
        content_pairs.groupby("query_id")["content_id"]
        .nunique()
        .rename("content_relevant_count")
    )

    comparison = pd.concat(
        [doc_counts, content_counts],
        axis=1
    ).fillna(0)

    comparison["collapsed_count"] = (
        comparison["document_relevant_count"]
        - comparison["content_relevant_count"]
    )

    affected = comparison[
        comparison["collapsed_count"] > 0
    ]

    # 检查空文档对应查询
    query_map = dict(zip(
        queries["query_id"],
        queries["query"]
    ))

    empty_details = []

    for _, row in empty_qrels.iterrows():

        qid = row["query_id"]

        total = qrels[
            qrels["query_id"] == qid
        ]

        searchable = searchable_qrels[
            searchable_qrels["query_id"] == qid
        ]

        empty_details.append({
            "query_id": qid,
            "query": query_map.get(qid, ""),
            "total_relevant_docs": total["doc_id"].nunique(),
            "searchable_relevant_docs": searchable["doc_id"].nunique(),
            "unsearchable_relevant_docs": (
                total["doc_id"].nunique()
                - searchable["doc_id"].nunique()
            )
        })

    summary = {
        "original_qrels_count": len(qrels),
        "searchable_qrels_count": len(document_pairs),
        "empty_document_qrels_count": len(empty_qrels),
        "content_level_qrels_count": len(content_pairs),
        "collapsed_qrels_count": (
            len(document_pairs) - len(content_pairs)
        ),
        "queries_affected_by_duplicates": len(affected),
        "total_queries_with_searchable_qrels": len(comparison),
        "empty_document_query_details": empty_details
    }

    comparison.to_csv(
        OUTPUT_DIR / "per_query_content_comparison.csv",
        encoding="utf-8-sig"
    )

    content_pairs.to_csv(
        OUTPUT_DIR / "content_level_qrels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(empty_details).to_csv(
        OUTPUT_DIR / "empty_qrels_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        OUTPUT_DIR / "content_audit_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== 内容级评价审计 ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n审计完成！")
    print("结果目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
