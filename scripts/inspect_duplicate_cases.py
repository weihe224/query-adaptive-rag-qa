
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROCESSED_DIR / "duplicate_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def main():

    print("加载数据...")

    raw = pd.read_parquet(
        DATA_DIR / "corpus.parquet"
    )

    clean = pd.read_parquet(
        PROCESSED_DIR / "corpus_clean.parquet"
    )

    queries = pd.read_parquet(
        DATA_DIR / "queries.parquet"
    )

    qrels = pd.read_parquet(
        DATA_DIR / "qrels.parquet"
    )

    for df, col in [
        (raw, "doc_id"),
        (clean, "doc_id"),
        (qrels, "doc_id"),
        (queries, "query_id"),
        (qrels, "query_id")
    ]:
        df[col] = df[col].astype(str)

    raw_map = dict(zip(raw["doc_id"], raw["text"]))

    query_map = dict(zip(
        queries["query_id"], queries["query"]
    ))

    clean["clean_hash"] = (
        clean["text"].fillna("").map(get_hash)
    )

    # 找到重复组
    valid = clean[
        clean["text"].fillna("").str.strip().ne("")
    ]

    duplicate_groups = valid[
        valid.duplicated("clean_hash", keep=False)
    ].groupby("clean_hash")

    summary_rows = []
    detail_rows = []

    for group_hash, group in duplicate_groups:

        doc_ids = group["doc_id"].tolist()

        relevant = qrels[
            qrels["doc_id"].isin(doc_ids)
        ]

        multi_query_counts = (
            relevant.groupby("query_id")["doc_id"]
            .nunique()
        )

        multi_query_ids = multi_query_counts[
            multi_query_counts > 1
        ].index.tolist()

        raw_texts = [
            str(raw_map[doc_id])
            for doc_id in doc_ids
        ]

        group_id = group_hash[:12]

        summary_rows.append({
            "group_id": group_id,
            "doc_count": len(doc_ids),
            "doc_ids": " | ".join(doc_ids),
            "raw_unique_count": len(set(raw_texts)),
            "qrels_count": len(relevant),
            "query_count": relevant["query_id"].nunique(),
            "multi_relevant_query_count": len(
                multi_query_ids
            ),
            "example_clean_text": group.iloc[0]["text"][:500]
        })

        for _, doc in group.iterrows():

            doc_id = str(doc["doc_id"])

            raw_text = str(raw_map[doc_id])

            doc_qrels = relevant[
                relevant["doc_id"] == doc_id
            ]

            query_ids = doc_qrels["query_id"].tolist()

            query_texts = [
                query_map.get(qid, "")
                for qid in query_ids
            ]

            detail_rows.append({
                "group_id": group_id,
                "doc_id": doc_id,
                "raw_length": len(raw_text),
                "clean_length": len(str(doc["text"])),
                "raw_text": raw_text,
                "clean_text": doc["text"],
                "query_ids": " | ".join(query_ids),
                "query_texts": " | ".join(query_texts)
            })

    summary = pd.DataFrame(summary_rows)

    details = pd.DataFrame(detail_rows)

    summary.to_csv(
        OUTPUT_DIR / "duplicate_group_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    details.to_csv(
        OUTPUT_DIR / "duplicate_doc_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 检查空文档
    empty_docs = clean[
        clean["text"].fillna("").str.strip().eq("")
    ]

    empty_rows = []

    for _, doc in empty_docs.iterrows():

        doc_id = str(doc["doc_id"])

        original = raw_map.get(doc_id, "")

        related = qrels[
            qrels["doc_id"] == doc_id
        ]

        if len(related) == 0:
            empty_rows.append({
                "doc_id": doc_id,
                "raw_text": original,
                "query_id": "",
                "query": ""
            })

        for _, rel in related.iterrows():

            qid = str(rel["query_id"])

            empty_rows.append({
                "doc_id": doc_id,
                "raw_text": original,
                "query_id": qid,
                "query": query_map.get(qid, "")
            })

    pd.DataFrame(empty_rows).to_csv(
        OUTPUT_DIR / "empty_document_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 检查完成 ==========")
    print("重复组数量：", len(summary))
    print("重复文档数量：", len(details))

    print("\n前10个重复组：")

    if len(summary):
        print(
            summary.head(10)[
                [
                    "group_id",
                    "doc_count",
                    "raw_unique_count",
                    "qrels_count",
                    "multi_relevant_query_count"
                ]
            ].to_string(index=False)
        )

    print("\n空文档详情：")

    for item in empty_rows:
        print(item)

    print("\n结果目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
