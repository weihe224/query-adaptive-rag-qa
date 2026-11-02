from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROCESSED_DIR / "duplicate_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def text_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def main():

    print("读取原始和清洗后的数据...")

    raw = pd.read_parquet(DATA_DIR / "corpus.parquet")
    clean = pd.read_parquet(
        PROCESSED_DIR / "corpus_clean.parquet"
    )
    qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")

    raw["doc_id"] = raw["doc_id"].astype(str)
    clean["doc_id"] = clean["doc_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)

    raw_map = dict(zip(raw["doc_id"], raw["text"]))

    # 建立原始文本和清洗文本的哈希
    clean["clean_hash"] = clean["text"].fillna("").map(
        text_hash
    )

    raw["raw_hash"] = raw["text"].fillna("").map(
        text_hash
    )

    # 只检查非空正文
    valid = clean[clean["text"].str.strip().ne("")].copy()

    # 找到清洗后重复文档组
    duplicate_groups = valid[
        valid.duplicated("clean_hash", keep=False)
    ].groupby("clean_hash")

    qrels["doc_id"] = qrels["doc_id"].astype(str)

    doc_to_group = {}
    group_rows = []

    for group_id, group in duplicate_groups:

        doc_ids = group["doc_id"].tolist()

        for doc_id in doc_ids:
            doc_to_group[doc_id] = group_id

        relevant = qrels[
            qrels["doc_id"].isin(doc_ids)
        ]

        # 检查清洗前是否已经重复
        raw_hashes = [
            text_hash(raw_map[doc_id])
            for doc_id in doc_ids
        ]

        # 检查同一查询是否标注了重复文档中的多篇
        multi_relevant = (
            relevant.groupby("query_id")["doc_id"]
            .nunique()
        )

        group_rows.append({
            "group_id": group_id,
            "doc_count": len(doc_ids),
            "doc_ids": " | ".join(doc_ids),
            "raw_unique_text_count": len(set(raw_hashes)),
            "qrels_count": len(relevant),
            "relevant_query_count": relevant["query_id"].nunique(),
            "queries_with_multiple_copies": int(
                (multi_relevant > 1).sum()
            ),
            "example_text": group.iloc[0]["text"][:300]
        })

    group_df = pd.DataFrame(group_rows)

    if len(group_df):
        group_df = group_df.sort_values(
            "doc_count", ascending=False
        )

    # 所有重复文档 ID
    duplicate_ids = set(doc_to_group.keys())

    # 查看这些文档是否出现在相关性标注中
    duplicate_qrels = qrels[
        qrels["doc_id"].isin(duplicate_ids)
    ].copy()

    if len(duplicate_qrels):
        duplicate_qrels["group_id"] = (
            duplicate_qrels["doc_id"].map(doc_to_group)
        )

    # 检查空文档
    empty_docs = clean[
        clean["text"].fillna("").str.strip().eq("")
    ].copy()

    empty_qrels = qrels[
        qrels["doc_id"].isin(empty_docs["doc_id"])
    ]

    # 检查查询文本是否重复
    queries = pd.read_parquet(
        DATA_DIR / "queries.parquet"
    )

    # 兼容不同版本的数据字段
    query_col = (
        "query" if "query" in queries.columns
        else "text"
    )

    query_duplicates = queries[
        queries.duplicated(query_col, keep=False)
    ].sort_values(query_col)

    # 输出结果
    group_df.to_csv(
        OUTPUT_DIR / "duplicate_groups.csv",
        index=False,
        encoding="utf-8-sig"
    )

    duplicate_qrels.to_csv(
        OUTPUT_DIR / "duplicate_qrels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    empty_docs.to_csv(
        OUTPUT_DIR / "empty_documents.csv",
        index=False,
        encoding="utf-8-sig"
    )

    empty_qrels.to_csv(
        OUTPUT_DIR / "empty_document_qrels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    query_duplicates.to_csv(
        OUTPUT_DIR / "duplicate_queries.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 重复文档审计 ==========")

    print("重复文档组数量：", len(group_df))
    print("重复文档涉及的 ID 数量：", len(duplicate_ids))
    print("重复文档额外记录数量：", (
        len(duplicate_ids) - len(group_df)
    ))

    print(
        "清洗后重复组中，原始正文不同的组数量：",
        int(
            (group_df["raw_unique_text_count"] > 1).sum()
        ) if len(group_df) else 0
    )

    print(
        "重复文档涉及的相关性标注数量：",
        len(duplicate_qrels)
    )

    print(
        "同一查询标注多个重复文档的组数：",
        int(
            group_df[
                "queries_with_multiple_copies"
            ].gt(0).sum()
        ) if len(group_df) else 0
    )

    print("\n========== 空文档检查 ==========")

    print("空文档数量：", len(empty_docs))
    print("空文档对应 Qrels 数量：", len(empty_qrels))

    if len(empty_docs):
        print("空文档 ID：", empty_docs["doc_id"].tolist())

    print("\n========== 查询重复检查 ==========")

    print(
        "重复查询记录数量：",
        len(query_duplicates)
    )

    print("\n审计完成！")
    print("结果目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
