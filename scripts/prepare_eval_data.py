from pathlib import Path
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROCESSED_DIR / "evaluation"
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
        PROCESSED_DIR / "queries_clean.parquet"
    )

    for df, col in [
        (corpus, "doc_id"),
        (qrels, "doc_id"),
        (qrels, "query_id"),
        (queries, "query_id")
    ]:
        df[col] = df[col].astype(str)

    # 找到空文档
    empty_mask = (
        corpus["text"].fillna("").str.strip().eq("")
    )

    empty_ids = set(corpus.loc[empty_mask, "doc_id"])

    # 可检索文档
    searchable_corpus = corpus[~empty_mask].copy()

    # 排除无法检索的相关性标注
    excluded_qrels = qrels[
        qrels["doc_id"].isin(empty_ids)
    ].copy()

    searchable_qrels = qrels[
        ~qrels["doc_id"].isin(empty_ids)
    ].copy()

    # 构建内容 ID
    searchable_corpus["content_id"] = (
        searchable_corpus["text"].map(content_hash)
    )

    doc_to_content = dict(zip(
        searchable_corpus["doc_id"],
        searchable_corpus["content_id"]
    ))

    # 内容级 Qrels
    content_qrels = searchable_qrels.copy()

    content_qrels["content_id"] = (
        content_qrels["doc_id"].map(doc_to_content)
    )

    content_qrels = content_qrels.drop_duplicates(
        ["query_id", "content_id"]
    )

    # 检查所有查询是否仍有相关文档
    covered_queries = set(searchable_qrels["query_id"])

    uncovered_queries = (
        set(queries["query_id"]) - covered_queries
    )

    # 保存数据
    searchable_corpus.to_parquet(
        OUTPUT_DIR / "corpus_searchable.parquet",
        index=False
    )

    searchable_qrels.to_parquet(
        OUTPUT_DIR / "qrels_searchable.parquet",
        index=False
    )

    content_qrels.to_parquet(
        OUTPUT_DIR / "qrels_content_level.parquet",
        index=False
    )

    excluded_qrels.to_parquet(
        OUTPUT_DIR / "qrels_excluded.parquet",
        index=False
    )

    print("\n========== 评测数据准备完成 ==========")

    print("可检索文档数量：", len(searchable_corpus))
    print("原始 Qrels：", len(qrels))
    print("主实验 Qrels：", len(searchable_qrels))
    print("内容级 Qrels：", len(content_qrels))
    print("排除的 Qrels：", len(excluded_qrels))
    print("没有相关文档的查询：", len(uncovered_queries))

    print("\n保存位置：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
