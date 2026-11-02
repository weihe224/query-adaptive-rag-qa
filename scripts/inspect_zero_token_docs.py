
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data" / "benchmark" / "t2retrieval"
    / "processed"
)

EVAL_DIR = BASE / "evaluation"

def main():

    corpus = pd.read_parquet(
        EVAL_DIR / "corpus_searchable.parquet"
    )

    qrels = pd.read_parquet(
        EVAL_DIR / "qrels_searchable.parquet"
    )

    queries = pd.read_parquet(
        BASE / "queries_clean.parquet"
    )

    zero_file = (
        ROOT / "data" / "indexes" / "bm25_jieba"
        / "zero_token_documents.csv"
    )

    zero_docs = pd.read_csv(
        zero_file,
        dtype={"doc_id": str}
    )

    for df, col in [
        (corpus, "doc_id"),
        (qrels, "doc_id"),
        (qrels, "query_id"),
        (queries, "query_id")
    ]:
        df[col] = df[col].astype(str)

    query_map = dict(zip(
        queries["query_id"],
        queries["query"]
    ))

    print("\n========== 零有效词项文档检查 ==========")

    details = []

    for doc_id in zero_docs["doc_id"]:

        doc = corpus[
            corpus["doc_id"] == doc_id
        ]

        related = qrels[
            qrels["doc_id"] == doc_id
        ]

        print(f"\n文档 ID：{doc_id}")

        if len(doc):
            text = str(doc.iloc[0]["text"])
            print("正文长度：", len(text))
            print("正文内容：", repr(text[:500]))
        else:
            print("未在可检索文档库中找到")

        print("相关性标注数量：", len(related))

        query_ids = related["query_id"].tolist()

        for qid in query_ids:
            print(
                f"相关查询：{qid} - "
                f"{query_map.get(qid, '查询不在验证集')}"
            )

        details.append({
            "doc_id": doc_id,
            "text": str(doc.iloc[0]["text"]) if len(doc) else "",
            "qrels_count": len(related),
            "query_ids": "|".join(query_ids)
        })

    pd.DataFrame(details).to_csv(
        ROOT / "results" / "zero_token_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n检查完成！")
    print("结果保存至 results/zero_token_audit.csv")


if __name__ == "__main__":
    main()
