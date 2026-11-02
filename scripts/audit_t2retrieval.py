
from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
REPORT_DIR = DATA_DIR / "audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def describe(name, series):
    print(f"\n========== {name} ==========")
    print(series.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).round(2))


def main():
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")

    # 1. 查询长度
    queries["query_chars"] = (
        queries["query"].fillna("").astype(str).str.len()
    )

    describe("查询字符长度", queries["query_chars"])

    # 2. 文档长度
    text = corpus["text"].fillna("").astype(str)

    corpus["text_chars"] = text.str.len()

    describe("文档字符长度", corpus["text_chars"])

    # 3. 标题情况
    titles = corpus["title"].fillna("").astype(str).str.strip()

    empty_title_count = int(titles.eq("").sum())

    print("\n========== 标题检查 ==========")
    print("标题为空的文档：", empty_title_count)
    print("标题为空比例：", round(
        empty_title_count / len(corpus), 4
    ))

    # 4. HTML 标签
    html_pattern = re.compile(r"<[^>]+>")

    html_count = int(
        text.str.contains(html_pattern).sum()
    )

    print("\n========== HTML 检查 ==========")
    print("包含 HTML 标签的文档：", html_count)
    print("HTML 文档比例：", round(
        html_count / len(corpus), 4
    ))

    # 统计常见标签
    tags = Counter()

    for document in text:
        found = re.findall(
            r"</?([a-zA-Z][a-zA-Z0-9]*)\b",
            document
        )
        tags.update(tag.lower() for tag in found)

    print("常见 HTML 标签：", tags.most_common(15))

    # 5. 相关文档数量
    relevant_counts = (
        qrels.groupby("query_id")["doc_id"].nunique()
    )

    describe("每条查询的相关文档数量", relevant_counts)

    # 6. Qrels 重复检查
    duplicate_qrels = qrels.duplicated(
        ["query_id", "doc_id"]
    ).sum()

    print("\n重复查询-文档标注对：", duplicate_qrels)

    # 7. 保存每条查询的相关文档数量
    query_details = queries[
        ["query_id", "query", "query_chars"]
    ].merge(
        relevant_counts.rename("relevant_doc_count"),
        left_on="query_id",
        right_index=True,
        how="left"
    )

    query_details.to_csv(
        REPORT_DIR / "query_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 8. 保存文档统计
    doc_details = corpus[
        ["doc_id", "title", "text_chars"]
    ].copy()

    doc_details["has_html"] = text.str.contains(
        html_pattern
    ).values

    doc_details["empty_title"] = titles.eq("").values

    doc_details.to_csv(
        REPORT_DIR / "document_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 9. 保存整体报告
    summary = {
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "qrels_count": len(qrels),
        "empty_title_count": empty_title_count,
        "html_document_count": html_count,
        "duplicate_qrels": int(duplicate_qrels),
        "query_length_mean": float(
            queries["query_chars"].mean()
        ),
        "document_length_mean": float(
            corpus["text_chars"].mean()
        ),
        "relevant_docs_per_query_mean": float(
            relevant_counts.mean()
        )
    }

    with open(
        REPORT_DIR / "audit_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary, f,
            ensure_ascii=False,
            indent=4
        )

    print("\n审计完成，结果保存至：", REPORT_DIR)


if __name__ == "__main__":
    main()
