
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import json
import re

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "benchmark" / "t2retrieval"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    """
    清理 HTML 标签并保留正文内容。
    不修改原始数据。
    """

    if pd.isna(value):
        return ""

    raw = str(value)

    soup = BeautifulSoup(raw, "html.parser")

    # 删除脚本和样式内容
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 保留图片的 alt 文本，没有 alt 则删除图片标签
    for img in soup.find_all("img"):
        alt = img.get("alt", "").strip()

        if alt:
            img.replace_with(f" {alt} ")
        else:
            img.decompose()

    # 提取可见文字，标签之间加入空格
    text = soup.get_text(separator=" ", strip=True)

    # 删除零宽字符
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)

    # 统一空白字符
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("读取原始数据...")

    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")

    # 保存原始长度，方便比较
    raw_lengths = corpus["text"].fillna("").astype(str).str.len()

    print("清洗文档...")

    corpus_clean = pd.DataFrame({
        "doc_id": corpus["doc_id"].astype(str),
        "text": corpus["text"].apply(clean_text)
    })

    print("清洗查询...")

    queries_clean = pd.DataFrame({
        "query_id": queries["query_id"].astype(str),
        "query": queries["query"].apply(clean_text)
    })

    # 统计清洗前后长度
    clean_lengths = corpus_clean["text"].str.len()

    report = {
        "document_count": len(corpus_clean),
        "query_count": len(queries_clean),

        "raw_document_mean_length": float(
            raw_lengths.mean()
        ),

        "clean_document_mean_length": float(
            clean_lengths.mean()
        ),

        "raw_document_median_length": float(
            raw_lengths.median()
        ),

        "clean_document_median_length": float(
            clean_lengths.median()
        ),

        "empty_clean_documents": int(
            corpus_clean["text"].eq("").sum()
        ),

        "empty_clean_queries": int(
            queries_clean["query"].eq("").sum()
        ),

        "duplicate_clean_documents": int(
            corpus_clean["text"].duplicated().sum()
        )
    }

    # 保存清洗后的数据
    corpus_clean.to_parquet(
        OUTPUT_DIR / "corpus_clean.parquet",
        index=False
    )

    queries_clean.to_parquet(
        OUTPUT_DIR / "queries_clean.parquet",
        index=False
    )

    # 保存清洗前后样例，供人工检查
    samples = corpus.sample(
        n=min(30, len(corpus)),
        random_state=42
    )

    sample_rows = []

    clean_map = dict(zip(
        corpus_clean["doc_id"],
        corpus_clean["text"]
    ))

    for _, row in samples.iterrows():
        doc_id = str(row["doc_id"])

        sample_rows.append({
            "doc_id": doc_id,
            "raw_text": row["text"],
            "clean_text": clean_map[doc_id]
        })

    pd.DataFrame(sample_rows).to_csv(
        OUTPUT_DIR / "cleaning_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 保存空文档清单，暂时不删除
    corpus_clean[
        corpus_clean["text"].eq("")
    ].to_csv(
        OUTPUT_DIR / "empty_documents.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        OUTPUT_DIR / "cleaning_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== 清洗结果 ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n清洗完成！")
    print("输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
