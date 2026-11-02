from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_retrieval.bm25 import tokenize_zh


DATA_DIR = (
    ROOT / "data" / "benchmark" / "t2retrieval"
    / "processed" / "evaluation"
)

INDEX_DIR = ROOT / "data" / "indexes" / "bm25_jieba"


def main():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = DATA_DIR / "corpus_searchable.parquet"

    if not corpus_path.exists():
        raise FileNotFoundError(
            "找不到 corpus_searchable.parquet，"
            "请先运行 prepare_eval_data.py"
        )

    print("读取文档库...")

    corpus = pd.read_parquet(corpus_path)

    corpus["doc_id"] = corpus["doc_id"].astype(str)

    texts = (
        corpus["text"].fillna("").astype(str).tolist()
    )

    doc_ids = corpus["doc_id"].tolist()

    print("文档数量：", len(texts))

    print("\n开始构建 Jieba 中文词频矩阵...")
    print("首次构建可能需要几分钟，请耐心等待。")

    vectorizer = CountVectorizer(
        tokenizer=tokenize_zh,
        token_pattern=None,
        lowercase=False,
        dtype=np.int32,
        min_df=1
    )

    # 文档-词项矩阵
    matrix = vectorizer.fit_transform(texts).tocsc()

    # 统计每篇文档的词项数量
    doc_lengths = np.asarray(
        matrix.sum(axis=1)
    ).ravel().astype(np.int32)

    avgdl = float(doc_lengths.mean())

    # 保存稀疏矩阵
    save_npz(
        INDEX_DIR / "term_doc_matrix.npz",
        matrix,
        compressed=True
    )

    # 保存词典
    vocabulary = {
        term: int(index)
        for term, index in vectorizer.vocabulary_.items()
    }

    with open(
        INDEX_DIR / "vocabulary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            vocabulary,
            f,
            ensure_ascii=False
        )

    # 保存文档 ID
    with open(
        INDEX_DIR / "doc_ids.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            doc_ids,
            f,
            ensure_ascii=False
        )

    # 保存文档长度
    np.save(
        INDEX_DIR / "doc_lengths.npy",
        doc_lengths
    )

    # 检查有正文但没有有效分词的文档
    zero_token_ids = [
        doc_ids[i]
        for i in np.where(doc_lengths == 0)[0]
    ]

    pd.DataFrame({
        "doc_id": zero_token_ids
    }).to_csv(
        INDEX_DIR / "zero_token_documents.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metadata = {
        "model": "BM25",
        "tokenizer": "Jieba",
        "document_count": len(doc_ids),
        "vocabulary_size": len(vocabulary),
        "matrix_nnz": int(matrix.nnz),
        "average_document_length": avgdl,
        "zero_token_document_count": len(zero_token_ids),
        "k1": 1.5,
        "b": 0.75,
        "data": "T2Retrieval",
        "random_seed": 42
    }

    with open(
        INDEX_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== BM25 索引构建完成 ==========")
    print("文档数量：", len(doc_ids))
    print("词汇表大小：", len(vocabulary))
    print("矩阵非零元素：", matrix.nnz)
    print("平均文档长度：", round(avgdl, 2))
    print("零有效词项文档：", len(zero_token_ids))
    print("索引位置：", INDEX_DIR)


if __name__ == "__main__":
    main()
