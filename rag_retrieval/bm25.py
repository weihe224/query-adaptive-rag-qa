
import json
from pathlib import Path

import jieba
import numpy as np
from scipy.sparse import load_npz


def tokenize_zh(text):
    """
    中文分词。
    保留中文、英文和数字，去除纯标点符号。
    """

    tokens = jieba.lcut(str(text), cut_all=False)

    return [
        token.strip().lower()
        for token in tokens
        if token.strip() and any(ch.isalnum() for ch in token)
    ]


class BM25Retriever:

    def __init__(self, index_dir, k1=1.5, b=0.75):

        self.index_dir = Path(index_dir)

        # 加载倒排矩阵
        self.matrix = load_npz(
            self.index_dir / "term_doc_matrix.npz"
        ).tocsc()

        # 加载词典
        with open(
            self.index_dir / "vocabulary.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.vocabulary = json.load(f)

        # 加载文档 ID
        with open(
            self.index_dir / "doc_ids.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.doc_ids = json.load(f)

        # 加载文档长度
        self.doc_lengths = np.load(
            self.index_dir / "doc_lengths.npy"
        )

        self.k1 = float(k1)
        self.b = float(b)

        self.n_docs = len(self.doc_ids)

        self.avgdl = max(
            float(self.doc_lengths.mean()), 1e-12
        )

        if self.matrix.shape[0] != self.n_docs:
            raise ValueError("索引中的文档数量不一致")

        # 计算每个词的文档频率 DF
        df = np.diff(self.matrix.indptr).astype(
            np.float64
        )

        # Robertson/Sparck Jones 风格的正值 IDF
        self.idf = np.log1p(
            (self.n_docs - df + 0.5) / (df + 0.5)
        )

    def search(self, query, top_k=10):

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        # 去除重复查询词，避免重复词被重复计分
        tokens = list(dict.fromkeys(tokenize_zh(query)))

        doc_parts = []
        score_parts = []

        for token in tokens:

            term_id = self.vocabulary.get(token)

            if term_id is None:
                continue

            start = self.matrix.indptr[term_id]
            end = self.matrix.indptr[term_id + 1]

            if start == end:
                continue

            # 当前词对应的文档编号
            doc_indices = self.matrix.indices[start:end]

            # 当前词在各文档中的词频
            tf = self.matrix.data[start:end].astype(
                np.float64
            )

            doc_lengths = self.doc_lengths[
                doc_indices
            ].astype(np.float64)

            norm = self.k1 * (
                1 - self.b
                + self.b * doc_lengths / self.avgdl
            )

            scores = (
                self.idf[term_id]
                * tf * (self.k1 + 1)
                / (tf + norm)
            )

            doc_parts.append(doc_indices)
            score_parts.append(scores)

        # 查询词均不在词典中
        if not doc_parts:
            return []

        # 合并各查询词的候选文档及得分
        all_docs = np.concatenate(doc_parts)
        all_scores = np.concatenate(score_parts)

        # 相同文档的不同词项得分累加
        unique_docs, inverse = np.unique(
            all_docs,
            return_inverse=True
        )

        candidate_scores = np.bincount(
            inverse,
            weights=all_scores
        )

        # 选取 Top-K
        top_n = min(top_k, len(unique_docs))

        if top_n < len(unique_docs):
            selected = np.argpartition(
                candidate_scores, -top_n
            )[-top_n:]
        else:
            selected = np.arange(len(unique_docs))

        # 按得分降序排列，同分时按文档索引排序
        order = selected[
            np.lexsort((
                unique_docs[selected],
                -candidate_scores[selected]
            ))
        ]

        results = []

        for idx in order:

            doc_index = int(unique_docs[idx])

            results.append({
                "doc_id": self.doc_ids[doc_index],
                "score": float(candidate_scores[idx])
            })

        return results
