
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "benchmark" / "t2retrieval" / "processed"
EVAL_DIR = PROCESSED_DIR / "evaluation"
OUTPUT_DIR = EVAL_DIR / "splits"

SEED = 42


def main():

    # 检查前一步的数据准备结果
    required_files = [
        PROCESSED_DIR / "queries_clean.parquet",
        EVAL_DIR / "qrels_searchable.parquet"
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到文件：{path}\n"
                "请先运行 prepare_eval_data.py"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(
        PROCESSED_DIR / "queries_clean.parquet"
    )

    qrels = pd.read_parquet(
        EVAL_DIR / "qrels_searchable.parquet"
    )

    queries["query_id"] = queries["query_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)

    # 统计每条查询的相关文档数量
    relevant_counts = (
        qrels.groupby("query_id")["doc_id"]
        .nunique()
    )

    queries["relevant_doc_count"] = (
        queries["query_id"]
        .map(relevant_counts)
        .fillna(0)
        .astype(int)
    )

    if (queries["relevant_doc_count"] == 0).any():
        raise ValueError("存在没有相关文档的查询，请检查 Qrels")

    # 固定随机种子，保证实验可复现
    rng = np.random.default_rng(SEED)

    indices = rng.permutation(len(queries))

    n = len(queries)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": queries.iloc[train_idx].copy(),
        "validation": queries.iloc[val_idx].copy(),
        "test": queries.iloc[test_idx].copy()
    }

    # 保存查询划分和对应 Qrels
    split_records = []

    for name, query_df in splits.items():

        query_ids = set(query_df["query_id"])

        split_qrels = qrels[
            qrels["query_id"].isin(query_ids)
        ].copy()

        query_df.to_parquet(
            OUTPUT_DIR / f"{name}_queries.parquet",
            index=False
        )

        split_qrels.to_parquet(
            OUTPUT_DIR / f"{name}_qrels.parquet",
            index=False
        )

        for _, row in query_df.iterrows():
            split_records.append({
                "query_id": row["query_id"],
                "split": name,
                "relevant_doc_count": row["relevant_doc_count"]
            })

        print(f"\n========== {name.upper()} ==========")
        print("查询数量：", len(query_df))
        print("Qrels 数量：", len(split_qrels))
        print(
            "平均相关文档数：",
            round(query_df["relevant_doc_count"].mean(), 3)
        )

    # 保存完整划分记录
    split_df = pd.DataFrame(split_records)

    split_df.to_csv(
        OUTPUT_DIR / "query_splits.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metadata = {
        "random_seed": SEED,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "total_queries": n,
        "train_queries": len(splits["train"]),
        "validation_queries": len(splits["validation"]),
        "test_queries": len(splits["test"]),
        "split_method": "random query-level split",
        "note": "Internal experimental split, not official benchmark split"
    }

    with open(
        OUTPUT_DIR / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("\n========== 数据划分完成 ==========")
    print("总查询数量：", n)
    print("随机种子：", SEED)
    print("输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
