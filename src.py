from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def load_data():
    docs = pd.read_csv(DATA / "documents.csv").fillna("")
    qs = pd.read_csv(DATA / "queries.csv").fillna("")
    return docs, qs

def tokenize(text):
    # Simple tokenization for a small bilingual demo corpus.
    # For serious Chinese IR, replace with jieba/custom tokenizer or a Chinese embedding model.
    text = str(text).lower()
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    latin_tokens = re.findall(r'[a-z0-9]+', text)
    # Character bigrams help preserve some Chinese phrase matching.
    compact = ''.join(chinese_chars)
    char_bigrams = [compact[i:i+2] for i in range(max(0, len(compact)-1))]
    return latin_tokens + chinese_chars + char_bigrams

def make_classifiers():
    return {
        "NaiveBayes": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1, sublinear_tf=True)),
            ("clf", MultinomialNB())
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ]),
        "LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42))
        ])
    }


def fit_classifiers(qs):
    # Keep a fixed holdout test split. Do not choose the adaptive model using test scores.
    train_idx, test_idx = train_test_split(
        np.arange(len(qs)), test_size=0.30, random_state=42, stratify=qs["label"]
    )
    train_qs = qs.iloc[train_idx]
    test_qs = qs.iloc[test_idx]
    X_train, X_test = train_qs["query"], test_qs["query"]
    y_train, y_test = train_qs["label"], test_qs["label"]
    metrics = []
    reports = []
    fitted = {}
    for name, model in make_classifiers().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "test_size": len(y_test)
        })
        reports.append(f"===== {name} =====\n" + classification_report(y_test, pred, zero_division=0))
        fitted[name] = model
    pd.DataFrame(metrics).to_csv(RESULTS/"classification_metrics.csv", index=False, encoding="utf-8-sig")
    (RESULTS/"classification_report.txt").write_text("\n\n".join(reports), encoding="utf-8")
    # Predeclare Logistic Regression as the routing model; avoid selecting on the test set.
    routing_model = fitted["LogisticRegression"]
    return routing_model, "LogisticRegression", test_qs.copy()

def build_retrievers(docs):
    texts = (docs["title"] + " " + docs["text"]).tolist()
    tokenized_docs = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_docs)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return texts, bm25, vectorizer, matrix

def retrieve(query, docs, retrievers, method="bm25", top_k=3):
    texts, bm25, vectorizer, matrix = retrievers
    if method == "bm25":
        scores = np.asarray(bm25.get_scores(tokenize(query)))
    elif method == "tfidf":
        qv = vectorizer.transform([query])
        scores = (matrix @ qv.T).toarray().ravel()
    elif method == "hybrid":
        b = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
        qv = vectorizer.transform([query])
        t = (matrix @ qv.T).toarray().ravel()
        # Rank-based fusion is more robust to different raw score scales.
        brank = np.argsort(np.argsort(-b)) + 1
        trank = np.argsort(np.argsort(-t)) + 1
        scores = 1/(60+brank) + 1/(60+trank)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    inds = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in inds]

def evaluate_retrieval(qs, docs, retrievers, classifier=None):
    rows = []
    mapping = {"fact": "bm25", "explain": "tfidf", "compare": "hybrid"}
    for _, row in qs.iterrows():
        gold = str(row["relevant_doc_id"])
        for method in ["bm25", "tfidf", "hybrid"]:
            ranked = retrieve(row["query"], docs, retrievers, method=method, top_k=3)
            doc_ids = [str(docs.iloc[i]["doc_id"]) for i, _ in ranked]
            hit = int(gold in doc_ids)
            rank = doc_ids.index(gold)+1 if gold in doc_ids else None
            rows.append({"query_id": row["query_id"], "system": method, "hit_at_3": hit,
                         "reciprocal_rank": 1/rank if rank else 0})
        if classifier is not None:
            predicted = classifier.predict([row["query"]])[0]
            method = mapping[predicted]
            ranked = retrieve(row["query"], docs, retrievers, method=method, top_k=3)
            doc_ids = [str(docs.iloc[i]["doc_id"]) for i, _ in ranked]
            hit = int(gold in doc_ids)
            rank = doc_ids.index(gold)+1 if gold in doc_ids else None
            rows.append({"query_id": row["query_id"], "system": "adaptive", "hit_at_3": hit,
                         "reciprocal_rank": 1/rank if rank else 0})
    raw = pd.DataFrame(rows)
    summary = raw.groupby("system", as_index=False).agg(
        recall_at_3=("hit_at_3","mean"),
        mrr_at_3=("reciprocal_rank","mean")
    )
    summary.to_csv(RESULTS/"retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    raw.to_csv(RESULTS/"retrieval_per_query.csv", index=False, encoding="utf-8-sig")
    return summary

def main():
    docs, qs = load_data()
    model, model_name, test_qs = fit_classifiers(qs)
    print(f"分类器演示采用：{model_name}")
    print("分类指标：")
    print(pd.read_csv(RESULTS/"classification_metrics.csv").to_string(index=False))
    retrievers = build_retrievers(docs)
    # Evaluate retrieval baselines and adaptive routing on the same held-out test queries only.
    summary = evaluate_retrieval(test_qs, docs, retrievers, classifier=model)
    print("\n检索指标（配套小型演示语料）：")
    print(summary.to_string(index=False))
    print("\n结果已写入 results/。注意：该小型标注集仅用于验证流程，正式报告应扩充数据并人工核验。")

if __name__ == "__main__":
    main()
