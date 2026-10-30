import os
import pandas as pd
import requests
from dotenv import load_dotenv
from src import ROOT, load_data, build_retrievers, retrieve

load_dotenv(ROOT / ".env")
base = os.getenv("LLM_BASE_URL", "").rstrip("/")
key = os.getenv("LLM_API_KEY", "")
model = os.getenv("LLM_MODEL", "")
if not (base and key and model):
    raise SystemExit("请先复制 .env.example 为 .env，并填写 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。")

docs, qs = load_data()
retrievers = build_retrievers(docs)
query = input("请输入问题：").strip()
ranked = retrieve(query, docs, retrievers, method="hybrid", top_k=3)
context = "\n\n".join(
    f"[{docs.iloc[i]['doc_id']}] {docs.iloc[i]['title']}: {docs.iloc[i]['text']}"
    for i, _ in ranked
)
prompt = f"""请仅依据以下资料回答问题。如果资料不足，请明确说明。回答末尾标注使用的文档编号。
问题：{query}

资料：
{context}
"""
url = base + "/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"},
                     json={"model":model, "messages":[{"role":"user","content":prompt}], "temperature":0.1},
                     timeout=90)
resp.raise_for_status()
print("\n检索证据：\n" + context)
print("\n模型回答：\n" + resp.json()["choices"][0]["message"]["content"])
