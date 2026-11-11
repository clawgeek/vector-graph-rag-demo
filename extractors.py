# -*- coding: utf-8 -*-
"""
可插拔的「生产级」适配器（可选依赖，本 Demo 默认用启发式，无需联网/密钥）。

把下面三个组件替换掉，即可得到接近 Zilliz 官方 Vector Graph RAG 的效果：
  - Embedder      -> sentence-transformers（bge / e5 等语义向量）
  - TripleExtractor-> OpenAI / 任意 LLM（从文档抽取三元组）
  - AnswerGenerator-> LLM（基于检索路径生成自然语言答案）

仅作示例，未安装对应包时不会被执行。
"""
from vector_graph_rag import VectorStore


# ------------------------- 语义向量（可选） -------------------------
class SentenceTransformerEmbedder:
    """需要: pip install sentence-transformers"""

    def __init__(self, model="BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed(self, text):
        import numpy as np
        v = self.model.encode(text, normalize_embeddings=True)
        return {f"d{i}": float(x) for i, x in enumerate(v)}


# ------------------------- LLM 抽取三元组（可选） -------------------------
class OpenAIExtractor:
    """需要: pip install openai，并设置 OPENAI_API_KEY。"""

    PROMPT = (
        "从下面的文本中抽取知识三元组，格式为 (主语, 关系, 宾语)，"
        "每行一个，不要解释。\n\n文本:\n{text}"
    )

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def extract(self, passages):
        import re, json
        triples = []
        for p in passages:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.PROMPT.format(text=p)}],
            )
            for line in resp.choices[0].message.content.strip().splitlines():
                m = re.findall(r"[(（](.+?),\s*(.+?),\s*(.+?)[)）]", line)
                if m:
                    triples.append(tuple(x.strip() for x in m[0]))
        return triples


# ------------------------- LLM 生成答案（可选） -------------------------
class OpenAIAnswerGenerator:
    """需要: pip install openai，并设置 OPENAI_API_KEY。"""

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, question, path):
        ctx = "\n".join(f"{s} --{r}--> {o}" for s, r, o in path)
        prompt = (
            f"已知以下检索路径：\n{ctx}\n\n"
            f"请回答用户问题：{question}\n用中文简洁作答。"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


# ------------------------- 可选：Milvus 后端（可选） -------------------------
class MilvusVectorStore(VectorStore):
    """需要: pip install pymilvus。零服务器的 Milvus Lite 也可。"""

    def __init__(self, embedder, collection="vgraph", dim=768):
        super().__init__(embedder)
        from pymilvus import MilvusClient
        self.client = MilvusClient("./milvus_demo.db")
        if not self.client.has_collection(collection):
            self.client.create_collection(collection, dimension=dim)
        self.coll = collection
        self.ids = []

    def add(self, text, meta=None):
        vec = list(self.embedder.embed(text).values())
        self.ids.append(len(self.ids) + 1)
        self.client.insert(self.coll, [{"id": self.ids[-1], "vector": vec, "text": text}])

    def search(self, query_vec, top_k=5):
        vec = list(query_vec.values())
        res = self.client.search(self.coll, [vec], limit=top_k)
        return [(hit["distance"], {"text": hit["entity"]["text"]}) for hit in res[0]]
