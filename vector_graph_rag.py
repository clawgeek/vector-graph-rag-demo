# -*- coding: utf-8 -*-
"""
Vector Graph RAG —— 从零实现的最小可运行 Demo（零硬依赖，纯标准库）

对应 Zilliz 团队开源的 Vector Graph RAG 思路：
  - 不依赖图数据库，把「实体」和「关系」都向量化，用纯向量检索完成多跳推理；
  - 查询阶段：实体抽取 → 向量检索 → 子图扩展 → 单轮重排序 → 生成答案；
  - 两阶段评分：关系评分(relation) + 段落评分(passage)，达到质量-延迟的帕累托最优；
  - 单轮重排序：避免传统 Graph RAG 的迭代搜索循环，降低延迟。

本 Demo 用 TF-IDF 向量 + 启发式抽取，完全离线可跑；真实生产可把
Embedder / TripleExtractor / AnswerGenerator 三个适配器替换为
sentence-transformers / OpenAI 等（见 extractors.py）。
"""

import math
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1) 向量化：TF-IDF（纯 Python，离线）
# ---------------------------------------------------------------------------
class TfidfEmbedder:
    def __init__(self):
        self.idf = {}
        self.N = 0

    def _tokenize(self, text):
        # 英文/数字按词，中文按字；小写化
        text = (text or "").lower()
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)

    def fit(self, corpus):
        docs = [self._tokenize(d) for d in corpus]
        self.N = max(len(docs), 1)
        df = defaultdict(int)
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log((self.N + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vec(self, toks):
        tf = defaultdict(float)
        for t in toks:
            tf[t] += 1
        vec = {t: w * self.idf.get(t, 1.0) for t, w in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def embed(self, text):
        return self._vec(self._tokenize(text))


def cosine(a, b):
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-9)


# ---------------------------------------------------------------------------
# 2) 向量存储：内存版（可选换成 Milvus，见 backends/milvus_store.py）
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, embedder):
        self.e = embedder
        self.items = []  # {'text', 'vec', 'meta'}

    def add(self, text, meta=None):
        self.items.append({"text": text, "vec": self.e.embed(text), "meta": meta or {}})

    def search(self, query_vec, top_k=5):
        scored = [(cosine(query_vec, it["vec"]), it) for it in self.items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 3) 三元组抽取（启发式；生产可换 OpenAI，见 extractors.py）
# ---------------------------------------------------------------------------
class HeuristicExtractor:
    """从自然语言段落中抽取 (subject, relation, object) 三元组。

    针对本 Demo 的示例语料设计了一组中文关系模式；真实场景应替换为 LLM 抽取。
    """

    RULES = [
        (r"(.+?)由(.+?)开发", "开发"),
        (r"(.+?)由(.+?)开源", "开源"),
        (r"(.+?)总部位于(.+?)[。.\n]", "总部位于"),
        (r"(.+?)的?创始人是?(.+?)[。.\n]", "创始人"),
        (r"(.+?)毕业于(.+?)[。.\n]", "毕业于"),
        (r"(.+?)基于(.+?)[。.\n]", "基于"),
        (r"(.+?)替代了?(.+?)[。.\n]", "替代"),
        (r"(.+?)的?存储后端是?(.+?)[。.\n]", "存储后端是"),
    ]

    def extract(self, passages):
        triples = []
        for p in passages:
            p = p.strip()
            if not p:
                continue
            # 列表型句子：A 支持 X、Y、Z。
            m = re.search(r"(.+?)支持(.+?)[。.\n]", p)
            if m:
                subj = m.group(1).strip()
                for obj in re.split(r"[、,，]", m.group(2)):
                    obj = obj.strip()
                    if obj:
                        triples.append((subj, "支持", obj))
                continue
            for pat, rel in self.RULES:
                m = re.search(pat, p)
                if m:
                    subj = m.group(1).strip()
                    obj = m.group(2).strip()
                    # 去掉关系词里可能带出的多余字
                    obj = re.sub(r"^[的是]+", "", obj).strip()
                    triples.append((subj, rel, obj))
                    break
        return triples


# ---------------------------------------------------------------------------
# 4) 答案生成（启发式模板；生产可换 LLM）
# ---------------------------------------------------------------------------
class HeuristicAnswerGenerator:
    @staticmethod
    def generate(question, path):
        if not path:
            return "（未检索到相关路径）"
        steps = " → ".join(
            f"{s} --{r}--> {o}" for s, r, o in path
        )
        final = path[-1][2]
        return f"检索路径：{steps}\n答案：{final}"


# ---------------------------------------------------------------------------
# 5) Vector Graph RAG 主流程
# ---------------------------------------------------------------------------
class VectorGraphRAG:
    def __init__(self, embedder=None, extractor=None, answer_gen=None):
        self.embedder = embedder or TfidfEmbedder()
        self.extractor = extractor or HeuristicExtractor()
        self.answer_gen = answer_gen or HeuristicAnswerGenerator()
        self.entity_store = VectorStore(self.embedder)   # 实体向量
        self.relation_store = VectorStore(self.embedder) # 关系向量
        self.graph = defaultdict(list)   # entity_text -> [(rel, obj_text), ...]
        self.passages = []

    def build(self, passages):
        """索引：文档 → 抽取三元组 → 实体/关系嵌入 → 存储。"""
        self.passages = [p for p in passages if p.strip()]
        corpus = self.passages + ["Vector Graph RAG 多跳检索 实体 关系 向量"]
        self.embedder.fit(corpus)

        triples = self.extractor.extract(self.passages)
        entities = set()
        for s, r, o in triples:
            entities.add(s)
            entities.add(o)
            self.graph[s].append((r, o))

        for e in entities:
            self.entity_store.add(e, {"kind": "entity"})
        for s, r, o in triples:
            self.relation_store.add(f"{r} {o}", {"subj": s, "rel": r, "obj": o})
        return triples

    def _passage_of(self, triple):
        s, r, o = triple
        for p in self.passages:
            if s in p and o in p:
                return p
        return ""

    def query(self, question, max_hops=3, top_k_seeds=3):
        """查询：向量检索种子实体 → 子图向量扩展 → 单轮重排序 → 生成答案。"""
        qvec = self.embedder.embed(question)

        # 种子实体：与问题最相似的 top_k 个实体
        seeds = self.entity_store.search(qvec, top_k=top_k_seeds)

        best_path = None
        best_score = -1.0

        def score_path(path):
            # 两阶段评分（单轮重排序）：
            #   relation 评分 = 每跳「关系+宾语」与问题的相似度
            #   passage  评分 = 每跳所在段落与问题的相似度
            #   最终用「整条路径的关系链文本」与问题的相似度做重排序，
            #   这样越能覆盖问题中关键词的（更长）正确路径会胜出。
            rel_scores, pas_scores = [], []
            for (s, r, o) in path:
                rel_scores.append(cosine(self.embedder.embed(f"{r} {o}"), qvec))
                pas = self._passage_of((s, r, o))
                pas_scores.append(cosine(self.embedder.embed(pas), qvec) if pas else 0.0)
            rel = sum(rel_scores) / max(len(rel_scores), 1)
            pas = sum(pas_scores) / max(len(pas_scores), 1)
            chain_text = " ".join(f"{r} {o}" for _, r, o in path)
            chain = cosine(self.embedder.embed(chain_text), qvec)
            # 关系权重最高，链式覆盖次之，段落兜底
            return 0.5 * rel + 0.3 * chain + 0.2 * pas

        def dfs(entity, path, depth):
            nonlocal best_path, best_score
            if depth > max_hops:
                return
            for (r, o) in self.graph.get(entity, []):
                new_path = path + [(entity, r, o)]
                sc = score_path(new_path)
                if sc > best_score:
                    best_score, best_path = sc, new_path
                if o in self.graph:  # 对象是实体，可继续扩展
                    dfs(o, new_path, depth + 1)

        for _, seed in seeds:
            dfs(seed["text"], [], 1)

        return best_path, round(best_score, 4)
