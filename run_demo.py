# -*- coding: utf-8 -*-
"""Vector Graph RAG Demo 运行入口（零依赖，离线可跑）。

用法:
    python3 run_demo.py

输出: 索引阶段 + 多跳查询阶段的终端日志，并分别写入
      screenshots/out_index.txt / screenshots/out_query.txt 供生成截图。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vector_graph_rag import VectorGraphRAG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PASSAGES_PATH = os.path.join(HERE, "data", "passages.txt")


def load_passages(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    passages = load_passages(PASSAGES_PATH)

    idx_lines = []
    q_lines = []

    def iprint(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        idx_lines.append(s)

    def qprint(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        q_lines.append(s)

    iprint("=" * 64)
    iprint("Vector Graph RAG · 索引阶段（纯向量检索，无图数据库）")
    iprint("=" * 64)

    vg = VectorGraphRAG()
    triples = vg.build(passages)

    iprint(f"[1] 读取文档: {len(passages)} 段")
    iprint(f"[2] 启发式抽取三元组: {len(triples)} 条")
    iprint("[3] 实体向量库: %d 个实体 | 关系向量库: %d 条关系"
          % (len(vg.entity_store.items), len(vg.relation_store.items)))
    iprint("-" * 64)
    iprint("抽取到的知识图谱（三元组）:")
    for i, (s, r, o) in enumerate(triples, 1):
        iprint(f"  {i:2d}. ({s}, {r}, {o})")
    iprint("")
    iprint("提示: 实体/关系均被向量化存入内存向量库，查询时只用相似度检索，")
    iprint("      不依赖任何图数据库（Neo4j 等）。生产可换 Milvus（见 extractors.py）。")

    # ---------------- 查询阶段 ----------------
    qprint("=" * 64)
    qprint("Vector Graph RAG · 查询阶段（多跳推理）")
    qprint("=" * 64)

    questions = [
        "睿检索 的存储后端由哪家公司开源？",
        "开发 睿检索 的公司总部在哪个城市？",
        "睿检索 背后公司的创始人毕业于哪所院校？",
        "向量检索 替代了什么技术？",
    ]

    for qi, q in enumerate(questions, 1):
        qprint("")
        qprint(f"Q{qi}: {q}")
        path, score = vg.query(q, max_hops=3, top_k_seeds=3)
        qprint(f"    检索评分(两阶段): {score}")
        if path:
            chain = " → ".join(f"{s} --{r}--> {o}" for s, r, o in path)
            qprint(f"    检索路径: {chain}")
            qprint(f"    答案: {path[-1][2]}")
        else:
            qprint("    答案: （未检索到相关路径）")
        qprint("")

    # 写出供截图用的文本
    with open(os.path.join(HERE, "screenshots", "out_index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx_lines))
    with open(os.path.join(HERE, "screenshots", "out_query.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(q_lines))


if __name__ == "__main__":
    main()
