# -*- coding: utf-8 -*-
"""
检索失败归因脚本：对指定评测题做"通道级"黄金块排名追溯
--------------------------------------------------------------------
对每道题打印黄金块在四层中的排名：
    vector 通道(前 K) -> bm25 通道(前 K) -> RRF 融合(生产池) -> rerank 后
用于区分失败发生在哪一层，指导调参（top_k / 候选池 / rerank 截断）：
  - 排名为 None           => 该层召回不到，需要加大 K 或改善块内容
  - 融合排名 > 候选池上限  => 提高 rerank_candidates / 单路 top_k
  - rerank 后掉出 5 名     => 精排模型问题（可考虑换更大 reranker）

保真性：向量/BM25 通道排名用深层 K 诊断值（黄金块真实召回的深度），
但**融合与 rerank 候选严格按生产参数复现**（通道先截到 vector_top_k /
bm25_top_k 再融合取 rerank_candidates）——曾因用深池融合导致 rerank
名次与线上不一致（评测命中而本工具显示大排名，反之亦然）。
用法：
    python debug_recall.py                 # 诊断默认 4 道已知失败题
    python debug_recall.py seed_001 ...    # 诊断指定题号
--------------------------------------------------------------------
"""
from __future__ import annotations
# --- 项目重构引导：支持 01_indexing / 02_retrieval 下带序号文件名的导入 ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "01_indexing", _ROOT / "02_retrieval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
import _bootstrap as _bootstrap
_bootstrap.init()
# --- 引导结束 ---


import json
import logging
import sys
from pathlib import Path

from config import settings
from evaluator import _gold_match
from hybrid_retriever import BGEReranker, RRFusion

# 最近一轮全量评测（hybrid_rerank）的残余失败题
DEFAULT_IDS = ["seed_009", "seed_028", "seed_034", "seed_035", "seed_038",
               "seed_040", "seed_041", "seed_042", "seed_045", "gen_090"]


def main(argv: list) -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from main import EHSRAGApplication  # 延迟导入：加载 BGE 等重组件

    app = EHSRAGApplication()
    ds = {r["id"]: r for r in json.load(
        open(Path(settings.eval_dir) / settings.eval_cfg.dataset_file,
             encoding="utf-8"))}
    ids = argv or DEFAULT_IDS
    K = 100
    rc = settings.retrieval

    for iid in ids:
        item = ds.get(iid)
        if item is None:
            print(f"[跳过] 评测集无 {iid}")
            continue
        q = item["question"]

        vecD = app.vector_store.similarity_search_with_scores(q, k=K)
        kwD = app.retriever.bm25.search(q, k=K)
        # 生产参数复现：通道截到生产 top_k 再融合（与
        # HybridRetriever.retrieve 的池构造逐字一致），避免深池名次失真
        vec, kw = vecD[: rc.vector_top_k], kwD[: rc.bm25_top_k]
        fused = RRFusion.fuse([vec, kw], rrf_k=rc.rrf_k,
                              topn=max(rc.rerank_candidates, rc.final_top_k))
        cands = [d for d, _ in fused[: rc.rerank_candidates]]
        reranked = BGEReranker(settings).rerank(q, cands)

        def rank(lst) -> int | None:
            for i, (d, _s) in enumerate(lst, 1):
                if _gold_match(d, item):
                    return i
            return None

        rv, rb = rank(vecD), rank(kwD)          # 深层通道名次（诊断）
        rf, rr = rank(fused), rank(reranked)    # 生产池融合/精排名次
        print(f"\n== {iid}  file={item['source_file'][:36]}  section={item.get('section')!r}")
        print(f"   {q}")
        print(f"   黄金块排名（通道=深{K}诊断，融合/rerank=生产参数复现）:")
        print(f"     vector={rv}  bm25={rb}  融合={rf}  "
              f"rerank(池{len(cands)})={rr}")
        gold = next((d for d, _ in fused if _gold_match(d, item)), None)
        if gold is not None:
            m = gold.metadata
            print(f"   黄金块: section_title={m.get('section_title')!r} "
                  f"page={m.get('page')} chunk_id={str(m.get('chunk_id'))[:16]}")
            print(f"   正文: {gold.page_content[:90]!r}")
        elif rv or rb:
            print(f"   黄金块被生产候选池截断（深层通道有名次，"
                  f"但未进生产池前 {rc.rerank_candidates}）")
        else:
            print(f"   深层通道（k={K}）均未召回黄金块")


if __name__ == "__main__":
    main(sys.argv[1:])
