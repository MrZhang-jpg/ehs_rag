# -*- coding: utf-8 -*-
"""
检索候选策略对比实验（hybrid_rerank 通道，仅检索侧，无 LLM 调用）
--------------------------------------------------------------------
背景：RRF 融合对"单通道命中"的专业术语块不友好 —— 只在某一路出现的块，
在 rrf_k=60 下融合分约 1/(60+r)，会被压到候选池 30 名开外，导致 reranker
根本看不到它（debug_recall 实测：单路第 14 名 → 融合第 55 名）。

本脚本在同一评测集上对比 4 种"送入 reranker 的候选池"构造方式：
  rrf30   现状：RRF(topn=30) -> rerank -> top5
  rrf40   RRF 全部候选直接送精排（不提前截断）
  union   vec(top_k) ∪ bm25(top_k) 去重后全部送精排
  union2  vec(30) ∪ bm25(30) 去重后送精排（更大池）

输出：各策略 Hit@5 / MRR / ContextRecall + 相对 rrf30 的逐题翻转明细。
用法：
    python tune_retrieval.py                 # 全量 99 题
    python tune_retrieval.py --ids seed_009 seed_037   # 指定题
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


import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config import settings
from evaluator import _gold_match, _point_coverage
from hybrid_retriever import BGEReranker, RRFusion

STRATS = ("rrf30", "rrf40", "union", "union2")


def dedup(docs: List[Document]) -> List[Document]:
    """按 chunk_id 去重（与 RRF 的身份键一致）。"""
    seen, out = set(), []
    for d in docs:
        key = d.metadata.get("chunk_id") or d.page_content[:50]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="检索候选策略对比")
    ap.add_argument("--ids", nargs="*", default=None, help="只跑指定题号")
    ap.add_argument("--sample", type=int, default=None, help="只跑前 N 题")
    args = ap.parse_args()

    from main import EHSRAGApplication

    app = EHSRAGApplication()
    ds_path = Path(settings.eval_dir) / settings.eval_cfg.dataset_file
    items = json.load(open(ds_path, encoding="utf-8"))
    if args.ids:
        items = [x for x in items if x["id"] in set(args.ids)]
    elif args.sample:
        items = items[: args.sample]
    print(f"评测题 {len(items)} 道；策略 {STRATS}")

    rc = settings.retrieval
    rer = BGEReranker(settings)
    K = rc.final_top_k

    results: Dict[str, Dict[str, Tuple[bool, float, float]]] = {s: {} for s in STRATS}
    t0 = time.time()
    for n, item in enumerate(items, 1):
        q = item["question"]
        # 通道候选一次取足（30 条），各策略按需切片，避免重复检索
        vec30 = app.vector_store.similarity_search_with_scores(q, k=30)
        kw30 = app.retriever.bm25.search(q, k=30)
        vec20, kw20 = vec30[: rc.vector_top_k], kw30[: rc.bm25_top_k]

        pools = {
            "rrf30": [d for d, _ in RRFusion.fuse([vec20, kw20], rrf_k=rc.rrf_k,
                                                  topn=rc.rerank_candidates)],
            "rrf40": [d for d, _ in RRFusion.fuse([vec20, kw20], rrf_k=rc.rrf_k,
                                                  topn=10 ** 6)],
            "union": dedup([d for d, _ in vec20] + [d for d, _ in kw20]),
            "union2": dedup([d for d, _ in vec30] + [d for d, _ in kw30]),
        }
        for name, pool in pools.items():
            ranked = rer.rerank(q, pool)[:K]
            gold_ranks = [i + 1 for i, (d, _) in enumerate(ranked)
                          if _gold_match(d, item)]
            hit = bool(gold_ranks)
            rr = 1.0 / gold_ranks[0] if gold_ranks else 0.0
            ctx = "\n".join(d.page_content for d, _ in ranked)
            cr, _ = _point_coverage(item.get("key_points", []), ctx)
            results[name][item["id"]] = (hit, rr, cr)
        if n % 10 == 0:
            print(f"  ... {n}/{len(items)}（{time.time() - t0:.0f}s）")

    print("\n" + "=" * 72)
    print(f" 候选策略对比（{len(items)} 题，hybrid_rerank，耗时 {time.time() - t0:.0f}s）")
    print("=" * 72)
    base = results["rrf30"]
    for name in STRATS:
        r = results[name]
        hits = sum(1 for v in r.values() if v[0]) / len(r)
        mrr = sum(v[1] for v in r.values()) / len(r)
        cr = sum(v[2] for v in r.values()) / len(r)
        delta = ""
        if name != "rrf30":
            up = [i for i in r if r[i][0] and not base[i][0]]
            down = [i for i in r if not r[i][0] and base[i][0]]
            delta = f"  [vs rrf30] miss->hit {len(up)} {up} | hit->miss {len(down)} {down}"
        print(f"  {name:7s} Hit@5={hits:.4f}  MRR={mrr:.4f}  CR={cr:.4f}{delta}")


if __name__ == "__main__":
    main()
