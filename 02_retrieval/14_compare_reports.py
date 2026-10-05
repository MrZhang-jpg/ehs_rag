# -*- coding: utf-8 -*-
"""
评测报告对比：基线 vs 修复后
--------------------------------------------------------------------
输出：
  1) 各检索模式的 Hit Rate@5 / MRR / Context Recall 变化（含 delta）
  2) 逐题翻转明细（miss -> hit 的题、hit -> miss 的题），并标注分类
用法：
    python compare_reports.py evaluation/baseline_eval_report.json evaluation/eval_report.json
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
import sys


def _fmt(base: float, new: float) -> str:
    d = new - base
    arrow = "↑" if d > 1e-9 else ("↓" if d < -1e-9 else "=")
    return f"{base:.3f} -> {new:.3f}  (Δ{d:+.3f} {arrow})"


def main() -> None:
    bpath, npath = sys.argv[1], sys.argv[2]
    base = json.load(open(bpath, encoding="utf-8"))
    new = json.load(open(npath, encoding="utf-8"))

    print("=" * 72)
    print(" 检索指标对比（Hit Rate@5 / MRR / Context Recall）")
    print("=" * 72)
    for mode in new["retrieval_summary"]:
        bs = base["retrieval_summary"].get(mode)
        ns = new["retrieval_summary"][mode]
        if not bs:
            continue
        print(f"  {mode}")
        print(f"    Hit Rate@5     {_fmt(bs['hit_rate'], ns['hit_rate'])}")
        print(f"    MRR            {_fmt(bs['mrr'], ns['mrr'])}")
        print(f"    Context Recall {_fmt(bs['context_recall'], ns['context_recall'])}")

    bd = {d["id"]: d for d in base["details"]}
    nd = {d["id"]: d for d in new["details"]}
    for mode in new["retrieval_summary"]:
        up, down = [], []
        for iid, d in nd.items():
            nb = d["retrieval"].get(mode, {}).get("hit")
            ob = bd.get(iid, {}).get("retrieval", {}).get(mode, {}).get("hit")
            if nb is None or ob is None:
                continue
            if nb and not ob:
                up.append(iid)
            elif ob and not nb:
                down.append(iid)
        if up or down:
            print(f"\n  [{mode}] miss->hit {len(up)} 题: {up}")
            print(f"  [{mode}] hit->miss {len(down)} 题: {down}")
            for iid in down:
                print(f"      {iid}: {nd[iid]['question'][:44]}")

    gs = new.get("generation_summary")
    if gs:
        print("\n" + "=" * 72)
        print(" 生成指标（本次）")
        print("=" * 72)
        for k, v in gs.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
