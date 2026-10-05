# -*- coding: utf-8 -*-
"""
评测集标签审计：逐题检查"黄金标签在源文件中是否可达"
--------------------------------------------------------------------
动机：评测集里若某题的标签（section / match_terms）在其 source_file 的
已入库内容中逐字不存在，则该校验点永远不可能命中 —— 指标被"标签噪声"
压住，而检索系统本身可能并无问题（反之也掩盖真实缺陷）。本脚本把
"逐题人工核对"固化成可重复执行的体检：

  1) 用 evaluator._gold_match 的**真实匹配语义**（不复制逻辑）检查：
     该题的 source_file 中是否存在至少一个块能判为黄金来源；
  2) 检查每个 key_point 是否能在该文件的任意块中逐字命中（归一化后），
     给出"不可匹配要点"清单与占比；
  3) 汇总输出，便于逐条修正（补 match_terms 原文锚点 / 修正 source_file）。

用法：
    python audit_labels.py                      # 全量审计
    python audit_labels.py --only-failing       # 仅打印有问题的题
    python audit_labels.py --dataset evaluation/ehs_eval_dataset.json
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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from config import settings
from evaluator import _gold_match, _point_coverage, normalize_for_match
from vector_store import EHSVectorStore


def load_items(path: Path) -> List[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return data if isinstance(data, list) else data.get("items", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="评测集标签可达性审计")
    ap.add_argument("--dataset", default="evaluation/ehs_eval_dataset.json")
    ap.add_argument("--only-failing", action="store_true")
    args = ap.parse_args()

    items = load_items(Path(args.dataset))
    store = EHSVectorStore(settings)
    docs = store.all_documents()
    print(f"向量库共 {len(docs)} 块；评测集 {len(items)} 题")

    by_file: Dict[str, List[Document]] = defaultdict(list)
    for d in docs:
        by_file[d.metadata.get("file_name") or "?"].append(d)

    no_chunks, no_label_hit = [], []
    bad_points_total = bad_points_items = 0
    points_total = 0

    for it in items:
        fname = it.get("source_file") or ""
        fdocs = by_file.get(fname, [])
        if not fdocs:
            no_chunks.append(it)
            continue
        # 1) 标签可达性：是否存在可判为黄金来源的块
        if not any(_gold_match(d, it) for d in fdocs):
            no_label_hit.append(it)
        # 2) key_points 可达性
        alltext = "\n".join(d.page_content for d in fdocs)
        pts = it.get("key_points") or []
        if pts:
            points_total += len(pts)
            cov, hits = _point_coverage(pts, alltext)
            miss = [p for p, ok in zip(pts, hits) if not ok]
            if miss:
                bad_points_items += 1
                bad_points_total += len(miss)
                it["_miss_points"] = miss

    print("\n" + "=" * 72)
    print(f" ① 源文件在库中无任何块（待修复/未入库）: {len(no_chunks)} 题")
    for it in no_chunks:
        print(f"    {it['id']}  {it.get('source_file')}")
    print(f"\n ② 标签逐字不可达（黄金来源永远判不中）: {len(no_label_hit)} 题")
    for it in no_label_hit:
        print(f"    {it['id']}  section={it.get('section')!r}  terms={it.get('match_terms')}")
        print(f"        Q: {it['question'][:56]}")
        print(f"        file: {it.get('source_file')}")
    print(f"\n ③ key_point 不可逐字匹配: {bad_points_total}/{points_total} "
          f"({bad_points_total / max(1, points_total):.1%})，涉及 {bad_points_items} 题")
    if args.only_failing:
        for it in items:
            if it.get("_miss_points"):
                print(f"    {it['id']}: {it['_miss_points']}")
    print("=" * 72)

    sys.exit(1 if (no_chunks or no_label_hit) else 0)


if __name__ == "__main__":
    main()
