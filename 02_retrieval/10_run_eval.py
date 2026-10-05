# -*- coding: utf-8 -*-
"""
评测入口脚本
--------------------------------------------------------------------
推荐流程（检索全量 + 生成抽样，节省免费 LLM 额度）：
  1) python run_eval.py --sample 99 --no-generation   # 全量检索消融（纯本地）
  2) python run_eval.py --gen-only --gen-size 40       # 在其基础上抽样做生成评测

也可一次跑完：
  python run_eval.py --sample 60                       # 检索+生成都用 60 条
  python run_eval.py --sample 30 --no-generation       # 只评检索
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
import logging

from evaluator import RAGEvaluator
from hybrid_retriever import HybridRetriever


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="EHS RAG 评测")
    parser.add_argument("--sample", type=int, default=None, help="评测样本数")
    parser.add_argument("--no-generation", action="store_true", help="仅评测检索")
    parser.add_argument("--gen-only", action="store_true",
                        help="在已有全量检索报告上增量补做生成评测")
    parser.add_argument("--gen-size", type=int, default=40, help="生成评测抽样条数")
    parser.add_argument("--gen-mode", default="hybrid_rerank",
                        choices=list(HybridRetriever.VALID_MODES),
                        help="生成评测所用检索模式")
    parser.add_argument("--modes", nargs="+",
                        default=["vector", "bm25", "hybrid", "hybrid_rerank"],
                        choices=list(HybridRetriever.VALID_MODES),
                        help="参与检索消融对比的模式")
    args = parser.parse_args()

    evaluator = RAGEvaluator()

    if args.gen_only:
        report = evaluator.run_generation_only(
            gen_size=args.gen_size, gen_mode=args.gen_mode)
    else:
        report = evaluator.run(
            sample_size=args.sample,
            modes=args.modes,
            gen_mode=args.gen_mode,
            do_generation=not args.no_generation,
        )

    print("\n" + "=" * 60)
    print(" 检索指标（Hit Rate@5 / MRR / Context Recall）")
    print("=" * 60)
    for mode, s in report["retrieval_summary"].items():
        print(f"  {mode:14s}  {s['hit_rate']:.3f} / {s['mrr']:.3f} / {s['context_recall']:.3f}")
    if report.get("generation_summary"):
        g = report["generation_summary"]
        print("\n" + "=" * 60)
        print(f" 生成指标（模式 {report['gen_mode']}，样本 {report.get('n_generation', report['n_samples'])} 条）")
        print("=" * 60)
        print(f"  问答准确率   : {g['answer_accuracy']:.1%}")
        print(f"  忠实度       : {g['faithfulness']:.1%}")
        print(f"  裁判平均分   : {g['avg_judge_score']:.2f} / 5")
        print(f"  要点覆盖率   : {g['avg_keypoint_coverage']:.1%}")
        print(f"  平均延迟     : {g['avg_latency']:.1f} s")
    print("\n报告已写入 evaluation/eval_report.md 与 eval_report.json")


if __name__ == "__main__":
    main()
