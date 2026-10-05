# -*- coding: utf-8 -*-
"""
RAG 评测模块（面向对象）
--------------------------------------------------------------------
检索侧指标（对 vector / bm25 / hybrid / hybrid_rerank 四种模式做消融对比）：
  - Hit Rate@k   黄金来源是否出现在 top-k
  - MRR          黄金来源首次出现位置的倒数
  - Context Recall（确定性） 评分要点被检索上下文覆盖的比例
生成侧指标（默认只评 hybrid_rerank，节省免费 LLM 额度）：
  - Key-point Coverage  答案要点命中率（确定性，字符串匹配）
  - LLM-as-Judge        apodex 裁判：正确性 1~5 分 + 是否忠实于原文
  - Answer Accuracy     裁判 score>=4 视为正确
  - 平均端到端延迟
产物：
  evaluation/eval_report.json（机器可读，含逐条明细）
  evaluation/eval_report.md  （人类可读，表格汇总）
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
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings, settings
from hybrid_retriever import HybridRetriever
from rag_chain import EHSLLM, EHSRAGChain, RAGAnswer

logger = logging.getLogger(__name__)


# ====================================================================
# 工具
# ====================================================================
# 要点匹配归一化：法规文本里同一数值常有多种写法（中文数字/阿拉伯数字、
# PDF 提取产生的空格、全角字符），不归一会让 Context Recall 被系统性低估。
_CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "壹": 1, "二": 2, "两": 2, "贰": 2,
              "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
              "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8,
              "九": 9, "玖": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
_CN_RUN_RE = re.compile(r"[〇零一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖]{1,8}")
# 数字只有后接这些量词/单位时才转换（"一般""一律"等普通词不受影响）
_UNIT_GATE = set("人日年月个次分米mM%倍元级层跨度步道种台套处项条款段根组件张块号天周旬小时")
_FULLWIDTH_MAP = str.maketrans("０１２３４５６７８９％（）　，；：！？",
                               "0123456789%(),;:!? ")


def _cn_run_to_arabic(run: str) -> str:
    """中文数字串 -> 阿拉伯数字（三十->30、一百二十->120）；失败原样返回。"""
    total, num = 0, 0
    for ch in run:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return run
    total += num
    return str(total) if total else run


def normalize_for_match(s: str) -> str:
    """匹配前归一化：去空白、全角转半角、数字统一为阿拉伯数字。"""
    s = re.sub(r"\s+", "", s or "").translate(_FULLWIDTH_MAP)
    s = re.sub(r"[～－—‑–]", "-", s)

    def _rep(m: re.Match) -> str:
        nxt = s[m.end():m.end() + 1]
        return _cn_run_to_arabic(m.group(0)) if nxt and nxt in _UNIT_GATE \
            else m.group(0)
    return _CN_RUN_RE.sub(_rep, s)


def _gold_match(doc: Document, item: dict) -> bool:
    """判断检索结果是否为该题黄金来源。"""
    cid = item.get("source_chunk_id")
    if cid and doc.metadata.get("chunk_id") == cid:
        return True
    if doc.metadata.get("file_name") != item.get("source_file"):
        return False
    # 语义标签题（如 section="三同时"，源文档中无此字面词）用 match_terms
    # 给出的原文实词作锚点；锚点必须逐字存在于黄金块中
    terms = item.get("match_terms") or []
    if terms:
        dsec = normalize_for_match(doc.metadata.get("section_title") or "")
        body = normalize_for_match(doc.page_content)
        return any(t in dsec or t in body for t in map(normalize_for_match, terms))
    sec = normalize_for_match(item.get("section") or "")
    if not sec:
        return True   # 只标注了来源文件
    dsec = normalize_for_match(doc.metadata.get("section_title") or "")
    if dsec:
        # 条款号双向子串匹配（容忍"第五十九条"vs"第五十九"等写法）
        core = sec.replace("条", "").replace("章", "").replace("节", "")
        dcore = dsec.replace("条", "").replace("章", "").replace("节", "")
        if sec in dsec or dsec in sec or (core and core in dcore) \
                or (dcore and dcore in core):
            return True
    # 多条款合并块：黄金条款号/图表号出现在正文中即视为命中
    # （分块可能合并多个相邻条款，section_title 只记录首个条款号）
    return sec in normalize_for_match(doc.page_content)


def _point_coverage(points: Sequence[str], text: str) -> Tuple[float, List[bool]]:
    """要点在文本中的覆盖率（归一化后做子串匹配）。"""
    if not points:
        return 0.0, []
    ntext = normalize_for_match(text)
    hits = []
    for p in points:
        p = normalize_for_match(p.strip())
        if not p:
            continue
        hits.append(p in ntext)
    if not hits:
        return 0.0, []
    return sum(hits) / len(hits), hits


def _is_daily_rate_limit(exc: BaseException) -> bool:
    """遍历异常因果链，判断是否为 OpenRouter 免费模型每日额度 429。"""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur)
        if "429" in msg and ("free-models-per-day" in msg or "Rate limit" in msg):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# ====================================================================
# 检索评测
# ====================================================================
class RetrieverEvaluator:
    def __init__(self, retriever: HybridRetriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def evaluate(self, item: dict, mode: str) -> dict:
        ranked = self.retriever.retrieve(item["question"], mode=mode)
        top = ranked[: self.k]
        gold_ranks = [i + 1 for i, (d, _s) in enumerate(top) if _gold_match(d, item)]

        hit = bool(gold_ranks)
        rr = 1.0 / gold_ranks[0] if gold_ranks else 0.0

        context_text = "\n".join(d.page_content for d, _ in top)
        ctx_recall, _ = _point_coverage(item.get("key_points", []), context_text)
        return {"hit": hit, "rr": rr, "context_recall": ctx_recall,
                "n_retrieved": len(top)}


# ====================================================================
# 生成评测
# ====================================================================
JUDGE_SYSTEM = (
    "你是严格的 EHS（环境、健康、安全）问答评分专家。请依据【评分要点/参考答案】"
    "与【检索原文】评判【候选答案】。评分标准：5 分=完全正确且要点完整、无编造；"
    "4 分=正确但有轻微遗漏或表述瑕疵；3 分=部分正确；2 分=大部分错误；1 分=完全错误或"
    "脱离原文编造。若候选答案包含检索原文无法支撑的法规、数值，视为不忠实(faithful=false)。"
    '只输出 JSON：{"score":整数1-5,"correct":true/false,"faithful":true/false,"reason":"一句话理由"}'
)


class AnswerEvaluator:
    def __init__(self, config: Settings):
        self.config = config
        self.llm = EHSLLM(config)

    @staticmethod
    def key_point_coverage(item: dict, answer: str) -> float:
        cov, _ = _point_coverage(item.get("key_points", []), answer)
        return cov

    def llm_judge(self, item: dict, answer: str, contexts: List[str]) -> dict:
        points = "；".join(item.get("key_points", []))
        ref = item.get("answer", "")
        ctx = "\n---\n".join(contexts)[:6000]
        human = (
            f"【评分要点】{points}\n【参考答案】{ref}\n\n"
            f"【检索原文】\n{ctx}\n\n【候选答案】\n{answer}"
        )
        messages = [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=human)]
        try:
            raw = self.llm.invoke(messages)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            return {
                "score": int(data.get("score", 0)),
                "correct": bool(data.get("correct", False)),
                "faithful": bool(data.get("faithful", False)),
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            if _is_daily_rate_limit(e):
                raise  # 每日限额：交由上层批量任务终止
            logger.warning("裁判解析失败: %s", e)
            return {"score": 0, "correct": False, "faithful": False, "reason": f"judge_error:{e}"}


# ====================================================================
# 顶层评测器（门面）
# ====================================================================
@dataclass
class EvalSummary:
    n: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_recall: float = 0.0
    answer_accuracy: Optional[float] = None
    faithfulness: Optional[float] = None
    avg_judge_score: Optional[float] = None
    avg_keypoint_coverage: Optional[float] = None
    avg_latency: Optional[float] = None


class RAGEvaluator:
    """对外评测入口：
        evaluator = RAGEvaluator()
        report = evaluator.run(sample_size=60)
    """

    def __init__(self, config: Settings = settings):
        self.config = config
        self.rag = EHSRAGChain(config)
        self.retriever = self.rag.retriever
        self.ret_evaluator = RetrieverEvaluator(self.retriever, k=config.eval_cfg.hit_rate_k)
        self.ans_evaluator = AnswerEvaluator(config)
        self.report_json = Path(config.eval_dir) / config.eval_cfg.report_file
        self.report_md = Path(config.eval_dir) / config.eval_cfg.report_md

    # ----------------------------------------------------------------
    def _load_dataset(self, sample_size: int) -> List[dict]:
        from eval_dataset import EvalDatasetBuilder
        path = Path(self.config.eval_dir) / self.config.eval_cfg.dataset_file
        if not path.exists():
            raise FileNotFoundError("评测集不存在，请先运行 eval_dataset.build()")
        records = EvalDatasetBuilder(self.config).load()
        # 分层抽样：优先保证各类与种子/生成题均衡
        if sample_size and sample_size < len(records):
            seeds = [r for r in records if r.get("is_seed")]
            gens = [r for r in records if not r.get("is_seed")]
            random.seed(7)
            random.shuffle(gens)
            take_gen = max(0, sample_size - len(seeds))
            records = seeds + gens[:take_gen]
            records = records[:sample_size] if len(records) > sample_size else records
        logger.info("评测样本 %d 条", len(records))
        return records

    # ----------------------------------------------------------------
    def run(self, sample_size: Optional[int] = None,
            modes: Sequence[str] = ("vector", "bm25", "hybrid", "hybrid_rerank"),
            gen_mode: str = "hybrid_rerank",
            do_generation: bool = True) -> dict:
        sample_size = sample_size if sample_size is not None \
            else self.config.eval_cfg.default_sample_size
        records = self._load_dataset(sample_size)

        # 每条 x 每模式的检索结果
        per_item: List[dict] = []
        retrieval_acc: Dict[str, List[dict]] = {m: [] for m in modes}

        for idx, item in enumerate(records):
            row = {"id": item["id"], "question": item["question"],
                   "category": item["category"], "retrieval": {}, "generation": None}
            for mode in modes:
                r = self.ret_evaluator.evaluate(item, mode)
                row["retrieval"][mode] = r
                retrieval_acc[mode].append(r)
            per_item.append(row)
            logger.info("检索评测进度 %d/%d", idx + 1, len(records))

        # 生成评测（仅 gen_mode）
        gen_records: List[dict] = []
        if do_generation:
            for idx, item in enumerate(records):
                t0 = time.time()
                rag_answer: RAGAnswer = self.rag.answer(item["question"], mode=gen_mode)
                latency = time.time() - t0
                kpc = self.ans_evaluator.key_point_coverage(item, rag_answer.answer)
                judge = self.ans_evaluator.llm_judge(
                    item, rag_answer.answer, rag_answer.contexts)
                g = {"latency": round(latency, 2), "keypoint_coverage": round(kpc, 3),
                     "judge": judge, "pred_answer": rag_answer.answer,
                     "sources": rag_answer.sources}
                per_item[idx]["generation"] = g
                gen_records.append(g)
                logger.info("生成评测进度 %d/%d：score=%s correct=%s",
                            idx + 1, len(records), judge["score"], judge["correct"])

        report = self._build_report(records, retrieval_acc, gen_records, modes, gen_mode)
        report["details"] = per_item
        self._save(report)
        return report

    # ----------------------------------------------------------------
    def run_generation_only(self, gen_size: int = 40,
                            gen_mode: str = "hybrid_rerank") -> dict:
        """在已完成的全量检索评测（eval_report.json）上，增量补做生成评测。

        特性：
          - 断点续跑：启动时加载已有报告，自动跳过已完成样本；
          - 检查点：每完成 1 条立即落盘，中断/限流不丢已完成结果；
          - 遇 OpenRouter 每日限额（429 free-models-per-day）快速终止并保留成果，
            限额重置后重跑同一命令即可继续。
        """
        from eval_dataset import EvalDatasetBuilder
        if not self.report_json.exists():
            raise FileNotFoundError("请先运行：run_eval.py --sample 99 --no-generation")
        prev = json.load(open(self.report_json, encoding="utf-8"))
        details = prev["details"]
        dataset = {r["id"]: r for r in EvalDatasetBuilder(self.config).load()}

        # 固定随机顺序；断点续跑时只取尚未完成的
        random.seed(11)
        order = list(range(len(details)))
        random.shuffle(order)
        pending = [i for i in order if not details[i].get("generation")][:gen_size]

        daily_limited = False
        for idx in pending:
            row = details[idx]
            item = dataset[row["id"]]
            try:
                t0 = time.time()
                rag_answer: RAGAnswer = self.rag.answer(item["question"], mode=gen_mode)
                latency = time.time() - t0
                kpc = self.ans_evaluator.key_point_coverage(item, rag_answer.answer)
                judge = self.ans_evaluator.llm_judge(
                    item, rag_answer.answer, rag_answer.contexts)
            except Exception as e:  # noqa: BLE001
                if _is_daily_rate_limit(e):
                    logger.warning("触发 OpenRouter 每日限额，已暂停并保留 %d 条结果；"
                                   "限额重置后重跑本命令将自动续跑。",
                                   sum(1 for r in details if r.get("generation")))
                    daily_limited = True
                    break
                raise
            row["generation"] = {
                "latency": round(latency, 2), "keypoint_coverage": round(kpc, 3),
                "judge": judge, "pred_answer": rag_answer.answer,
                "sources": rag_answer.sources}
            # 检查点：每条立即落盘
            self._save(self._assemble_from_details(
                details, prev["modes"], gen_mode, daily_limited))
            logger.info("生成评测进度：累计 %d 条（本条 score=%s correct=%s）",
                        sum(1 for r in details if r.get("generation")),
                        judge["score"], judge["correct"])

        report = self._assemble_from_details(details, prev["modes"], gen_mode,
                                             daily_limited)
        self._save(report)
        return report

    # ----------------------------------------------------------------
    @staticmethod
    def _assemble_from_details(details: List[dict], modes: Sequence[str],
                               gen_mode: str, daily_limited: bool = False) -> dict:
        """从 details（含检索、可能含生成）汇总出完整报告。"""
        summaries: Dict[str, dict] = {}
        for m in modes:
            rows_m = [r["retrieval"][m] for r in details
                      if m in r.get("retrieval", {})]
            s = RAGEvaluator._summarize_retrieval(rows_m)
            summaries[m] = {"hit_rate": s.hit_rate, "mrr": s.mrr,
                            "context_recall": s.context_recall}

        gen_pairs = [(r, r["generation"]) for r in details if r.get("generation")]
        gen_summary = None
        by_cat: Dict[str, dict] = {}
        if gen_pairs:
            n = len(gen_pairs)
            scores = [g["judge"]["score"] for _r, g in gen_pairs
                      if g["judge"]["score"] > 0]
            gen_summary = {
                "answer_accuracy": round(
                    sum(g["judge"]["correct"] for _r, g in gen_pairs) / n, 4),
                "faithfulness": round(
                    sum(g["judge"]["faithful"] for _r, g in gen_pairs) / n, 4),
                "avg_judge_score": round(sum(scores) / len(scores), 3) if scores else 0,
                "avg_keypoint_coverage": round(
                    sum(g["keypoint_coverage"] for _r, g in gen_pairs) / n, 4),
                "avg_latency": round(
                    sum(g["latency"] for _r, g in gen_pairs) / n, 2),
            }
            cat_bucket: Dict[str, List[dict]] = {}
            for row, g in gen_pairs:
                cat_bucket.setdefault(row["category"], []).append(g)
            by_cat = {
                cat: {"n": len(gs),
                      "accuracy": round(
                          sum(x["judge"]["correct"] for x in gs) / len(gs), 4),
                      "avg_keypoint_coverage": round(
                          sum(x["keypoint_coverage"] for x in gs) / len(gs), 3)}
                for cat, gs in cat_bucket.items()}

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": len(details),
            "n_generation": len(gen_pairs),
            "modes": list(modes),
            "gen_mode": gen_mode if gen_pairs else None,
            "daily_limited": daily_limited,
            "retrieval_summary": summaries,
            "generation_summary": gen_summary,
            "category_summary": by_cat,
            "details": details,
        }

    # ----------------------------------------------------------------
    @staticmethod
    def _summarize_retrieval(rows: List[dict]) -> EvalSummary:
        n = len(rows)
        if not n:
            return EvalSummary()
        return EvalSummary(
            n=n,
            hit_rate=round(sum(r["hit"] for r in rows) / n, 4),
            mrr=round(sum(r["rr"] for r in rows) / n, 4),
            context_recall=round(sum(r["context_recall"] for r in rows) / n, 4),
        )

    def _build_report(self, records, retrieval_acc, gen_records, modes, gen_mode) -> dict:
        summaries: Dict[str, dict] = {}
        for mode in modes:
            s = self._summarize_retrieval(retrieval_acc[mode])
            summaries[mode] = {
                "hit_rate": s.hit_rate, "mrr": s.mrr, "context_recall": s.context_recall}

        gen_summary = None
        if gen_records:
            n = len(gen_records)
            scores = [g["judge"]["score"] for g in gen_records
                      if g["judge"]["score"] > 0]
            gen_summary = {
                "answer_accuracy": round(sum(g["judge"]["correct"] for g in gen_records) / n, 4),
                "faithfulness": round(sum(g["judge"]["faithful"] for g in gen_records) / n, 4),
                "avg_judge_score": round(sum(scores) / len(scores), 3) if scores else 0,
                "avg_keypoint_coverage": round(sum(g["keypoint_coverage"] for g in gen_records) / n, 4),
                "avg_latency": round(sum(g["latency"] for g in gen_records) / n, 2),
            }
            summaries[gen_mode].update(gen_summary)

        # 分类目准确率（基于裁判 correct）
        by_cat: Dict[str, dict] = {}
        if gen_records:
            cat_bucket: Dict[str, List[dict]] = {}
            for item, g in zip(records, gen_records):
                cat_bucket.setdefault(item["category"], []).append(g)
            for cat, gs in cat_bucket.items():
                by_cat[cat] = {
                    "n": len(gs),
                    "accuracy": round(sum(x["judge"]["correct"] for x in gs) / len(gs), 4),
                    "avg_keypoint_coverage": round(
                        sum(x["keypoint_coverage"] for x in gs) / len(gs), 3),
                }

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": len(records),
            "modes": list(modes),
            "gen_mode": gen_mode if gen_records else None,
            "retrieval_summary": summaries,
            "generation_summary": gen_summary,
            "category_summary": by_cat,
        }

    # ----------------------------------------------------------------
    def _save(self, report: dict) -> None:
        with open(self.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.report_md.write_text(self._to_markdown(report), encoding="utf-8")
        logger.info("评测报告已保存：%s / %s", self.report_json, self.report_md)

    @staticmethod
    def _to_markdown(report: dict) -> str:
        lines = ["# EHS 知识库 RAG 问答系统 —— 评测报告", "",
                 f"- 生成时间：{report['generated_at']}",
                 f"- 检索评测样本数：{report['n_samples']}",
                 f"- 检索模式：{', '.join(report['modes'])}"]
        if report.get("n_generation"):
            lines.append(f"- 生成评测样本数：{report['n_generation']}")
        if report.get("daily_limited"):
            lines.append("- 注意：生成评测因 OpenRouter 免费模型每日限额中断；限额重置"
                         "（北京时间次日 08:00）后重跑 `run_eval.py --gen-only` 即可自动续跑")
        lines.append("")

        lines += ["## 一、检索指标（消融对比）", "",
                  "| 检索模式 | Hit Rate@5 | MRR | Context Recall |",
                  "|---|---|---|---|"]
        for mode, s in report["retrieval_summary"].items():
            lines.append(f"| {mode} | {s['hit_rate']:.3f} | {s['mrr']:.3f} | {s['context_recall']:.3f} |")

        if report.get("generation_summary"):
            g = report["generation_summary"]
            lines += ["", f"## 二、生成指标（模式：{report['gen_mode']}，"
                          f"样本 {report.get('n_generation')} 条）", "",
                      "| 指标 | 数值 |", "|---|---|",
                      f"| 问答准确率（裁判判定正确） | **{g['answer_accuracy']:.1%}** |",
                      f"| 忠实度（无原文外编造） | {g['faithfulness']:.1%} |",
                      f"| 裁判平均分（1~5） | {g['avg_judge_score']:.2f} |",
                      f"| 答案要点覆盖率 | {g['avg_keypoint_coverage']:.1%} |",
                      f"| 平均端到端延迟（秒） | {g['avg_latency']:.1f} |"]

        if report.get("category_summary"):
            lines += ["", "## 三、分业务场景准确率", "",
                      "| 场景 | 样本数 | 准确率 | 要点覆盖率 |", "|---|---|---|---|"]
            for cat, s in report["category_summary"].items():
                lines.append(f"| {cat} | {s['n']} | {s['accuracy']:.1%} | {s['avg_keypoint_coverage']:.1%} |")
        lines.append("")
        return "\n".join(lines)
