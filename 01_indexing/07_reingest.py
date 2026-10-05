# -*- coding: utf-8 -*-
"""
索引质量修复脚本：自动识别并重建"乱码/噪声"文件的向量块
--------------------------------------------------------------------
背景：部分 PDF 的文本层不可用（扫描版、CID 字体乱码、仅含水印），或
pdfplumber 在图片页上误检表格，导致入库的是无意义内容 —— 这类文件无论
用什么检索模式都召回不到，是检索召回率的主要损失来源。

本脚本流程：
  1) 体检：扫描向量库中每个文件的已入库块，按"汉字占比 / 表格碎片密度 /
     块数"判定低质量文件（无需人工列名单）；
  2) 清理：删除这些文件在向量库中的旧块；
  3) 重建：用修复后的解析链路（逐页 OCR 兜底 + 假表格过滤）并行重解析，
     分块后重新入库（chunk_id 确定性，天然去重）；
  4) 重建 BM25 索引（按向量库全量内容，消除删除带来的悬空文档）。

用法：
    python reingest.py --dry-run          # 只体检，列出将修复的文件
    python reingest.py                    # 体检并修复
    python reingest.py --files a.pdf b.pdf  # 只修复指定文件
    python reingest.py --workers 6        # 指定 OCR 并行进程数
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
import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from config import Settings, settings
from document_loader import cjk_ratio, table_looks_garbage
from vector_store import EHSVectorStore

logger = logging.getLogger(__name__)

# 判定阈值
MIN_CJK_RATIO = 0.40           # 文件级汉字占比下限（乱码文件远低于此值）
MAX_FRAGMENT_PER_CHUNK = 3.0   # 平均每块允许的表格碎片数上限
# "可读块"判定：中英双语规范（英文目次页汉字 0%）、含表格/水印的文件
# 会拉低整体占比，但只要存在足量可读中文块，检索就是可达的 —— 应以
# "可读块数量"为准，占比/碎片只作为辅助理由，避免误报触发无谓重入库。
READABLE_MIN_LEN = 150         # 可读块最小长度
READABLE_MIN_CJK = 0.50        # 可读块汉字占比下限
MIN_READABLE_RATIO = 0.05      # 可读块占比下限（至少 max(2, 5%)）


# ====================================================================
# 1) 体检：找出需要修复的文件
# ====================================================================
def _fragment_count(text: str) -> int:
    """统计文本中"单字单元格"表格碎片数量（pdfplumber 噪声特征）。"""
    import re
    return len(re.findall(r"\|\s*[一-鿿]\s*\|\s*[一-鿿]\s*\|", text))


def diagnose(docs: List[Document]) -> Dict[str, dict]:
    """按文件聚合已入库块的质量指标。

    返回 {source_path: {"chunks": n, "cjk_ratio": r, "frag_per_chunk": f,
                        "bad": bool, "reason": str}}
    """
    agg: Dict[str, dict] = {}
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_name") or "?"
        a = agg.setdefault(src, {"chunks": 0, "cjk": 0.0, "chars": 0,
                                 "frags": 0, "readable": 0})
        a["chunks"] += 1
        a["chars"] += len(d.page_content)
        n = len(d.page_content)
        if n:
            r = cjk_ratio(d.page_content)
            a["cjk"] += r * n                              # 按字符数加权
            if n >= READABLE_MIN_LEN and r >= READABLE_MIN_CJK:
                a["readable"] += 1
        a["frags"] += _fragment_count(d.page_content)

    out: Dict[str, dict] = {}
    for src, a in agg.items():
        chars = max(1, a["chars"])
        ratio = a["cjk"] / chars
        frag = a["frags"] / max(1, a["chunks"])
        readable = a["readable"]
        need_readable = max(2, int(a["chunks"] * MIN_READABLE_RATIO))
        reasons = []
        # 核心判据：可读中文块过少（真乱码文件全册无块达标）；
        # 占比/碎片仅作辅助证据，避免双语/含表格文件被误伤
        if readable < need_readable and (ratio < MIN_CJK_RATIO
                                         or frag > MAX_FRAGMENT_PER_CHUNK):
            if ratio < MIN_CJK_RATIO:
                reasons.append(f"汉字占比 {ratio:.1%}（疑似乱码）")
            if frag > MAX_FRAGMENT_PER_CHUNK:
                reasons.append(f"表格碎片 {frag:.1f}/块（疑似假表格噪声）")
            reasons.append(f"可读块仅 {readable}/{a['chunks']}")
        out[src] = {"chunks": a["chunks"], "cjk_ratio": ratio,
                    "frag_per_chunk": frag, "readable": readable,
                    "bad": bool(reasons), "reason": "；".join(reasons)}
    return out


def find_bad_files(store: EHSVectorStore,
                   config: Settings = settings) -> List[Tuple[Path, dict]]:
    """返回 [(文件路径, 诊断信息)]：低质量文件 + 知识库中存在但库中缺失的文件。

    "缺失"检测用于自愈中断的入库/修复（上一轮修复中断时，已删块未重入库的
    文件在库中完全没有记录，仅靠质量体检发现不了）。
    """
    docs = store.all_documents()
    diag = diagnose(docs)
    bad: List[Tuple[Path, dict]] = []
    for src, info in sorted(diag.items()):
        if info["bad"]:
            bad.append((Path(src), info))

    # 缺失检测以入库清单为准（清单记录了"曾入库过"的文件），避免把知识库
    # 目录里的 README 等说明文件误当法规入库
    present = set(diag)
    present_names = {Path(s).name for s in diag}
    manifest_path = Path(config.data_dir) / "ingest_manifest.json"
    missing = 0
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("读取入库清单失败：%s", e)
            manifest = {}
        for src in sorted(manifest):
            fp = Path(src)
            if not fp.is_file():
                continue
            if str(fp) in present or fp.name in present_names:
                continue
            bad.append((fp, {"chunks": 0, "cjk_ratio": 0.0, "frag_per_chunk": 0.0,
                             "bad": True, "reason": "向量库中缺失（上次入库/修复未完成）"}))
            missing += 1

    logger.info("体检完成：%d 个文件有块，低质量 %d 个，库中缺失 %d 个",
                len(diag), len(bad) - missing, missing)
    return bad


# ====================================================================
# 2) 并行重解析（含 OCR 兜底）
# ====================================================================
PARSE_CACHE_DIR = settings.data_dir / "parse_cache"


def _cache_path(path_str: str) -> Path:
    """解析结果缓存路径：以 路径+mtime+大小 为键，源文件变化自动失效。"""
    st = Path(path_str).stat()
    key = hashlib.sha1(
        f"{path_str}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    return PARSE_CACHE_DIR / f"{key}.pkl"


def _parse_worker(path_str: str) -> tuple:
    """多进程 worker：解析单个文件（必要时自动 OCR），结果落盘缓存。

    OCR 是分钟级重操作：缓存可保证后续流程（如入库因故中断）重跑时
    无需重新 OCR —— 上一版因解析结果只在内存中，崩溃即丢失 90 分钟算力。
    """
    cp = _cache_path(path_str)
    if cp.exists():
        try:
            with open(cp, "rb") as f:
                return path_str, pickle.load(f)
        except Exception:  # 缓存损坏：回退重新解析
            logger.warning("解析缓存读取失败，重新解析：%s", Path(path_str).name)
    from document_loader import EHSDocumentLoader
    loader = EHSDocumentLoader()
    docs = loader.load_file(path_str)
    items = [(d.page_content, d.metadata) for d in docs]
    try:
        PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(items, f)
    except Exception as e:  # 缓存写失败不影响主流程
        logger.warning("解析缓存写入失败：%s", e)
    return path_str, items


def parse_files_parallel(files: List[Path], workers: int) -> Dict[str, List[Document]]:
    """并行解析文件（OCR 是 CPU 密集操作，多进程可线性加速）。"""
    out: Dict[str, List[Document]] = {}
    if not files:
        return out
    if workers <= 1 or len(files) == 1:
        for fp in files:
            _, items = _parse_worker(str(fp))
            out[str(fp)] = [Document(page_content=c, metadata=m) for c, m in items]
        return out

    logger.info("并行解析 %d 个文件（%d 进程，含 OCR）...", len(files), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_parse_worker, str(fp)): fp for fp in files}
        for i, fut in enumerate(as_completed(futures), 1):
            src, items = fut.result()
            out[src] = [Document(page_content=c, metadata=m) for c, m in items]
            logger.info("解析完成 %d/%d：%s（%d 页有内容）",
                        i, len(files), Path(src).name, len(items))
    return out


# ====================================================================
# 3) 主流程
# ====================================================================
def run(config: Settings = settings,
        files: Optional[List[Path]] = None,
        dry_run: bool = False,
        workers: Optional[int] = None,
        rebuild_bm25: bool = True) -> dict:
    from main import EHSRAGApplication

    app = EHSRAGApplication(config)
    workers = workers or config.ocr.workers

    # ---- 1) 体检 ----
    if files:
        targets = [(fp, {"chunks": "?", "chunks_str": "", "reason": "手动指定"})
                   for fp in files]
        diag = {}
    else:
        bad = find_bad_files(app.vector_store)
        diag = {str(fp): info for fp, info in bad}
        targets = bad

    if not targets:
        logger.info("未发现需要修复的文件，索引质量良好。")
        return {"repaired": [], "chunks_added": 0}

    print("\n" + "=" * 70)
    print(" 待修复文件（入库内容为乱码/噪声，检索不可达）")
    print("=" * 70)
    for fp, info in targets:
        n = info.get("chunks", "?")
        print(f"  [{n} 块] {fp.name}")
        if info.get("reason"):
            print(f"          {info['reason']}")
    print("=" * 70)

    if dry_run:
        logger.info("--dry-run：仅体检，不修改索引。")
        return {"repaired": [str(fp) for fp, _ in targets], "chunks_added": 0}

    # ---- 2) 清理旧块 ----
    for fp, _ in targets:
        n = app.vector_store.delete_by_file(fp)
        if n:
            logger.info("清理旧块：%s（%d 个）", fp.name, n)

    # ---- 3) 并行重解析 + 分块 + 入库 ----
    t0 = time.time()
    parsed = parse_files_parallel([fp for fp, _ in targets], workers)
    all_chunks: List[Document] = []
    failed: List[str] = []
    for fp, _ in targets:
        docs = parsed.get(str(fp), [])
        if not docs:
            logger.warning("重解析后无内容（文件可能损坏）：%s", fp.name)
            failed.append(fp.name)
            continue
        try:
            chunks = app.splitter.split_documents(docs)
            # 单文件内按 chunk_id 去重（不同页可能产出同 id 的重复块）
            seen: Set[str] = set()
            uniq: List[Document] = []
            for c in chunks:
                cid = c.metadata.get("chunk_id")
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                uniq.append(c)
            n_vec = app.vector_store.add_documents(uniq)
            all_chunks.extend(uniq)
            logger.info("重建入库：%s -> %d 页文档 / %d 块 / 新增 %d",
                        fp.name, len(docs), len(uniq), n_vec)
        except Exception as e:  # 单文件失败不中断整体修复（其余文件继续入库）
            logger.error("重建入库失败（跳过，可重跑本脚本补齐）：%s：%s", fp.name, e)
            failed.append(fp.name)

    # ---- 4) 重建 BM25（删除操作无法增量维护，按全量重建） ----
    if rebuild_bm25:
        logger.info("重建 BM25 索引（全量 %d 块）...", app.vector_store.count())
        app.retriever.bm25.build(app.vector_store.all_documents())

    # ---- 5) 清单同步 ----
    manifest_path = config.data_dir / "ingest_manifest.json"
    manifest = app._load_manifest(manifest_path)
    app._update_manifest(manifest_path, manifest, all_chunks)

    # ---- 6) 索引已修复：清空 Redis 缓存，避免旧乱码时期的结果残留 ----
    try:
        from redis_cache import RAGRedisCache
        RAGRedisCache.get_instance(config).invalidate_all()
    except Exception:
        pass

    result = {
        "repaired": [str(fp) for fp, _ in targets],
        "failed": failed,
        "chunks_added": len(all_chunks),
        "vectorstore_total": app.vector_store.count(),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    logger.info("修复完成：%s", result)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="索引质量修复（乱码/噪声文件重建）")
    parser.add_argument("--dry-run", action="store_true", help="只体检不改索引")
    parser.add_argument("--files", nargs="*", default=None,
                        help="只修复指定文件（完整路径或相对知识库路径）")
    parser.add_argument("--workers", type=int, default=None, help="OCR 并行进程数")
    parser.add_argument("--no-bm25", action="store_true", help="跳过 BM25 重建")
    args = parser.parse_args()

    files = None
    if args.files:
        files = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                p = settings.knowledge_base_dir / f
            if not p.exists():
                raise FileNotFoundError(f)
            files.append(p)

    run(files=files, dry_run=args.dry_run, workers=args.workers,
        rebuild_bm25=not args.no_bm25)


if __name__ == "__main__":
    main()
