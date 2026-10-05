# -*- coding: utf-8 -*-
"""
EHS 知识库 RAG 问答系统 —— 应用编排与命令行入口（面向对象）
--------------------------------------------------------------------
把 文档加载 -> 语义分块 -> 向量入库 -> BM25 构建 -> 混合检索 -> 问答
完整串联，提供 CLI：

  # 1) 全量入库（D 盘 ehs_rag1 全部文件）
  python main.py ingest

  # 2) 仅入库某个子目录（冒烟/分批入库）
  python main.py ingest --subdir "01_法律"

  # 3) 单轮问答（默认 混合检索+rerank）
  python main.py ask "脚手架立杆垫板有什么要求？"
  python main.py ask "安全生产法规定从业人员有哪些义务？" --mode vector

  # 4) 交互式多轮问答
  python main.py chat

检索模式：vector / bm25 / hybrid / hybrid_rerank
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
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config import Settings, settings
from document_loader import EHSDocumentLoader
from hybrid_retriever import HybridRetriever
from rag_chain import EHSRAGChain, RAGAnswer
from text_splitter import EHSTextSplitter
from vector_store import EHSVectorStore


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ====================================================================
# 应用编排类
# ====================================================================
class EHSRAGApplication:
    """RAG 系统顶层应用类（门面），统一编排各组件。"""

    def __init__(self, config: Settings = settings):
        self.config = config
        self.loader = EHSDocumentLoader(config)
        self.splitter = EHSTextSplitter(config)
        self.vector_store = EHSVectorStore(config)
        self.retriever = HybridRetriever(config, vector_store=self.vector_store)
        self.rag = EHSRAGChain(config, retriever=self.retriever)

    # ----------------------------------------------------------------
    def ingest(self, subdir: Optional[str] = None,
               reset: bool = False) -> dict:
        """文档入库全流程。返回统计信息。
        策略：
          - reset=True：清空向量库与 BM25，全量重建。
          - 否则：基于 data/ingest_manifest.json 做增量 —— 跳过未变文件，
            变化文件先删旧块再重新入库，重复内容自动去重。
        """
        root = (Path(self.config.knowledge_base_dir) / subdir) if subdir \
            else self.config.knowledge_base_dir
        if reset:
            self.vector_store.reset()
            self.retriever.bm25.reset()

        manifest_path = self.config.data_dir / "ingest_manifest.json"
        manifest = self._load_manifest(manifest_path)

        logging.info("==== 1/4 加载文档（跳过未变文件） ====")
        raw_docs: List[Document] = []
        skipped = 0
        for doc in self.loader.load_directory(root):
            fp = Path(doc.metadata.get("source", ""))
            info = self._manifest_entry(manifest, fp)
            if info and self._file_unchanged(fp, info):
                skipped += 1
                continue
            # 内容变化：先删除该文件在向量库中的旧块
            if info:
                deleted = self.vector_store.delete_by_file(fp)
                if deleted:
                    logging.info("  清理旧块：%s 删除 %d 个", fp.name, deleted)
            raw_docs.append(doc)
        logging.info("已跳过 %d 个未变文件，加载 %d 个文件", skipped, len(raw_docs))

        logging.info("==== 2/4 语义分块 ====")
        chunks = self.splitter.split_documents(raw_docs)
        # 全局去重：相同 chunk_id 只保留一份（须先登记再判断，避免同批次重复 id）
        seen: set[str] = set()
        deduped: List[Document] = []
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            if cid:
                if cid in seen:
                    continue
                seen.add(cid)
            deduped.append(c)
        chunks = deduped

        logging.info("==== 3/4 向量入库（BGE + Chroma）====")
        n_added = self.vector_store.add_documents(chunks)

        logging.info("==== 4/4 增量合并 BM25 索引 ====")
        n_appended = self.retriever.bm25.append(chunks)

        self._update_manifest(manifest_path, manifest, chunks)

        # 索引发生变更：清空 Redis 检索/答案缓存，避免返回与新索引不一致的旧结果
        if raw_docs:
            try:
                from redis_cache import RAGRedisCache
                RAGRedisCache.get_instance(self.config).invalidate_all()
            except Exception:
                pass

        stats = {
            "source_dir": str(root),
            "raw_documents": len(raw_docs),
            "skipped": skipped,
            "chunks_new": len(chunks),
            "vectors_added": n_added,
            "bm25_appended": n_appended,
            "bm25_total": len(self.retriever.bm25.documents)
                          + len(self.retriever.bm25.staged_documents),
            "vectorstore_total": self.vector_store.count(),
        }
        logging.info("入库完成：%s", stats)
        return stats

    # ----------------------------------------------------------------
    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning("读取 manifest 失败，从全新开始：%s", e)
            return {}

    @staticmethod
    def _manifest_entry(manifest: Dict[str, dict], fp: Path) -> Optional[dict]:
        """取清单中该文件的记录：新格式键为完整路径，兼容旧格式（文件名）。"""
        return manifest.get(str(fp)) or manifest.get(fp.name)

    @staticmethod
    def _file_unchanged(fp: Path, info: dict) -> bool:
        """缓存命中：mtime_ns + size 一致即认为文件内容未变。"""
        try:
            st = fp.stat()
            return st.st_mtime_ns == info.get("mtime_ns") and st.st_size == info.get("size")
        except OSError:
            return False

    @staticmethod
    def _update_manifest(path: Path, manifest: Dict[str, dict], chunks: List[Document]) -> None:
        """按"文件 -> 本次生成的 chunk_ids"整体覆盖写入（键为完整路径）。

        覆盖而不是追加：文件内容变化后重建时，旧 chunk_id 不应残留。
        """
        by_file: Dict[str, List[str]] = {}
        for c in chunks:
            src = c.metadata.get("source", "")
            cid = c.metadata.get("chunk_id")
            if src and cid:
                by_file.setdefault(str(src), []).append(cid)
        for src, cids in by_file.items():
            fp = Path(src)
            try:
                st = fp.stat()
            except OSError:
                continue
            manifest[str(fp)] = {"file_name": fp.name,
                                 "mtime_ns": st.st_mtime_ns,
                                 "size": st.st_size, "chunk_ids": cids}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    @staticmethod
    def _print_answer(result: RAGAnswer) -> None:
        """格式化输出 RAG 答案与溯源信息（CLI / chat 复用）。"""
        print("\n" + result.answer)
        if result.sources:
            print("\n—— 溯源 ——")
            for s in result.sources:
                page = f" 第{s['page']}页" if s.get("page") else ""
                print(f"  · {s['file_name']} {s.get('section_title', '')}{page}")
        print(f"（耗时 {result.latency_seconds}s，模式 {result.mode}）")

    # ----------------------------------------------------------------
    def ask(self, question: str, mode: str = "hybrid_rerank") -> RAGAnswer:
        return self.rag.answer(question, mode=mode)

    def chat(self, mode: str = "hybrid_rerank") -> None:
        """交互式问答 REPL。"""
        print("=" * 64)
        print(" EHS 知识库问答（输入问题回车；:mode 切换检索模式；quit 退出）")
        print("=" * 64)
        while True:
            try:
                q = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.startswith(":mode"):
                mode = q.split()[-1]
                print("已切换模式：", mode)
                continue
            result = self.ask(q, mode=mode)
            self._print_answer(result)


# ====================================================================
# CLI
# ====================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EHS 知识库 RAG 问答系统")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="文档入库")
    p_ing.add_argument("--subdir", default=None, help="仅入库知识库下某子目录")
    p_ing.add_argument("--reset", action="store_true", help="入库前清空向量库")

    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--mode", default="hybrid_rerank",
                       choices=list(HybridRetriever.VALID_MODES))

    p_chat = sub.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--mode", default="hybrid_rerank",
                        choices=list(HybridRetriever.VALID_MODES))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    app = EHSRAGApplication()

    if args.command == "ingest":
        app.ingest(subdir=args.subdir, reset=args.reset)
    elif args.command == "ask":
        r = app.ask(args.question, mode=args.mode)
        app._print_answer(r)
    elif args.command == "chat":
        app.chat(mode=args.mode)


if __name__ == "__main__":
    main(sys.argv[1:])
