# -*- coding: utf-8 -*-
"""
混合检索模块（面向对象）
--------------------------------------------------------------------
通道：
  1) 向量检索（Chroma + 本地 BGE，语义召回）
  2) BM25 关键词检索（jieba 分词 + rank_bm25，精确术语/法规编号召回）
融合：
  RRF（Reciprocal Rank Fusion）对两路排名倒数加权融合，无需分数标定。
重排：
  BAAI/bge-reranker-base CrossEncoder 对融合候选做 query-doc 精排。
检索模式（便于消融评测）：
  vector / bm25 / hybrid / hybrid_rerank（默认）
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


import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from config import Settings, settings, resolve_device
from vector_store import EHSVectorStore

logger = logging.getLogger(__name__)

# BM25 索引持久化文件
BM25_INDEX_FILE = "bm25_index.pkl"
BM25_STAGED_FILE = "bm25_staged.pkl"

# 轻量标点停用集合。注意保留数字：法规条款号（"第5条"）、标准编号、参数值
# （"3.6m"、"30日"）都依赖数字 token 做精确召回，早期版本误将单个数字一并过滤，
# 导致"第5条"这类查询的编号线索丢失。
_PUNCT = set(" \n\r\t，。；：、！？“”‘’\"'（）()[]【】《》〈〉—-·.,;:!?/\\|")


def jieba_tokenize(text: str) -> List[str]:
    """jieba 中文分词，去空白与纯标点（保留数字）。"""
    import jieba
    jieba.setLogLevel(logging.WARN)
    return [t for t in jieba.lcut(text) if t.strip() and t not in _PUNCT]


# ====================================================================
# 1) BM25 检索器
# ====================================================================
class BM25Retriever:
    """BM25 关键词检索器，支持 pickle 持久化与增量追加。
    设计（避免重复分词与重建）：
      - 全量构建(build)：对语料分词并构建 rank_bm25 索引，持久化 bm25_index.pkl。
      - 增量追加(append)：新文档只分词后暂存 bm25_staged.pkl，不重做全量分词与建索引。
      - 首次检索时自动把暂存块与全量合并、构建/重建索引并持久化，之后搜索走内存索引。
    """

    def __init__(self, config: Settings = settings):
        self.config = config
        self.documents: List[Document] = []         # 全量文档（含已合并的暂存）
        self.corpus_tokens: List[List[str]] = []    # 全量分词结果
        self.staged_documents: List[Document] = []  # 尚未合并的暂存文档
        self.staged_tokens: List[List[str]] = []    # 尚未合并的暂存分词
        self.bm25 = None
        self._index_path = Path(self.config.bm25_dir) / BM25_INDEX_FILE
        self._staged_path = Path(self.config.bm25_dir) / BM25_STAGED_FILE

    # ----------------------------------------------------------------
    def reset(self) -> None:
        """清空索引（用于 --reset 场景）。"""
        for _p in (self._index_path, self._staged_path):
            try:
                _p.unlink()
            except OSError:
                pass
        self.documents = []
        self.corpus_tokens = []
        self.staged_documents = []
        self.staged_tokens = []
        self.bm25 = None

    # ----------------------------------------------------------------
    def build(self, chunks: Sequence[Document]) -> None:
        """全量构建（用于 --reset 等从头入库场景）。"""
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        logger.info("BM25 全量构建中，对 %d 个块分词 ...", len(chunks))
        self.corpus_tokens = [jieba_tokenize(c.page_content) for c in self.documents]
        self.staged_documents = []
        self.staged_tokens = []
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.save()
        self._remove_staged()
        logger.info("BM25 全量构建完成并持久化：%d 块。", len(self.documents))

    def append(self, chunks: Sequence[Document]) -> int:
        """增量追加新块：只对新块分词后暂存，不重建全量索引。"""
        if not chunks:
            return 0
        if not self.documents:
            self.load()
        existing_ids = {d.metadata.get("chunk_id") for d in self.documents
                        if d.metadata.get("chunk_id")}
        n = 0
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            if cid and cid in existing_ids:
                continue
            self.staged_documents.append(c)
            self.staged_tokens.append(jieba_tokenize(c.page_content))
            if cid:
                existing_ids.add(cid)
            n += 1
        self._save_staged()
        logger.info("BM25 增量暂存 %d 个块（首次检索时自动合并构建索引）。", n)
        return n

    def _merge_staged(self) -> None:
        """把暂存块合并进全量，并构建/重建 rank_bm25 索引（仅在首次检索时调用）。"""
        from rank_bm25 import BM25Okapi
        self.documents = self.documents + self.staged_documents
        self.corpus_tokens = self.corpus_tokens + self.staged_tokens
        logger.info("BM25 合并暂存块 %d 个，正在构建索引 ...", len(self.staged_documents))
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.staged_documents = []
        self.staged_tokens = []
        self.save()
        self._remove_staged()
        logger.info("BM25 索引已构建并持久化：%d 块。", len(self.documents))

    def save(self) -> None:
        with open(self._index_path, "wb") as f:
            pickle.dump({"documents": self.documents,
                         "corpus_tokens": self.corpus_tokens,
                         "bm25": self.bm25}, f)

    def _save_staged(self) -> None:
        with open(self._staged_path, "wb") as f:
            pickle.dump({"documents": self.staged_documents,
                         "tokens": self.staged_tokens}, f)

    def _remove_staged(self) -> None:
        try:
            self._staged_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ----------------------------------------------------------------
    def load(self) -> bool:
        """加载全量索引；若暂存文件存在，自动合并进全量并持久化。"""
        staged_ok = False
        if self._staged_path.exists():
            with open(self._staged_path, "rb") as f:
                staged = pickle.load(f)
            self.staged_documents = staged.get("documents", [])
            self.staged_tokens = staged.get("tokens", [])
            staged_ok = True
        if self._index_path.exists():
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.corpus_tokens = data.get("corpus_tokens", [])
            self.bm25 = data.get("bm25")
        if staged_ok:
            self._merge_staged()
            return True
        return bool(self._index_path.exists())

    # ----------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        if self.bm25 is None:
            self.load()
        if self.staged_documents:
            self._merge_staged()
        tokens = jieba_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.documents[i], float(s)) for i, s in ranked if s > 0]


# ====================================================================
# 2) RRF 融合
# ====================================================================
class RRFusion:
    """Reciprocal Rank Fusion。"""

    @staticmethod
    def fuse(ranked_lists: List[List[Tuple[Document, float]]],
             rrf_k: int = 60,
             topn: int = 20) -> List[Tuple[Document, float]]:
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, (doc, _score) in enumerate(ranked):
                cid = doc.metadata.get("chunk_id") or doc.page_content[:50]
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map[cid] = doc
        ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[cid], score) for cid, score in ordered[:topn]]


# ====================================================================
# 3) BGE Reranker（CrossEncoder，可选）
# ====================================================================
class BGEReranker:
    """BGE CrossEncoder 重排序器（线程安全单例，惰性加载）。"""

    _instance: Optional["BGEReranker"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Settings = settings):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst.config = config
                    inst.model = None
                    cls._instance = inst
        return cls._instance

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            rc = self.config.reranker
            device = resolve_device(rc.device)
            self.device = device
            logger.info("加载本地 Reranker：%s (device=%s)", rc.model_name, device)
            self.model = CrossEncoder(rc.model_name, device=device, max_length=rc.max_length)

    def rerank(self, query: str, candidates: List[Document]) -> List[Tuple[Document, float]]:
        if not candidates:
            return []
        self._ensure_model()
        rc = self.config.reranker
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(pairs, batch_size=rc.batch_size)
        ranked = sorted(zip(candidates, [float(s) for s in scores]),
                        key=lambda x: x[1], reverse=True)
        return ranked


# ====================================================================
# 4) 混合检索器（门面 / 核心）
# ====================================================================
class HybridRetriever:
    """对外统一检索入口：
        retriever = HybridRetriever()
        retriever.build_index(chunks)      # 构建 BM25（向量库另行入库）
        results = retriever.retrieve(query, mode="hybrid_rerank")
    """

    VALID_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")

    def __init__(self, config: Settings = settings,
                 vector_store: Optional[EHSVectorStore] = None):
        self.config = config
        self.vector_store = vector_store or EHSVectorStore(config)
        self.bm25 = BM25Retriever(config)
        self._reranker: Optional[BGEReranker] = None

    # ----------------------------------------------------------------
    def build_index(self, chunks: Sequence[Document]) -> None:
        self.bm25.build(chunks)

    def load_indexes(self) -> None:
        self.bm25.load()

    # ----------------------------------------------------------------
    def _vector_search(self, query: str) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_scores(
            query, k=self.config.retrieval.vector_top_k)

    def _bm25_search(self, query: str) -> List[Tuple[Document, float]]:
        return self.bm25.search(query, k=self.config.retrieval.bm25_top_k)

    # ----------------------------------------------------------------
    def retrieve(self, query: str, mode: str = "hybrid_rerank"
                 ) -> List[Tuple[Document, float]]:
        if mode not in self.VALID_MODES:
            raise ValueError(f"未知检索模式 {mode}，可选 {self.VALID_MODES}")

        # ---- Redis 检索缓存：命中则跳过 召回 / RRF / rerank 全流程 ----
        cache = None
        try:
            from redis_cache import RAGRedisCache
            cache = RAGRedisCache.get_instance(self.config)
            if cache.available():
                hit = cache.get_retrieval(query, mode)
                if hit is not None:
                    logger.info("检索缓存命中（%s）：%s", mode, query[:30])
                    return hit
        except Exception:
            cache = None

        rc = self.config.retrieval

        if mode == "vector":
            results = self._vector_search(query)
        elif mode == "bm25":
            results = self._bm25_search(query)
        else:
            vec = self._vector_search(query)
            kw = self._bm25_search(query)
            results = RRFusion.fuse(
                [vec, kw], rrf_k=rc.rrf_k,
                topn=max(rc.rerank_candidates, rc.final_top_k))
            if mode == "hybrid_rerank" and rc.use_reranker:
                cands = [d for d, _ in results[:rc.rerank_candidates]]
                results = BGEReranker(self.config).rerank(query, cands)

        # 分数阈值过滤（仅对 rerank 后的 raw logits 宽松处理）
        if mode == "hybrid_rerank":
            results = [(d, s) for d, s in results]  # bge-reranker 分数可负，不做硬阈值
        final_results = results[:rc.final_top_k if mode != "vector" and mode != "bm25"
                                else max(rc.vector_top_k, rc.final_top_k)]

        # ---- 写入检索缓存（异常不影响返回）----
        if cache is not None:
            try:
                cache.set_retrieval(query, mode, final_results)
            except Exception:
                pass
        return final_results

    def retrieve_documents(self, query: str, mode: str = "hybrid_rerank"
                           ) -> List[Document]:
        return [d for d, _ in self.retrieve(query, mode=mode)]
