# -*- coding: utf-8 -*-
"""
本地 BGE 嵌入模型封装（面向对象，CPU 运行）
--------------------------------------------------------------------
- 加载 HuggingFace 上的 BAAI/bge-base-zh-v1.5（首次自动下载，之后走本地缓存）；
- 实现 LangChain Embeddings 标准接口，可直接喂给 Chroma / Retriever；
- 查询侧自动拼接 BGE 官方中文 instruction，文档侧不加（官方推荐用法）；
- 单例加载，避免同一进程内重复占用内存。
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
import threading
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from config import Settings, settings, resolve_device

logger = logging.getLogger(__name__)


class _InstructionEmbeddings(Embeddings):
    """Embeddings 装饰器：查询侧自动拼接指令（用于 BGE 检索），文档侧原样透传。"""

    def __init__(self, base: Embeddings, query_instruction: str = ""):
        self.base = base
        self.query_instruction = query_instruction

    @staticmethod
    def _sanitize(text: str) -> str:
        """防御性清洗：确保为 str，并清除孤立代理字符。"""
        if not isinstance(text, str):
            text = str(text)
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [self._sanitize(t) for t in texts]
        # 过滤清洗后为空的文本，避免 tokenizer 报错
        return self.base.embed_documents([t or " " for t in texts])

    def embed_query(self, text: str) -> List[float]:
        text = self._sanitize(text)
        if self.query_instruction and not text.startswith(self.query_instruction):
            text = self.query_instruction + text
        return self.base.embed_query(text)


class BGEEmbeddingModel:
    """BGE 嵌入模型管理器（线程安全单例）。"""

    _instance: Optional["BGEEmbeddingModel"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Settings = settings) -> "BGEEmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init_model(config)
                    cls._instance = inst
        return cls._instance

    def _init_model(self, config: Settings) -> None:
        self.config = config
        ec = config.embedding
        device = resolve_device(ec.device)
        self.device = device
        logger.info("加载本地 BGE 嵌入模型：%s (device=%s)", ec.model_name, device)
        from langchain_huggingface import HuggingFaceEmbeddings

        base = HuggingFaceEmbeddings(
            model_name=ec.model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": ec.normalize_embeddings,
                "batch_size": ec.batch_size,
            },
        )
        # 新版 HuggingFaceEmbeddings 移除了 instruction 参数；
        # 用装饰器在查询侧拼接 BGE 官方中文指令，文档侧不加（bge 官方推荐用法）。
        self._embeddings: Embeddings = _InstructionEmbeddings(
            base=base, query_instruction=ec.query_instruction)
        # 自检
        v = self._embeddings.embed_query("自检：安全生产")
        logger.info("BGE 模型加载完成，向量维度 = %d", len(v))

    # ----------------------------------------------------------------
    @property
    def embeddings(self) -> Embeddings:
        """返回 LangChain Embeddings 对象。"""
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)


def get_embeddings(config: Settings = settings) -> Embeddings:
    """便捷函数：拿到 LangChain Embeddings 单例。"""
    return BGEEmbeddingModel(config).embeddings
