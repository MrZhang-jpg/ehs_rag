# -*- coding: utf-8 -*-
"""
Redis 缓存模块（检索侧基础设施）
--------------------------------------------------------------------
缓存两层结果，降低重复问题的 CPU/LLM 开销：
  · retrieval 检索层：(问题, mode) -> ranked [(Document, score)]，
    命中可跳过 向量/BM25 召回 与 Rerank 精排（CPU 上的主要耗时）；
  · answer 问答层：(问题, mode) -> RAGAnswer 字典，命中直接返回，
    连生成都跳过（answer 层的序列化由 rag_chain 负责，本模块提供
    通用 get_json / set_json，避免与 rag_chain 循环导入）。

健壮性：
  · 未安装 redis-py / Redis 服务未启动 / 任何读写异常 -> 自动旁路，
    问答主流程完全不受影响（enabled=False）；
  · 索引发生变更（ingest / reingest 修复）后，调用 invalidate_all()
    清空本前缀，避免返回与新索引不一致的旧答案。

配置见 config.RedisConfig；也可用环境变量覆盖：
  REDIS_URL / REDIS_HOST / REDIS_PORT / REDIS_DB / REDIS_PASSWORD /
  EHS_REDIS_DISABLED=1
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

import hashlib
import json
import logging
import os
import threading
import unicodedata
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import Settings, settings

logger = logging.getLogger(__name__)


class RAGRedisCache:
    """RAG 检索/问答结果的 Redis 缓存（进程内单例）。

    用法：
        cache = RAGRedisCache.get_instance()
        if cache.available():
            hit = cache.get_retrieval(question, mode)
            ...
        cache.invalidate_all()   # 索引变更后
    """

    _global_instance: Optional["RAGRedisCache"] = None
    _global_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: Settings = settings) -> "RAGRedisCache":
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    cls._global_instance = cls(config)
        return cls._global_instance

    # ----------------------------------------------------------------
    def __init__(self, config: Settings = settings):
        self.config = config
        rc = config.redis
        self.prefix = rc.key_prefix.strip().rstrip(":")
        self.answer_ttl = rc.answer_ttl
        self.retrieval_ttl = rc.retrieval_ttl
        self.client = None
        self.enabled = False

        if not rc.enabled or os.getenv("EHS_REDIS_DISABLED"):
            logger.info("Redis 缓存已关闭（config.redis.enabled=False）")
            return
        try:
            import redis
        except ImportError:
            logger.warning("未安装 redis-py，缓存旁路；执行 pip install redis 后启用")
            return

        try:
            common = dict(
                decode_responses=True,
                socket_timeout=rc.socket_timeout,
                socket_connect_timeout=rc.socket_connect_timeout,
            )
            url = os.getenv("REDIS_URL")
            if url:
                self.client = redis.Redis.from_url(url, **common)
                desc = "REDIS_URL"
            else:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST", rc.host),
                    port=int(os.getenv("REDIS_PORT", rc.port)),
                    db=int(os.getenv("REDIS_DB", rc.db)),
                    password=os.getenv("REDIS_PASSWORD") or rc.password or None,
                    **common,
                )
                desc = f"{os.getenv('REDIS_HOST', rc.host)}:{os.getenv('REDIS_PORT', rc.port)}"
            self.client.ping()
            self.enabled = True
            logger.info("Redis 缓存已启用（%s），键前缀「%s」", desc, self.prefix)
        except Exception as e:
            # 连接失败：静默旁路，仅 warning，不影响后续问答
            logger.warning("Redis 连接失败，缓存旁路（不影响问答）：%s", e)
            self.client = None
            self.enabled = False

    # ----------------------------------------------------------------
    # 键构造
    # ----------------------------------------------------------------
    @staticmethod
    def normalize_question(question: str) -> str:
        """问题归一化：NFKC 兼容分解、小写、折叠空白，提高缓存命中率。"""
        q = unicodedata.normalize("NFKC", question).strip().lower()
        return " ".join(q.split())

    def make_key(self, kind: str, question: str, mode: str) -> str:
        payload = f"{self.normalize_question(question)}|{mode}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
        return f"{self.prefix}:{kind}:{digest}"

    # ----------------------------------------------------------------
    # 通用 JSON 读写（answer 层复用）
    # ----------------------------------------------------------------
    @staticmethod
    def _json_safe(obj):
        """递归把非 JSON 原生类型（Path 等）转为字符串。"""
        if isinstance(obj, dict):
            return {str(k): RAGRedisCache._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RAGRedisCache._json_safe(v) for v in obj]
        if isinstance(obj, bool) or obj is None:
            return obj
        if isinstance(obj, (str, int, float)):
            return obj
        return str(obj)

    def get_json(self, key: str):
        if not self.enabled:
            return None
        try:
            raw = self.client.get(key)
        except Exception as e:
            logger.debug("Redis 读取异常，按未命中处理：%s", e)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_json(self, key: str, obj, ttl: int) -> bool:
        if not self.enabled:
            return False
        try:
            payload = json.dumps(self._json_safe(obj), ensure_ascii=False)
            if ttl and ttl > 0:
                self.client.set(key, payload, ex=int(ttl))
            else:
                self.client.set(key, payload)
            return True
        except Exception as e:
            logger.debug("Redis 写入异常，跳过缓存：%s", e)
            return False

    # ----------------------------------------------------------------
    # 检索结果缓存（ranked: List[Tuple[Document, score]]）
    # ----------------------------------------------------------------
    @staticmethod
    def encode_ranked(ranked: List[Tuple[Document, float]]) -> list:
        return [
            {"page_content": d.page_content, "metadata": d.metadata, "score": float(s)}
            for d, s in ranked
        ]

    @staticmethod
    def decode_ranked(data: list) -> List[Tuple[Document, float]]:
        out: List[Tuple[Document, float]] = []
        for item in data or []:
            meta = item.get("metadata") or {}
            doc = Document(page_content=item.get("page_content", ""), metadata=meta)
            out.append((doc, float(item.get("score", 0.0))))
        return out

    def get_retrieval(self, question: str, mode: str):
        data = self.get_json(self.make_key("retrieval", question, mode))
        return self.decode_ranked(data) if data is not None else None

    def set_retrieval(self, question: str, mode: str,
                      ranked: List[Tuple[Document, float]]) -> None:
        self.set_json(
            self.make_key("retrieval", question, mode),
            self.encode_ranked(ranked), self.retrieval_ttl,
        )

    # ----------------------------------------------------------------
    # 失效：索引变更后清空本前缀
    # ----------------------------------------------------------------
    def invalidate_all(self) -> int:
        """删除本系统前缀下的全部缓存键，返回删除条数。"""
        if not self.enabled:
            return 0
        deleted = 0
        try:
            batch: List[str] = []
            for key in self.client.scan_iter(f"{self.prefix}:*", count=200):
                batch.append(key)
                if len(batch) >= 200:
                    deleted += int(self.client.delete(*batch))
                    batch = []
            if batch:
                deleted += int(self.client.delete(*batch))
            logger.info("检测到索引变更，已清空 Redis 缓存 %d 条", deleted)
        except Exception as e:
            logger.warning("清空 Redis 缓存失败（可手动 FLUSHDB）：%s", e)
        return deleted

    def available(self) -> bool:
        return self.enabled
