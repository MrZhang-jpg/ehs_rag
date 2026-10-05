# -*- coding: utf-8 -*-
"""
向量库管理模块（面向对象，Milvus Lite + 本地 BGE）
--------------------------------------------------------------------
- EHSVectorStore 负责 Milvus 集合的创建、持久化、批量入库、增量更新、检索；
- 采用 Milvus Lite（本地单文件 ehs_milvus.db），无需 Docker / 无需起服务；
- 入库按 chunk_id 去重（chunk_id 作为主键），支持知识库增量更新；
- 批量分批写入，避免一次性占用过多内存。


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
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import Settings, settings
from embeddings import BGEEmbeddingModel

logger = logging.getLogger(__name__)


def _clean_metadata(meta: dict) -> dict:
    """清洗 metadata 为 Milvus 可存储的标量字段：
    - 丢弃 None（Milvus 动态字段不接受空值）；
    - 仅保留 str/int/float/bool，其余（Path/list 等）转为字符串。
    """
    clean: Dict[str, object] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


class EHSVectorStore:
    """Milvus 向量库管理器。"""

    def __init__(self, config: Settings = settings):
        self.config = config
        # 嵌入模型单例（device 由 config 自动解析，GPU 环境下为 cuda）
        self.embedding = BGEEmbeddingModel(config).embeddings
        self._store: Optional[VectorStore] = None

    # ----------------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        """惰性创建 Milvus（Milvus Lite 本地文件，自动建集合与索引）。"""
        if self._store is None:
            from langchain_milvus import Milvus

            mc = self.config.milvus
            logger.info("连接 Milvus Lite：%s", self.config.milvus_db)
            self._store = Milvus(
                embedding_function=self.embedding,
                collection_name=mc.collection_name,
                connection_args={"uri": str(self.config.milvus_db)},
                index_params={
                    "metric_type": mc.metric_type,
                    "index_type": mc.index_type,
                    "params": {},
                },
                search_params={"metric_type": mc.metric_type, "params": {}},
                auto_id=mc.auto_id,
                drop_old=False,
                primary_field=mc.primary_field,
                text_field=mc.text_field,
                vector_field=mc.vector_field,
                enable_dynamic_field=mc.enable_dynamic_field,
            )
            logger.info("Milvus 向量库已就绪：%s（已有 %d 条）",
                        self.config.milvus_db, self.count())
        return self._store

    def _ensure_loaded(self) -> None:
        """确保集合已 load 到内存（Milvus Lite 中 released 状态无法 query/search）。"""
        try:
            client = self.store.client
            name = self.config.milvus.collection_name
            if client.has_collection(name):
                client.load_collection(name)
        except Exception as e:
            logger.warning("load_collection 失败: %s", e)

    # ----------------------------------------------------------------
    def count(self) -> int:
        try:
            client = self.store.client
            name = self.config.milvus.collection_name
            if not client.has_collection(name):
                return 0
            return int(client.get_collection_stats(name).get("row_count", 0))
        except Exception as e:
            logger.warning("统计向量条数失败: %s", e)
            return 0

    def existing_chunk_ids(self) -> set:
        """返回库中已有的 chunk_id（主键）集合，用于去重。"""
        try:
            mc = self.config.milvus
            client = self.store.client
            if not client.has_collection(mc.collection_name):
                return set()
            self._ensure_loaded()
            expr = f"{mc.primary_field} != ''"
            rows = client.query(
                mc.collection_name,
                filter=expr,
                output_fields=[mc.primary_field],
                limit=16384,
            )
            return {r.get(mc.primary_field) for r in rows}
        except Exception as e:
            logger.warning("读取已有 chunk_id 失败: %s", e)
            return set()

    # ----------------------------------------------------------------
    def add_documents(self, chunks: Sequence[Document], batch_size: int = 256,
                      skip_existing: bool = True) -> int:
        """批量 / 增量写入文档块，返回实际写入条数。"""
        if not chunks:
            return 0

        # 清洗 metadata（复制 Document，不改原对象）
        prepared: List[Document] = []
        for c in chunks:
            prepared.append(Document(page_content=c.page_content,
                                     metadata=_clean_metadata(c.metadata)))
        chunks = prepared

        if skip_existing:
            existed = self.existing_chunk_ids()
            chunks = [c for c in chunks
                      if c.metadata.get("chunk_id") not in existed]
        if not chunks:
            logger.info("所有块均已存在，无需写入。")
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = list(chunks[i:i + batch_size])
            ids = [c.metadata["chunk_id"] for c in batch]
            self.store.add_documents(batch, ids=ids)
            total += len(batch)
            logger.info("向量入库进度：%d/%d", total, len(chunks))
        logger.info("向量入库完成，新增 %d 条，当前共 %d 条", total, self.count())
        return total

    def build(self, chunks: Sequence[Document]) -> int:
        """全量构建（语义等同 add_documents，保留语义化方法名）。"""
        return self.add_documents(chunks)

    # ----------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 10,
                          where: Optional[Dict] = None) -> List[Document]:
        if where:
            return self.store.similarity_search(query, k=k, expr=where)
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_scores(self, query: str, k: int = 10
                                      ) -> List[tuple]:
        """返回 (Document, 相似度分数)。
        Milvus COSINE 指标返回的 distance 即余弦相似度（越大越相似，
        归一化向量下范围约 0~1），与 Chroma 返回"距离"的语义相反，无需换算。
        """
        pairs = self.store.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in pairs]

    def as_retriever(self, k: int = 10):
        return self.store.as_retriever(search_kwargs={"k": k})

    def all_documents(self) -> List[Document]:
        """拉取库中全部块（正文 + 元数据），用于重建 BM25 等派生索引。"""
        mc = self.config.milvus
        client = self.store.client
        if not client.has_collection(mc.collection_name):
            return []
        self._ensure_loaded()
        rows = client.query(
            mc.collection_name,
            filter=f"{mc.primary_field} != ''",
            output_fields=["*"],
            limit=16384,
        )
        docs: List[Document] = []
        for r in rows:
            r.pop(mc.vector_field, None)
            text = r.pop(mc.text_field, "") or ""
            r.pop(mc.primary_field, None)
            docs.append(Document(page_content=text, metadata=r))
        return docs

    def delete_by_file(self, file_path: Path | str) -> int:
        """删除某文件在向量库中的所有块，返回删除条数（用于内容更新后的增量替换）。

        优先用完整路径（source）匹配；老数据只有 file_name 时回退按文件名匹配。
        先查询命中主键再删除，避免文件路径反斜杠的表达式转义问题。
        """
        path = Path(file_path)
        mc = self.config.milvus
        client = self.store.client
        if not client.has_collection(mc.collection_name):
            return 0
        self._ensure_loaded()
        rows = client.query(
            mc.collection_name,
            filter=f"{mc.primary_field} != ''",
            output_fields=[mc.primary_field, "source", "file_name"],
            limit=16384,
        )
        ids = []
        for r in rows:
            src = r.get("source")
            fn = r.get("file_name")
            if src == str(path) or ((not src) and fn == path.name):
                ids.append(r.get(mc.primary_field))
        if not ids:
            return 0
        self.store.delete(ids=ids)
        logger.info("向量库删除 %s 的 %d 个块", path.name, len(ids))
        return len(ids)

    # ----------------------------------------------------------------
    def reset(self) -> None:
        """删除集合并重建空集合（危险操作，仅在显式调用时执行）。"""
        logger.warning("正在重置 Milvus 向量库 ...")
        try:
            self.store.drop()
        except Exception as e:
            logger.error("重置失败: %s", e)
        self._store = None
        # 触发空集合重建
        _ = self.store
