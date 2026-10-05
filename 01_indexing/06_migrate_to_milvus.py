# -*- coding: utf-8 -*-
"""
向量库迁移工具：旧 Chroma -> Milvus Lite
--------------------------------------------------------------------
- 写入 Milvus，
  无需加载/运行 BGE（向量与原库完全一致，迁移快、不占 GPU）；
- 用 pymilvus 直接建立与 langchain 约定一致的集合（pk/text/vector + 动态字段，
  COSINE + AUTOINDEX）；
- 默认幂等：跳过 Milvus 中已存在的主键。

用法：
  python migrate_to_milvus.py --dry-run   # 体检：条数 / 维度 / metadata keys
  python migrate_to_milvus.py             # 执行迁移
  python migrate_to_milvus.py --reset     # 清空 Milvus 集合后全量迁移
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
import logging

from config import settings
from vector_store import _clean_metadata

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("migrate")

CHROMA_COLLECTION = "ehs_knowledge_base"


def load_from_chroma():
    """直接读取旧 Chroma（不加载嵌入模型，只用底层 collection.get）。"""
    from langchain_chroma import Chroma
    old = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(settings.vectorstore_dir),
    )
    raw = old._collection.get(  # type: ignore[attr-defined]
        include=["documents", "metadatas", "embeddings"])
    return raw


def ensure_milvus_collection(reset: bool):
    """用 pymilvus 建立（或重置）与 langchain 约定一致的集合与索引，返回 client。"""
    from pymilvus import DataType, MilvusClient

    mc = settings.milvus
    client = MilvusClient(uri=str(settings.milvus_db))
    if client.has_collection(mc.collection_name):
        if reset:
            client.drop_collection(mc.collection_name)
        else:
            return client

    schema = client.create_schema(enable_dynamic_field=mc.enable_dynamic_field)
    schema.add_field(mc.primary_field, DataType.VARCHAR,
                     max_length=65535, is_primary=True)
    schema.add_field(mc.text_field, DataType.VARCHAR, max_length=65535)
    schema.add_field(mc.vector_field, DataType.FLOAT_VECTOR,
                     dim=settings.embedding.dimension)
    idx = client.prepare_index_params()
    idx.add_index(field_name=mc.vector_field,
                  index_type=mc.index_type,
                  metric_type=mc.metric_type, params={})
    client.create_collection(mc.collection_name, schema=schema, index_params=idx)
    logger.info("已创建 Milvus 集合：%s", mc.collection_name)
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description="Chroma -> Milvus 迁移")
    parser.add_argument("--dry-run", action="store_true", help="只体检不写入")
    parser.add_argument("--reset", action="store_true", help="迁移前清空 Milvus 集合")
    args = parser.parse_args()

    logger.info("读取旧 Chroma：%s", settings.vectorstore_dir)
    raw = load_from_chroma()

    def _got(key):
        v = raw.get(key)
        return list(v) if v is not None else []

    ids = _got("ids")
    docs = _got("documents")
    metas = _got("metadatas")
    vecs = _got("embeddings")
    n = len(ids)
    dim = len(vecs[0]) if n else 0
    meta_keys = set()
    for m in metas:
        meta_keys.update((m or {}).keys())

    logger.info("Chroma 条数 = %d，向量维度 = %d", n, dim)
    logger.info("metadata keys = %s", sorted(meta_keys))

    if args.dry_run:
        logger.info("dry-run 结束，未写入。")
        return

    mc = settings.milvus
    client = ensure_milvus_collection(args.reset)

    existing: set = set()
    if client.has_collection(mc.collection_name):
        client.load_collection(mc.collection_name)
        rows = client.query(mc.collection_name,
                            filter=f"{mc.primary_field} != ''",
                            output_fields=[mc.primary_field], limit=1000000)
        existing = {r.get(mc.primary_field) for r in rows}

    batch, total, skipped = [], 0, 0
    for cid, doc, meta, vec in zip(ids, docs, metas, vecs):
        if cid in existing:
            skipped += 1
            continue
        m = _clean_metadata(meta)
        m.pop(mc.primary_field, None)
        m.pop(mc.text_field, None)
        m.pop(mc.vector_field, None)
        batch.append({
            mc.primary_field: cid,
            mc.text_field: doc or "",
            mc.vector_field: [float(x) for x in vec],
            **m,
        })
        if len(batch) >= 256:
            client.insert(mc.collection_name, batch)
            total += len(batch)
            batch = []
            logger.info("迁移进度：%d/%d（跳过已存在 %d）", total, n, skipped)
    if batch:
        client.insert(mc.collection_name, batch)
        total += len(batch)

    client.flush(mc.collection_name)
    final_rows = client.get_collection_stats(mc.collection_name).get("row_count", 0)
    logger.info("迁移完成：本次写入 %d，跳过 %d，Milvus 当前共 %d 条",
                total, skipped, final_rows)


if __name__ == "__main__":
    main()
