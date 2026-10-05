# -*- coding: utf-8 -*-
"""
01_indexing —— 索引侧（离线建库链路）
--------------------------------------------------------------------
按执行顺序：
  01_document_loader   文档解析（PDF/Word/TXT/MD/Excel + 表格结构化）
  02_ocr_module        扫描版/乱码 PDF 的 OCR 识别与入库
  03_text_splitter     语义分块（结构感知 + 表格保护）
  04_embeddings        本地 BGE 嵌入封装
  05_vector_store      Milvus/Chroma 向量库（持久化、去重、增量）
  06_migrate_to_milvus 旧 Chroma -> Milvus Lite 迁移工具
  07_reingest          索引质量体检与修复（乱码/噪声文件重建）
  08_batch_ingest      批量入库脚本
"""
