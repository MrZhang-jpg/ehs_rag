# -*- coding: utf-8 -*-
"""
02_retrieval —— 检索侧（在线问答链路 + 评测）
--------------------------------------------------------------------
按执行顺序：
  01_hybrid_retriever  BM25 + 向量 + RRF + BGE Rerank
  02_redis_cache       Redis 缓存（检索结果 / 问答结果，可优雅降级）
  03_rag_chain         RAG 问答链（防幻觉 + 溯源）
  04_api_server        FastAPI 服务
  05_app_streamlit     Streamlit 对话前端
  06_main              应用编排与 CLI（ingest / ask / chat）
  07_eval_dataset      评测集构建
  08_build_dataset     评测集批量构建脚本
  09_evaluator         检索/生成多指标评测器
  10_run_eval          评测入口脚本
  11_debug_recall      检索失败归因
  12_audit_labels      评测集标签可达性审计
  13_tune_retrieval    候选池策略对比实验
  14_compare_reports   评测报告前后对比
  15_test_api          API 接口测试脚本
"""
