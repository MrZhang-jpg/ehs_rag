# -*- coding: utf-8 -*-
"""
批量入库脚本：顺序入库 03~07 全部子目录（模型只加载一次）。
用法：python batch_ingest.py
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
import time

from main import EHSRAGApplication, setup_logging

SUBDIRS = [
    "03_部门规章及规范性文件",
    "04_国家标准_通用规范及安全规范",
    "05_行业标准_JGJ_基础安全",
    "06_行业标准_JGJ_机械吊装及防护",
    "07_环保排放标准及环境规章",
]


def main() -> None:
    setup_logging()
    app = EHSRAGApplication()
    t0 = time.time()
    all_stats = []
    for sub in SUBDIRS:
        logging.info(">>>>>>>>>> 开始入库：%s <<<<<<<<<<", sub)
        try:
            stats = app.ingest(subdir=sub)
            all_stats.append(stats)
        except Exception as e:
            logging.exception("目录 %s 入库失败：%s", sub, e)
    logging.info(">>>>>>>>>> 全部入库完成，耗时 %.1f 分钟 <<<<<<<<<<",
                 (time.time() - t0) / 60)
    for s in all_stats:
        logging.info("  %s: chunks_new=%s, 向量库累计=%s",
                     s["source_dir"], s["chunks_new"], s["vectorstore_total"])


if __name__ == "__main__":
    main()
