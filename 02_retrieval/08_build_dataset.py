# -*- coding: utf-8 -*-
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

"""构建 200 条评测集（可传参指定规模）。"""
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
from eval_dataset import EvalDatasetBuilder

target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
records = EvalDatasetBuilder().build(target)
print("DATASET_SIZE =", len(records))
