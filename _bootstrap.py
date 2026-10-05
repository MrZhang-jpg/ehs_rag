# -*- coding: utf-8 -*-
"""
运行引导（公共模块，置于项目根目录）
--------------------------------------------------------------------
项目重构后，代码分别位于 01_indexing / 02_retrieval 两个目录，且每个
文件名带两位序号前缀（如 05_vector_store.py）。数字开头的目录名与文件
名无法直接使用 `import` 语法，本模块负责：

  1) 把 项目根目录、01_indexing、02_retrieval 加入 sys.path；
  2) 安装一个 meta_path 别名导入器，让"去掉 NN_ 序号前缀"的原模块名
     仍可正常导入，例如
        from vector_store import EHSVectorStore
     会自动解析到 01_indexing/05_vector_store.py；通过带序号限定名
     （01_indexing.05_vector_store）导入时也返回同一个模块对象，
     保证单例不重复初始化。

用法：每个代码文件在顶部插入
        import sys as _sys
        from pathlib import Path as _Path
        _ROOT = _Path(__file__).resolve().parents[1]
        for _p in (_ROOT, _ROOT / "01_indexing", _ROOT / "02_retrieval"):
            if str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        import _bootstrap as _bootstrap
        _bootstrap.init()
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

# 模块别名 -> (所在子目录, 带序号文件名[不含.py])
_ALIAS_MAP = {
    # ---- 01_indexing 索引侧 ----
    "document_loader": ("01_indexing", "01_document_loader"),
    "ocr_module": ("01_indexing", "02_ocr_module"),
    "text_splitter": ("01_indexing", "03_text_splitter"),
    "embeddings": ("01_indexing", "04_embeddings"),
    "vector_store": ("01_indexing", "05_vector_store"),
    "migrate_to_milvus": ("01_indexing", "06_migrate_to_milvus"),
    "reingest": ("01_indexing", "07_reingest"),
    "batch_ingest": ("01_indexing", "08_batch_ingest"),
    # ---- 02_retrieval 检索侧 ----
    "hybrid_retriever": ("02_retrieval", "01_hybrid_retriever"),
    "redis_cache": ("02_retrieval", "02_redis_cache"),
    "rag_chain": ("02_retrieval", "03_rag_chain"),
    "api_server": ("02_retrieval", "04_api_server"),
    "app_streamlit": ("02_retrieval", "05_app_streamlit"),
    "main": ("02_retrieval", "06_main"),
    "eval_dataset": ("02_retrieval", "07_eval_dataset"),
    "build_dataset": ("02_retrieval", "08_build_dataset"),
    "evaluator": ("02_retrieval", "09_evaluator"),
    "run_eval": ("02_retrieval", "10_run_eval"),
    "debug_recall": ("02_retrieval", "11_debug_recall"),
    "audit_labels": ("02_retrieval", "12_audit_labels"),
    "tune_retrieval": ("02_retrieval", "13_tune_retrieval"),
    "compare_reports": ("02_retrieval", "14_compare_reports"),
    "test_api": ("02_retrieval", "15_test_api"),
}

# 带序号限定名 -> 别名（反向映射）
_REVERSE_MAP = {f"{subdir}.{filename}": alias
                for alias, (subdir, filename) in _ALIAS_MAP.items()}


def project_root() -> Path:
    """项目根目录（本文件所在目录）。"""
    return Path(__file__).resolve().parent


# ====================================================================
# 别名 Loader：两种名字共享同一模块对象，模块代码只执行一次
# ====================================================================
class _AliasLoader(importlib.abc.Loader):

    def __init__(self, alias: str, real_name: str, file_path: Path):
        self.alias = alias                    # 无序号规范名，如 vector_store
        self.real_name = real_name           # 带序号限定名
        self.file_path = Path(file_path)

    def _build_real_spec(self):
        return importlib.util.spec_from_file_location(
            self.real_name, self.file_path)

    def create_module(self, spec):
        existing = sys.modules.get(self.alias)
        if existing is not None:
            # 已加载（可能通过另一个名字）：直接共享，保证单例
            sys.modules.setdefault(self.real_name, existing)
            sys.modules.setdefault(self.alias, existing)
            return existing
        real_spec = self._build_real_spec()
        module = importlib.util.module_from_spec(real_spec)
        self._real_spec = real_spec
        sys.modules[self.alias] = module
        sys.modules[self.real_name] = module
        return module

    def exec_module(self, module):
        if getattr(module, "_bootstrap_executed", False):
            return
        module._bootstrap_executed = True
        real_spec = getattr(self, "_real_spec", None) or self._build_real_spec()
        try:
            real_spec.loader.exec_module(module)
        except BaseException:
            # 执行失败：清理半成品注册，避免污染后续导入
            sys.modules.pop(self.alias, None)
            sys.modules.pop(self.real_name, None)
            raise
        sys.modules[self.alias] = module
        sys.modules[self.real_name] = module


# ====================================================================
# meta_path Finder
# ====================================================================
class _AliasFinder:
    """拦截别名与带序号限定名，重定向到实际文件。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _ALIAS_MAP:
            alias = fullname
            subdir, filename = _ALIAS_MAP[fullname]
            real_name = f"{subdir}.{filename}"
        elif fullname in _REVERSE_MAP:
            alias = _REVERSE_MAP[fullname]
            subdir, filename = _ALIAS_MAP[alias]
            real_name = fullname
        else:
            return None
        file_path = project_root() / subdir / f"{filename}.py"
        if not file_path.exists():
            return None
        loader = _AliasLoader(alias, real_name, file_path)
        return importlib.util.spec_from_loader(fullname, loader)


# ====================================================================
# 幂等初始化
# ====================================================================
def init() -> None:
    """补齐 sys.path 并安装别名 finder（可重复调用）。"""
    root = project_root()
    for p in (root, root / "01_indexing", root / "02_retrieval"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        # 置于最前，优先于默认 PathFinder，避免重复加载
        sys.meta_path.insert(0, _AliasFinder())
