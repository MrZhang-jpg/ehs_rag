# -*- coding: utf-8 -*-
"""
FastAPI 服务模块（面向对象）
--------------------------------------------------------------------
- API-Key 鉴权（X-API-Key 请求头）
- 每把 Key 每分钟请求数限流（滑动窗口）
- 阻塞式 RAG 在线程池中执行，单次问答超时熔断
- 接口：
    GET  /health                        健康检查
    GET  /                              豆包风格对话前端（static/index.html）
    POST /api/v1/ask                    RAG 问答（含溯源）
    GET  /api/v1/stats                  知识库统计（向量块 / BM25 / 文件数）
    POST /api/v1/ingest-file            上传单个文件并增量入库

启动：
    python api_server.py
    或 uvicorn api_server:app --host 0.0.0.0 --port 8000
测试：
    curl -X POST http://localhost:8000/api/v1/ask \
      -H "X-API-Key: ehs-demo-key-2026" -H "Content-Type: application/json" \
      -d '{"question":"脚手架垫板有什么要求？"}'
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


import asyncio
import json
import logging
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Settings, settings

logger = logging.getLogger(__name__)

# 前端静态资源目录（豆包风格对话界面）
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ====================================================================
# 鉴权 & 限流组件
# ====================================================================
class APIKeyAuthenticator:
    """API-Key 校验器。"""

    def __init__(self, valid_keys: tuple):
        self.valid_keys = set(valid_keys)

    def verify(self, key: Optional[str]) -> str:
        if not key or key not in self.valid_keys:
            raise HTTPException(status_code=401, detail="无效或缺失的 API-Key")
        return key


class SlidingWindowRateLimiter:
    """每 Key 滑动窗口限流（每分钟 N 次）。"""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            window = self._windows[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(now)
            return True


# ====================================================================
# 请求 / 响应模型
# ====================================================================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: str = Field("hybrid_rerank",
                      pattern="^(vector|bm25|hybrid|hybrid_rerank)$")
    return_contexts: bool = False


class SourceItem(BaseModel):
    file_name: Optional[str] = None
    section_title: Optional[str] = None
    page: Optional[int] = None
    category: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    contexts: List[str] = []
    mode: str
    latency_seconds: float


# ====================================================================
# RAG 惰性单例（避免导入即占用大量内存）
# ====================================================================
class RAGService:
    """RAG 组件进程内单例。"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    from main import EHSRAGApplication
                    app = EHSRAGApplication()
                    app.retriever.load_indexes()
                    cls._instance = app
        return cls._instance


# ====================================================================
# FastAPI 应用
# ====================================================================
class EHSAPIServer:
    """组装 FastAPI 应用与路由。"""

    def __init__(self, config: Settings = settings):
        self.config = config
        self.authenticator = APIKeyAuthenticator(config.service.api_keys)
        self.rate_limiter = SlidingWindowRateLimiter(
            config.service.rate_limit_per_minute)
        self.app = FastAPI(
            title="EHS 知识库 RAG 问答系统 API",
            version="1.0.0",
            description="基于 LangChain + 本地 BGE + 混合检索的 EHS 法规问答服务",
        )
        self._register_routes()

    # ----------------------------------------------------------------
    def _auth_dependency(self, x_api_key: Optional[str] = Header(None)) -> str:
        key = self.authenticator.verify(x_api_key)
        if not self.rate_limiter.allow(key):
            raise HTTPException(status_code=429, detail="请求过于频繁，已触发限流")
        return key

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        async def health():
            return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}

        # ---- 前端页面与静态资源 ----
        if STATIC_DIR.exists():
            index_file = STATIC_DIR / "index.html"

            @app.get("/", include_in_schema=False)
            async def index():
                return FileResponse(index_file)

            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
                      name="static")

        @app.get("/api/v1/stats")
        async def stats(_key: str = Depends(self._auth_dependency)):
            """知识库统计：向量块 / BM25 条目 / 已入库文件数。"""
            def _collect() -> dict:
                rag = RAGService.get()
                bm25 = rag.retriever.bm25
                try:
                    manifest = json.loads(
                        (self.config.data_dir / "ingest_manifest.json")
                        .read_text(encoding="utf-8"))
                    n_files = len(manifest)
                except Exception:
                    n_files = None
                return {
                    "vectors": rag.vector_store.count(),
                    "bm25": len(bm25.documents) + len(bm25.staged_documents),
                    "files": n_files,
                }

            loop = asyncio.get_running_loop()
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _collect),
                    timeout=self.config.service.answer_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="统计超时")

        @app.post("/api/v1/ask", response_model=AskResponse)
        async def ask(req: AskRequest, _key: str = Depends(self._auth_dependency)):
            rag = RAGService.get()
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: rag.ask(req.question, mode=req.mode)),
                    timeout=self.config.service.answer_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="问答处理超时")
            return AskResponse(
                question=result.question,
                answer=result.answer,
                sources=result.sources,
                contexts=result.contexts if req.return_contexts else [],
                mode=result.mode,
                latency_seconds=result.latency_seconds,
            )

        @app.post("/api/v1/ingest-file")
        async def ingest_file(file: UploadFile = File(...),
                              _key: str = Depends(self._auth_dependency)):
            if Path(file.filename).suffix.lower() not in self.config.document.supported_extensions:
                raise HTTPException(status_code=400, detail="不支持的文件类型")
            # 落临时文件 -> 解析 -> 分块 -> 入库 -> BM25 合并重建
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file.filename).suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            try:
                rag = RAGService.get()
                docs = rag.loader.load_file(tmp_path)
                # 统一改写成真实文件名元数据
                for d in docs:
                    d.metadata["file_name"] = file.filename
                    d.metadata["source"] = file.filename
                chunks = rag.splitter.split_documents(docs)
                n_vec = rag.vector_store.add_documents(chunks)
                # 增量追加 BM25（不再全量重建：只对新块分词并暂存）
                rag.retriever.bm25.append(chunks)
                # 索引变更：清空 Redis 检索/答案缓存
                try:
                    from redis_cache import RAGRedisCache
                    RAGRedisCache.get_instance(self.config).invalidate_all()
                except Exception:
                    pass
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return {"file": file.filename, "chunks": len(chunks), "vectors_added": n_vec}


def build_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    return EHSAPIServer().app


app = build_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.service.api_host, port=settings.service.api_port)
