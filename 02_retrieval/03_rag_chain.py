# -*- coding: utf-8 -*-
"""
RAG 问答链模块（面向对象，LangChain LCEL）
--------------------------------------------------------------------
- EHSLLM：封装生成后端（默认本地 Ollama，可切 OpenRouter），统一 num_ctx /
  生成上限 / 超时参数映射，内置最小请求间隔（云端免费模型限流保护）；
- EHSRAGChain：检索 -> 上下文拼装 -> Prompt -> LLM -> 解析，
  强制"仅依据原文作答"以缓解幻觉，答案附带条款/页码/文件来源，支持溯源；
- 输出 RAGAnswer（答案 + 来源 + 原文上下文），供前端展示与评测使用。
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
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import Settings, settings
from hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ====================================================================
# 数据结构
# ====================================================================
@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    mode: str = "hybrid_rerank"
    latency_seconds: float = 0.0


# ====================================================================
# LLM 封装
# ====================================================================
class EHSLLM:
    """生成大模型封装（线程安全单例）。

    支持两种后端（config.llm.provider）：
      - ollama     本地 Ollama 服务（离线、无限流，默认）
      - openrouter 云端 OpenRouter（需 API Key 与配额）

    注意：不同后端的构造参数名不同（Ollama 用 num_predict/num_ctx，OpenAI 协议
    用 max_tokens），传错会被静默忽略——例如 max_tokens 传给 ChatOllama 不报错
    但完全不生效。
    """

    _instance: Optional["EHSLLM"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Settings = settings):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init(config)
                    cls._instance = inst
        return cls._instance

    def _init(self, config: Settings) -> None:
        self.config = config
        lc = config.llm
        if lc.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            logger.info("初始化云端 LLM（OpenRouter）：%s @ %s", lc.model, lc.base_url)
            self.chat = ChatOpenAI(
                model=lc.model,
                api_key=config.api_key(),
                base_url=lc.base_url,
                temperature=lc.temperature,
                max_tokens=lc.max_tokens,
                timeout=lc.timeout,
                max_retries=lc.max_retries,
                model_kwargs={"reasoning": {"effort": lc.reasoning_effort}},
            )
        else:
            from langchain_ollama import ChatOllama
            logger.info("初始化本地 LLM（Ollama）：%s @ %s（num_ctx=%d）",
                        lc.model, lc.ollama_base_url, lc.num_ctx)
            self.chat = ChatOllama(
                model=lc.model,
                base_url=lc.ollama_base_url,
                temperature=lc.temperature,
                num_predict=lc.max_tokens,     # Ollama 的生成上限参数
                num_ctx=lc.num_ctx,            # 必须显式设置，否则默认窗口会截断检索原文
                reasoning=lc.reasoning,        # 关闭思考模式，直接作答
                client_kwargs={"timeout": lc.timeout},
            )
        self._last_call_ts = 0.0
        self._rate_lock = threading.Lock()
        self._chain = self.chat | StrOutputParser()  # 复用于所有请求，避免重复组装

    def _respect_rate_limit(self) -> None:
        """免费模型限流：保证两次请求之间至少间隔 min_request_interval（线程安全）。"""
        with self._rate_lock:
            wait = self.config.llm.min_request_interval - (time.time() - self._last_call_ts)
            if wait > 0:
                logger.debug("限流等待 %.2fs", wait)
                time.sleep(wait)
            self._last_call_ts = time.time()

    def invoke(self, messages) -> str:
        """messages: ChatPromptTemplate 格式化得到的消息列表。"""
        self._respect_rate_limit()
        return self._chain.invoke(messages)


# ====================================================================
# Prompt
# ====================================================================
SYSTEM_PROMPT = """你是企业 EHS（环境 Environment、健康 Health、安全 Safety）管理知识库问答助手，\
服务对象是施工现场一线安全管理人员。你必须严格依据提供的法规与制度原文片段回答问题。

回答要求：
1. 只依据【参考原文】作答，严禁编造法规条款、数值、参数；若原文不足以回答问题，\
直接回答"根据现有知识库无法确定该问题，建议查阅相关规范原文"，不要臆测。
2. 涉及条款编号、距离、尺寸、浓度、时限、人数等数值时，必须与原文完全一致，不得换算或估算。
3. 回答条理清晰，可用分点；内容完整但不啰嗦，优先复述原文的规范性表述。
4. 在回答末尾另起一行标注引用来源，格式：【来源：文件名；条款/章节；第X页】，多个来源依次列出。
5. 全程使用简体中文。"""

HUMAN_TEMPLATE = """【参考原文】
{context}

【问题】
{question}

请依据上述原文作答。"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# ====================================================================
# RAG 问答链
# ====================================================================
class EHSRAGChain:
    """对外 RAG 问答主类：
        rag = EHSRAGChain()
        result: RAGAnswer = rag.answer("脚手架立杆垫板有什么要求？")
    """

    def __init__(self, config: Settings = settings,
                 retriever: Optional[HybridRetriever] = None):
        self.config = config
        self.retriever = retriever or HybridRetriever(config)
        self.llm = EHSLLM(config)

    # ----------------------------------------------------------------
    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        blocks = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            loc = meta.get("section_title") or ""
            page = f"，第{meta['page']}页" if meta.get("page") else ""
            header = f"[片段{i}] 《{meta.get('file_name', '未知文件')}》{loc}{page}"
            blocks.append(f"{header}\n{d.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _to_sources(docs: Sequence[Document]) -> List[dict]:
        seen, sources = set(), []
        for d in docs:
            m = d.metadata
            key = (m.get("file_name"), m.get("section_title"), m.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "file_name": m.get("file_name"),
                "rel_path": m.get("rel_path"),
                "section_title": m.get("section_title"),
                "page": m.get("page"),
                "category": m.get("category"),
            })
        return sources

    # ----------------------------------------------------------------
    def retrieve(self, question: str, mode: str
                 ) -> List[Document]:
        return self.retriever.retrieve_documents(question, mode=mode)

    def answer(self, question: str, mode: str = "hybrid_rerank",
               return_contexts: bool = True) -> RAGAnswer:
        # ---- Redis 答案缓存：命中直接返回，跳过检索与生成 ----
        cache = None
        try:
            from redis_cache import RAGRedisCache
            cache = RAGRedisCache.get_instance(self.config)
            if cache.available():
                data = cache.get_json(cache.make_key("answer", question, mode))
                if data is not None:
                    cached_contexts = data.get("contexts", []) if return_contexts else []
                    cached = RAGAnswer(
                        question=data.get("question", question),
                        answer=data.get("answer", ""),
                        sources=data.get("sources", []),
                        contexts=cached_contexts,
                        mode=data.get("mode", mode),
                        latency_seconds=0.0,
                    )
                    logger.info("问答缓存命中（%s）：%s", mode, question[:30])
                    return cached
        except Exception:
            cache = None

        t0 = time.time()
        ranked = self.retriever.retrieve(question, mode=mode)
        docs = [d for d, _ in ranked]
        if not docs:
            return RAGAnswer(
                question=question,
                answer="根据现有知识库无法确定该问题（未检索到相关原文），建议查阅相关规范原文。",
                sources=[], contexts=[], mode=mode,
                latency_seconds=time.time() - t0,
            )

        context = self._format_context(docs)
        prompt = CHAT_PROMPT.format_messages(context=context, question=question)
        try:
            text = self.llm.invoke(prompt)
        except Exception as e:
            # OpenRouter 每日额度限流：不降级，向上抛出，让批量评测及时终止
            from evaluator import _is_daily_rate_limit
            if _is_daily_rate_limit(e):
                raise
            logger.exception("LLM 生成失败：%s", e)
            text = f"（生成失败：{e}）"

        result = RAGAnswer(
            question=question,
            answer=text.strip(),
            sources=self._to_sources(docs),
            contexts=[d.page_content for d in docs] if return_contexts else [],
            mode=mode,
            latency_seconds=round(time.time() - t0, 2),
        )
        logger.info("问答完成（%s），%d 个来源，耗时 %.2fs",
                    mode, len(result.sources), result.latency_seconds)

        # ---- 写入答案缓存（始终存 contexts；异常不影响返回）----
        if cache is not None:
            try:
                from dataclasses import asdict
                cache.set_json(
                    cache.make_key("answer", question, mode),
                    asdict(result), cache.answer_ttl,
                )
            except Exception:
                pass
        return result

    # ----------------------------------------------------------------
    def build_lcel_chain(self, mode: str = "hybrid_rerank"):
        """组装纯 LCEL 链（输入问题字符串，输出答案字符串），供服务层直接调用。"""
        def _retrieve_docs(question: str) -> List[Document]:
            return self.retriever.retrieve_documents(question, mode=mode)

        chain = (
            {
                "context": RunnablePassthrough()
                           | RunnableLambda(lambda q: self._format_context(_retrieve_docs(q))),
                "question": RunnablePassthrough(),
            }
            | CHAT_PROMPT
            | self.llm.chat
            | StrOutputParser()
        )
        return chain
