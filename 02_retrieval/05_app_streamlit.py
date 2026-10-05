# -*- coding: utf-8 -*-
"""
Streamlit 前端（面向对象）
--------------------------------------------------------------------
- 对话式 EHS 知识库问答，答案下方展示来源溯源与检索原文；
- 侧边栏切换检索模式（vector / bm25 / hybrid / hybrid_rerank）、查看知识库统计；
- RAG 组件通过 st.cache_resource 全局缓存，避免每次交互重新加载模型。

启动：
    streamlit run app_streamlit.py
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


import time
from typing import List

import streamlit as st

from config import Settings, settings


class EHSStreamlitApp:
    """Streamlit 应用类。"""

    def __init__(self, config: Settings = settings):
        self.config = config
        st.set_page_config(page_title="EHS 知识库 RAG 问答", page_icon="🛡", layout="wide")

    # ----------------------------------------------------------------
    @st.cache_resource(show_spinner="正在加载 RAG 系统（本地 BGE / 向量库 / BM25）...")
    @staticmethod
    def load_rag():
        from main import EHSRAGApplication
        app = EHSRAGApplication()
        app.retriever.load_indexes()
        return app

    # ----------------------------------------------------------------
    def render_sidebar(self) -> str:
        with st.sidebar:
            st.title("🛡 EHS 知识库问答")
            mode = st.selectbox(
                "检索模式",
                options=["hybrid_rerank", "hybrid", "vector", "bm25"],
                format_func=lambda m: {
                    "hybrid_rerank": "混合检索 + Rerank（推荐）",
                    "hybrid": "混合检索（BM25+向量）",
                    "vector": "仅向量检索",
                    "bm25": "仅 BM25 关键词",
                }[m],
            )
            st.divider()
            try:
                rag = self.load_rag()
                st.caption("知识库统计")
                st.write(f"向量条目：**{rag.vector_store.count()}**")
                st.write(f"BM25 条目：**{len(rag.retriever.bm25.documents)}**")
            except Exception as e:
                st.warning(f"知识库未就绪：{e}")
            st.divider()
            st.caption("回答均基于检索到的法规原文，数值与条款以原文为准。")
        return mode

    # ----------------------------------------------------------------
    @staticmethod
    def init_chat_state() -> None:
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "您好，我是 EHS 知识库问答助手，请提出安全法规、隐患判定、参数查阅类问题。",
                 "sources": [], "contexts": []}
            ]

    def render_chat(self, mode: str) -> None:
        self.init_chat_state()
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    self._render_sources(msg["sources"], msg.get("contexts", []))

        question = st.chat_input("请输入 EHS 问题，例如：脚手架垫板有什么要求？")
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question, "sources": [], "contexts": []})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("检索与生成中 ..."):
                    rag = self.load_rag()
                    t0 = time.time()
                    result = rag.ask(question, mode=mode)
                st.markdown(result.answer)
                self._render_sources(
                    [s for s in result.sources], result.contexts)
                st.caption(f"耗时 {result.latency_seconds}s · 模式 {result.mode}")
            st.session_state.messages.append({
                "role": "assistant", "content": result.answer,
                "sources": result.sources, "contexts": result.contexts})

    @staticmethod
    def _render_sources(sources: List[dict], contexts: List[str]) -> None:
        if not sources:
            return
        with st.expander(f"📎 来源溯源（{len(sources)}）"):
            for i, s in enumerate(sources):
                page = f" · 第{s.get('page')}页" if s.get("page") else ""
                st.markdown(
                    f"**{i+1}. {s.get('file_name')}** · "
                    f"{s.get('section_title', '')}{page} · {s.get('category', '')}")
            if contexts:
                st.divider()
                st.caption("检索原文片段")
                for i, c in enumerate(contexts, 1):
                    st.text_area(f"片段 {i}", c, height=160, key=f"ctx_{i}_{time.time_ns()}")

    # ----------------------------------------------------------------
    def run(self) -> None:
        mode = self.render_sidebar()
        st.title("EHS 知识库 RAG 问答系统")
        self.render_chat(mode)


if __name__ == "__main__":
    EHSStreamlitApp(settings).run()
