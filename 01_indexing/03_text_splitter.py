# -*- coding: utf-8 -*-
"""
语义分块模块（面向对象，结构感知）
--------------------------------------------------------------------
- 表格块（doc_type=table）优先整体保留，超长表格按行切分并重复表头，
  保证"表格、关联说明不被粗暴拆分"；
- 普通文本按法规结构（第X条 / 第X章 / X.X.X / 一、）做第一层语义切分；
- 超长结构单元再按段落 -> 句子递归切分（带 overlap）；
- 连续短单元贪心合并到目标块大小，减少碎片、降低 embedding 成本；
- 每个 chunk 记录所属条款标题(section_title)，供答案溯源。
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


import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

from config import Settings, settings

logger = logging.getLogger(__name__)


# ====================================================================
# 抽象基类
# ====================================================================
class BaseSplitter(ABC):
    def __init__(self, config: Settings):
        self.config = config

    @abstractmethod
    def split(self, doc: Document) -> List[Document]:
        raise NotImplementedError

    # 常量：句子结束标点
    SENTENCE_ENDS = "。；！？;"


# ====================================================================
# 表格分块器
# ====================================================================
class TableSplitter(BaseSplitter):
    """表格整体保留；超长时按行切分、重复表头与表注。"""

    def split(self, doc: Document) -> List[Document]:
        text = doc.page_content.strip()
        max_size = self.config.chunk.table_max_size
        if len(text) <= max_size:
            return [self._child(doc, text, 0)]

        lines = text.split("\n")
        # 定位表注（【表格...】）、表头行、分隔行、数据行
        caption = ""
        header_idx = None
        for i, ln in enumerate(lines):
            if ln.startswith("【表格"):
                caption = ln
            if ln.startswith("|") and header_idx is None:
                header_idx = i
                break
        if header_idx is None:  # 非预期格式，退回普通切分
            return SemanticTextSplitter(self.config).split(doc)

        header_line = lines[header_idx]
        sep_line = lines[header_idx + 1] if header_idx + 1 < len(lines) else ""
        data_lines = [ln for ln in lines[header_idx + 2:] if ln.startswith("|")]

        chunks, group, idx = [], [], 0
        head_len = len(caption) + len(header_line) + len(sep_line) + 4

        def flush():
            nonlocal group, idx
            if not group:
                return
            body = "\n".join(x for x in [caption, header_line, sep_line] + group if x)
            chunks.append(self._child(doc, body, idx))
            idx += 1
            group = []

        for row in data_lines:
            if head_len + sum(len(x) + 1 for x in group) + len(row) > max_size:
                flush()
            group.append(row)
        flush()
        logger.debug("表格切分为 %d 块：%s", len(chunks), doc.metadata.get("file_name"))
        return chunks

    @staticmethod
    def _child(parent: Document, content: str, part_idx: int) -> Document:
        """用内容哈希生成确定性 chunk_id：相同文件/位置/内容 → 同一 id，便于去重与缓存。"""
        meta = dict(parent.metadata)
        # 哈希须含页码：idx 是"页内序号"，不同页可能出现 idx 与内容前缀都相同
        # 的块（如重复的版权/空白页），仅用 source+idx+前缀会撞 id
        chunk_id = hashlib.sha256(
            f"{meta.get('source', '')}:{meta.get('page', '')}:{part_idx}:{content[:256]}".encode()
        ).hexdigest()[:16]
        meta.update({"chunk_id": chunk_id, "table_part": part_idx,
                     "section_title": "表格"})
        return Document(page_content=content, metadata=meta)


# ====================================================================
# 语义文本分块器
# ====================================================================
class SemanticTextSplitter(BaseSplitter):
    def __init__(self, config: Settings):
        super().__init__(config)
        self.size = config.chunk.chunk_size
        self.overlap = config.chunk.chunk_overlap
        self.structure_res = [re.compile(p) for p in config.chunk.section_separators]
        # 条款标题提取
        self.title_res = [
            re.compile(r"第[一二三四五六七八九十百零〇\d]+[章节条]"),
            re.compile(r"\d+\.\d+\.\d+"),
            re.compile(r"[一二三四五六七八九十]+、"),
        ]

    # ----------------------------------------------------------------
    def split(self, doc: Document) -> List[Document]:
        units = self._split_by_structure(doc.page_content)
        chunks: List[Document] = []

        # 贪心合并小单元；拆分大单元
        buf: List[tuple] = []   # (title, text)
        buf_len = 0

        def flush_buf():
            nonlocal buf, buf_len
            if not buf:
                return
            title = buf[0][0]
            content = "\n".join(t for _, t in buf if t).strip()
            if content:
                chunks.append(self._child(doc, content, title, len(chunks)))
            buf, buf_len = [], 0

        for title, unit in units:
            if len(unit) > self.size:
                # 先冲刷缓冲，再递归切分大单元
                flush_buf()
                for piece in self._split_long_unit(unit):
                    chunks.append(self._child(doc, piece, title, len(chunks)))
                continue

            if buf_len + len(unit) + 1 > self.size:
                flush_buf()
            buf.append((title, unit))
            buf_len += len(unit) + 1
        flush_buf()
        return chunks

    # ----------------------------------------------------------------
    def _split_by_structure(self, text: str) -> List[tuple]:
        """按法规结构边界切成 (section_title, unit_text) 列表。"""
        text = text.strip()
        # 收集全部结构边界位置
        cuts = [0]
        for rx in self.structure_res:
            for m in rx.finditer(text):
                pos = m.start()
                # 跳过开头的换行，定位到真正的标题字符
                while pos < len(text) and text[pos] == "\n":
                    pos += 1
                cuts.append(pos)
        cuts = sorted(set(cuts))
        parts = []
        for a, b in zip(cuts, cuts[1:] + [len(text)]):
            seg = text[a:b].strip()
            if seg:
                parts.append((self._extract_title(seg), seg))
        return parts or [("", text)]

    def _extract_title(self, seg: str) -> str:
        head = seg[:60].replace("\n", " ")
        for rx in self.title_res:
            m = rx.search(head)
            if m:
                return m.group(0)
        # 否则取首行前 24 字作为定位标题
        first = seg.split("\n")[0][:24]
        return first

    # ----------------------------------------------------------------
    def _split_long_unit(self, unit: str) -> List[str]:
        """超长单元：按段落 -> 句子贪心打包，相邻块带 overlap。"""
        paragraphs = [p.strip() for p in unit.split("\n") if p.strip()]
        sentences: List[str] = []
        for p in paragraphs:
            sentences.extend(self._split_sentences(p))

        chunks, cur, cur_len = [], [], 0
        for s in sentences:
            if cur_len + len(s) > self.size and cur:
                chunks.append("".join(cur))
                # overlap：携带上一块尾部内容
                tail = chunks[-1][-self.overlap:]
                cur, cur_len = [tail], len(tail)
            cur.append(s)
            cur_len += len(s)
        if cur:
            chunks.append("".join(cur))

        # 单句就超长的极端情况：硬切
        final = []
        for c in chunks:
            if len(c) <= self.size * 1.5:
                final.append(c)
            else:
                for i in range(0, len(c), self.size - self.overlap):
                    final.append(c[i:i + self.size])
        return final

    def _split_sentences(self, paragraph: str) -> List[str]:
        out, buf = [], ""
        for ch in paragraph:
            buf += ch
            if ch in self.SENTENCE_ENDS:
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
        return out

    # ----------------------------------------------------------------
    @staticmethod
    def _child(parent: Document, content: str, title: str, idx: int) -> Document:
        """用内容哈希生成确定性 chunk_id：相同文件/位置/内容 → 同一 id，便于去重与缓存。"""
        content = content.strip()
        meta = dict(parent.metadata)
        # 哈希须含页码（同表格块：避免不同页的相同前缀块撞 id）
        chunk_id = hashlib.sha256(
            f"{meta.get('source', '')}:{meta.get('page', '')}:{idx}:{content[:256]}".encode()
        ).hexdigest()[:16]
        meta.update({
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "section_title": title or meta.get("section_title", ""),
        })
        return Document(page_content=content, metadata=meta)


# ====================================================================
# 门面类
# ====================================================================
class EHSTextSplitter:
    """对外统一分块入口：
        splitter = EHSTextSplitter()
        chunks = splitter.split_documents(docs)
    """

    def __init__(self, config: Settings = settings):
        self.config = config
        self.table_splitter = TableSplitter(config)
        self.text_splitter = SemanticTextSplitter(config)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        n_table = 0
        for doc in documents:
            if doc.metadata.get("doc_type") == "table":
                pieces = self.table_splitter.split(doc)
                n_table += 1
            else:
                pieces = self.text_splitter.split(doc)
            chunks.extend(pieces)

        # 全局重排 chunk_index（按文件/页顺序），并统计
        logger.info("分块完成：%d 个原始文档（含 %d 个表格文档）-> %d 个 chunk",
                    len(documents), n_table, len(chunks))
        return chunks

    def split_one(self, doc: Document) -> List[Document]:
        if doc.metadata.get("doc_type") == "table":
            return self.table_splitter.split(doc)
        return self.text_splitter.split(doc)
