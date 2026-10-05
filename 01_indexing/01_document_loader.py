# -*- coding: utf-8 -*-
"""
文档加载模块（面向对象）
--------------------------------------------------------------------
统一把 EHS 知识库中的 TXT / MD / PDF / Word / Excel 解析为 LangChain
Document，并对表格做结构化文本转换、对图注做识别标记。

设计：
    BaseLoader                 抽象加载器，约定 load() 接口
    TextFileLoader             .txt / .md 纯文本与 Markdown
    PDFLoader                  .pdf  pypdf 文本 + pdfplumber 表格 + 图注
    WordLoader                 .docx 段落 + 表格（保持文档顺序）
    ExcelLoader                .xlsx / .xls 工作表转结构化文本
    EHSDocumentLoader          门面类：目录遍历 + 后缀分发 + 元数据注入
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
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config import Settings, settings

logger = logging.getLogger(__name__)

# 图注 / 表注编号：图3-1、图 3.1.2、表5.2-3 等
_FIGURE_CAPTION_RE = re.compile(r"^\s*(图|表)\s*\d+([-.．]\d+)*")
# 页眉页脚里常见的纯编号行（单独成行的数字），用于轻量清洗
_ONLY_DIGITS_RE = re.compile(r"^\s*\d{1,4}\s*$")
# CJK 统一表意文字（用于判断提取文本是否为乱码）
_CJK_RE = re.compile(r"[一-鿿]")


# ====================================================================
# 文本层质量判定（识别"乱码 PDF"，触发 OCR 兜底）
# ====================================================================
def cjk_ratio(text: str) -> float:
    """非空白字符中 CJK 汉字的占比。"""
    s = re.sub(r"\s", "", text or "")
    if not s:
        return 0.0
    return len(_CJK_RE.findall(s)) / len(s)


def text_is_garbled(text: str, min_cjk_ratio: float = 0.40) -> bool:
    """判断提取文本是否为乱码。

    部分 PDF 使用 CID 字体但缺少 ToUnicode 映射，pypdf 会提取出形如
    "OOb?W蜰a^鷭緪鐿醏oQl_" 的乱码：长度看似正常，汉字占比却极低。
    """
    s = (text or "").strip()
    if not s:
        return True
    return cjk_ratio(s) < min_cjk_ratio


def table_looks_garbage(rows: List[List[Optional[str]]],
                        cell_ratio: float = 0.7, min_cells: int = 8) -> bool:
    """判断 pdfplumber 抽出的表格是否为噪声（在图片页上误检）。

    噪声表格的单元格普遍只剩 1~2 个字（如 | 以 | 黑 |），正常表格单元格
    多为词/短语。碎片占比超过阈值即判定为噪声。
    """
    cells = [str(c).strip() for row in rows for c in row if c not in (None, "")]
    if len(cells) < min_cells:
        return False
    frag = sum(1 for c in cells if len(c) <= 2)
    return frag / len(cells) > cell_ratio


# ====================================================================
# 工具函数
# ====================================================================
def clean_text(text: str) -> str:
    """轻量文本清洗：统一换行、去除多余空白行、清除非法 Unicode（孤立代理字符）。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF 提取可能产生孤立代理字符（如 \udbc0 复选框乱码），Rust tokenizer 会拒绝，
    # 用 replace 规范化为 U+FFFD
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # 3 个以上换行压缩为 2 个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除每行尾部空白
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def table_to_markdown(rows: List[List[Optional[str]]], caption: str = "") -> str:
    """把二维表格数据转成 Markdown 表格文本（结构化文本，利于检索与 LLM 理解）。"""
    # 去掉整行全空的行
    rows = [["" if c is None else str(c).strip().replace("\n", " ") for c in row]
            for row in rows if row and any(c not in (None, "") for c in row)]
    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    lines = []
    if caption:
        lines.append(f"【表格：{caption.strip()}】")
    else:
        lines.append("【表格】")
    lines.append("| " + " | ".join(h or " " for h in header) + " |")
    lines.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in body:
        lines.append("| " + " | ".join(c or " " for c in r) + " |")
    return "\n".join(lines)


# ====================================================================
# 抽象基类
# ====================================================================
class BaseLoader(ABC):
    """文档加载器抽象基类。"""

    def __init__(self, config: Settings):
        self.config = config

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """把单个文件解析为若干 Document。"""
        raise NotImplementedError

    @staticmethod
    def _base_metadata(file_path: Path) -> dict:
        """所有文档共用的来源元数据。"""
        kb_root = settings.knowledge_base_dir
        try:
            rel = file_path.relative_to(kb_root)
            category = rel.parts[0] if len(rel.parts) > 1 else "根目录"
            rel_path = str(rel)
        except ValueError:
            category, rel_path = "外部", str(file_path)
        return {
            "source": str(file_path),
            "file_name": file_path.name,
            "rel_path": rel_path,
            "category": category,
        }


# ====================================================================
# 1) TXT / Markdown
# ====================================================================
class TextFileLoader(BaseLoader):
    def load(self, file_path: Path) -> List[Document]:
        text = None
        last_err = None
        for enc in self.config.document.text_encodings:
            try:
                text = file_path.read_text(encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        if text is None:
            logger.warning("文件 %s 编码无法识别，已忽略错误字节: %s", file_path, last_err)
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        text = clean_text(text)
        if not text:
            return []

        # 识别图注/表注行并单独标记（txt 法规中较少，md 中可能有）
        doc_type = "markdown" if file_path.suffix.lower() == ".md" else "text"
        meta = self._base_metadata(file_path)
        meta.update({"page": None, "doc_type": doc_type})
        return [Document(page_content=text, metadata=meta)]


# ====================================================================
# 2) PDF（文本 + 表格 + 图注）
# ====================================================================
class PDFLoader(BaseLoader):
    """PDF 加载器：
    - 默认 pypdf 快速抽取每页文本；
    - 文本层不可用（扫描版 / CID 乱码 / 仅含水印）时自动 OCR 兜底：
      · 整册不可用 -> 全册 OCR；个别页为空/乱码 -> 仅该页 OCR（混合型 PDF）；
    - 中小文件额外用 pdfplumber 抽取表格，转 Markdown 结构化文本，
      并过滤掉在图片页上误检出的碎片化"假表格"；
    - 大文件（>large_file_mb 或页数超阈值）自动降级为纯文本，防止 CPU 卡死；
    - 图注行（图X-X ...）打上 figure_caption 标记。
    """

    def __init__(self, config: Settings = settings):
        super().__init__(config)
        self._ocr = None            # 惰性加载的 OCR 引擎（整进程共享单例）
        self._pdfium_doc = None     # 当前文件的 pypdfium2 文档句柄
        self._pdfium_path: Optional[str] = None

    # ---------------- OCR 兜底 ----------------
    def _ensure_ocr(self):
        if self._ocr is None:
            from ocr_module import RapidOCRExtractor
            self._ocr = RapidOCRExtractor()
        return self._ocr

    def _get_pdfium(self, file_path: Path):
        import pypdfium2 as pdfium
        if self._pdfium_doc is None or self._pdfium_path != str(file_path):
            self._close_pdfium()
            self._pdfium_doc = pdfium.PdfDocument(str(file_path))
            self._pdfium_path = str(file_path)
        return self._pdfium_doc

    def _close_pdfium(self) -> None:
        if self._pdfium_doc is not None:
            try:
                self._pdfium_doc.close()
            except Exception:
                pass
        self._pdfium_doc, self._pdfium_path = None, None

    def _ocr_page(self, file_path: Path, idx: int) -> str:
        """把 PDF 第 idx 页渲染为图片后 OCR，返回识别文本（失败返回空串）。"""
        try:
            doc = self._get_pdfium(file_path)
            image = doc[idx].render(scale=self.config.ocr.render_dpi / 72.0).to_pil()
            return self._ensure_ocr().recognize(image)
        except Exception as e:
            logger.warning("第 %d 页 OCR 失败（%s）：%s", idx + 1, e, file_path.name)
            return ""

    def _text_layer_unusable(self, pages_text: List[str], n_pages: int) -> bool:
        """判断整册 PDF 的文本层是否无实际内容（需全册 OCR）。"""
        dc = self.config.document
        if n_pages == 0:
            return False
        # 平均每页文字过少 -> 只是水印/页眉，无正文
        if sum(len(t) for t in pages_text) / n_pages < dc.min_chars_per_page:
            return True
        nonempty = [t for t in pages_text if t.strip()]
        if not nonempty:
            return True
        # 过半页面是乱码 -> 文本层不可用
        garbled = sum(1 for t in nonempty if text_is_garbled(t, dc.min_cjk_ratio))
        return garbled / len(nonempty) > 0.5

    def _need_tables(self, file_path: Path, n_pages: int) -> bool:
        if not self.config.document.extract_pdf_tables:
            return False
        size_mb = file_path.stat().st_size / 1024 / 1024
        if size_mb > self.config.document.large_file_mb:
            logger.info("PDF %.1fMB 超过 %.0fMB，跳过表格提取：%s",
                        size_mb, self.config.document.large_file_mb, file_path.name)
            return False
        if n_pages > self.config.document.max_pages_for_table:
            logger.info("PDF %d 页超过 %d 页阈值，跳过表格提取：%s",
                        n_pages, self.config.document.max_pages_for_table, file_path.name)
            return False
        return True

    def load(self, file_path: Path, ocr_fallback: bool = True) -> List[Document]:
        from pypdf import PdfReader

        docs: List[Document] = []
        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            logger.error("pypdf 无法打开 %s: %s", file_path, e)
            return []

        dc = self.config.document
        n_pages = len(reader.pages)

        # ---- 第一遍：抽取每页文本层 ----
        pages_text: List[str] = []
        for i, page in enumerate(reader.pages):
            try:
                t = clean_text(page.extract_text() or "")
            except Exception as e:  # 单页失败不影响整体
                logger.warning("pypdf 第 %d 页提取失败(%s)：%s", i + 1, e, file_path.name)
                t = ""
            pages_text.append(t)

        # ---- 判断文本层是否可用；不可用则整册 OCR ----
        ocr_all = ocr_fallback and self._text_layer_unusable(pages_text, n_pages)
        if ocr_all:
            logger.info("文本层不可用（扫描版/乱码/仅水印），转 OCR：%s", file_path.name)

        # OCR 的页面不再抽取 pdfplumber 表格（图片页上只会抽出碎片噪声）
        extract_tables = self._need_tables(file_path, n_pages) and not ocr_all
        plumber_pages = self._open_pdfplumber(file_path) if extract_tables else {}

        try:
            for i, page_text in enumerate(pages_text):
                page_no = i + 1
                is_ocr = False

                # 逐页 OCR 兜底：空页 / 乱码页（混合型 PDF 的图片页）
                if ocr_fallback and (ocr_all or text_is_garbled(page_text, dc.min_cjk_ratio)):
                    ocr_text = clean_text(self._ocr_page(file_path, i))
                    if ocr_text and not text_is_garbled(ocr_text, dc.min_cjk_ratio):
                        page_text, is_ocr = ocr_text, True

                table_md = ""
                if extract_tables and not is_ocr:
                    table_md = self._extract_page_tables(plumber_pages, i, page_no)

                if not page_text and not table_md:
                    continue

                # 图注标记：本页是否含图注
                has_figure = any(_FIGURE_CAPTION_RE.match(ln) for ln in page_text.split("\n"))
                # 去掉孤立页码行
                page_text = "\n".join(
                    ln for ln in page_text.split("\n") if not _ONLY_DIGITS_RE.match(ln)
                ).strip()

                content = page_text
                if table_md:
                    content = (content + "\n\n" + table_md).strip()
                if not content:
                    continue

                meta = self._base_metadata(file_path)
                meta.update({
                    "page": page_no,
                    "total_pages": n_pages,
                    "doc_type": "ocr_text" if is_ocr and not table_md else (
                        "mixed" if table_md else
                        ("figure_caption" if has_figure else "text")),
                    "has_table": bool(table_md),
                    "has_figure": has_figure,
                    "is_ocr": is_ocr,
                })
                docs.append(Document(page_content=content, metadata=meta))
        finally:
            if plumber_pages:
                for pg in plumber_pages.values():
                    try:
                        pg.close()
                    except Exception:
                        pass
            self._close_pdfium()

        n_ocr = sum(1 for d in docs if d.metadata.get("is_ocr"))
        logger.info("PDF 解析完成：%s，%d 页 -> %d 个文档（其中 OCR %d 页）",
                    file_path.name, n_pages, len(docs), n_ocr)
        return docs

    def _open_pdfplumber(self, file_path: Path) -> dict:
        try:
            import pdfplumber
            pdf = pdfplumber.open(str(file_path))
            return {pg.page_number - 1: pg for pg in pdf.pages}
        except Exception as e:
            logger.warning("pdfplumber 打开 %s 失败，降级纯文本: %s", file_path.name, e)
            return {}

    def _extract_page_tables(self, plumber_pages: dict, idx: int, page_no: int) -> str:
        page = plumber_pages.get(idx)
        if page is None:
            return ""
        try:
            tables = page.extract_tables()
        except Exception as e:
            logger.warning("pdfplumber 第 %d 页表格提取失败: %s", page_no, e)
            return ""
        blocks = []
        dc = self.config.document
        for tbl in tables or []:
            # 过滤 pdfplumber 在图片页上误检出的碎片化"假表格"
            if table_looks_garbage(tbl, dc.table_garbage_cell_ratio,
                                   dc.table_garbage_min_cells):
                continue
            md = table_to_markdown(tbl)
            if md:
                blocks.append(md)
        return "\n\n".join(blocks)


# ====================================================================
# 3) Word（.docx，段落 + 表格保持顺序）
# ====================================================================
class WordLoader(BaseLoader):
    def load(self, file_path: Path) -> List[Document]:
        import docx
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        try:
            doc = docx.Document(str(file_path))
        except Exception as e:
            logger.error("python-docx 无法打开 %s: %s", file_path, e)
            return []

        para_buf: List[str] = []
        table_docs: List[Document] = []
        all_docs: List[Document] = []

        def flush_paragraphs():
            text = clean_text("\n".join(para_buf))
            para_buf.clear()
            if text:
                meta = WordLoader._base_metadata(file_path)
                meta.update({"page": None, "doc_type": "text"})
                all_docs.append(Document(page_content=text, metadata=meta))

        # 按 body 真实顺序遍历段落与表格
        for child in doc.element.body.iterchildren():
            if child.tag == qn("w:p"):
                p = Paragraph(child, doc)
                t = clean_text(p.text)
                if t:
                    # 记录标题层级，便于语义分块
                    style = (p.style.name or "") if p.style else ""
                    if style.lower().startswith("heading"):
                        para_buf.append(f"\n## {t}")
                    else:
                        para_buf.append(t)
            elif child.tag == qn("w:tbl"):
                flush_paragraphs()
                tbl = Table(child, doc)
                rows = [[cell.text for cell in row.cells] for row in tbl.rows]
                md = table_to_markdown(rows)
                if md:
                    meta = self._base_metadata(file_path)
                    meta.update({"page": None, "doc_type": "table"})
                    table_docs.append(Document(page_content=md, metadata=meta))
                    all_docs.append(table_docs[-1])

        flush_paragraphs()
        logger.info("Word 解析完成：%s，段落文档 %d，表格 %d",
                    file_path.name, len(all_docs) - len(table_docs), len(table_docs))
        return all_docs


# ====================================================================
# 4) Excel（.xlsx / .xls）
# ====================================================================
class ExcelLoader(BaseLoader):
    MAX_ROWS_PER_SHEET = 2000  # 单表最大行数，防止超大台账拖慢

    def load(self, file_path: Path) -> List[Document]:
        try:
            import openpyxl
        except ImportError:
            logger.error("未安装 openpyxl，无法解析 Excel：%s", file_path)
            return []

        try:
            wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        except Exception as e:
            logger.error("openpyxl 无法打开 %s: %s", file_path, e)
            return []

        docs: List[Document] = []
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= self.MAX_ROWS_PER_SHEET:
                    break
                rows.append(list(row))
            md = table_to_markdown(rows, caption=f"工作表 {ws.title}")
            if md:
                meta = self._base_metadata(file_path)
                meta.update({"page": None, "doc_type": "table", "sheet": ws.title})
                docs.append(Document(page_content=md, metadata=meta))
        return docs


# ====================================================================
# 门面类：统一入口
# ====================================================================
class EHSDocumentLoader:
    """对外暴露的文档加载门面：
        loader = EHSDocumentLoader()
        docs = loader.load_directory()           # 加载整个知识库
        docs = loader.load_file(path)            # 加载单个文件（增量入库用）
    """

    def __init__(self, config: Settings = settings):
        self.config = config
        self._loaders = {
            ".txt": TextFileLoader(config),
            ".md": TextFileLoader(config),
            ".pdf": PDFLoader(config),
            ".docx": WordLoader(config),
            ".doc": WordLoader(config),
            ".xlsx": ExcelLoader(config),
            ".xls": ExcelLoader(config),
        }

    def load_file(self, file_path: Path | str) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        suffix = file_path.suffix.lower()
        loader = self._loaders.get(suffix)
        if loader is None:
            logger.warning("不支持的文件类型，已跳过：%s", file_path)
            return []
        try:
            return loader.load(file_path)
        except Exception as e:
            logger.exception("解析 %s 时发生未预期错误: %s", file_path, e)
            return []

    def load_directory(self, root: Path | str | None = None,
                       exclude: tuple = ("README.md",)) -> List[Document]:
        root = Path(root) if root else self.config.knowledge_base_dir
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在：{root}")

        all_docs: List[Document] = []
        files = sorted(p for p in root.rglob("*") if p.is_file())
        supported = self.config.document.supported_extensions
        n_used = 0
        for fp in files:
            if fp.suffix.lower() not in supported:
                continue
            if fp.name in exclude:
                # README 是目录说明，默认不入库（避免污染检索）
                continue
            docs = self.load_file(fp)
            if docs:
                all_docs.extend(docs)
                n_used += 1
        logger.info("目录加载完成：%s，扫描 %d 个文件，产出 %d 个原始文档",
                    root, n_used, len(all_docs))
        return all_docs
