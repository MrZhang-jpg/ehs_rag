# -*- coding: utf-8 -*-
"""
扫描版 PDF 的 OCR 识别模块（面向对象，CPU 运行）
--------------------------------------------------------------------
- RapidOCRExtractor：封装 rapidocr-onnxruntime（纯 onnxruntime，轻量、中文友好）；
- ScannedPDFOCRLoader：用 pypdfium2 把扫描版 PDF 每页渲染为图片（200dpi），
  逐页 OCR，产出与 PDFLoader 同构的 Document（doc_type=ocr_text），保留页码；
- find_scanned_pdfs：自动检测"文本提取为空"的扫描版 PDF；
- ingest_scanned_pdfs：OCR -> 语义分块 -> 向量入库 -> BM25 合并重建，全流程闭环。

用法：
    python ocr_module.py                 # 自动检测并 OCR 入库全部扫描版 PDF
    python ocr_module.py --file xxx.pdf  # 仅处理指定文件
    python ocr_module.py --max-pages 10  # 每个文件最多处理前 N 页（快速验证）
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


import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config import Settings, settings
from document_loader import cjk_ratio, clean_text, text_is_garbled

logger = logging.getLogger(__name__)

RENDER_DPI = 200                 # 渲染分辨率（默认值，可由 config.ocr.render_dpi 覆盖）
SCANNED_SAMPLE_PAGES = 6         # 检测扫描版时抽样页数


# ====================================================================
# OCR 引擎封装
# ====================================================================
class RapidOCRExtractor:
    """RapidOCR（onnxruntime）封装，惰性单例。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._engine = None
            cls._instance = inst
        return cls._instance

    def _ensure(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("加载 RapidOCR 引擎（首次会初始化 onnx 模型）...")
            threads = getattr(settings.ocr, "intra_op_threads", 4)
            try:
                self._engine = RapidOCR(intra_op_num_threads=threads)
            except TypeError:
                # 老版本无该参数
                self._engine = RapidOCR()
        return self._engine

    def recognize(self, image) -> str:
        """对 PIL.Image / np.ndarray 做 OCR，返回按行拼接的文本。"""
        import numpy as np
        if hasattr(image, "mode"):  # PIL Image
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)
        engine = self._ensure()
        result, _elapse = engine(image)
        if not result:
            return ""
        # result: [[box(4点), text, score], ...]，通常已自上而下排序
        lines = [item[1].strip() for item in result if item and item[1]]
        return "\n".join(ln for ln in lines if ln)


# ====================================================================
# 扫描版 PDF 检测 + OCR 加载
# ====================================================================
class ScannedPDFOCRLoader:
    """把扫描版 PDF 经 OCR 转为 Document。"""

    def __init__(self, config: Settings = settings,
                 ocr: Optional[RapidOCRExtractor] = None):
        self.config = config
        self.ocr = ocr or RapidOCRExtractor()

    @staticmethod
    def needs_ocr(file_path: Path | str, config: Settings = settings) -> bool:
        """判断 PDF 是否需要 OCR（抽样多页综合判定）：

        1) 抽样页提取不到文本（空白）-> 扫描版；
        2) 抽样页文字是乱码（CID 字体缺 ToUnicode 映射，汉字占比极低）-> 假文本层；
        3) 抽样页平均字符数过低（仅水印/页眉，无正文）-> 图片版。

        旧实现只统计"总字符数 < 20"，会漏掉 2)、3) 两类（每页都有几十个
        水印字符或乱码字符的 PDF），导致这些规范的正文从未入库。
        """
        from pypdf import PdfReader

        dc = config.document
        try:
            reader = PdfReader(str(file_path))
        except Exception:
            return False
        n = len(reader.pages)
        if n == 0:
            return True

        # 均匀抽样：前若干页 + 正文中段落，避免只看封面/目录误判
        idxs = list(range(min(n, SCANNED_SAMPLE_PAGES)))
        if n > SCANNED_SAMPLE_PAGES * 2:
            step = max(1, (n - SCANNED_SAMPLE_PAGES) // SCANNED_SAMPLE_PAGES)
            idxs += list(range(SCANNED_SAMPLE_PAGES, n, step))[:SCANNED_SAMPLE_PAGES]

        texts: List[str] = []
        for i in idxs:
            try:
                texts.append((reader.pages[i].extract_text() or "").strip())
            except Exception:
                texts.append("")
        if not texts:
            return True
        if sum(len(t) for t in texts) == 0:
            return True
        # 平均每页字符过少 -> 文本层只是水印/页眉
        if sum(len(t) for t in texts) / len(texts) < dc.min_chars_per_page:
            return True
        # 过半抽样页为乱码 -> 文本层不可用
        nonempty = [t for t in texts if t]
        garbled = sum(1 for t in nonempty if text_is_garbled(t, dc.min_cjk_ratio))
        return bool(nonempty) and garbled > len(nonempty) / 2

    # 兼容旧调用名
    is_scanned = needs_ocr

    def find_scanned_pdfs(self, root: Optional[Path] = None) -> List[Path]:
        root = root or self.config.knowledge_base_dir
        scanned = []
        for p in sorted(root.rglob("*.pdf")):
            if self.needs_ocr(p, self.config):
                scanned.append(p)
        logger.info("检测到 %d 个需要 OCR 的 PDF：%s", len(scanned),
                    [p.name for p in scanned])
        return scanned

    def load(self, file_path: Path | str,
             max_pages: Optional[int] = None) -> List[Document]:
        import pypdfium2 as pdfium
        file_path = Path(file_path)
        pdf = pdfium.PdfDocument(str(file_path))
        n = len(pdf)
        limit = min(n, max_pages) if max_pages else n
        scale = self.config.ocr.render_dpi / 72.0

        docs: List[Document] = []
        t0 = time.time()
        for i in range(limit):
            page = pdf[i]
            image = page.render(scale=scale).to_pil()
            text = self.ocr.recognize(image)
            if text and text.strip():
                meta = self._metadata(file_path, i + 1, n)
                docs.append(Document(page_content=text.strip(), metadata=meta))
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                logger.info("OCR 进度：%s %d/%d 页（%.1f 页/秒）",
                            file_path.name, i + 1, limit, rate)
        logger.info("OCR 完成：%s，%d 页 -> %d 个文档，耗时 %.1fs",
                    file_path.name, limit, len(docs), time.time() - t0)
        return docs

    @staticmethod
    def _metadata(file_path: Path, page_no: int, total: int) -> dict:
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
            "page": page_no,
            "total_pages": total,
            "doc_type": "ocr_text",
            "has_table": False,
            "has_figure": False,
            "is_ocr": True,
        }


# ====================================================================
# 全流程：OCR -> 分块 -> 入库
# ====================================================================
def _ocr_file_worker(args: tuple) -> tuple:
    """多进程 worker：OCR 单个 PDF，返回 (文件名, [(正文, 元数据), ...])。

    OCR 是 CPU 密集且单页耗时数秒，多进程并行（每进程一份 onnx 模型）
    可将全库 OCR 从数小时压到几十分钟。
    """
    path_str, max_pages = args
    loader = ScannedPDFOCRLoader()
    docs = loader.load(Path(path_str), max_pages=max_pages)
    return Path(path_str).name, [(d.page_content, d.metadata) for d in docs]


def ingest_scanned_pdfs(config: Settings = settings,
                        files: Optional[List[Path]] = None,
                        max_pages: Optional[int] = None,
                        workers: Optional[int] = None) -> dict:
    from main import EHSRAGApplication

    loader = ScannedPDFOCRLoader(config)
    files = files or loader.find_scanned_pdfs()
    if not files:
        logger.info("没有需要 OCR 的 PDF。")
        return {"files": [], "chunks": 0}

    workers = workers or config.ocr.workers
    docs_by_file: dict = {}
    if workers > 1 and len(files) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info("OCR 并行入库：%d 个文件 / %d 进程", len(files), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ocr_file_worker,
                                   (str(fp), max_pages)): fp for fp in files}
            for fut in as_completed(futures):
                name, items = fut.result()
                docs_by_file[name] = items
                logger.info("OCR 完成：%s（%d 页有文字）", name, len(items))
    else:
        docs_by_file = {fp.name: [(d.page_content, d.metadata)
                                  for d in loader.load(fp, max_pages=max_pages)]
                        for fp in files}

    # 主进程顺序做分块 / 向量入库 / BM25 追加（嵌入与索引非进程安全）
    app = EHSRAGApplication(config)
    total_chunks = 0
    for fp in files:
        items = docs_by_file.get(fp.name, [])
        docs = [Document(page_content=c, metadata=m) for c, m in items]
        chunks = app.splitter.split_documents(docs) if docs else []
        n_vec = app.vector_store.add_documents(chunks)
        app.retriever.bm25.append(chunks)
        total_chunks += len(chunks)
        logger.info("  %s：OCR文档 %d，块 %d，入库 %d",
                    fp.name, len(docs), len(chunks), n_vec)

    result = {"files": [f.name for f in files],
              "chunks": total_chunks,
              "vectorstore_total": app.vector_store.count()}
    logger.info("扫描版 OCR 入库全部完成：%s", result)
    return result


# ====================================================================
# CLI
# ====================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="扫描版 PDF OCR 入库")
    parser.add_argument("--file", default=None, help="仅处理指定 PDF")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="每个文件最多处理页数")
    parser.add_argument("--workers", type=int, default=None,
                        help="OCR 并行进程数（默认取 config.ocr.workers）")
    args = parser.parse_args()

    if args.file:
        ingest_scanned_pdfs(files=[Path(args.file)], max_pages=args.max_pages,
                            workers=args.workers)
    else:
        ingest_scanned_pdfs(max_pages=args.max_pages, workers=args.workers)


if __name__ == "__main__":
    main()
