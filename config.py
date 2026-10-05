# -*- coding: utf-8 -*-
"""
EHS 知识库 RAG 问答系统 —— 全局配置
所有模块统一从这里读取配置，避免硬编码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ----------------------------------------------------------------------
# 0a) 设备（CPU / GPU）选择 —— 代码级硬约束（必须在 import torch 之前处理）
#     用环境变量 EHS_DEVICE 控制，三档：
#       · auto（默认）：有可用 GPU 就用 GPU，否则回退 CPU；
#       · cpu：在 import torch 前设置 CUDA_VISIBLE_DEVICES=""，让进程彻底
#               "看不见"任何 GPU，从根上杜绝误走 GPU；
#       · cuda：强制 GPU（无可用 GPU 时由 resolve_device() 回退 CPU）。
#     注意：若环境中安装的是 CPU 版 torch（版本号带 +cpu），即使选 auto/cuda
#     也仍会回落到 CPU —— 需另行安装 CUDA 版 torch 才能真正调用 GPU。
# ----------------------------------------------------------------------
DEVICE_PREFERENCE = os.environ.get("EHS_DEVICE", "auto").strip().lower()
if DEVICE_PREFERENCE not in ("auto", "cpu", "cuda"):
    DEVICE_PREFERENCE = "auto"
if DEVICE_PREFERENCE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def resolve_device(preferred: str = "auto") -> str:
    """在 torch 可导入后解析实际运行设备（模型加载时调用，此时 torch 已可用）。

    - preferred="cpu"：恒为 cpu；
    - preferred="cuda"：CUDA 可用则 cuda，否则回退 cpu；
    - preferred="auto"：自动按 CUDA 可用性选择。
    """
    preferred = (preferred or "auto").strip().lower()
    if preferred == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    if preferred == "cuda":
        return "cuda" if cuda_ok else "cpu"
    return "cuda" if cuda_ok else "cpu"

# ----------------------------------------------------------------------
# 0b) HuggingFace 环境变量（必须在 import sentence-transformers / huggingface
#    之前设置）：默认走 hf-mirror 镜像；模型已在本地缓存时直接离线加载，
#    避免每次启动联网 HEAD 检查（huggingface.co 不通会卡顿数分钟）。
# ----------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _is_model_cached(repo_id: str) -> bool:
    """判断 HuggingFace 本地缓存中是否已存在某模型（含 config.json）。"""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    # 也尊重 HF_HOME
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub = Path(hf_home) / "hub"
    repo_dir = hub / ("models--" + repo_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    snap_root = repo_dir / "snapshots"
    return any((s / "config.json").exists() for s in snap_root.glob("*"))


# 所需本地模型均已缓存 -> 开启离线模式（新环境首次下载时这些变量不要预设）
_REQUIRED_LOCAL_MODELS = ("BAAI/bge-base-zh-v1.5", "BAAI/bge-reranker-base")
if all(_is_model_cached(m) for m in _REQUIRED_LOCAL_MODELS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ----------------------------------------------------------------------
# 1) 路径配置
# ----------------------------------------------------------------------
# 项目根目录（本文件所在目录）
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# 原始 EHS 知识库数据目录（D:\ehs_rag1）
KNOWLEDGE_BASE_DIR: Path = Path(r"D:\ehs_rag1")

# 持久化产物目录
DATA_DIR: Path = PROJECT_ROOT / "data"
VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"          # 旧 Chroma 持久化目录（保留备份）
MILVUS_DIR: Path = DATA_DIR / "milvus"                    # Milvus Lite 数据目录
MILVUS_DB: Path = MILVUS_DIR / "ehs_milvus.db"           # Milvus Lite 单文件数据库
BM25_DIR: Path = DATA_DIR / "bm25"                        # BM25 索引持久化目录
EVAL_DIR: Path = PROJECT_ROOT / "evaluation"              # 评测集与评测报告
LOGS_DIR: Path = PROJECT_ROOT / "logs"                    # 运行日志

for _d in (DATA_DIR, VECTORSTORE_DIR, MILVUS_DIR, BM25_DIR, EVAL_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 2) OpenRouter 云端大模型配置（问答 / LLM 裁判）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    # 生成后端：ollama（本地，离线）/ openrouter（云端，需 API Key 与配额）
    provider: str = "ollama"
    model: str = "deepseek-r1"
    # 本地 Ollama 服务地址（provider=ollama 时生效）
    ollama_base_url: str = "http://127.0.0.1:11434"
    # 云端 OpenRouter（provider=openrouter 时生效；model 需换成对应云端模型名）
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""                 # 请通过环境变量 OPENROUTER_API_KEY 注入
    temperature: float = 0.2          # 法规问答要求稳定、低发散
    max_tokens: int = 4096            # 生成上限（Ollama 映射为 num_predict）
    # 上下文窗口：RAG 检索到的原文（5 块×约 450 字）远超 Ollama 默认 2048，
    # 不显式放大 num_ctx 会导致检索到的原文被静默截断，答案缺依据。
    num_ctx: int = 8192
    # 推理模型是否启用"思考"：问答场景关闭以提高响应速度、避免思考文本混入答案
    reasoning: bool = False
    reasoning_effort: str = "low"     # provider=openrouter 时的推理强度：low/medium/high
    timeout: int = 180                # 单次请求超时（秒）
    max_retries: int = 5              # 限流/超时最大重试次数
    # 限流保护：两次请求之间的最小间隔（秒）；本地 Ollama 无需限流
    min_request_interval: float = 0.0


# ----------------------------------------------------------------------
# 3) 本地 BGE 嵌入模型配置（device=auto 时自动选择 CPU/GPU）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    # 本地 BGE 中文嵌入模型（首次运行自动从 hf-mirror 下载，之后走本地缓存）
    model_name: str = "BAAI/bge-base-zh-v1.5"
    # auto：自动选择（有 CUDA 用 GPU，否则 CPU）；也可显式写 "cpu" / "cuda"
    device: str = "auto"
    # bge-base-zh-v1.5 输出维度 768；bge-small-zh-v1.5 为 512
    dimension: int = 768
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    # CPU 批量编码大小（过大占内存，过小速度慢）
    batch_size: int = 64
    # 为 BGE 检索添加的中文指令（bge 官方建议检索侧加 instruction，文档侧不加）
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："


# ----------------------------------------------------------------------
# 4) 本地 Rerank 重排序模型配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    # auto：自动选择（有 CUDA 用 GPU，否则 CPU）；也可显式写 "cpu" / "cuda"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512


# ----------------------------------------------------------------------
# 4b) Milvus 向量库配置（Milvus Lite，本地单文件，无需 Docker）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MilvusConfig:
    collection_name: str = "ehs_knowledge_base"   # 集合名
    # Milvus Lite 推荐 AUTOINDEX（自动选择合适索引）；也可 HNSW / FLAT
    index_type: str = "AUTOINDEX"
    metric_type: str = "COSINE"                    # 向量已归一化，用余弦相似度
    auto_id: bool = False                          # 用 chunk_id 作主键，不自动生成
    primary_field: str = "pk"                      # 主键字段名
    text_field: str = "text"                       # 正文字段名
    vector_field: str = "vector"                   # 向量字段名
    # 开启动态字段：metadata 各类 key（source/file_name/page 等）以动态列存储，
    # 无需为每个 key 预定义 schema，且可用于 filter 表达式（如按 source 删除）
    enable_dynamic_field: bool = True


# ----------------------------------------------------------------------
# 5) 文档解析与语义分块配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentConfig:
    # 支持的文件后缀
    supported_extensions: tuple = (".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls")
    # PDF 解析：是否启用 pdfplumber 表格提取（大文件较慢，可关闭）
    extract_pdf_tables: bool = True
    # 单文件大小上限（MB），超过则表格提取降级为纯文本提取，防止 CPU 上卡死
    large_file_mb: float = 30.0
    # 超大 PDF 页数上限（防止 66MB 规范全量表格解析过慢）
    max_pages_for_table: int = 120
    # 文本编码（读取 txt/md 时尝试的编码顺序）
    text_encodings: tuple = ("utf-8", "gb18030", "gbk")
    # ---- 文本层质量判定（用于自动识别"乱码 PDF"并触发 OCR）----
    # 正常中文法规文本的 CJK 字符占比远高于此值；CID 字体缺 ToUnicode 映射的
    # PDF 会提取出乱码（如 "OOb?W蜰a^鷭緪鐿醏oQl_"），CJK 占比极低。
    min_cjk_ratio: float = 0.40
    # 平均每页文本字符数低于此值 -> 文本层无实际内容（纯水印/页眉），需 OCR
    min_chars_per_page: int = 150
    # 表格"碎片化"判定：单元格普遍只有 1~2 个字的表格是 pdfplumber 在图片页上
    # 误检出的噪声（如 JGJ215），直接丢弃
    table_garbage_cell_ratio: float = 0.7
    table_garbage_min_cells: int = 8


@dataclass(frozen=True)
class OCRConfig:
    # 渲染分辨率（DPI）：200 对 10.5pt 印刷体识别准确率高
    render_dpi: int = 200
    # OCR 并行进程数（每个进程加载一份 onnx 模型；20 核机器建议 4~6）
    workers: int = 5
    # 每个 OCR 进程的 onnxruntime 线程数（workers * intra_threads <= CPU 核数）
    intra_op_threads: int = 3


@dataclass(frozen=True)
class ChunkConfig:
    # 语义分块目标大小（中文字符数；BGE 最大 512 token，约对应 400~500 汉字）
    chunk_size: int = 450
    chunk_overlap: int = 80
    # 表格/图注块的最大长度，超过才二次切分；表格块优先整体保留
    table_max_size: int = 900
    # 法规条款层级标题的正则（第X条 / 第X章 / X.X.X / 一、 等）
    section_separators: tuple = (
        r"\n第[一二三四五六七八九十百零〇\d]+条",   # 法律条款
        r"\n第[一二三四五六七八九十百零〇\d]+章",
        r"\n第[一二三四五六七八九十百零〇\d]+节",
        r"\n\d+\.\d+\.\d+",                          # 国标编号 3.1.2
        r"\n\d+\.\d+(?!\d)",                         # 3.1
        r"\n[一二三四五六七八九十]+、",               # 中文序号
    )


# ----------------------------------------------------------------------
# 6) 混合检索配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    # 单路召回候选数：调大能显著提升"黄金块是否进入候选池"的概率；
    # 代价是 rerank 候选变多（CPU 上精排耗时近似线性）。
    vector_top_k: int = 20          # 向量检索候选数
    bm25_top_k: int = 20            # BM25 检索候选数
    rrf_k: int = 60                 # RRF 融合常数（标准值 60）
    vector_weight: float = 0.55     # 加权融合时向量通道权重（RRF 模式下不使用）
    bm25_weight: float = 0.45
    fusion_method: str = "rrf"      # rrf / weighted
    use_reranker: bool = True       # 是否启用 Rerank 重排
    rerank_candidates: int = 30     # 进入 rerank 的最大候选数
    final_top_k: int = 5            # 最终喂给大模型的上下文片段数
    score_threshold: float = 0.0    # 最终相关性下限（rerank 分数低于阈值丢弃）


# ----------------------------------------------------------------------
# 7) 评测配置
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    dataset_file: str = "ehs_eval_dataset.json"   # 评测集文件名
    report_file: str = "eval_report.json"         # 机器可读报告
    report_md: str = "eval_report.md"             # 人类可读报告
    target_size: int = 200                         # 目标评测条数（简历口径）
    # 实际批量评测规模（免费 LLM 限流，可先用子集跑通）
    default_sample_size: int = 60
    hit_rate_k: int = 5
    judge_max_tokens: int = 4096


# ----------------------------------------------------------------------
# 8) 服务化配置（FastAPI / Streamlit）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ServiceConfig:
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # 简单 API-Key 鉴权（演示用；生产应放环境变量/数据库）
    api_keys: tuple = ("ehs-demo-key-2026",)
    # 每把 key 每分钟最大请求数（令牌桶限流）
    rate_limit_per_minute: int = 20
    # 熔断：单次问答超时（秒）
    answer_timeout: int = 120


# ----------------------------------------------------------------------
# 8b) Redis 缓存配置（检索结果 / 问答结果；服务不可用时自动旁路）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RedisConfig:
    enabled: bool = True                  # 总开关；False 时完全不连接 Redis
    host: str = "127.0.0.1"
    # 本机 6379 已被 Langfuse 自托管栈的 Redis 占用，EHS 使用独立容器 ehs-redis
    # （docker run -d --name ehs-redis --restart always -p 6380:6379 redis:7-alpine）
    port: int = 6380
    db: int = 0
    password: str = ""                    # 无密码留空
    key_prefix: str = "ehs:rag"           # 本系统所有缓存键前缀
    answer_ttl: int = 3600                # 问答结果有效期（秒）
    retrieval_ttl: int = 1800             # 检索结果有效期（秒）
    # 超时必须短：Redis 异常时不能拖慢在线问答（失败即旁路）
    socket_timeout: float = 2.0
    socket_connect_timeout: float = 1.0


# ----------------------------------------------------------------------
# 聚合配置（对外只暴露一个 settings 单例）
# ----------------------------------------------------------------------
@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval_cfg: EvalConfig = field(default_factory=EvalConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR                       # 向量库 / BM25 /  manifest 等产物
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR
    vectorstore_dir: Path = VECTORSTORE_DIR
    milvus_dir: Path = MILVUS_DIR
    milvus_db: Path = MILVUS_DB
    bm25_dir: Path = BM25_DIR
    eval_dir: Path = EVAL_DIR
    logs_dir: Path = LOGS_DIR

    # 允许用环境变量覆盖 OpenRouter API Key
    def api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", self.llm.api_key)


settings = Settings()
