# EHS 知识库 RAG 问答系统

面向企业 **EHS（Environment 环境 / Health 健康 / Safety 安全）** 管理场景的垂直领域知识库问答系统。针对安全制度文件、法规规范、操作规程分散，且文档含大量表格、参数、现场示意图，一线人员查阅规范难、人工答疑负担重的问题，基于 **LangChain** 构建完整 RAG 链路，并完成检索与生成质量评测。

- 开发环境：`cpu_default`（纯 CPU，Windows）
- 技术范式：**面向对象（OOP）** 封装，模块职责单一、可独立替换
- 语言模型：本地 Ollama `deepseek-r1:1.5b`（默认，离线）；可切换 OpenRouter 云端
  （`config.py` 改 `llm.provider`，API Key 经环境变量 `OPENROUTER_API_KEY` 注入）
- 嵌入模型：本地 `BAAI/bge-base-zh-v1.5`（768 维，离线运行）
- 重排序模型：本地 `BAAI/bge-reranker-base`
- 扫描版/乱码文档：RapidOCR（onnxruntime，多进程并行）识别

---

## 一、系统架构

```
                          ┌──────────────────────────────────────┐
   EHS 原始文档           │  文档解析层 document_loader           │
  (PDF/Word/TXT/MD/Excel) │  · pypdf 文本 + pdfplumber 表格结构化 │
        │                 │  · python-docx 段落/表格（保持顺序）   │
        ▼                 │  · 图注识别 + 非法字符清洗             │
 ┌──────────────┐         └───────────────┬──────────────────────┘
 │ 语义分块层     │                         │
 │ text_splitter │ ◀── 结构感知：第X条/章节/编号，表格整体保留
 └──────┬───────┘                         │
        │                                ▼
        │            ┌──────────────────────────────────────┐
        ├──────────▶ │  混合检索层 hybrid_retriever          │
        │            │  向量(BGE+Chroma)  +  BM25(jieba)     │
        │            │  RRF 融合  →  BGE Reranker 精排        │
        │            └───────────────┬──────────────────────┘
        │                            ▼
        │            ┌──────────────────────────────────────┐
        └──────────▶ │  生成层 rag_chain                     │
                     │  仅依据原文作答（缓解幻觉）+ 来源溯源   │
                     └───────────────┬──────────────────────┘
                                     ▼
              ┌────────────┬──────────────────┬──────────────┐
              │ CLI main.py │ FastAPI api_server│ Streamlit 前端│
              └────────────┴──────────────────┴──────────────┘
                                     │
                                     ▼
                    评测 evaluator（Hit Rate / MRR /
                    上下文召回 / 答案准确率 / 忠实度）
```

### 核心优化点
1. **多格式结构化解析**：表格转 Markdown 结构化文本，图注（图X-X / 表X-X）识别标记，PDF 提取的孤立代理字符自动清洗。
2. **语义分块**：按法规条款/章节结构切分，表格优先整体保留、超长表格按行切分并重复表头，避免表格与关联说明被粗暴拆分。
3. **混合检索**：向量语义召回（解决同义表述）+ BM25 精确召回（解决法规编号、专业术语），RRF 排名融合，无需分数标定。
4. **Rerank 精排**：CrossEncoder 对融合候选做 query-doc 相关性精排，显著提升 top-k 命中质量。
5. **防幻觉 + 溯源**：Prompt 强制仅依据检索原文作答，答案末尾标注文件/条款/页码，前端可定位原文片段。
6. **文本层质量判定 + OCR 兜底**：不只检测"提取为空"的扫描版 —— 对 CID 字体缺
   ToUnicode 映射产生的"形似正常、实为乱码"的 PDF（汉字占比极低），以及仅含水印的
   PDF，按页判定并逐页 OCR 兜底；pdfplumber 在图片页误检出的碎片化"假表格"自动过滤。
7. **索引质量修复工具 `reingest.py`**：无需人工列名单，自动体检向量库中每个文件的
   汉字占比/表格碎片密度，识别"入库即乱码"的低质量文件，并行重解析（含 OCR）后重建
   向量块与 BM25，消除"检索永远不可达"的死档。
8. **工程化**：FastAPI 提供 API-Key 鉴权、滑动窗口限流、超时熔断；Streamlit 对话式前端；支持文档增量更新。
9. **Redis 两级缓存**：检索结果与问答结果缓存，重复问题跳过 Rerank 与 LLM；问题归一化提高命中率，索引变更后按前缀自动失效，Redis 异常时自动旁路。

---

## 二、目录结构

```
D:\ehs_rag
├── config.py            # 全局配置（模型、路径、分块/检索/评测/服务/缓存参数）
├── _bootstrap.py        # 运行引导（为带序号文件名提供别名导入）
├── 01_indexing/         # 索引侧 —— 离线建库（文件名按执行顺序编号）
│   ├── 01_document_loader.py   # 文档解析（PDF/Word/TXT/MD/Excel + 表格结构化）
│   ├── 02_ocr_module.py        # 扫描版/乱码 PDF 的 OCR 识别与入库（多进程）
│   ├── 03_text_splitter.py     # 语义分块（结构感知 + 表格保护）
│   ├── 04_embeddings.py        # 本地 BGE 嵌入封装（查询侧 instruction）
│   ├── 05_vector_store.py      # Milvus 向量库（持久化、去重、增量）
│   ├── 06_migrate_to_milvus.py # 旧 Chroma -> Milvus Lite 迁移工具
│   ├── 07_reingest.py          # 索引质量体检与修复（乱码/噪声文件重建）
│   └── 08_batch_ingest.py      # 批量入库脚本
├── 02_retrieval/        # 检索侧 —— 在线问答 + 评测（文件名按执行顺序编号）
│   ├── 01_hybrid_retriever.py  # BM25 + 向量 + RRF + BGE Rerank
│   ├── 02_redis_cache.py       # Redis 缓存（检索/答案，可优雅降级）
│   ├── 03_rag_chain.py         # RAG 问答链（防幻觉 + 溯源）
│   ├── 04_api_server.py        # FastAPI 服务
│   ├── 05_app_streamlit.py     # Streamlit 对话前端
│   ├── 06_main.py              # 应用编排与 CLI（ingest / ask / chat）
│   ├── 07_eval_dataset.py      # 评测集构建（人工种子 + LLM 扩充）
│   ├── 08_build_dataset.py     # 评测集批量构建脚本
│   ├── 09_evaluator.py         # 检索/生成多指标评测器（含消融对比）
│   ├── 10_run_eval.py          # 评测入口脚本
│   ├── 11_debug_recall.py      # 检索失败归因（黄金块各层排名）
│   ├── 12_audit_labels.py      # 评测集标签可达性审计
│   ├── 13_tune_retrieval.py    # 候选池策略对比实验
│   ├── 14_compare_reports.py   # 评测报告前后对比
│   └── 15_test_api.py          # API 接口测试脚本
├── requirements.txt     # 依赖清单
├── data/                # 持久化 Milvus / BM25 / parse_cache
├── evaluation/          # 评测集与评测报告
└── logs/                # 运行日志
```

---

## 三、环境安装

```powershell
conda activate cpu_default
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install rapidocr-onnxruntime -i https://mirrors.aliyun.com/pypi/simple
```

首次运行会从 `hf-mirror.com` 自动下载 BGE 嵌入与 Reranker 模型，之后走本地缓存、离线加载。

---

## 四、快速使用

### 1. 文档入库

```powershell
# 全量入库
python 02_retrieval\06_main.py ingest

# 仅入库某子目录（分批/冒烟）
python 02_retrieval\06_main.py ingest --subdir "01_法律"

# 扫描版/乱码 PDF 自动 OCR 入库（多进程并行）
python 01_indexing\02_ocr_module.py --workers 5

# 索引质量体检与修复：自动找出"入库即乱码/噪声"的文件并重建
python 01_indexing\07_reingest.py --dry-run     # 只体检，列出将修复的文件
python 01_indexing\07_reingest.py --workers 5   # 体检并修复（逐页 OCR 兜底 + 假表格过滤）
```

### 1.1 检索失败归因与评测集体检

```powershell
# 打印指定题目的黄金块在 向量/BM25/RRF融合/rerank 各层的排名，
# 区分"召回不到"与"排不进前5"
python 02_retrieval\11_debug_recall.py seed_009 seed_037

# 评测集标签审计：检查每题的 section/match_terms 与 key_point 能否在
# 其 source_file 的已入库内容中逐字命中（标签噪声会让指标失真）
python 02_retrieval\12_audit_labels.py --only-failing

# 候选池策略对比：RRF(topn=30) 截断 vs 单路并集直通 reranker
# （单通道命中的专业术语块会被 RRF 压出候选池，reranker 看不到）
python 02_retrieval\13_tune_retrieval.py
```

候选池实测结论（99 题）：`rrf30`（现状）在 Hit@5 / MRR / Context Recall 上
最优或并列最优；`rrf40` / `union` 与现状互为对称交换（各 1 题 miss↔hit，无净
增益），更大的 `union2` 反而净亏 1 题——扩大候选池不划算，保留 RRF 截断 30
（精排候选更少也更快）。

### 2. 问答

```powershell
# 单轮问答（默认 混合检索 + Rerank）
python 02_retrieval\06_main.py ask "脚手架立杆垫板有什么要求？"

# 指定检索模式：vector / bm25 / hybrid / hybrid_rerank
python 02_retrieval\06_main.py ask "安全生产法规定从业人员有哪些义务？" --mode vector

# 交互式多轮问答
python 02_retrieval\06_main.py chat
```

### 2.5 Redis 缓存（可选，默认开启）

对重复问题做两级 Redis 缓存，降低 CPU 与 LLM 开销：

- **检索缓存**：`(问题, mode)` → 召回 + Rerank 结果，命中跳过 向量/BM25/RRF/Rerank；
- **答案缓存**：`(问题, mode)` → 完整 `RAGAnswer`，命中直接返回、连 LLM 生成都跳过。

问题经 NFKC 归一化（小写、折叠空白），空格或大小写不同也能命中；不同 mode 相互隔离。
`ingest` / `reingest` / 上传入库后自动失效，且只删除本系统 `ehs:rag:*` 前缀，
不影响同一 Redis 中的其他业务（如 Langfuse）。

本机部署（EHS 专用容器；因 6379 已被 Langfuse 的 Redis 占用，使用 **6380**，
该端口已写入 `config.py` 默认值）：

```powershell
docker run -d --name ehs-redis --restart always -p 6380:6379 redis:7-alpine
```

其他环境部署标准 Redis（6379）后，用环境变量覆盖连接参数：
`REDIS_URL` / `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` / `REDIS_PASSWORD`；
设 `EHS_REDIS_DISABLED=1` 可完全关闭缓存。

**Redis 不可用时自动旁路**（仅一条 WARNING），问答照常进行，不阻断主流程。
实测（GPU）：同一问题首次检索 21.6s，缓存命中 0.003s。

---

### 3. 评测

```powershell
# 构建评测集（人工种子 45 条 + LLM 扩充至 200 条）
python -c "import _bootstrap; _bootstrap.init(); from eval_dataset import EvalDatasetBuilder; EvalDatasetBuilder().build(200)"

# 推荐两步：先全量检索消融（纯本地，无 LLM 配额消耗），再抽样生成评测
python 02_retrieval\10_run_eval.py --sample 99 --no-generation      # Hit Rate@5 / MRR / Context Recall
python 02_retrieval\10_run_eval.py --gen-only --gen-size 40         # 生成指标，本地 Ollama 执行

# 与基线对比（总体指标 delta + 逐题 miss->hit 翻转明细）
python 02_retrieval\14_compare_reports.py evaluation/baseline_eval_report.json evaluation/eval_report.json
```

### 4. 服务与前端

```powershell
# FastAPI（http://localhost:8000/docs）
python 02_retrieval\04_api_server.py

# Streamlit（对话式界面）
streamlit run 02_retrieval\05_app_streamlit.py
```

FastAPI 调用示例：

```powershell
curl -X POST http://localhost:8000/api/v1/ask `
  -H "X-API-Key: ehs-demo-key-2026" -H "Content-Type: application/json" `
  -d '{\"question\":\"脚手架垫板有什么要求？\"}'
```

---

## 五、评测指标说明

| 类别 | 指标 | 含义 |
|---|---|---|
| 检索 | Hit Rate@5 | 黄金来源是否出现在 top-5 |
| 检索 | MRR | 黄金来源首次出现位置的倒数 |
| 检索 | Context Recall | 评分要点被检索上下文覆盖的比例 |
| 生成 | Answer Accuracy | LLM 裁判评分 ≥4（满分 5）的比例 |
| 生成 | Faithfulness | 答案无原文外编造（忠实度）的比例 |
| 生成 | Key-point Coverage | 答案命中评分要点的比例 |
| 生成 | Latency | 平均端到端延迟 |

评测报告：`evaluation/eval_report.md`（人类可读）、`evaluation/eval_report.json`（逐条明细）。

---

## 六、常见问题

1. **启动时长时间卡在加载模型**：模型已缓存时会自动离线加载；若仍卡顿，检查 `C:\Users\<用户>\.cache\huggingface\hub` 是否完整，或手动设置 `$env:HF_HUB_OFFLINE="1"`。
2. **某 PDF 检索不到内容**：可能是扫描版，或 CID 字体缺 ToUnicode 映射导致提取出
   "形似正常实为乱码"的文本。运行 `python 01_indexing\07_reingest.py --dry-run` 体检
   （按汉字占比与表格碎片密度识别），再 `python 01_indexing\07_reingest.py` 自动重解析
   （含逐页 OCR）重建索引。
3. **免费模型限流（429）**：系统已内置请求间隔与自动重试，可在 `config.py` 调大 `min_request_interval`。
4. **API-Key 鉴权**：演示 Key 为 `ehs-demo-key-2026`，生产环境请通过环境变量/密钥管理替换。
