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

"""用 requests 以严格 UTF-8 测试 FastAPI，排除 PowerShell 编码干扰。"""
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8000"
KEY = "ehs-demo-key-2026"

# 1) 健康检查
h = requests.get(f"{BASE}/health", timeout=10)
print("GET /health ->", h.status_code, h.json())

# 2) 问答（UTF-8）
payload = {"question": "脚手架立杆垫板有什么要求？", "return_contexts": True}
t = time.time()
r = requests.post(f"{BASE}/api/v1/ask",
                  headers={"X-API-Key": KEY},
                  json=payload, timeout=240)
wall = time.time() - t
print("\nPOST /api/v1/ask ->", r.status_code,
      "| Content-Type:", r.headers.get("content-type"),
      f"| 墙钟 {wall:.1f}s")
r.encoding = "utf-8"
data = r.json()
print("回显 question:", data.get("question"))
print("\nanswer:\n", data.get("answer"))
print("\nsources:")
for s in data.get("sources", []):
    print(f"  - {s.get('file_name')} | p{s.get('page')} | {s.get('section_title')}")
print("\n服务端 latency_seconds:", data.get("latency_seconds"))
print("mode:", data.get("mode"))
