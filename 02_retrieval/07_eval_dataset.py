# -*- coding: utf-8 -*-
"""
评测集构建模块（面向对象）
--------------------------------------------------------------------
- SeedQuestionBank：内置约 45 条人工编写的高质量种子问题，覆盖 7 大法规
  目录与"制度查询 / 隐患判定 / 参数查阅 / 图纸解读"四类典型业务场景，
  每条含问题、答案要点(key_points)、来源文件与条款；
- EvalDatasetBuilder：
  1) 从已入库 chunk 中挑选信息量大的条款块；
  2) 分批调用 apodex，让模型基于原文块生成"问题 + 标准答案 + 要点"；
  3) 与种子问题合并、去重、编号，产出目标 200 条评测集 JSON。

评测集条目 schema：
  {id, question, answer, key_points:[...], source_file, section,
   category, source_chunk_id?}
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


import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config import Settings, settings

logger = logging.getLogger(__name__)


# ====================================================================
# 人工种子问题（答案均可在对应文件中找到依据）
# ====================================================================
class SeedQuestionBank:
    """人工种子评测题。section 做子串匹配，可留空。"""

    # category 取值
    C_SYSTEM = "制度查询"
    C_HAZARD = "隐患判定"
    C_PARAM = "参数查阅"
    C_FIGURE = "图纸解读"

    SEED: List[dict] = [
        # ---------------- 01 法律 ----------------
        {"question": "安全生产法规定从业人员发现事故隐患或者其他不安全因素后应当怎么做？",
         "key_points": ["立即报告", "现场安全生产管理人员", "本单位负责人", "及时处理"],
         "source_file": "安全生产法_2021年修正.txt", "section": "第五十九条", "category": C_HAZARD},
        {"question": "安全生产法规定从业人员有哪些安全生产义务？",
         "key_points": ["遵守规章制度", "操作规程", "服从管理", "劳动防护用品", "教育培训", "隐患报告"],
         "source_file": "安全生产法_2021年修正.txt", "section": "从业人员", "category": C_SYSTEM},
        {"question": "从业人员有权拒绝违章指挥和强令冒险作业，这在安全生产法中是如何规定的？",
         "key_points": ["拒绝", "违章指挥", "强令冒险作业"],
         "source_file": "安全生产法_2021年修正.txt", "section": "权利", "category": C_SYSTEM},
        {"question": "安全生产法规定生产经营单位的主要负责人对本单位安全生产工作负有哪些职责？",
         "key_points": ["建立健全", "安全生产责任制", "规章制度", "投入", "隐患排查", "应急预案"],
         "source_file": "安全生产法_2021年修正.txt", "section": "主要负责人", "category": C_SYSTEM},
        {"question": "生产经营单位发生生产安全事故后，现场有关人员应当如何报告？",
         "key_points": ["立即报告", "本单位负责人"],
         "source_file": "安全生产法_2021年修正.txt", "section": "事故报告", "category": C_HAZARD},
        {"question": "消防法规定，进行电焊、气焊等具有火灾危险作业的人员应当符合什么要求？",
         "key_points": ["持证上岗", "消防安全"],
         "source_file": "消防法_2021年修正.txt", "section": "电焊", "category": C_SYSTEM},
        {"question": "职业病防治法规定用人单位对从事接触职业病危害作业的劳动者应当如何组织职业健康检查？",
         "key_points": ["上岗前", "在岗期间", "离岗时", "职业健康检查"],
         "source_file": "职业病防治法_2018年修正.txt", "section": "健康检查", "category": C_SYSTEM},
        {"question": "特种设备安全法规定特种设备使用单位应当在什么期限内办理使用登记？",
         "key_points": ["投入使用前", "投入使用后三十日内", "使用登记"],
         "source_file": "特种设备安全法_2013年通过.txt", "section": "使用登记", "category": C_PARAM},
        {"question": "特种设备作业人员应当具备什么条件方可上岗？",
         "key_points": ["取得", "资格", "证书"],
         "source_file": "特种设备安全法_2013年通过.txt", "section": "作业人员", "category": C_SYSTEM},
        {"question": "突发事件应对法把自然灾害、事故灾难等突发事件分为哪几级？",
         "key_points": ["特别重大", "重大", "较大", "一般"],
         "source_file": "突发事件应对法_2024年修订.txt", "section": "分级", "category": C_SYSTEM},
        {"question": "危险化学品安全法对危险化学品的储存有哪些基本安全要求？",
         "key_points": ["专用仓库", "专人管理", "禁忌物料", "分类存放"],
         "source_file": "危险化学品安全法_2025年通过.txt", "section": "储存", "category": C_SYSTEM},
        {"question": "建筑法规定，涉及建筑主体和承重结构变动的装修工程应当如何处理？",
         "key_points": ["设计方案", "原设计单位", "相应资质"],
         "source_file": "建筑法_2019年修正.txt", "section": "装修", "category": C_SYSTEM},

        # ---------------- 02 行政法规 ----------------
        {"question": "生产安全事故报告和调查处理条例中，特别重大事故的死亡人数和直接经济损失是如何规定的？",
         "key_points": ["30人以上", "1亿元以上"],
         "source_file": "生产安全事故报告和调查处理条例_国务院令第493号.pdf", "section": "事故等级", "category": C_PARAM},
        {"question": "事故发生后，单位负责人接到报告应当于多长时间内向事故发生地县级以上监管部门报告？",
         "key_points": ["1小时内"],
         "source_file": "生产安全事故报告和调查处理条例_国务院令第493号.pdf", "section": "报告", "category": C_PARAM},
        {"question": "建设工程安全生产管理条例规定，哪些分部分项工程需要编制专项施工方案？",
         "key_points": ["基坑支护", "降水", "土方开挖", "模板", "起重吊装", "脚手架", "拆除爆破"],
         "source_file": "建设工程安全生产管理条例_国务院令第393号.pdf", "section": "专项施工方案", "category": C_SYSTEM},
        {"question": "对达到一定规模的危险性较大分部分项工程，专项施工方案还应履行什么程序？",
         "key_points": ["专家论证", "审查"],
         "source_file": "建设工程安全生产管理条例_国务院令第393号.pdf", "section": "专家论证", "category": C_SYSTEM},
        {"question": "建设工程安全生产管理条例规定施工单位应当为施工现场从事危险作业的人员办理什么保险？",
         "key_points": ["意外伤害保险", "施工单位支付"],
         "source_file": "建设工程安全生产管理条例_国务院令第393号.pdf", "section": "意外伤害保险", "category": C_SYSTEM},
        {"question": "安全生产许可证的有效期为几年？期满需要延期的应在何时办理？",
         "key_points": ["3年", "期满前3个月"],
         "source_file": "安全生产许可证条例_国务院令第397号(2014年第二次修订).pdf", "section": "有效期", "category": C_PARAM},
        {"question": "建设项目环境保护管理的三同时制度是指什么？",
         "key_points": ["同时设计", "同时施工", "同时投产使用"],
         "source_file": "建设项目环境保护管理条例_国务院令第253号(2017年第682号修订).pdf", "section": "三同时", "category": C_SYSTEM},
        {"question": "特种设备安全监察条例规定特种设备在投入使用前或投入使用后多少日内办理使用登记？",
         "key_points": ["30日"],
         "source_file": "特种设备安全监察条例_国务院令第549号(2009年修订).pdf", "section": "使用登记", "category": C_PARAM},

        # ---------------- 03 部门规章 ----------------
        {"question": "危险性较大的分部分项工程安全管理规定中，专项施工方案应当由谁签字后方可实施？",
         "key_points": ["施工单位技术负责人", "总监理工程师"],
         "source_file": "危险性较大的分部分项工程安全管理规定_住建部令第37号(47号令修正).pdf", "section": "方案", "category": C_SYSTEM},
        {"question": "超过一定规模的危大工程专家论证会对专家人数有什么要求？",
         "key_points": ["5名以上", "符合相关专业要求"],
         "source_file": "危险性较大的分部分项工程安全管理规定_住建部令第37号(47号令修正).pdf", "section": "专家", "category": C_PARAM},
        {"question": "建筑施工企业的安管人员三类人员分别指什么？",
         "key_points": ["主要负责人", "项目负责人", "专职安全生产管理人员"],
         "source_file": "建筑施工企业主要负责人项目负责人和专职安全生产管理人员安全生产管理规定_住建部令第17号.pdf", "section": "安管人员", "category": C_SYSTEM},
        {"question": "生产安全事故应急预案分为哪几类？",
         "key_points": ["综合应急预案", "专项应急预案", "现场处置方案"],
         "source_file": "生产安全事故应急预案管理办法_安监总局令第88号(应急部令第2号修正).txt", "section": "预案", "category": C_SYSTEM},
        {"question": "生产经营单位应当多久组织一次综合或专项应急预案演练？现场处置方案演练频次是多少？",
         "key_points": ["每年至少一次", "每半年至少一次"],
         "source_file": "生产安全事故应急预案管理办法_安监总局令第88号(应急部令第2号修正).txt", "section": "演练", "category": C_PARAM},
        {"question": "特种作业操作证多长时间复审一次？有效期为几年？",
         "key_points": ["3年复审", "6年"],
         "source_file": "特种作业人员安全技术培训考核管理规定_应急管理部令第19号.txt", "section": "复审", "category": C_PARAM},
        {"question": "房屋市政工程生产安全重大事故隐患判定标准（2024版）主要用于什么工作？",
         "key_points": ["重大事故隐患", "判定"],
         "source_file": "房屋市政工程生产安全重大事故隐患判定标准（2024版）_建质规〔2024〕5号.pdf", "section": "判定标准", "category": C_HAZARD},

        # ---------------- 04 国家标准 ----------------
        {"question": "建筑与市政施工现场安全卫生与职业健康通用规范GB55034规定，坠落高度基准面多少米及以上的临边应设置防护栏杆？",
         "key_points": ["2m", "防护栏杆"],
         "source_file": "建筑与市政施工现场安全卫生与职业健康通用规范_GB55034-2022.pdf", "section": "临边", "category": C_PARAM},
        {"question": "GB55034规定施工现场临边防护栏杆的高度不应低于多少米？",
         "key_points": ["1.2m"],
         "source_file": "建筑与市政施工现场安全卫生与职业健康通用规范_GB55034-2022.pdf", "section": "防护栏杆", "category": C_PARAM},
        {"question": "施工脚手架通用规范GB55023规定脚手架立杆底部应设置什么构件？",
         "key_points": ["垫板", "底座", "扫地杆"],
         "source_file": "施工脚手架通用规范_GB55023-2022.pdf", "section": "立杆", "category": C_SYSTEM},
        {"question": "建筑防火通用规范要求消防车道的净宽度和净空高度分别不应小于多少米？",
         "key_points": ["4.0m", "4.0m"],
         "source_file": "建筑防火通用规范_GB55037-2022.pdf", "section": "消防车道", "category": C_PARAM},
        {"question": "建设工程施工现场消防安全技术标准GB50720对施工现场动火作业有哪些管理要求？",
         "key_points": ["动火许可证", "动火监护人", "灭火器材"],
         "source_file": "建设工程施工现场消防安全技术标准_GB-T50720-2011_2025局部修订.pdf", "section": "动火", "category": C_HAZARD},
        {"question": "GB50720规定施工现场临时宿舍、办公用房的建筑构件燃烧性能等级应为哪一级？",
         "key_points": ["A级", "不燃"],
         "source_file": "建设工程施工现场消防安全技术标准_GB-T50720-2011_2025局部修订.pdf", "section": "临时用房", "category": C_PARAM},

        # ---------------- 05 JGJ 基础安全 ----------------
        {"question": "施工现场临时用电JGJ/T46规定的三级配电、两级保护分别指什么？",
         "key_points": ["总配电箱", "分配电箱", "开关箱", "漏电保护"],
         "source_file": "建筑与市政工程施工现场临时用电安全技术标准_JGJ-T46-2024.pdf", "section": "配电", "category": C_SYSTEM},
        {"question": "临时用电一机一闸一漏一箱的具体要求是什么？",
         "key_points": ["一台设备", "一个开关箱", "一个隔离开关", "一个漏电保护器"],
         "source_file": "建筑与市政工程施工现场临时用电安全技术标准_JGJ-T46-2024.pdf", "section": "开关箱", "category": C_SYSTEM},
        {"question": "建筑施工安全检查标准JGJ59中，建筑施工安全检查评定的优良和合格分数界限是多少？",
         "key_points": ["80分", "70分"],
         "source_file": "建筑施工安全检查标准_JGJ59-2011.pdf", "section": "评定", "category": C_PARAM},
        {"question": "扣件式钢管脚手架JGJ130对脚手板的垫板规格有什么要求？",
         "key_points": ["垫板", "长度不少于2跨", "厚度不小于50mm"],
         "source_file": "建筑施工扣件式钢管脚手架安全技术规范_JGJ130-2011.pdf", "section": "垫板", "category": C_PARAM},
        {"question": "扣件式钢管脚手架JGJ130对连墙件的设置有什么要求？",
         "key_points": ["连墙件", "两步三跨", "刚性"],
         "source_file": "建筑施工扣件式钢管脚手架安全技术规范_JGJ130-2011.pdf", "section": "连墙件", "category": C_SYSTEM},
        {"question": "建筑施工高处作业安全技术规范JGJ80对高处作业是如何定义的？",
         "key_points": ["坠落高度基准面2m及以上", "有可能坠落"],
         "source_file": "建筑施工高处作业安全技术规范_JGJ80-2016.pdf", "section": "高处作业", "category": C_PARAM},
        {"question": "JGJ80规定安全带的正确使用原则是什么？",
         "key_points": ["高挂低用"],
         "source_file": "建筑施工高处作业安全技术规范_JGJ80-2016.pdf", "section": "安全带", "category": C_SYSTEM},
        {"question": "承插型盘扣式钢管脚手架JGJ/T231的连接方式有什么特点？",
         "key_points": ["盘扣节点", "插销", "立杆连接盘"],
         "source_file": "建筑施工承插型盘扣式钢管脚手架安全技术标准_JGJ-T231-2021.pdf", "section": "节点", "category": C_SYSTEM},

        # ---------------- 06 JGJ 机械吊装 ----------------
        {"question": "塔式起重机应设置哪些安全限位装置？",
         "key_points": ["力矩限制器", "起重量限制器", "起升高度限位", "幅度限位", "行程限位"],
         "source_file": "建筑施工塔式起重机安装使用拆卸安全技术规程_JGJ196-2010.pdf", "section": "安全装置", "category": C_SYSTEM},
        {"question": "施工升降机的防坠安全器在标定期限和坠落试验方面有什么要求？",
         "key_points": ["标定期限不超过一年", "每三个月", "坠落试验"],
         "source_file": "建筑施工升降机安装使用拆卸安全技术规程_JGJ215-2010.pdf", "section": "防坠安全器", "category": C_PARAM},
        {"question": "建筑深基坑工程JGJ311规定基坑支护设计和施工应重视哪些监测内容？",
         "key_points": ["水平位移", "沉降", "地下水位", "周边环境"],
         "source_file": "建筑深基坑工程施工安全技术规范_JGJ311-2013.pdf", "section": "监测", "category": C_SYSTEM},
        {"question": "建筑施工模板安全技术规范JGJ162对可调托撑螺杆伸出立杆顶端的长度有什么限制？",
         "key_points": ["可调托撑", "伸出长度"],
         "source_file": "建筑施工模板安全技术规范_JGJ162-2008.pdf", "section": "托撑", "category": C_PARAM},
        {"question": "龙门架及井架物料提升机JGJ88应设置哪些主要安全装置？",
         "key_points": ["安全停靠装置", "断绳保护装置", "上极限限位", "下极限限位"],
         "source_file": "龙门架及井架物料提升机安全技术规范_JGJ88-2010.pdf", "section": "安全装置", "category": C_SYSTEM},
        {"question": "建筑施工作业劳动防护用品配备标准JGJ184规定，架子工应配备哪些主要劳动防护用品？",
         "key_points": ["安全帽", "安全带", "防滑鞋"],
         "source_file": "建筑施工作业劳动防护用品配备及使用标准_JGJ184-2009.pdf", "section": "防护用品", "category": C_SYSTEM},
        {"question": "建筑拆除工程JGJ147规定，拆除施工前应做好哪些准备工作？",
         "key_points": ["专项方案", "切断管线", "围挡", "警戒"],
         "source_file": "建筑拆除工程安全技术规范_JGJ147-2016.pdf", "section": "拆除准备", "category": C_SYSTEM},

        # ---------------- 07 环保 ----------------
        {"question": "建筑施工场界环境噪声排放标准GB12523规定昼间和夜间的噪声排放限值分别是多少分贝？",
         "key_points": ["70dB", "55dB"],
         "source_file": "建筑施工噪声排放标准_GB12523-2025.pdf", "section": "限值", "category": C_PARAM},
        {"question": "防治城市扬尘污染技术规范以及施工扬尘六个百分百要求主要包括哪些内容？",
         "key_points": ["工地周边围挡", "物料堆放覆盖", "出入车辆冲洗", "路面硬化", "拆迁湿法作业", "密闭运输"],
         "source_file": "防治城市扬尘污染技术规范_HJ_T393-2007.pdf", "section": "扬尘", "category": C_SYSTEM},
        {"question": "排污许可管理办法规定排污许可证的有效期为几年？",
         "key_points": ["5年"],
         "source_file": "排污许可管理办法_生态环境部令第32号-2024.txt", "section": "有效期", "category": C_PARAM},
        {"question": "建筑垃圾污染控制技术规范HJ1462对建筑垃圾的处理处置提出了哪些基本要求？",
         "key_points": ["分类收集", "资源化利用", "无害化处置"],
         "source_file": "建筑垃圾污染控制技术规范_HJ1462-2026.pdf", "section": "处理", "category": C_SYSTEM},
    ]

    def as_records(self) -> List[dict]:
        records = []
        for i, item in enumerate(self.SEED, 1):
            rec = {
                "id": f"seed_{i:03d}",
                "question": item["question"],
                "answer": item.get("answer", ""),
                "key_points": item["key_points"],
                "source_file": item["source_file"],
                "section": item.get("section", ""),
                "category": item["category"],
                "is_seed": True,
            }
            records.append(rec)
        return records


# ====================================================================
# LLM 批量生成
# ====================================================================
GEN_SYSTEM = (
    "你是 EHS 法规考题编写专家。给定若干法规/规范原文片段，请为每个片段编写"
    "1 道中文问答题，题目必须能仅凭该片段内容回答，且覆盖制度规定、隐患判定、"
    "数值参数或图表说明。输出严格的 JSON 数组，每个元素形如："
    '{"question":"...","answer":"基于片段的简明参考答案","key_points":["得分要点1","得分要点2"]}。'
    "不要输出 JSON 以外的任何文字。"
)


class EvalDatasetBuilder:
    """评测集构建器。"""

    def __init__(self, config: Settings = settings):
        self.config = config
        self.path = Path(config.eval_dir) / config.eval_cfg.dataset_file

    # ----------------------------------------------------------------
    def load_all_chunks(self) -> List[Document]:
        """从 Milvus 向量库取出全部已入库 chunk。"""
        from vector_store import EHSVectorStore
        vs = EHSVectorStore(self.config)
        return vs.all_documents()

    @staticmethod
    def _is_good_source(doc: Document) -> bool:
        """挑选信息量大、含规范要求/数值的条款块。"""
        t = doc.page_content
        if len(t) < 120 or len(t) > 700:
            return False
        if doc.metadata.get("doc_type") == "table":
            return True
        return any(k in t for k in ("应当", "不得", "严禁", "不应", "必须", "不少于", "不大于"))

    def _sample_chunks(self, n: int) -> List[Document]:
        docs = [d for d in self.load_all_chunks() if self._is_good_source(d)]
        random.seed(42)
        random.shuffle(docs)
        return docs[:n]

    # ----------------------------------------------------------------
    def _generate_for_batch(self, batch: List[Document]) -> List[dict]:
        from rag_chain import EHSLLM
        llm = EHSLLM(self.config)
        blocks = "\n\n".join(
            f"【片段{i}】{d.page_content}" for i, d in enumerate(batch, 1))
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=GEN_SYSTEM),
                    HumanMessage(content=f"以下是 {len(batch)} 个片段：\n{blocks}")]
        raw = llm.invoke(messages)
        return self._parse_json_array(raw)

    @staticmethod
    def _parse_json_array(text: str) -> List[dict]:
        if not text:
            return []
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            return [d for d in data if d.get("question") and d.get("answer")]
        except Exception as e:
            logger.warning("JSON 解析失败: %s", e)
            return []

    # ----------------------------------------------------------------
    def generate(self, target_size: int, batch_size: int = 6,
                 verbose: bool = True) -> List[dict]:
        """用 LLM 生成评测题（在种子之外补足 target_size）。"""
        seeds = SeedQuestionBank().as_records()
        need = max(0, target_size - len(seeds))
        # 每个片段产出 1 题，多取 20% 余量应对解析失败
        source_chunks = self._sample_chunks(int(need * 1.3) + 5)

        generated: List[dict] = []
        for i in range(0, len(source_chunks), batch_size):
            if len(generated) >= need:
                break
            batch = source_chunks[i:i + batch_size]
            try:
                items = self._generate_for_batch(batch)
            except Exception as e:
                logger.warning("第 %d 批生成失败: %s", i, e)
                items = []
            for item, src in zip(items, batch):
                if len(generated) >= need:
                    break
                generated.append(self._record_from_generation(item, src))
            if verbose:
                logger.info("LLM 生成进度：%d/%d", len(generated), need)

        records = seeds + generated[:need]
        self._renumber(records)
        return records

    @staticmethod
    def _record_from_generation(item: dict, src: Document) -> dict:
        # 根据来源块内容粗略归类
        t = src.page_content
        if src.metadata.get("doc_type") == "table":
            cat = SeedQuestionBank.C_PARAM
        elif re.search(r"\d+\s*(m|mm|米|dB|%|倍|日|小时|年|人)", t):
            cat = SeedQuestionBank.C_PARAM
        elif any(k in t for k in ("隐患", "判定", "危险")):
            cat = SeedQuestionBank.C_HAZARD
        else:
            cat = SeedQuestionBank.C_SYSTEM
        if re.search(r"图\s*\d", t) and "图" in item["question"]:
            cat = SeedQuestionBank.C_FIGURE
        return {
            "id": "",
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
            "key_points": item.get("key_points", []),
            "source_file": src.metadata.get("file_name"),
            "section": src.metadata.get("section_title", ""),
            "source_chunk_id": src.metadata.get("chunk_id"),
            "category": cat,
            "is_seed": False,
        }

    @staticmethod
    def _renumber(records: List[dict]) -> None:
        for i, r in enumerate(records, 1):
            if not r.get("is_seed"):
                r["id"] = f"gen_{i:03d}"

    # ----------------------------------------------------------------
    def build(self, target_size: Optional[int] = None,
              use_llm: bool = True) -> List[dict]:
        target_size = target_size or self.config.eval_cfg.target_size
        if use_llm:
            records = self.generate(target_size)
        else:
            records = SeedQuestionBank().as_records()
        self.save(records)
        self._summary(records)
        return records

    def save(self, records: List[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("评测集已保存：%s（%d 条）", self.path, len(records))

    def load(self) -> List[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _summary(self, records: List[dict]) -> None:
        by_cat: Dict[str, int] = {}
        n_seed = 0
        for r in records:
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
            n_seed += 1 if r.get("is_seed") else 0
        logger.info("评测集统计：共 %d 条（人工种子 %d），分类：%s",
                    len(records), n_seed, by_cat)
