"""
第2层：解析加工层（Text Processing Layer）
功能：
  - 调用LLM对原始金融文本进行深度解析
  - 提取：投资逻辑、核心驱动、受益标的、时效周期
  - 生成标准化摘要、标签
  - 结果写入 InfoStore
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.data.info_store import InfoStore
from llm_quant.info_based_summarize import InfoParsingAgent
from llm_quant.utils.llm_client import LLMClient


class TextProcessor:
    """
    第2层：解析加工层。
    对原始信息进行LLM深度解析，结构化存入信息库。
    """

    def __init__(self, llm: LLMClient | None = None, info_store: InfoStore | None = None):
        self.llm = llm or LLMClient()
        self.parser = InfoParsingAgent(llm=self.llm)
        self.info_store = info_store or InfoStore()

    def process(self, raw_record: dict, max_summary_chars: int = 220) -> dict:
        """
        解析单条原始信息记录，返回增强后的结构化记录。
        """
        parsed = self.parser.analyze_record(
            record=raw_record,
            mode="realtime",
            max_chars=max_summary_chars,
        )

        topics = parsed.get("related_topics", [])
        industries = parsed.get("related_industries", [])
        assumptions = parsed.get("assumptions", [])
        uncertainties = parsed.get("uncertainties", [])
        core_viewpoints = parsed.get("core_viewpoints", [])

        sentiment_map = {"乐观": "positive", "中立": "neutral", "悲观": "negative"}

        # 合并原始记录 + 解析结果；兼容已有层使用的字段名
        enhanced = dict(raw_record)
        enhanced.update({
            "summary": parsed.get("refined_summary", ""),
            "logic": parsed.get("market_impact_chain", ""),
            "key_drivers": core_viewpoints,
            "affected_sectors": industries,
            "pricing_factors": topics,
            "time_horizon": "未知",
            "sentiment": sentiment_map.get(parsed.get("sentiment", "中立"), "neutral"),
            "tags": list(dict.fromkeys(topics + industries + parsed.get("related_companies", []))),
            "risk_warnings": "；".join(uncertainties),
            "source_meta": parsed.get("source", {}),
            "market_impact_chain": parsed.get("market_impact_chain", ""),
            "uncertainties": uncertainties,
            "assumptions": assumptions,
            "core_viewpoints": core_viewpoints,
            "key_metrics": parsed.get("key_metrics", []),
            "traceability": parsed.get("traceability", {}),
            "processing_mode": "realtime",
        })
        return enhanced

    def process_factor_literature(self, raw_record: dict) -> dict:
        """
        解析单篇因子文献（可由PDF抽取文本后传入content字段）。
        """
        parsed = self.parser.analyze_record(record=raw_record, mode="factor_literature")

        enhanced = dict(raw_record)
        enhanced.update(
            {
                "summary": f"提取到 {len(parsed.get('factors', []))} 个可计算因子",
                "logic": "文献因子提取结果见 factor_literature 字段",
                "key_drivers": [x.get("factor_name", "") for x in parsed.get("factors", []) if x.get("factor_name")],
                "affected_sectors": [],
                "pricing_factors": ["文献因子"],
                "time_horizon": "未知",
                "sentiment": "neutral",
                "tags": ["研报", "文献", "因子提取"],
                "risk_warnings": "文献抽取结果建议人工复核后使用。",
                "factor_literature": parsed,
                "processing_mode": "factor_literature",
            }
        )
        return enhanced

    def process_and_store(self, raw_records: list[dict]) -> list[str]:
        """
        批量解析并存入信息库，返回 record_id 列表。
        """
        ids = []
        for i, rec in enumerate(raw_records):
            print(f"[Layer2] 解析 {i+1}/{len(raw_records)}: {rec.get('title','')[:40]}...")
            enhanced = self.process(rec)
            rec_id   = self.info_store.add(enhanced)
            ids.append(rec_id)
        print(f"[Layer2] 解析完成，入库 {len(ids)} 条。")
        return ids

    def process_factor_literature_and_store(self, raw_records: list[dict]) -> list[str]:
        """
        批量处理包含因子构建方式的文献，并存入信息库。
        """
        ids = []
        for i, rec in enumerate(raw_records):
            print(f"[Layer2] 文献解析 {i+1}/{len(raw_records)}: {rec.get('title','')[:40]}...")
            enhanced = self.process_factor_literature(rec)
            rec_id = self.info_store.add(enhanced)
            ids.append(rec_id)
        print(f"[Layer2] 文献解析完成，入库 {len(ids)} 条。")
        return ids
