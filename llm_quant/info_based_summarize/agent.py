from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from llm_quant.utils.llm_client import LLMClient


SYSTEM_PROMPT = (
    "你是一名资深宏观与策略研究员，负责把某一天内的新闻与研报单条解读结果整合成日度投资综述。"
    "请从市场主线、定价逻辑、风险点和投资建议四个维度做归纳，只输出 JSON。"
)


USER_PROMPT_TEMPLATE = """请根据以下某一天内的单条分析结果，输出日度综述，只返回 JSON。

【日期】
{date}

【样本统计】
news_count: {news_count}
report_count: {report_count}
item_count: {item_count}

【单条分析摘录】
{context}

【输出要求】
1. 全部用中文。
2. 结论要以“当天市场信息综合”视角输出，不要逐条复述。
3. 建议审慎，说明不确定性来源。
4. 字段名固定，字段值中文。

【JSON Schema】
{{
  "market_overview": "对当天信息面的总体概括，150字以内",
  "dominant_themes": ["主线1", "主线2", "主线3"],
  "pricing_logic_summary": "归纳当天最重要的定价逻辑链条",
  "bullish_signals": ["偏利多信号1", "偏利多信号2"],
  "bearish_signals": ["偏利空信号1", "偏利空信号2"],
  "investor_advice": {{
    "stance": "偏积极|中性|偏谨慎",
    "short_term": "短线应对建议",
    "swing_term": "波段应对建议",
    "risk_controls": ["风控1", "风控2"]
  }},
  "watch_list": ["需要继续跟踪的方向1", "方向2"],
  "uncertainties": ["不确定性1", "不确定性2"]
}}
"""


class InfoBasedSummarizerAgent:
    def __init__(self, llm: LLMClient | None = None):
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = LLMClient()
            except Exception:
                self.llm = None

    def summarize_day(self, date: str, analyzed_rows: list[dict[str, Any]]) -> dict[str, Any]:
        news_count = sum(1 for x in analyzed_rows if x.get("source_record", {}).get("source_type") == "news")
        report_count = sum(1 for x in analyzed_rows if x.get("source_record", {}).get("source_type") == "report")
        context = self._build_context(analyzed_rows)

        if self.llm is None:
            summary = self._fallback_summary(analyzed_rows)
        else:
            prompt = USER_PROMPT_TEMPLATE.format(
                date=date,
                news_count=news_count,
                report_count=report_count,
                item_count=len(analyzed_rows),
                context=context[:12000],
            )
            try:
                resp = self.llm.chat(user_message=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=3000)
                summary = self._parse_json(resp)
            except Exception:
                summary = self._fallback_summary(analyzed_rows)

        return {
            "date": date,
            "counts": {
                "items": len(analyzed_rows),
                "news": news_count,
                "reports": report_count,
            },
            "summary": summary,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _build_context(analyzed_rows: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for idx, row in enumerate(analyzed_rows, start=1):
            src = row.get("source_record", {})
            ana = row.get("analysis", {})
            impact = ana.get("market_impact", {})
            suggestion = ana.get("investment_suggestion", {})
            chunks.append(
                f"[{idx}] 类型={src.get('source_type','')} 标题={src.get('title','')}\n"
                f"  主要内容={ana.get('main_content','')}\n"
                f"  定价逻辑={ana.get('pricing_logic','')}\n"
                f"  影响方向={impact.get('direction','')} 资产={','.join(impact.get('target_assets',[]) or [])}\n"
                f"  建议={suggestion.get('action','')} {suggestion.get('position_guidance','')}"
            )
        return "\n\n".join(chunks)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        t = text.strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3:
                t = "\n".join(lines[1:-1])
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
        obj = json.loads(t)
        obj.setdefault("market_overview", "")
        obj.setdefault("dominant_themes", [])
        obj.setdefault("pricing_logic_summary", "")
        obj.setdefault("bullish_signals", [])
        obj.setdefault("bearish_signals", [])
        obj.setdefault("investor_advice", {})
        obj.setdefault("watch_list", [])
        obj.setdefault("uncertainties", [])
        advice = obj["investor_advice"]
        advice.setdefault("stance", "中性")
        advice.setdefault("short_term", "轻仓观察")
        advice.setdefault("swing_term", "等待更多确认")
        advice.setdefault("risk_controls", ["控制仓位", "避免追高"])
        return obj

    @staticmethod
    def _fallback_summary(analyzed_rows: list[dict[str, Any]]) -> dict[str, Any]:
        directions = [x.get("analysis", {}).get("market_impact", {}).get("direction", "不确定") for x in analyzed_rows]
        cautious = sum(1 for x in directions if x in {"不确定", "中性偏空", "利空"})
        stance = "偏谨慎" if cautious >= max(1, len(directions) // 2) else "中性"
        return {
            "market_overview": "当天信息面已完成聚合，但当前为兜底汇总结果，建议结合更多样本和人工复核使用。",
            "dominant_themes": [],
            "pricing_logic_summary": "样本内未形成足够稳定的统一主线，建议继续观察政策、行业景气和业绩验证。",
            "bullish_signals": [],
            "bearish_signals": [],
            "investor_advice": {
                "stance": stance,
                "short_term": "短线轻仓观察，不依据单日样本激进交易。",
                "swing_term": "等待更高置信度主线形成后再做波段布局。",
                "risk_controls": ["控制仓位", "避免追高"]
            },
            "watch_list": [],
            "uncertainties": ["部分单条结论可能受样本质量影响", "需结合后续信息验证"]
        }
