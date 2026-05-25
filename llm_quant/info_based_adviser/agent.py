from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_quant.utils.llm_client import LLMClient


SYSTEM_PROMPT = (
    "你是一名资深投资研究员，擅长对单条财经新闻、研报和事件信息进行审慎解读。"
    "你的任务是从投资者视角，提炼主要内容、定价逻辑、潜在市场影响和可执行但保守的投资建议。"
    "请只输出 JSON，不要输出任何额外解释。"
)


USER_PROMPT_TEMPLATE = """请对以下单条信息进行投资视角解读，并只返回 JSON。

【输入信息】
section: {section}
title: {title}
publish_time: {publish_time}
url: {url}
content: {content}

【输出要求】
1. 输出内容全部使用中文。
2. 语气务必审慎，不要夸大确定性。
3. 如果信息不足、正文缺失或只是专题页标题，要明确指出信息有限，并给出保守建议。
4. 字段名保持不变，便于后续程序处理；字段值请用中文表达。
5. confidence 取值范围为 0 到 1。

【JSON Schema】
{{
    "main_content": "该信息的主要内容概述，100字以内",
    "pricing_logic": "定价逻辑，写清楚：冲击/事件 -> 传导路径 -> 可能影响的资产或板块",
    "market_impact": {{
        "direction": "利多|利空|中性偏多|中性偏空|不确定",
        "target_assets": ["受影响的资产/行业/风格"],
        "horizon_days": 1,
        "confidence": 0.0,
        "reasoning": "判断依据，说明为什么会有上述影响"
    }},
    "investment_suggestion": {{
        "action": "买入|持有|减仓|回避|观察",
        "for_investor": "短线|波段|中长期",
        "holding_period_days": 1,
        "position_guidance": "仓位或应对建议，强调审慎操作",
        "risk_controls": ["风控建议1", "风控建议2"]
    }},
    "risk_warnings": ["风险提示1", "风险提示2"],
    "evidence_quotes": ["来自原文的关键证据片段1", "关键证据片段2"]
}}
"""


class InfoBasedAdviserAgent:
    """Analyze one record into investment-oriented structured advice."""

    def __init__(self, llm: LLMClient | None = None):
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = LLMClient()
            except Exception:
                self.llm = None

    def analyze_record(self, record: dict[str, Any]) -> dict[str, Any]:
        effective_publish_time = self._get_effective_publish_time(record)
        prompt = USER_PROMPT_TEMPLATE.format(
            section=str(record.get("source_type", record.get("section", ""))),
            title=str(record.get("title", "")),
            publish_time=effective_publish_time,
            url=str(record.get("url", "")),
            content=str(record.get("content", ""))[:3500],
        )

        if self.llm is None:
            parsed = self._rule_based_parse(record)
        else:
            try:
                resp = self.llm.chat(user_message=prompt, system_prompt=SYSTEM_PROMPT)
                parsed = self._parse_json(resp)
            except Exception as exc:
                parsed = self._fallback(record, f"llm_error: {exc}")

        return self._with_metadata(record, parsed)

    def _with_metadata(self, record: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_record": {
                "source_type": record.get("source_type", "news"),
                "section": record.get("section", ""),
                "report_type": record.get("report_type", ""),
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "publish_time": self._get_effective_publish_time(record),
                "publish_date": self._get_effective_publish_date(record),
                "org": record.get("org", ""),
                "analyst": record.get("analyst", ""),
            },
            "analysis": parsed,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _get_effective_publish_time(record: dict[str, Any]) -> str:
        publish_time = str(record.get("publish_time", "")).strip()
        if publish_time:
            return publish_time
        publish_date = str(record.get("publish_date", "")).strip()
        return publish_date

    @staticmethod
    def _get_effective_publish_date(record: dict[str, Any]) -> str:
        publish_time = str(record.get("publish_time", "")).strip()
        if publish_time:
            return publish_time[:10]
        publish_date = str(record.get("publish_date", "")).strip()
        return publish_date[:10]

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

        # Light normalization to keep downstream stable.
        obj.setdefault("main_content", "")
        obj.setdefault("pricing_logic", "")
        obj.setdefault("market_impact", {})
        obj.setdefault("investment_suggestion", {})
        obj.setdefault("risk_warnings", [])
        obj.setdefault("evidence_quotes", [])

        impact = obj["market_impact"]
        impact.setdefault("direction", "不确定")
        impact.setdefault("target_assets", [])
        impact.setdefault("horizon_days", 5)
        impact.setdefault("confidence", 0.3)
        impact.setdefault("reasoning", "")

        sugg = obj["investment_suggestion"]
        sugg.setdefault("action", "观察")
        sugg.setdefault("for_investor", "波段")
        sugg.setdefault("holding_period_days", 5)
        sugg.setdefault("position_guidance", "轻仓观察，等待更多确认信号")
        sugg.setdefault("risk_controls", ["设置止损", "避免集中持仓"])

        # Clamp confidence
        try:
            c = float(impact.get("confidence", 0.3))
            impact["confidence"] = max(0.0, min(1.0, c))
        except Exception:
            impact["confidence"] = 0.3

        return obj

    @staticmethod
    def _fallback(record: dict[str, Any], reason: str) -> dict[str, Any]:
        title = str(record.get("title", ""))
        return {
            "main_content": title[:120],
            "pricing_logic": "可用信息不足，暂时无法建立可靠的定价传导链条，应等待更多证据。",
            "market_impact": {
                "direction": "不确定",
                "target_assets": [],
                "horizon_days": 5,
                "confidence": 0.2,
                "reasoning": reason,
            },
            "investment_suggestion": {
                "action": "观察",
                "for_investor": "波段",
                "holding_period_days": 5,
                "position_guidance": "暂不追价，等待更多信息确认后再决策。",
                "risk_controls": ["设置止损", "保持低仓位"],
            },
            "risk_warnings": ["当前结果为兜底输出，请人工复核。"],
            "evidence_quotes": [],
        }

    @staticmethod
    def _rule_based_parse(record: dict[str, Any]) -> dict[str, Any]:
        title = str(record.get("title", ""))
        content = str(record.get("content", ""))
        text = (title + " " + content).lower()

        bullish_words = ["improve", "growth", "rebound", "support", "cut", "stimulus", "up"]
        bearish_words = ["risk", "decline", "down", "pressure", "default", "fall", "weak"]

        bull_score = sum(1 for w in bullish_words if w in text)
        bear_score = sum(1 for w in bearish_words if w in text)

        if bull_score > bear_score:
            direction = "中性偏多"
            action = "持有"
            conf = 0.45
        elif bear_score > bull_score:
            direction = "中性偏空"
            action = "减仓"
            conf = 0.45
        else:
            direction = "不确定"
            action = "观察"
            conf = 0.3

        return {
            "main_content": title[:120],
            "pricing_logic": "当前为规则兜底模式：依据标题关键词与栏目特征，对潜在市场影响做保守推断。",
            "market_impact": {
                "direction": direction,
                "target_assets": [str(record.get("section", "market"))],
                "horizon_days": 5,
                "confidence": conf,
                "reasoning": "当前未获取到可用模型结果，因此采用规则兜底判断，结论仅供参考。",
            },
            "investment_suggestion": {
                "action": action,
                "for_investor": "波段",
                "holding_period_days": 5,
                "position_guidance": "建议轻仓观察，等待价格与成交量进一步确认。",
                "risk_controls": ["设置止损", "避免杠杆"],
            },
            "risk_warnings": ["当前为规则兜底结果，交易前请人工复核。"],
            "evidence_quotes": [title[:80]],
        }


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
