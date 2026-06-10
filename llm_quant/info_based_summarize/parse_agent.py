from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from llm_quant.prompts import (
        INFO_PARSING_SYSTEM_PROMPT,
        render_factor_paper_prompt,
        render_general_info_prompt,
)
from llm_quant.utils.llm_client import LLMClient


class InfoParsingAgent:
    """第2层解析加工Agent：支持一般信息解析与因子文献解析。"""

    def __init__(self, llm: LLMClient | None = None):
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = LLMClient()
            except Exception:
                self.llm = None

    def analyze_realtime_text(
        self,
        raw_text: str,
        metadata: dict[str, Any] | None = None,
        max_chars: int = 220,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        prompt = render_general_info_prompt(
            raw_text=(raw_text or "")[:12000],
            max_chars=max_chars,
            source=str(metadata.get("source", "")),
            publish_date=str(metadata.get("publish_date", "")),
            title=str(metadata.get("title", "")),
            url=str(metadata.get("url", "")),
        )

        if self.llm is None:
            parsed = self._fallback_realtime(raw_text=raw_text, metadata=metadata, max_chars=max_chars)
        else:
            try:
                resp = self.llm.chat(
                    user_message=prompt,
                    system_prompt=INFO_PARSING_SYSTEM_PROMPT,
                    max_tokens=3000,
                )
                parsed = self._parse_json_object(resp)
            except Exception:
                parsed = self._fallback_realtime(raw_text=raw_text, metadata=metadata, max_chars=max_chars)

        return self._normalize_realtime(parsed, metadata=metadata, max_chars=max_chars)

    def analyze_factor_literature(
        self,
        raw_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        prompt = render_factor_paper_prompt(
            document_name=str(metadata.get("document_name", "")),
            raw_text=(raw_text or "")[:30000],
        )

        if self.llm is None:
            parsed = self._fallback_factor_literature(metadata=metadata)
        else:
            try:
                resp = self.llm.chat(
                    user_message=prompt,
                    system_prompt=INFO_PARSING_SYSTEM_PROMPT,
                    max_tokens=4000,
                )
                parsed = self._parse_json_object(resp)
            except Exception:
                parsed = self._fallback_factor_literature(metadata=metadata)

        return self._normalize_factor_literature(parsed, metadata=metadata)

    def analyze_record(
        self,
        record: dict[str, Any],
        mode: str = "realtime",
        max_chars: int = 220,
    ) -> dict[str, Any]:
        title = str(record.get("title", ""))
        content = str(record.get("content", "") or record.get("summary", ""))
        raw_text = f"标题：{title}\n正文：{content}".strip()
        publish_date = str(record.get("publish_date", "") or record.get("date", "") or "")
        metadata = {
            "source": str(record.get("source", "")),
            "publish_date": publish_date,
            "title": title,
            "url": str(record.get("url", "")),
            "document_name": str(record.get("document_name", title)),
        }
        if mode == "factor_literature":
            return self.analyze_factor_literature(raw_text=raw_text, metadata=metadata)
        return self.analyze_realtime_text(raw_text=raw_text, metadata=metadata, max_chars=max_chars)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        t = text.strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3:
                t = "\n".join(lines[1:-1])
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
        return json.loads(t)

    @staticmethod
    def _clamp_summary(text: str, max_chars: int) -> str:
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        if len(clean) <= max_chars:
            return clean
        return clean[: max(1, max_chars - 1)] + "…"

    def _normalize_realtime(
        self,
        parsed: dict[str, Any],
        metadata: dict[str, Any],
        max_chars: int,
    ) -> dict[str, Any]:
        source = parsed.get("source", {}) if isinstance(parsed.get("source", {}), dict) else {}
        traceability = parsed.get("traceability", {}) if isinstance(parsed.get("traceability", {}), dict) else {}

        result = {
            "source": {
                "name": source.get("name", metadata.get("source", "")),
                "publish_date": source.get("publish_date", metadata.get("publish_date", "")),
                "title": source.get("title", metadata.get("title", "")),
                "url": source.get("url", metadata.get("url", "")),
            },
            "related_topics": self._ensure_str_list(parsed.get("related_topics", [])),
            "related_industries": self._ensure_str_list(parsed.get("related_industries", [])),
            "related_companies": self._ensure_str_list(parsed.get("related_companies", [])),
            "core_viewpoints": self._ensure_str_list(parsed.get("core_viewpoints", [])),
            "sentiment": self._normalize_sentiment(str(parsed.get("sentiment", "中立"))),
            "market_impact_chain": str(parsed.get("market_impact_chain", "")).strip(),
            "key_metrics": self._normalize_metrics(parsed.get("key_metrics", [])),
            "uncertainties": self._ensure_str_list(parsed.get("uncertainties", [])),
            "assumptions": self._ensure_str_list(parsed.get("assumptions", [])),
            "refined_summary": self._clamp_summary(str(parsed.get("refined_summary", "")), max_chars=max_chars),
            "traceability": {
                "evidence_quotes": self._ensure_str_list(traceability.get("evidence_quotes", [])),
            },
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

        if not result["refined_summary"]:
            fallback = str(metadata.get("title", "")) or "信息摘要待补充"
            result["refined_summary"] = self._clamp_summary(fallback, max_chars=max_chars)

        return result

    def _normalize_factor_literature(
        self,
        parsed: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        doc = parsed.get("document", {}) if isinstance(parsed.get("document", {}), dict) else {}
        factors = parsed.get("factors", []) if isinstance(parsed.get("factors", []), list) else []

        normalized_factors: list[dict[str, Any]] = []
        for x in factors:
            if not isinstance(x, dict):
                continue
            normalized_factors.append(
                {
                    "factor_name": str(x.get("factor_name", "文献未提及")).strip() or "文献未提及",
                    "factor_definition": str(x.get("factor_definition", "文献未提及")).strip() or "文献未提及",
                    "calculation_method": str(x.get("calculation_method", "文献未提及")).strip() or "文献未提及",
                    "economic_logic": str(x.get("economic_logic", "文献未提及")).strip() or "文献未提及",
                    "return_source": str(x.get("return_source", "文献未提及")).strip() or "文献未提及",
                    "required_inputs": self._ensure_str_list(x.get("required_inputs", [])),
                    "constraints_or_notes": str(x.get("constraints_or_notes", "文献未提及")).strip() or "文献未提及",
                    "missing_info": self._ensure_str_list(x.get("missing_info", [])),
                }
            )

        return {
            "document": {
                "name": str(doc.get("name", metadata.get("document_name", ""))).strip(),
                "source": str(doc.get("source", metadata.get("source", ""))).strip(),
                "publish_date": str(doc.get("publish_date", metadata.get("publish_date", ""))).strip(),
                "notes": str(doc.get("notes", "")).strip(),
            },
            "factors": normalized_factors,
            "extraction_notes": str(parsed.get("extraction_notes", "")).strip(),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        return []

    @staticmethod
    def _normalize_sentiment(sentiment: str) -> str:
        s = sentiment.strip()
        if s in {"乐观", "中立", "悲观"}:
            return s
        if s in {"positive", "bullish"}:
            return "乐观"
        if s in {"negative", "bearish"}:
            return "悲观"
        return "中立"

    @staticmethod
    def _normalize_metrics(metrics: Any) -> list[dict[str, str]]:
        if not isinstance(metrics, list):
            return []
        out: list[dict[str, str]] = []
        for x in metrics:
            if not isinstance(x, dict):
                continue
            out.append(
                {
                    "name": str(x.get("name", "")).strip(),
                    "value": str(x.get("value", "")).strip(),
                    "unit": str(x.get("unit", "")).strip(),
                    "context": str(x.get("context", "")).strip(),
                }
            )
        return out

    def _fallback_realtime(
        self,
        raw_text: str,
        metadata: dict[str, Any],
        max_chars: int,
    ) -> dict[str, Any]:
        title = str(metadata.get("title", ""))
        return {
            "source": {
                "name": str(metadata.get("source", "")),
                "publish_date": str(metadata.get("publish_date", "")),
                "title": title,
                "url": str(metadata.get("url", "")),
            },
            "related_topics": [],
            "related_industries": [],
            "related_companies": [],
            "core_viewpoints": [self._clamp_summary(raw_text or title, max_chars=80)],
            "sentiment": "中立",
            "market_impact_chain": "信息不足，暂无法建立稳定传导链条。",
            "key_metrics": [],
            "uncertainties": ["当前结果为兜底输出，需人工复核"],
            "assumptions": [],
            "refined_summary": self._clamp_summary(raw_text or title, max_chars=max_chars),
            "traceability": {
                "evidence_quotes": [self._clamp_summary(raw_text or title, max_chars=120)],
            },
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _fallback_factor_literature(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "document": {
                "name": str(metadata.get("document_name", "")),
                "source": str(metadata.get("source", "")),
                "publish_date": str(metadata.get("publish_date", "")),
                "notes": "模型不可用，返回兜底结构。",
            },
            "factors": [
                {
                    "factor_name": "文献未提及",
                    "factor_definition": "文献未提及",
                    "calculation_method": "文献未提及",
                    "economic_logic": "文献未提及",
                    "return_source": "文献未提及",
                    "required_inputs": [],
                    "constraints_or_notes": "文献未提及",
                    "missing_info": ["因子原始信息缺失"],
                }
            ],
            "extraction_notes": "当前为兜底输出，请补充文献文本后重试。",
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
