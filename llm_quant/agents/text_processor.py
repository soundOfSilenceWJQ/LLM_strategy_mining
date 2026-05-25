"""
第2层：解析加工层（Text Processing Layer）
功能：
  - 调用LLM对原始金融文本进行深度解析
  - 提取：投资逻辑、核心驱动、受益标的、时效周期
  - 生成标准化摘要、标签
  - 结果写入 InfoStore
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.utils.llm_client import LLMClient
from llm_quant.data.info_store import InfoStore

_SYSTEM_PROMPT = """你是一名顶级的金融研究分析师，专注于中国A股市场及全球资本市场的量化投研。
你的任务是对输入的金融文本进行专业的深度解析，提取其中隐含的投资逻辑和定价信息。
请始终以严谨、专业的态度进行分析，输出必须是有效的JSON格式。"""

_PARSE_PROMPT_TEMPLATE = """请对以下金融文本进行深度解析，输出JSON格式的结构化结果。

【原始文本】
标题：{title}
内容：{content}

【要求】
请严格按以下JSON结构输出，不要添加任何注释或说明：
{{
  "summary": "50字以内的核心摘要",
  "investment_logic": "提取文本中隐含的投资逻辑（宏观→行业→个股的传导路径，200字以内）",
  "key_drivers": ["驱动因素1", "驱动因素2", "驱动因素3"],
  "affected_sectors": ["行业1", "行业2"],
  "pricing_factors": ["可能与该信息相关的量化定价因子类型，如：估值、动量、盈利、政策敏感性等"],
  "time_horizon": "信息的投资有效期（如：短期1-4周/中期1-3月/长期6-12月）",
  "sentiment": "positive/negative/neutral",
  "tags": ["标签1", "标签2", "标签3"],
  "risk_warnings": "潜在风险提示（100字以内）"
}}"""


class TextProcessor:
    """
    第2层：解析加工层。
    对原始信息进行LLM深度解析，结构化存入信息库。
    """

    def __init__(self, llm: LLMClient | None = None, info_store: InfoStore | None = None):
        self.llm = llm or LLMClient()
        self.info_store = info_store or InfoStore()

    def process(self, raw_record: dict) -> dict:
        """
        解析单条原始信息记录，返回增强后的结构化记录。
        """
        title   = raw_record.get("title", "")
        content = raw_record.get("content", "") or raw_record.get("summary", "")

        # 内容太短则直接用标题
        if len(content) < 20:
            content = title

        prompt = _PARSE_PROMPT_TEMPLATE.format(
            title=title,
            content=content[:1500],   # 截断避免超长
        )

        try:
            response = self.llm.chat(user_message=prompt, system_prompt=_SYSTEM_PROMPT)
            parsed   = self._parse_json(response)
        except Exception as e:
            print(f"[Layer2] LLM解析失败 ({title[:30]}...): {e}")
            parsed = {
                "summary": title[:50],
                "investment_logic": "",
                "key_drivers": [],
                "affected_sectors": [],
                "pricing_factors": [],
                "time_horizon": "未知",
                "sentiment": "neutral",
                "tags": [],
                "risk_warnings": "",
            }

        # 合并原始记录 + 解析结果
        enhanced = dict(raw_record)
        enhanced.update({
            "summary":          parsed.get("summary", ""),
            "logic":            parsed.get("investment_logic", ""),
            "key_drivers":      parsed.get("key_drivers", []),
            "affected_sectors": parsed.get("affected_sectors", []),
            "pricing_factors":  parsed.get("pricing_factors", []),
            "time_horizon":     parsed.get("time_horizon", ""),
            "sentiment":        parsed.get("sentiment", "neutral"),
            "tags":             parsed.get("tags", []),
            "risk_warnings":    parsed.get("risk_warnings", ""),
        })
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

    # ── 辅助：JSON 解析容错 ───────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        """从LLM输出中提取JSON，容错处理Markdown代码块。"""
        text = text.strip()
        # 去除 ```json ... ``` 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        # 提取第一个 { } 块
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)
