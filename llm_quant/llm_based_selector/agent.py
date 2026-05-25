"""
LLM 因子选择 Agent

基于大盘信息和因子特征，使用 LLM 进行智能因子推荐。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from llm_quant.utils.llm_client import LLMClient


SYSTEM_PROMPT = (
    "你是一名资深量化因子研究员和宏观策略分析师。"
    "你的任务是根据当前的大盘信息、市场风格、宏观环境等因素，"
    "从可用的因子库中推荐最适合当前市场环境的因子组合。"
    "分析要基于逻辑、数据和市场常识，输出 JSON 格式的推荐结果。"
)


USER_PROMPT_TEMPLATE = """请根据以下大盘信息和因子特征，推荐当前最有效的因子组合。

【当前日期】
{date}

【大盘信息汇总】
{market_info}

【可用因子列表】
{factors_info}

【历史因子表现】
{factor_performance}

【输出要求】
1. 推荐 3-8 个最适合当前环境的因子。
2. 对每个推荐因子解释为什么在当前环境下有效。
3. 给出因子权重建议。
4. 识别需要关注的风险或需要回避的因子。
5. 所有内容必须用中文，输出格式必须是有效的 JSON。

【JSON Schema】
{{
  "market_analysis": {{
    "current_regime": "当前市场制度/风格描述",
    "dominant_factors": ["主导因子方向1", "主导因子方向2"],
    "risk_factors": ["当前风险因子1", "风险因子2"],
    "macroeconomic_outlook": "宏观经济展望"
  }},
  "recommended_factors": [
    {{
      "factor_name": "因子名称",
      "rationale": "推荐原因（为什么这个因子在当前环境下有效）",
      "historical_ic": 0.035,
      "recent_performance": "最近表现描述",
      "expected_performance": "预期表现",
      "weight": 0.15,
      "risk_level": "低|中|高"
    }}
  ],
  "portfolio_composition": {{
    "total_weight": 1.0,
    "concentration": "因子权重集中度描述",
    "diversification": "分散化程度描述"
  }},
  "factors_to_avoid": [
    {{
      "factor_name": "因子名称",
      "reason": "需要回避的原因"
    }}
  ],
  "implementation_suggestions": {{
    "rebalance_frequency": "调整周期",
    "entry_timing": "入场建议",
    "exit_signals": ["退出信号1", "退出信号2"],
    "monitoring_metrics": ["监控指标1", "指标2"]
  }},
  "confidence_level": "高|中|低",
  "uncertainty_sources": ["不确定性来源1", "来源2"],
  "next_review_date": "下次审视日期（ISO格式）"
}}
"""


class FactorSelectorAgent:
    """基于 LLM 的因子选择 Agent"""

    def __init__(self, llm: LLMClient | None = "auto"):
        """
        初始化 Agent
        
        Args:
            llm: LLMClient 实例，或 "auto"（自动创建），或 None（禁用 LLM，使用备用方案）
        """
        if llm == "auto":
            # 自动创建 LLMClient（原来的行为）
            try:
                self.llm = LLMClient()
            except Exception:
                self.llm = None
        else:
            # 显式指定 llm（包括 None）
            self.llm = llm

    def select_factors(
        self,
        date: str,
        market_info: str,
        factors_list: list[dict[str, Any]],
        factor_performance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        基于大盘信息和因子特征进行因子选择

        Args:
            date: 当前日期 (ISO 格式)
            market_info: 大盘信息汇总文本
            factors_list: 可用因子列表，每个元素为 {name, description, category, ...}
            factor_performance: 历史因子表现信息

        Returns:
            包含推荐因子的结构化 JSON 结果
        """
        factors_info = self._format_factors_info(factors_list)
        performance_info = (
            self._format_performance_info(factor_performance)
            if factor_performance
            else "暂无历史表现数据"
        )

        if self.llm is None:
            recommendation = self._fallback_recommendation(factors_list)
        else:
            prompt = USER_PROMPT_TEMPLATE.format(
                date=date,
                market_info=market_info[:8000],  # 限制上下文长度
                factors_info=factors_info[:5000],
                factor_performance=performance_info[:3000],
            )
            try:
                resp = self.llm.chat(
                    user_message=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=4000,
                )
                recommendation = self._parse_json(resp)
            except Exception as e:
                print(f"LLM 调用失败: {e}，使用备用推荐")
                recommendation = self._fallback_recommendation(factors_list)

        return {
            "date": date,
            "recommendation": recommendation,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _format_factors_info(factors_list: list[dict[str, Any]]) -> str:
        """格式化因子信息"""
        chunks = []
        for factor in factors_list[:50]:  # 最多 50 个因子
            name = factor.get("name", "未知")
            description = factor.get("description", "")
            category = factor.get("category", "")
            formula = factor.get("formula", "")

            chunk = f"- {name} [{category}]"
            if description:
                chunk += f"\n  描述: {description}"
            if formula:
                chunk += f"\n  公式: {formula}"
            chunks.append(chunk)

        return "\n".join(chunks)

    @staticmethod
    def _format_performance_info(performance: dict[str, Any]) -> str:
        """格式化因子表现信息"""
        chunks = []

        if "top_factors" in performance:
            chunks.append("【近期表现最好的因子】")
            for factor in performance["top_factors"][:5]:
                chunks.append(
                    f"- {factor.get('name')}: "
                    f"IC={factor.get('ic', 0):.4f}, "
                    f"年化收益={factor.get('annual_return', 0)*100:.2f}%, "
                    f"Sharpe={factor.get('sharpe', 0):.4f}"
                )

        if "bottom_factors" in performance:
            chunks.append("\n【近期表现最差的因子】")
            for factor in performance["bottom_factors"][:5]:
                chunks.append(
                    f"- {factor.get('name')}: "
                    f"IC={factor.get('ic', 0):.4f}, "
                    f"年化收益={factor.get('annual_return', 0)*100:.2f}%"
                )

        if "market_style" in performance:
            chunks.append(f"\n【市场风格】{performance['market_style']}")

        return "\n".join(chunks)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从 LLM 响应中解析 JSON"""
        t = text.strip()
        
        # 移除代码块标记
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3:
                t = "\n".join(lines[1:-1])
        
        # 提取 JSON 部分
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
        
        obj = json.loads(t)
        
        # 设置默认值
        obj.setdefault("market_analysis", {})
        obj.setdefault("recommended_factors", [])
        obj.setdefault("portfolio_composition", {})
        obj.setdefault("factors_to_avoid", [])
        obj.setdefault("implementation_suggestions", {})
        obj.setdefault("confidence_level", "中")
        obj.setdefault("uncertainty_sources", [])
        obj.setdefault("next_review_date", "")
        
        return obj

    @staticmethod
    def _fallback_recommendation(factors_list: list[dict[str, Any]]) -> dict[str, Any]:
        """LLM 失败时的备用推荐"""
        # 简单的备用策略：选择前几个因子，等权重
        selected = factors_list[:5] if len(factors_list) >= 5 else factors_list
        weight = 1.0 / len(selected) if selected else 0
        
        recommended = [
            {
                "factor_name": f.get("name", "未知"),
                "rationale": f"基于备用策略选择的因子",
                "historical_ic": f.get("ic", 0),
                "weight": weight,
                "risk_level": f.get("risk_level", "中"),
            }
            for f in selected
        ]

        return {
            "market_analysis": {
                "current_regime": "数据不可用，无法进行详细分析",
                "dominant_factors": [],
                "risk_factors": ["市场不确定性"],
                "macroeconomic_outlook": "中性观点",
            },
            "recommended_factors": recommended,
            "portfolio_composition": {
                "total_weight": 1.0,
                "concentration": "均匀分散",
            },
            "factors_to_avoid": [],
            "implementation_suggestions": {
                "rebalance_frequency": "1 个月",
                "entry_timing": "逐步建仓",
                "exit_signals": ["IC 转负", "排名跌出前 10"],
                "monitoring_metrics": ["IC", "夏普比率"],
            },
            "confidence_level": "低",
            "uncertainty_sources": ["LLM 不可用", "数据不完整"],
        }
