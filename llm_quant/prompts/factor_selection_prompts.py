FACTOR_SELECTOR_SYSTEM_PROMPT = (
    "你是一名资深量化因子研究员和宏观策略分析师。"
    "你的任务是根据当前的大盘信息、市场风格、宏观环境等因素，"
    "从可用的因子库中推荐最适合当前市场环境的因子组合。"
    "分析要基于逻辑、数据和市场常识，输出 JSON 格式的推荐结果。"
)

_FACTOR_SELECTOR_USER_TEMPLATE = """请根据以下大盘信息和因子特征，推荐当前最有效的因子组合。

【当前日期】
{date}

【大盘信息汇总】
{market_info}

【可用因子列表】
{factors_info}

【历史因子表现】
{factor_performance}

【输出要求】
1. 只允许从“可用因子列表”中推荐 2-5 个因子，禁止生成列表外因子名称。
2. 对每个推荐因子解释为什么在当前环境下有效。
3. 给出因子权重建议。
4. 识别需要关注的风险或需要回避的因子。
5. 所有内容必须用中文，输出格式必须是有效的 JSON。
6. 输出精炼：rationale/recent_performance/expected_performance 各不超过50字；uncertainty_sources 最多2项。

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


def render_factor_selector_user_prompt(
    date: str,
    market_info: str,
    factors_info: str,
    factor_performance: str,
) -> str:
    return _FACTOR_SELECTOR_USER_TEMPLATE.format(
        date=date,
        market_info=market_info,
        factors_info=factors_info,
        factor_performance=factor_performance,
    )
