_MARKET_ASSESS_TEMPLATE = """你是一名资深量化投资组合经理。
请根据以下最新市场信息，评估当前市场环境，并给出对各类因子的适配性调整建议。

【最新市场信息】
{market_context}

请输出JSON格式：
{{
  "market_regime": "bull/bear/volatile/sideways",
  "risk_appetite": "high/medium/low",
  "recommended_factor_types": ["momentum", "value", ...],
  "deprecated_factor_types":  ["...", "..."],
  "rotation_reason": "风格轮动原因说明（100字以内）"
}}
"""


def render_market_assess_prompt(market_context: str) -> str:
    return _MARKET_ASSESS_TEMPLATE.format(market_context=market_context)
