INFO_SUMMARIZER_SYSTEM_PROMPT = (
    "你是一名资深宏观与策略研究员，负责把某一天内的新闻与研报单条解读结果整合成日度投资综述。"
    "请从市场主线、定价逻辑、风险点和投资建议四个维度做归纳，只输出 JSON。"
)

_INFO_SUMMARIZER_USER_TEMPLATE = """请根据以下某一天内的单条分析结果，输出日度综述，只返回 JSON。

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
5. 输出精炼：overview <= 80字；pricing_logic_summary <= 90字；每个数组最多2项。

【JSON Schema】
{{
  "market_overview": "对当天信息面的总体概括，80字以内",
  "dominant_themes": ["主线1", "主线2", "主线3"],
  "pricing_logic_summary": "归纳当天最重要的定价逻辑链条（90字以内）",
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


def render_info_summarizer_user_prompt(
    date: str,
    news_count: int,
    report_count: int,
    item_count: int,
    context: str,
) -> str:
    return _INFO_SUMMARIZER_USER_TEMPLATE.format(
        date=date,
        news_count=news_count,
        report_count=report_count,
        item_count=item_count,
        context=context,
    )
