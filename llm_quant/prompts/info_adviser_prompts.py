INFO_ADVISER_SYSTEM_PROMPT = (
    "你是一名资深投资研究员，擅长对单条财经新闻、研报和事件信息进行审慎解读。"
    "你的任务是从投资者视角，提炼主要内容、定价逻辑、潜在市场影响和可执行但保守的投资建议。"
    "请只输出 JSON，不要输出任何额外解释。"
)

_INFO_ADVISER_USER_TEMPLATE = """请对以下单条信息进行投资视角解读，并只返回 JSON。

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
6. 输出要精炼：main_content <= 60字；pricing_logic <= 80字；reasoning <= 60字；risk_warnings 每条 <= 30字，最多2条；evidence_quotes 最多1条。

【JSON Schema】
{{
    "main_content": "该信息的主要内容概述（60字以内）",
    "pricing_logic": "定价逻辑（80字以内）：冲击/事件 -> 传导路径 -> 影响资产",
    "market_impact": {{
        "direction": "利多|利空|中性偏多|中性偏空|不确定",
        "target_assets": ["受影响的资产/行业/风格"],
        "horizon_days": 1,
        "confidence": 0.0,
        "reasoning": "判断依据（60字以内）"
    }},
    "investment_suggestion": {{
        "action": "买入|持有|减仓|回避|观察",
        "for_investor": "短线|波段|中长期",
        "holding_period_days": 1,
        "position_guidance": "仓位或应对建议，强调审慎操作",
        "risk_controls": ["风控建议1", "风控建议2"]
    }},
    "risk_warnings": ["风险提示1", "风险提示2（可选）"],
    "evidence_quotes": ["来自原文的关键证据片段（最多1条）"]
}}
"""


def render_info_adviser_user_prompt(
    section: str,
    title: str,
    publish_time: str,
    url: str,
    content: str,
) -> str:
    return _INFO_ADVISER_USER_TEMPLATE.format(
        section=section,
        title=title,
        publish_time=publish_time,
        url=url,
        content=content,
    )
