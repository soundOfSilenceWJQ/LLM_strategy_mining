INFO_PARSING_SYSTEM_PROMPT = (
    "你是一名资深的金融信息分析师。你的职责是精准理解并提炼金融新闻、研究报告中的核心信息，"
    "将复杂的观点转化为清晰的结构化描述。分析应准确、客观，避免过度解读，"
    "确保信息的真实性和可追溯性。"
)

_GENERAL_INFO_TEMPLATE = """请从以下文本中提取关键信息，并按照指定格式输出。

【原始文本】
{raw_text}

【提取要求】
1. 识别文本涉及的主题、行业、公司等。
2. 提取核心观点和数据。
3. 判断观点的情感倾向（乐观/中立/悲观）。
4. 判断该信息对资本市场可能造成的影响及传导链条。
5. 指出不确定性因素和假设条件。
6. 输出需包含新闻来源、发布日期、相关主题、相关行业、核心观点、情感倾向、关键数据指标和不确定因素等关键内容，输出为 JSON 格式。
7. 对文本进行精炼到不超过 {max_chars} 字的长度（对于长度超过 {max_chars} 字的文本）。
8. 输出精简：related_topics/related_industries/core_viewpoints/uncertainties/assumptions 各最多2项；key_metrics最多1项；evidence_quotes最多1条。

【元信息】
source: {source}
publish_date: {publish_date}
title: {title}
url: {url}

【JSON Schema】
{{
  "source": {{
    "name": "信息来源",
    "publish_date": "YYYY-MM-DD 或原文日期",
    "title": "标题",
    "url": "链接"
  }},
  "related_topics": ["主题1", "主题2"],
  "related_industries": ["行业1", "行业2"],
  "related_companies": ["公司1", "公司2"],
  "core_viewpoints": ["核心观点1", "核心观点2"],
  "sentiment": "乐观|中立|悲观",
  "market_impact_chain": "事件/信息 -> 传导环节 -> 可能受影响资产",
  "key_metrics": [
    {{"name": "指标名", "value": "数值", "unit": "单位", "context": "上下文说明"}}
  ],
  "uncertainties": ["不确定因素1", "不确定因素2"],
  "assumptions": ["假设条件1", "假设条件2"],
  "refined_summary": "精炼总结，不超过 {max_chars} 字",
  "traceability": {{
    "evidence_quotes": ["原文证据片段1", "原文证据片段2"]
  }}
}}
"""

_FACTOR_PAPER_TEMPLATE = """【附件】（文献对应的pdf文件）
【附件名称】{document_name}

【文献文本】
{raw_text}

【提取要求】
请对附件中的研究文献进行解读，深度解读附件中的研究文献，全面梳理、挖掘文献内全部可计算的量化因子，无遗漏、无主观删减，严格按照以下标准化规则完成信息提取与整理工作。逐条完整提取各因子核心信息，包含：因子名称及精准定义、详细计算方式、核心经济学逻辑与收益来源。

【残缺信息处理规则】
若文献中部分因子存在信息缺失，需如实标注“文献未提及”，禁止自行补充、主观推测内容。

【多因子处理规则】
若文献包含多个因子、因子变体、细分指标，需逐一拆分独立整理，不合并、不遗漏，清晰区分不同因子的差异定义与计算逻辑。

【输出质量规则】
梳理内容逻辑清晰、表述规整，专业术语与文献保持完全一致，剔除主观冗余表述，只保留文献客观研究内容，保证提取结果标准化、规范化，输出为 JSON 格式。
补充：若因子数量很多，仅保留可计算且定义清晰的前10项，其余在 extraction_notes 中概述。

【JSON Schema】
{{
  "document": {{
    "name": "文献名称",
    "source": "来源",
    "publish_date": "发布日期",
    "notes": "补充说明"
  }},
  "factors": [
    {{
      "factor_name": "因子名称",
      "factor_definition": "精准定义；若缺失写文献未提及",
      "calculation_method": "详细计算方式；若缺失写文献未提及",
      "economic_logic": "核心经济学逻辑；若缺失写文献未提及",
      "return_source": "收益来源；若缺失写文献未提及",
      "required_inputs": ["所需原始字段1", "字段2"],
      "constraints_or_notes": "适用范围、样本限制、参数设置等；若缺失写文献未提及",
      "missing_info": ["缺失项1", "缺失项2"]
    }}
  ],
  "extraction_notes": "只描述文献客观信息，不做主观延展"
}}
"""


def render_general_info_prompt(
    raw_text: str,
    max_chars: int,
    source: str,
    publish_date: str,
    title: str,
    url: str,
) -> str:
    return _GENERAL_INFO_TEMPLATE.format(
        raw_text=raw_text,
        max_chars=max_chars,
        source=source,
        publish_date=publish_date,
        title=title,
        url=url,
    )


def render_factor_paper_prompt(document_name: str, raw_text: str) -> str:
    return _FACTOR_PAPER_TEMPLATE.format(document_name=document_name, raw_text=raw_text)
