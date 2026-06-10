FACTOR_GENERATOR_SYSTEM_PROMPT = (
    "你是专业量化金融因子设计专家，需要基于结构化数据、学术文献及行业研报，"
    "完成存量因子复现、标准化代码转化与元数据生成。"
    "仅复刻已有成熟因子，输出具备可解释性、金融逻辑严谨、适配回测平台的有效因子，"
    "支撑因子库搭建和维护工作。"
)

DEFAULT_PLATFORM_SPEC = """- 输入: df (pandas.DataFrame)，列包含 CLOSE/OPEN/HIGH/LOW/VOLUME/VWAP（大小写兼容）
- 输出: pd.Series（index 为股票代码，值为最新时点截面因子）
- 只允许 pandas/numpy
- 禁止未来数据: 不允许 shift(-n)
- 推荐模板:
def compute_factor(df):
    import pandas as pd
    import numpy as np
    close = df['close'] if 'close' in df.columns else df['CLOSE']
    factor = close.iloc[-1] / close.iloc[-20] - 1
    return factor
"""

_FACTOR_GENERATION_TEMPLATE = """你需要将一个来自于文献的因子进行代码化复现，使得你的代码能够在回测平台运行。内容规整、术语标准，要求无重复错配、无主观冗余内容。

【因子文本信息】
因子计算方式在文献中的描述：{factor_description}
因子经济学逻辑在文献中的描述：{economic_logic}
其他因子相关信息备注：{other_info}

【回测平台代码规范】
{platform_spec}

【因子衍生数据生成规则】
同步生成配套元数据并与因子绑定入库，包含：经济学归因、适用市场环境、文献来源、因子类型、适配周期，保障因子可解释、可追溯。

【因子生成约束】
实现“因子值越高，资产预期收益越高”，反向因子通过取负号完成转化并留存备注。
输出精炼：logic/source_insight/risk_warnings 各不超过80字；metadata.applicable_market_environment 最多2项；data_requirements 最多5项。

【可用算子】
{operators}

【可参考上下文】
{context}

请严格按以下JSON格式输出：
{{
  "name": "因子名称（英文下划线命名）",
  "category": "因子类别，如 momentum/value/quality/growth/liquidity/volatility/reversal/fundamental/technical",
  "expression": "因子表达式（与文献定义一致）",
  "logic": "经济学逻辑与收益来源",
  "source_insight": "文献来源与锚点说明",
  "risk_warnings": "潜在风险与失效场景",
  "code": "def compute_factor(df):\\n    import pandas as pd\\n    import numpy as np\\n    ...\\n    return factor",
  "metadata": {{
    "economic_attribution": "经济学归因",
    "applicable_market_environment": ["适用市场环境1", "环境2"],
    "literature_source": "文献来源",
    "factor_type": "因子类型",
    "rebalance_cycle": "日频/周频/月频",
    "data_requirements": ["所需字段1", "字段2"]
  }},
  "directional_constraint": {{
    "higher_is_better": true,
    "original_direction": "positive|negative|unknown",
    "transformation": "none|negate",
    "note": "若为反向因子，说明已取负处理"
  }}
}}
"""

_FACTOR_REPAIR_TEMPLATE = """你需要对你生成的因子进行修正和错误处理，具体情况如下：

【报错信息】
之前生成的因子遇到了如下问题或报错：{error_info}；经分析，我们给出了如下的修正建议：{correct_advice}。

【当前因子JSON】
{current_factor_json}

【历史对话】
{history_context}

【迭代修复规则】
进行增量精准修复，无需全盘重写，针对性调整公式逻辑、参数配置、数据调用方式或优化底层投资逻辑。

【修复标准】
针对代码类错误，精准修正程序漏洞、规避未来数据、对齐原始因子逻辑；针对性能不达标问题，结合回测验证报告，复盘经济学逻辑与参数合理性，完成参数优化与逻辑迭代。

【输出要求】
1. 输出仍为完整JSON，结构与当前因子JSON保持一致。
2. 只做必要改动，不新增无关字段。
3. 保持“因子值越高，资产预期收益越高”；若需反向，明确 transformation=negate 并在 note 说明。
4. 仅改动和错误直接相关字段，避免扩写说明。
"""


def render_factor_generation_prompt(
    factor_description: str,
    economic_logic: str,
    other_info: str,
    platform_spec: str,
    operators: str,
    context: str,
) -> str:
    return _FACTOR_GENERATION_TEMPLATE.format(
        factor_description=factor_description,
        economic_logic=economic_logic,
        other_info=other_info,
        platform_spec=platform_spec,
        operators=operators,
        context=context,
    )


def render_factor_repair_prompt(
    error_info: str,
    correct_advice: str,
    current_factor_json: str,
    history_context: str,
) -> str:
    return _FACTOR_REPAIR_TEMPLATE.format(
        error_info=error_info,
        correct_advice=correct_advice,
        current_factor_json=current_factor_json,
        history_context=history_context,
    )
