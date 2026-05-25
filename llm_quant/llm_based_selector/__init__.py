"""
LLM 因子选择框架

基于 LLM 的智能因子选择系统，可以根据大盘信息和因子特征
推荐最适合当前市场环境的因子组合。

使用示例:
    from llm_quant.llm_based_selector import FactorSelectorFramework
    
    framework = FactorSelectorFramework()
    result = framework.run_pipeline()
    FactorSelectorFramework.print_recommendation(result)
"""

from llm_quant.llm_based_selector.agent import FactorSelectorAgent
from llm_quant.llm_based_selector.selector import FactorSelectorFramework
from llm_quant.llm_based_selector.utils import (
    create_sample_market_info,
    create_sample_factors_list,
    create_sample_factor_performance,
    setup_sample_data,
)

__all__ = [
    "FactorSelectorAgent",
    "FactorSelectorFramework",
    "create_sample_market_info",
    "create_sample_factors_list",
    "create_sample_factor_performance",
    "setup_sample_data",
]
