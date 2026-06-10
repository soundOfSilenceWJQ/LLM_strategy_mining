# Prompt Templates

本目录统一管理 llm_quant 的提示词模板，便于集中维护和快速迭代。

## 目录说明

- `info_adviser_prompts.py`：单条新闻/研报解读模板
- `info_summarizer_prompts.py`：日度汇总模板
- `info_parsing_prompts.py`：解析加工层（实时文本、因子文献）模板
- `factor_generation_prompts.py`：因子复现与错误修复模板
- `factor_selection_prompts.py`：LLM 因子选择模板
- `market_regime_prompts.py`：市场环境评估模板

## 使用规范

- 模板字符串与代码逻辑分离，业务代码只调用 `render_*` 函数。
- 所有可变字段通过函数参数注入，不在业务代码中手工拼接提示词。
- 调整提示词时尽量保持占位符名称不变，避免影响调用方。
