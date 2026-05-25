"""
第3层：策略生成层（Strategy Generation Layer）
功能：
  - RAG：从信息库检索与当前市场环境相关的信息
  - 调用LLM生成标准化阿尔法因子（公式 + Python代码）
  - 保证因子的经济逻辑与代码可执行性
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.utils.llm_client import LLMClient
from llm_quant.data.info_store import InfoStore
from llm_quant.config import ALLOWED_OPERATORS, RAG_TOP_K

_SYSTEM_PROMPT = """你是一名顶级量化投资研究员，专注于A股市场Alpha因子挖掘。
你需要根据提供的市场信息和投资逻辑，生成具有严谨经济学含义的可执行Alpha因子。
因子代码必须符合规定的算子体系，不能引入未来数据。
输出必须是有效的JSON格式。"""

_FACTOR_GENERATION_PROMPT = """请根据以下市场信息，生成一个新的Alpha因子。

【参考信息（来自最新市场研究）】
{context}

【生成要求】
1. 因子必须基于以下标准数据列：CLOSE, OPEN, HIGH, LOW, VOLUME, VWAP
2. 可用算子：{operators}
3. 因子必须有清晰的经济逻辑，不得含有未来函数
4. 因子类别从以下选择：momentum, reversal, volatility, fundamental, liquidity, value, quality, growth, technical

【Python代码规范】
代码接受参数：df（DataFrame，含CLOSE/OPEN/HIGH/LOW/VOLUME列，日期为索引，股票代码为列名）
返回：pd.Series（因子值，index为股票代码）
使用pandas/numpy，不得import其他库。

请严格按以下JSON格式输出：
{{
  "name": "因子名称（英文，如: momentum_14d）",
  "category": "因子类别",
  "expression": "因子表达式（数学公式或简洁描述，如：14日价格动量 = CLOSE - DELAY(CLOSE, 14)）",
  "logic": "经济学逻辑说明（100-200字）",
  "source_insight": "来源于哪条信息/研究（简述）",
  "risk_warnings": "潜在风险（50字）",
  "code": "def compute_factor(df):\\n    import pandas as pd\\n    import numpy as np\\n    # 计算因子值\\n    close = df['close'] if 'close' in df.columns else df['CLOSE']\\n    factor = ...\\n    return factor.iloc[-1]  # 返回最新截面因子值"
}}"""


class StrategyGenerator:
    """
    第3层：策略生成层。
    使用RAG检索相关信息，驱动LLM生成Alpha因子。
    """

    def __init__(self, llm: LLMClient | None = None, info_store: InfoStore | None = None):
        self.llm = llm or LLMClient()
        self.info_store = info_store or InfoStore()

    def generate_factor(
        self,
        query: str = "A股市场当前有效的超额收益来源",
        n_context: int = RAG_TOP_K,
    ) -> dict | None:
        """
        生成单个Alpha因子。
        1. RAG检索相关信息
        2. 构建 context
        3. LLM生成因子
        """
        # Step 1: RAG 检索
        related_records = self.info_store.search(query, top_k=n_context)
        if not related_records:
            print("[Layer3] 信息库为空，使用通用知识生成因子...")
            context = "（暂无实时信息，基于通用金融市场知识）"
        else:
            context = "\n\n".join([
                f"[{i+1}] 标题：{r.get('title','')}\n"
                f"  投资逻辑：{r.get('logic','') or r.get('summary','')}\n"
                f"  关键驱动：{', '.join(r.get('key_drivers',[]))}\n"
                f"  时效：{r.get('time_horizon','')}"
                for i, r in enumerate(related_records)
            ])

        # Step 2: 构建 prompt
        prompt = _FACTOR_GENERATION_PROMPT.format(
            context=context[:3000],
            operators=", ".join(ALLOWED_OPERATORS),
        )

        # Step 3: LLM 生成
        try:
            response  = self.llm.chat(user_message=prompt, system_prompt=_SYSTEM_PROMPT)
            factor    = self._parse_json(response)
            factor["source_query"]   = query
            factor["context_records"] = [r.get("id","") for r in related_records]
            print(f"[Layer3] 生成因子：{factor.get('name','unknown')} ({factor.get('category','')})")
            return factor
        except Exception as e:
            print(f"[Layer3] 因子生成失败: {e}")
            return None

    def generate_batch(
        self,
        queries: list[str] | None = None,
        n_factors: int = 5,
    ) -> list[dict]:
        """批量生成多个因子。"""
        if queries is None:
            queries = [
                "A股市场价量动量效应",
                "A股市场均值回归与反转策略",
                "A股市场流动性溢价因子",
                "A股市场估值与盈利因子",
                "A股市场政策敏感性与行业轮动",
                "A股市场波动率与风险因子",
                "A股市场财务质量因子",
            ]

        factors = []
        for i in range(min(n_factors, len(queries))):
            print(f"\n[Layer3] 生成第 {i+1}/{n_factors} 个因子...")
            f = self.generate_factor(query=queries[i % len(queries)])
            if f:
                factors.append(f)
        return factors

    # ── 辅助：JSON 解析容错 ───────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)
