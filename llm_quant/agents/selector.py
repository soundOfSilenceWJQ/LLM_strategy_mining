"""
第5层：策略筛选层（Strategy Selection Layer）
功能：
  - 基于多维度评分从PAR构建IAR（投资级因子库）
  - 因子全生命周期动态管理（轮动、淘汰、优化）
  - 基于IAR构建多因子组合策略
  - 生成投资组合权重
"""

from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.data.factor_store import FactorStore
from llm_quant.data.info_store import InfoStore
from llm_quant.prompts import render_market_assess_prompt
from llm_quant.utils.llm_client import LLMClient
from llm_quant.config import IAR_MAX_FACTORS, PAR_TOP_RATIO


class Selector:
    """
    第5层：策略筛选层。
    实现IAR动态轮动、投资组合构建、因子生命周期管理。
    """

    def __init__(
        self,
        factor_store: FactorStore | None = None,
        info_store: InfoStore | None = None,
        llm: LLMClient | None = None,
    ):
        self.factor_store = factor_store or FactorStore()
        self.info_store   = info_store or InfoStore()
        self.llm          = llm         # 可选，用于LLM辅助市场环境评估

    # ── IAR 轮动 ──────────────────────────────────────────
    def rotate_iar(self, use_llm_assessment: bool = False) -> dict:
        """
        执行一次IAR轮动。
        1. （可选）用LLM评估市场环境，调整评分权重
        2. 对PAR全量因子重新评分
        3. 选Top 30%因子入IAR，保持类别均衡
        返回轮动摘要。
        """
        print("\n[Layer5] 开始IAR因子轮动...")

        market_regime = "unknown"
        if use_llm_assessment and self.llm:
            market_regime = self._assess_market()
            print(f"  当前市场环境评估: {market_regime}")

        # 按综合评分排序并轮动
        self.factor_store.rotate_iar()

        iar = self.factor_store.get_iar()
        par_stats = self.factor_store.par_stats()
        iar_stats  = self.factor_store.iar_stats()

        summary = {
            "timestamp":     datetime.now().isoformat(),
            "market_regime": market_regime,
            "par_stats":     par_stats,
            "iar_stats":     iar_stats,
            "iar_factor_names": [f.get("name","?") for f in iar],
        }
        print(f"  PAR统计: {par_stats}")
        print(f"  IAR统计: {iar_stats}")
        return summary

    # ── 生命周期管理：因子淘汰 ───────────────────────────
    def retire_weak_factors(
        self,
        ic_decay_threshold: float = 0.01,
        icir_decay_threshold: float = 0.2,
    ):
        """
        将绩效持续下滑的因子从PAR中标记为失效。
        判断标准：IC均值低于阈值 且 ICIR低于阈值。
        """
        par = self.factor_store.get_par()
        retired = []
        for f in par:
            if (
                abs(f.get("ic_mean", 0)) < ic_decay_threshold
                and abs(f.get("icir", 0)) < icir_decay_threshold
                and f.get("validated", False)  # 只淘汰经过验证的因子
            ):
                self.factor_store.deactivate_par(f["id"])
                retired.append(f.get("name","?"))

        if retired:
            print(f"[Layer5] 淘汰 {len(retired)} 个失效因子: {retired}")
        else:
            print("[Layer5] 无因子被淘汰。")
        return retired

    # ── 多因子组合构建 ────────────────────────────────────
    def build_portfolio(
        self,
        panel: pd.DataFrame | None = None,
        top_n_stocks: int = 30,
        method: str = "equal_weight",   # "equal_weight" | "ic_weight"
    ) -> dict:
        """
        基于IAR因子库构建投资组合。
        简化版：等权或IC加权合成Alpha信号，选Top N股票。
        返回：{symbol: weight}
        """
        iar = self.factor_store.get_iar()
        if not iar:
            print("[Layer5] IAR为空，无法构建组合。")
            return {}

        if panel is None or panel.empty:
            print("[Layer5] 无行情数据，跳过组合构建。")
            return {}

        # 合成 Alpha 信号
        alpha_signals = self._synthesize_alpha(iar, panel)
        if alpha_signals.empty:
            return {}

        # 排序选股
        alpha_signals = alpha_signals.dropna().sort_values(ascending=False)
        selected = alpha_signals.head(top_n_stocks)

        # 等权分配
        n = len(selected)
        weights = {sym: round(1.0 / n, 4) for sym in selected.index}

        print(f"[Layer5] 组合构建完成: {n}只股票，等权配置。")
        return weights

    # ── Alpha 信号合成 ────────────────────────────────────
    def _synthesize_alpha(
        self,
        factors: list[dict],
        panel: pd.DataFrame,
    ) -> pd.Series:
        """
        简化的Alpha合成：
        对每个因子按IC权重合成横截面得分。
        """
        close = panel["close"].unstack("symbol")
        if close.empty:
            return pd.Series(dtype=float)

        # 计算各因子的最新截面值（这里用内置的技术因子代理）
        factor_matrix = []
        ic_weights    = []

        for f in factors:
            code = f.get("code", "")
            ic   = abs(f.get("ic_mean", 0.01))
            try:
                exec_globals = {"pd": pd, "np": np, "__builtins__": {}}
                exec(code, exec_globals)
                fn = exec_globals.get("compute_factor")
                if fn:
                    vals = fn(close)
                    if isinstance(vals, pd.Series) and not vals.empty:
                        # 截面标准化
                        std = vals.std()
                        if std > 1e-8:
                            vals = (vals - vals.mean()) / std
                        factor_matrix.append(vals)
                        ic_weights.append(ic)
            except Exception:
                pass

        if not factor_matrix:
            return pd.Series(dtype=float)

        # IC 加权合成
        total_ic = sum(ic_weights)
        alpha = pd.Series(0.0, index=factor_matrix[0].index)
        for vals, w in zip(factor_matrix, ic_weights):
            aligned = vals.reindex(alpha.index).fillna(0)
            alpha   += aligned * (w / total_ic)

        return alpha

    # ── LLM 市场环境评估 ──────────────────────────────────
    def _assess_market(self) -> str:
        """调用LLM评估当前市场环境。"""
        recent = self.info_store.get_recent(n=5)
        if not recent:
            return "unknown"
        context = "\n".join([
            f"- {r.get('title','')}: {r.get('summary','')[:80]}"
            for r in recent
        ])
        prompt = render_market_assess_prompt(market_context=context)
        try:
            resp = self.llm.chat(user_message=prompt)
            data = json.loads(self._extract_json(resp))
            return data.get("market_regime", "unknown")
        except Exception:
            return "unknown"

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        start = text.find("{")
        end   = text.rfind("}")
        return text[start:end+1] if start != -1 else "{}"

    # ── 报告输出 ──────────────────────────────────────────
    def print_summary(self):
        """打印当前PAR/IAR状态摘要。"""
        par_stats = self.factor_store.par_stats()
        iar_stats  = self.factor_store.iar_stats()
        iar = self.factor_store.get_iar()

        print("\n" + "="*60)
        print("【策略筛选层状态摘要】")
        print(f"  PAR（原始因子库）: 共{par_stats['total']}个因子，{par_stats['validated']}个经验证")
        print(f"  类别分布: {par_stats['categories']}")
        print(f"  IAR（投资级因子库）: 共{iar_stats['total']}个因子")
        print(f"  IAR平均IC: {iar_stats['avg_ic']:.4f}  平均综合得分: {iar_stats['avg_score']:.1f}")
        if iar:
            print("  IAR因子列表:")
            for f in iar:
                print(f"    - [{f.get('category','?')}] {f.get('name','?')} | IC={f.get('ic_mean',0):.4f} 得分={f.get('score',0):.1f}")
        print("="*60)
