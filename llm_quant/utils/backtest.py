"""
简化回测引擎
计算因子的 IC、ICIR、IC月胜率、年化多空超额收益等指标
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats
from typing import Callable


# ── 因子 IC 计算 ──────────────────────────────────────────
def calc_ic(
    factor_values: pd.Series,          # MultiIndex (date, symbol)
    returns: pd.Series,                # MultiIndex (date, symbol)
    method: str = "rank",              # "pearson" or "rank"
) -> pd.Series:
    """
    计算每个截面日期的 IC（信息系数）。
    返回 Series，index 为交易日期。
    """
    df = pd.concat([factor_values.rename("factor"), returns.rename("ret")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    ic_series = {}
    for date, grp in df.groupby(level="date"):
        if len(grp) < 5:
            continue
        f = grp["factor"].values
        r = grp["ret"].values
        if method == "rank":
            ic, _ = stats.spearmanr(f, r)
        else:
            ic, _ = stats.pearsonr(f, r)
        ic_series[date] = ic

    return pd.Series(ic_series)


# ── 因子绩效摘要 ──────────────────────────────────────────
def factor_performance(ic_series: pd.Series) -> dict:
    """根据日频IC序列计算综合绩效指标。"""
    if ic_series.empty or ic_series.isna().all():
        return {"ic_mean": 0, "icir": 0, "ic_win_rate": 0, "valid_days": 0}

    ic = ic_series.dropna()
    ic_mean = ic.mean()
    ic_std  = ic.std()
    icir    = ic_mean / ic_std if ic_std > 1e-8 else 0.0
    win_rate = (ic > 0).mean()

    return {
        "ic_mean":    round(ic_mean, 4),
        "icir":       round(icir, 4),
        "ic_win_rate": round(win_rate, 4),
        "valid_days": int(len(ic)),
    }


# ── 多空组合超额收益 ──────────────────────────────────────
def long_short_return(
    factor_values: pd.Series,
    returns: pd.Series,
    n_groups: int = 5,
) -> dict:
    """
    构建多空组合：做多Top分组，做空Bottom分组。
    返回年化超额收益、夏普比率、最大回撤。
    """
    df = pd.concat([factor_values.rename("factor"), returns.rename("ret")], axis=1).dropna()
    if df.empty:
        return {}

    ls_daily = []
    for date, grp in df.groupby(level="date"):
        if len(grp) < n_groups * 2:
            continue
        grp = grp.sort_values("factor")
        n  = len(grp)
        k  = max(1, n // n_groups)
        bottom_ret = grp["ret"].iloc[:k].mean()
        top_ret    = grp["ret"].iloc[-k:].mean()
        ls_daily.append({"date": date, "ls_ret": top_ret - bottom_ret})

    if not ls_daily:
        return {}

    ls = pd.DataFrame(ls_daily).set_index("date")["ls_ret"]
    nav = (1 + ls).cumprod()

    total_days = len(ls)
    ann_ret = (nav.iloc[-1] ** (250 / total_days) - 1) if total_days > 0 else 0
    ann_vol = ls.std() * np.sqrt(250)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-8 else 0
    drawdown = (nav / nav.cummax() - 1).min()

    return {
        "annual_return":   round(ann_ret, 4),
        "annual_vol":      round(ann_vol, 4),
        "sharpe":          round(sharpe, 4),
        "max_drawdown":    round(drawdown, 4),
    }


# ── 综合评分 ──────────────────────────────────────────────
def compute_factor_score(perf: dict, ls: dict) -> float:
    """
    综合得分（0~100）：
      IC均值 × 40% + ICIR × 30% + IC月胜率 × 20% + 夏普比率 × 10%
    """
    ic_score    = min(perf.get("ic_mean", 0) / 0.05 * 40, 40)    # 0.05为满分锚点
    icir_score  = min(perf.get("icir", 0) / 1.0 * 30, 30)        # 1.0为满分锚点
    win_score   = min((perf.get("ic_win_rate", 0) - 0.5) / 0.2 * 20, 20)
    sharpe_score = min(ls.get("sharpe", 0) / 2.0 * 10, 10)
    return round(max(ic_score + icir_score + win_score + sharpe_score, 0), 2)
