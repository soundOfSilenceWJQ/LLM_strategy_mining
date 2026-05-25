"""
Strategy Backtest Script
将月度因子选择与因子框架结合，进行策略回测分析
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================================
# 配置和常量
# ============================================================================

MONTHLY_SELECTION_FILE = Path("D:/wjq/learning/grad_project/data/factors/output/monthly_factor_sampling_2023_2025_3_3.csv")
FACTOR_FRAME_FILE = Path("D:/wjq/learning/grad_project/data/factors/factors/output/multi_factor_2023_2025_4/multi_factor_2023_2025_4_frame.csv")
RESULT_DIR = Path("D:/wjq/learning/grad_project/data/factors/output/strategy_backtest_2023_2025_4")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_MONTH_LIMIT = None  # 如果需要仅处理前N个月，设置此值，否则为None表示处理所有月份

# ============================================================================
# 数据加载函数
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载必要的数据文件"""
    monthly_selection = pd.read_csv(MONTHLY_SELECTION_FILE)
    factor_frame = pd.read_csv(FACTOR_FRAME_FILE, parse_dates=["datetime"])
    return monthly_selection, factor_frame


# ============================================================================
# 工具函数
# ============================================================================

def parse_selected_factors(raw_value: object) -> list[str]:
    """
    解析选中的因子列表
    支持多种格式：列表、元组、集合、ndarray、字符串（分号分隔）
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, np.ndarray):
        return [str(item).strip() for item in raw_value.tolist() if str(item).strip()]
    try:
        if pd.isna(raw_value):  # type: ignore
            return []
    except (TypeError, ValueError):
        pass
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def total_return_from_daily(daily_return: pd.Series) -> float:
    """从日收益计算总收益"""
    daily_return = daily_return.dropna()
    if daily_return.empty:
        return 0.0
    return float((1.0 + daily_return).prod()) - 1.0  # type: ignore


def annualized_return_from_daily(daily_return: pd.Series) -> float:
    """从日收益计算年化收益"""
    daily_return = daily_return.dropna()
    if daily_return.empty:
        return 0.0
    equity = float((1.0 + daily_return).prod())  # type: ignore
    return (equity ** (252.0 / len(daily_return))) - 1.0


def annualized_volatility_from_daily(daily_return: pd.Series) -> float:
    """从日收益计算年化波动率"""
    daily_return = daily_return.dropna()
    if len(daily_return) <= 1:
        return 0.0
    return float(daily_return.std(ddof=1) * np.sqrt(252.0))  # type: ignore


def sharpe_from_daily(daily_return: pd.Series) -> float:
    """从日收益计算夏普比率（无风险率=0）"""
    annual_return = annualized_return_from_daily(daily_return)
    annual_volatility = annualized_volatility_from_daily(daily_return)
    return float(annual_return / annual_volatility) if annual_volatility > 0 else 0.0


def max_drawdown_from_daily(daily_return: pd.Series) -> float:
    """从日收益计算最大回撤"""
    daily_return = daily_return.dropna()
    if daily_return.empty:
        return 0.0
    equity_curve = (1.0 + daily_return).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    return float(drawdown.min())  # type: ignore


# ============================================================================
# 策略回测函数
# ============================================================================

def build_monthly_rotation_strategy(
    factor_frame: pd.DataFrame,
    monthly_selection: pd.DataFrame,
    selected_months: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建月度轮动策略（仅计算策略相关指标）
    
    Args:
        factor_frame: 因子日频收益数据框
        monthly_selection: 月度选中因子列表
        selected_months: 限制处理的月份列表（可选）
    
    Returns:
        tuple: (strategy_daily, all_factor_strategy)
            - strategy_daily: 策略日收益数据框
            - all_factor_strategy: 全因子等权策略基准
    """
    factor_frame = factor_frame.copy()
    factor_frame["datetime"] = pd.to_datetime(factor_frame["datetime"])
    factor_frame["month"] = factor_frame["datetime"].dt.to_period("M").astype(str)

    monthly_selection = monthly_selection.copy()
    monthly_selection["month"] = monthly_selection["month"].astype(str)
    monthly_selection["selected_factors"] = monthly_selection["selected_factors"].apply(parse_selected_factors)

    if selected_months is not None:
        monthly_selection = monthly_selection[monthly_selection["month"].isin(selected_months)].copy()

    strategy_daily_parts = []

    # 构建全因子等权基准策略
    available_factors = set(factor_frame["factor"].dropna().unique().tolist())
    all_factor_daily = factor_frame.groupby("datetime")["net_return"].mean().sort_index()
    all_factor_strategy = pd.DataFrame(
        {
            "datetime": all_factor_daily.index,
            "strategy_name": "all_factors_equal_weight",
            "daily_return": all_factor_daily.values,
        }
    )

    # 逐月构建轮动策略
    for _, row in monthly_selection.iterrows():
        month = row["month"]
        selected_factors = [factor for factor in row["selected_factors"] if factor in available_factors]
        if not selected_factors:
            continue

        month_slice = factor_frame[(factor_frame["month"] == month) & (factor_frame["factor"].isin(selected_factors))]
        if month_slice.empty:
            continue

        # 计算月度策略日收益
        monthly_daily = month_slice.groupby("datetime")["net_return"].mean().sort_index()
        strategy_daily_parts.append(
            pd.DataFrame(
                {
                    "datetime": monthly_daily.index,
                    "strategy_name": "monthly_rotation",
                    "daily_return": monthly_daily.values,
                    "month": month,
                    "n_factors": len(selected_factors),
                }
            )
        )

    if not strategy_daily_parts:
        raise RuntimeError("No monthly rotation strategy data was generated.")

    strategy_daily = pd.concat(strategy_daily_parts, ignore_index=True).sort_values("datetime")
    strategy_daily = strategy_daily.drop_duplicates(subset=["datetime", "strategy_name"], keep="first")
    strategy_daily = strategy_daily.reset_index(drop=True)

    return strategy_daily, all_factor_strategy


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序：加载数据、构建策略、生成结果"""
    
    print("=" * 80)
    print("策略回测程序启动")
    print("=" * 80)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    monthly_selection_df, factor_frame_df = load_data()
    
    print(f"  monthly_selection_df 形状: {monthly_selection_df.shape}")
    print(f"  factor_frame_df 形状: {factor_frame_df.shape}")
    
    # 准备月度选择数据
    print("\n[2/3] 准备月度选择数据...")
    monthly_selection_df = monthly_selection_df.copy()
    monthly_selection_df["month"] = monthly_selection_df["month"].astype(str)
    monthly_selection_df["selected_factors"] = monthly_selection_df["selected_factors"].apply(parse_selected_factors)
    monthly_selection_df = monthly_selection_df.sort_values("month").reset_index(drop=True)
    
    if SMOKE_MONTH_LIMIT is not None:
        selected_months = monthly_selection_df["month"].head(SMOKE_MONTH_LIMIT).tolist()
        monthly_selection_view = monthly_selection_df[monthly_selection_df["month"].isin(selected_months)].copy()
    else:
        selected_months = monthly_selection_df["month"].tolist()
        monthly_selection_view = monthly_selection_df.copy()
    
    # 构建月度轮动策略
    print("\n[3/3] 构建月度轮动策略...")
    strategy_daily_df, all_factor_strategy_df = build_monthly_rotation_strategy(
        factor_frame_df,
        monthly_selection_view,
        selected_months=selected_months,
    )
    
    # 计算策略指标
    rotation_daily_return = strategy_daily_df["daily_return"].astype(float)
    all_factor_daily_return = all_factor_strategy_df["daily_return"].astype(float)
    
    selected_factor_names = sorted({factor for factors in monthly_selection_view["selected_factors"] for factor in factors if factor})
    
    # 构建策略总结数据框
    rotation_strategy_summary = pd.DataFrame(
        [
            {
                "strategy_name": "monthly_rotation",
                "total_return": total_return_from_daily(rotation_daily_return),
                "annual_return": annualized_return_from_daily(rotation_daily_return),
                "annual_volatility": annualized_volatility_from_daily(rotation_daily_return),
                "sharpe": sharpe_from_daily(rotation_daily_return),
                "max_drawdown": max_drawdown_from_daily(rotation_daily_return),
                "n_days": int(rotation_daily_return.dropna().shape[0]),
                "n_months": len(selected_months),
                "n_unique_factors": len(selected_factor_names),
            },
            {
                "strategy_name": "all_factors_equal_weight",
                "total_return": total_return_from_daily(all_factor_daily_return),
                "annual_return": annualized_return_from_daily(all_factor_daily_return),
                "annual_volatility": annualized_volatility_from_daily(all_factor_daily_return),
                "sharpe": sharpe_from_daily(all_factor_daily_return),
                "max_drawdown": max_drawdown_from_daily(all_factor_daily_return),
                "n_days": int(all_factor_daily_return.dropna().shape[0]),
                "n_months": len(selected_months),
                "n_unique_factors": 0,
            },
        ]
    )
    
    # 输出结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    
    print(f"\n选中月份数: {len(selected_months)}")
    print(f"选中唯一因子数: {len(selected_factor_names)}")
    
    if len(selected_months) > 0:
        print(f"月份范围: {selected_months[0]} 至 {selected_months[-1]}")
    
    print("\n策略总结:")
    print(rotation_strategy_summary.to_string(index=False))
    
    # 保存结果文件
    print("\n" + "=" * 80)
    print("保存结果文件...")
    print("=" * 80)
    
    rotation_strategy_summary.to_csv(RESULT_DIR / "rotation_strategy_summary.csv", index=False, encoding="utf-8-sig")
    print(f"✓ {RESULT_DIR / 'rotation_strategy_summary.csv'}")
    
    strategy_daily_df.to_csv(RESULT_DIR / "monthly_rotation_strategy_daily.csv", index=False, encoding="utf-8-sig")
    print(f"✓ {RESULT_DIR / 'monthly_rotation_strategy_daily.csv'}")
    
    all_factor_strategy_df.to_csv(RESULT_DIR / "all_factor_equal_weight_strategy_daily.csv", index=False, encoding="utf-8-sig")
    print(f"✓ {RESULT_DIR / 'all_factor_equal_weight_strategy_daily.csv'}")
    
    # 构建结果摘要
    smoke_test_result = {
        "selected_months": selected_months,
        "selected_factor_count": len(selected_factor_names),
        "rotation_strategy_summary": rotation_strategy_summary.to_dict(orient="records"),
        "rotation_daily_rows": int(strategy_daily_df.shape[0]),
        "all_factor_daily_rows": int(all_factor_strategy_df.shape[0]),
    }
    
    result_json_path = RESULT_DIR / "backtest_result_summary.json"
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(smoke_test_result, f, ensure_ascii=False, indent=2)
    print(f"✓ {result_json_path}")
    
    print("\n" + "=" * 80)
    print("程序完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
