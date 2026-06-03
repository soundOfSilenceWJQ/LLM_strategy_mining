from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ====== 写死参数区 ======
INPUT_CSV = Path(
    r"D:\wjq\working\citic\codes_wjq(1)\factors\output\one_factor_backtest_py312_rerun\one_factor_py312_rerun_ic_series.csv"
)
OUTPUT_PNG = INPUT_CSV.with_name("one_factor_py312_rerun_ic_weekly_bar.png")
WEEK_FREQ = "W-FRI"  # 以周五为周末，便于和A股交易周对齐
AGG_METHOD = "mean"  # 可选: mean / sum / median
FIGSIZE = (16, 7)


def _aggregate_weekly(series: pd.Series, method: str) -> pd.Series:
    if method == "sum":
        return series.resample(WEEK_FREQ).sum()
    if method == "median":
        return series.resample(WEEK_FREQ).median()
    return series.resample(WEEK_FREQ).mean()


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "datetime" not in df.columns:
        raise ValueError("CSV must contain 'datetime' column")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")

    # 用于检查日历分布（周几），方便排查周频聚合是否异常
    df["weekday"] = df.index.day_name()

    ic_cols = [c for c in df.columns if c.endswith("_ic") and not c.endswith("_rank_ic")]
    rank_ic_cols = [c for c in df.columns if c.endswith("_rank_ic")]

    if not ic_cols or not rank_ic_cols:
        raise ValueError("Cannot find IC/Rank IC columns. Expected columns like xxx_ic and xxx_rank_ic")

    # 按前缀匹配同一因子对（例如 gap_ic 与 gap_rank_ic）
    factor_prefixes = [c[: -len("_ic")] for c in ic_cols]
    paired = []
    for prefix in factor_prefixes:
        rank_col = f"{prefix}_rank_ic"
        ic_col = f"{prefix}_ic"
        if rank_col in df.columns:
            paired.append((prefix, ic_col, rank_col))

    if not paired:
        raise ValueError("No matched IC/Rank IC pair found")

    n = len(paired)
    fig, axes = plt.subplots(n, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * n), squeeze=False)

    for i, (prefix, ic_col, rank_col) in enumerate(paired):
        ax = axes[i, 0]

        weekly_ic = _aggregate_weekly(df[ic_col], AGG_METHOD)
        weekly_rank_ic = _aggregate_weekly(df[rank_col], AGG_METHOD)

        weekly = pd.concat([weekly_ic, weekly_rank_ic], axis=1)
        weekly.columns = ["ic", "rank_ic"]
        weekly = weekly.dropna(how="all")

        x = np.arange(len(weekly))
        width = 0.42

        ax.bar(x - width / 2, weekly["ic"].fillna(0).values, width=width, label="IC", alpha=0.85)
        ax.bar(x + width / 2, weekly["rank_ic"].fillna(0).values, width=width, label="Rank IC", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_title(f"Weekly IC vs Rank IC - {prefix} ({AGG_METHOD}, {WEEK_FREQ})")
        ax.set_ylabel("Correlation")
        ax.legend(loc="upper right")

        labels = [dt.strftime("%Y-%m-%d") for dt in weekly.index]
        step = max(1, len(labels) // 16)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")

    axes[-1, 0].set_xlabel("Week End Date")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)

    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_PNG}")
    print("Weekday count in source daily data:")
    print(df["weekday"].value_counts().to_string())


if __name__ == "__main__":
    main()
