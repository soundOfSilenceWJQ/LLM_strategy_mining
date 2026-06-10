"""
Multi-factor backtest framework using qlib symbolic expressions.

This module supports loading and backtesting multiple factors simultaneously,
with merged output results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.config import MultiFactorBacktestConfig
from factors.multi_factor_load import (
    infer_effective_date_range_from_frames,
    load_factor_cache,
    load_factors_from_csv,
)

logger = logging.getLogger(__name__)


def run_top_quantile_backtest(
    factor_df: pd.DataFrame,
    top_quantile: float = 0.2,
    transaction_cost: float = 0.0015,
) -> Dict:
    """
    Run long-short backtest for a single factor.

    Long: top quantile stocks (e.g., top 20%)
    Short: bottom quantile stocks (e.g., bottom 20%)

    Args:
        factor_df: DataFrame with 'factor' and 'label' columns, indexed by (date, instrument)
        top_quantile: Quantile threshold (e.g., 0.2 for top/bottom 20%)
        transaction_cost: Transaction cost as fraction per side

    Returns:
        Dict containing daily results, IC series, and aggregated metrics.
    """
    # Ensure proper multi-index
    if not isinstance(factor_df.index, pd.MultiIndex):
        raise ValueError("factor_df must have (date, instrument) MultiIndex")

    index_names = list(factor_df.index.names)
    if "datetime" in index_names:
        datetime_level = index_names.index("datetime")
    else:
        # Fallback for unnamed index: prefer DatetimeIndex-like level.
        datetime_level = None
        for i in range(factor_df.index.nlevels):
            level_values = factor_df.index.get_level_values(i)
            if pd.api.types.is_datetime64_any_dtype(level_values):
                datetime_level = i
                break
        if datetime_level is None:
            datetime_level = 0

    instrument_level = 1 - datetime_level if factor_df.index.nlevels == 2 else -1

    records = []
    daily_ics = []
    daily_rank_ics = []
    prev_long_holdings = set()
    prev_short_holdings = set()

    for dt, group in factor_df.groupby(level=datetime_level, sort=True):
        try:
            # group is a DataFrame with index level 1 (instrument) and columns [factor, label]
            group_clean = group.dropna()
            if len(group_clean) < 2:
                continue

            # Compute IC (Pearson) and Rank IC (Spearman) for the day
            pr_result: Any = pearsonr(group_clean["factor"].values, group_clean["label"].values)
            ic_raw = pr_result.statistic if hasattr(pr_result, "statistic") else pr_result[0]
            ic_value = float(ic_raw)
            if not np.isnan(ic_value):
                daily_ics.append((dt, ic_value))

            sp_result: Any = spearmanr(group_clean["factor"].values, group_clean["label"].values)
            rank_ic_raw = sp_result.statistic if hasattr(sp_result, "statistic") else sp_result[0]
            rank_ic_value = float(rank_ic_raw)
            if not np.isnan(rank_ic_value):
                daily_rank_ics.append((dt, rank_ic_value))

            # Long: top quantile selection
            top_threshold = group_clean["factor"].quantile(1 - top_quantile)
            long_selected = group_clean[group_clean["factor"] >= top_threshold]
            current_long_holdings = set(long_selected.index.get_level_values(instrument_level))

            # Short: bottom quantile selection
            bottom_threshold = group_clean["factor"].quantile(top_quantile)
            short_selected = group_clean[group_clean["factor"] <= bottom_threshold]
            current_short_holdings = set(short_selected.index.get_level_values(instrument_level))

            # Turnover for long and short separately
            long_turnover = 0.0 if not prev_long_holdings else 1.0 - len(prev_long_holdings & current_long_holdings) / max(len(prev_long_holdings), 1)
            short_turnover = 0.0 if not prev_short_holdings else 1.0 - len(prev_short_holdings & current_short_holdings) / max(len(prev_short_holdings), 1)
            
            # Total turnover and costs (transaction cost on both sides)
            avg_turnover = (long_turnover + short_turnover) / 2.0
            cost = avg_turnover * transaction_cost

            # Daily return: long mean - short mean (long-short hedge)
            long_return = long_selected["label"].mean() if len(long_selected) > 0 else 0.0
            short_return = short_selected["label"].mean() if len(short_selected) > 0 else 0.0
            
            # Long-short return (long positive return, short negative return)
            daily_return = long_return - short_return
            net_return = daily_return - cost

            records.append({
                "datetime": dt,
                "num_stocks": len(group_clean),
                "num_long": len(long_selected),
                "num_short": len(short_selected),
                "long_return": long_return,
                "short_return": short_return,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "daily_return": daily_return,
                "transaction_cost": cost,
                "net_return": net_return,
            })

            prev_long_holdings = current_long_holdings
            prev_short_holdings = current_short_holdings
        except Exception as e:
            logger.debug(f"Backtest failed for {dt}: {e}")
            continue

    if not records:
        raise RuntimeError("No valid backtest results generated.")

    daily_df = pd.DataFrame(records).set_index("datetime").sort_index()
    daily_df["cumulative_pnl"] = (1.0 + daily_df["net_return"]).cumprod() - 1.0
    daily_df["drawdown"] = daily_df["cumulative_pnl"] - daily_df["cumulative_pnl"].cummax()

    return {
        "daily_results": daily_df,
        "daily_ics": daily_ics,
        "daily_rank_ics": daily_rank_ics,
    }


def summarize(
    factor_name: str,
    backtest_output: Dict,
) -> Dict:
    """
    Summarize backtest metrics for a single factor.

    Args:
        factor_name: Factor name
        backtest_output: Output from run_top_quantile_backtest

    Returns:
        Summary dict with annual metrics.
    """
    daily_results = backtest_output["daily_results"]
    daily_ics = backtest_output["daily_ics"]
    daily_rank_ics = backtest_output["daily_rank_ics"]

    days = len(daily_results)

    net_returns = daily_results["net_return"].values
    cumulative_pnl = daily_results["cumulative_pnl"]

    equity = float((1.0 + net_returns).prod()) if days > 0 else 1.0
    annual_return = equity ** (252.0 / days) - 1.0 if days > 0 else 0.0
    annual_volatility = net_returns.std(ddof=1) * np.sqrt(252) if len(net_returns) > 1 else 0.0
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    max_drawdown = float(daily_results["drawdown"].min()) if len(cumulative_pnl) > 0 else 0.0

    ic_values = [ic for _, ic in daily_ics]
    ic_mean = np.mean(ic_values) if ic_values else 0.0
    ic_std = np.std(ic_values, ddof=1) if len(ic_values) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    rank_ic_values = [rank_ic for _, rank_ic in daily_rank_ics]
    rank_ic_mean = np.mean(rank_ic_values) if rank_ic_values else 0.0
    rank_ic_std = np.std(rank_ic_values, ddof=1) if len(rank_ic_values) > 1 else 0.0
    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0

    return {
        "factor_name": factor_name,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": rank_ic_ir,
        "days": days,
    }


def run_backtest_from_factor_frames(
    config: MultiFactorBacktestConfig,
    factor_frames: Dict[str, pd.DataFrame],
    factor_errors: List[Dict[str, str]],
    effective_start_date: str,
    effective_end_date: str,
    output_prefix: str = "multi_factor",
) -> Dict:
    """Execute backtest once factor frames are already prepared."""
    config.validate()

    # Run backtest for each factor
    all_summaries = []
    all_daily_results = []
    all_ic_series = []
    all_backtest_curves = []

    total_factors = len(factor_frames)
    pbar = tqdm(
        factor_frames.items(),
        total=total_factors,
        desc="Backtest",
        unit="factor",
        file=sys.stderr,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    for idx, (factor_name, factor_df) in enumerate(pbar, start=1):
        if sys.stderr.isatty():
            pbar.set_postfix(factor=factor_name)
        else:
            logger.info(f"Backtest progress {idx}/{total_factors} | factor={factor_name}")
        try:
            backtest_output = run_top_quantile_backtest(
                factor_df,
                top_quantile=config.top_quantile,
                transaction_cost=config.transaction_cost,
            )

            summary = summarize(factor_name, backtest_output)
            all_summaries.append(summary)

            # Prepare output data
            daily_df = backtest_output["daily_results"].copy()
            daily_df["factor"] = factor_name
            all_daily_results.append(daily_df)

            ic_df = pd.DataFrame(backtest_output["daily_ics"], columns=["datetime", f"{factor_name}_ic"]).set_index("datetime")
            rank_ic_df = pd.DataFrame(backtest_output["daily_rank_ics"], columns=["datetime", f"{factor_name}_rank_ic"]).set_index("datetime")
            ic_df = ic_df.join(rank_ic_df, how="outer")
            all_ic_series.append(ic_df)

            curve_df = pd.DataFrame(
                {
                    "datetime": daily_df.index,
                    factor_name: daily_df["cumulative_pnl"].values,
                }
            )
            curve_df = curve_df.set_index("datetime")
            all_backtest_curves.append(curve_df)
        except Exception as e:
            logger.warning(f"Skip factor '{factor_name}' in backtest stage: {e}")
            expr = next((x[1] for x in config.factors if x[0] == factor_name), "")
            factor_errors.append(
                {
                    "factor": factor_name,
                    "stage": "backtest",
                    "error": str(e),
                    "expression": expr,
                }
            )
            continue

    # Merge outputs
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    summary_json = config.output_dir / f"{output_prefix}_summary.json"
    with open(summary_json, "w") as f:
        json.dump(
            {
                "config": {
                    "provider_uri": config.provider_uri,
                    "region": config.region,
                    "instruments": config.instruments,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "effective_start_date": effective_start_date,
                    "effective_end_date": effective_end_date,
                    "factors": [{"name": name, "expression": expr} for name, expr in config.factors],
                    "top_quantile": config.top_quantile,
                    "transaction_cost": config.transaction_cost,
                },
                "summary": all_summaries,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Saved summary: {summary_json}")

    # 1.1 Errors CSV (if any)
    if factor_errors:
        error_csv = config.output_dir / f"{output_prefix}_errors.csv"
        pd.DataFrame(factor_errors).to_csv(error_csv, index=False)
        logger.info(f"Saved factor errors: {error_csv}")
    else:
        error_csv = None

    # 2. Merged factor frame CSV
    if all_daily_results:
        factor_frame_df = pd.concat(all_daily_results, ignore_index=False)
        factor_frame_csv = config.output_dir / f"{output_prefix}_frame.csv"
        factor_frame_df.to_csv(factor_frame_csv)
        logger.info(f"Saved factor frame: {factor_frame_csv}")
    else:
        factor_frame_csv = None

    # 3. Merged IC series CSV
    if all_ic_series:
        ic_series_df = pd.concat(all_ic_series, axis=1)
        ic_series_csv = config.output_dir / f"{output_prefix}_ic_series.csv"
        ic_series_df.to_csv(ic_series_csv)
        logger.info(f"Saved IC series: {ic_series_csv}")
    else:
        ic_series_csv = None

    # 4. Merged backtest curves CSV
    if all_backtest_curves:
        curves_df = pd.concat(all_backtest_curves, axis=1)
        curves_csv = config.output_dir / f"{output_prefix}_backtest_curves.csv"
        curves_df.to_csv(curves_csv)
        logger.info(f"Saved backtest curves: {curves_csv}")
    else:
        curves_csv = None

    return {
        "config": config,
        "summary": all_summaries,
        "errors": factor_errors,
        "files": {
            "summary_json": str(summary_json),
            "errors_csv": str(error_csv) if error_csv else None,
            "factor_frame_csv": str(factor_frame_csv) if factor_frame_csv else None,
            "ic_series_csv": str(ic_series_csv) if ic_series_csv else None,
            "backtest_curves_csv": str(curves_csv) if curves_csv else None,
        },
    }


def run_multi_factor_backtest_from_cache(
    config: MultiFactorBacktestConfig,
    cache_dir: Path,
    output_prefix: str = "multi_factor",
) -> Dict:
    """Load cached factors from disk and run backtest without recomputing factors."""
    config.validate()
    factor_frames, cache_errors = load_factor_cache(config.factors, cache_dir)
    if not factor_frames:
        raise RuntimeError(f"No cached factor frame available under {cache_dir}")

    effective_start_date, effective_end_date = infer_effective_date_range_from_frames(factor_frames)
    return run_backtest_from_factor_frames(
        config,
        factor_frames,
        cache_errors,
        effective_start_date,
        effective_end_date,
        output_prefix=output_prefix,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ====== 写死参数区 ======
    provider_uri = "D:/qlib_data/cn_data"
    region = "cn"
    instruments = "csi300"
    input_file = "D:/wjq/working/citic/codes_wjq(1)/factors/input/factors_financial_stmt_formulas.csv"
    start_date = "2023-09-01"
    end_date = "2025-12-31"
    top_quantile = 0.2
    transaction_cost = 0.0015
    output_dir = "D:/wjq/working/citic/codes_wjq(1)/factors/output/financial_stmt_backtest_py312"
    output_prefix = "financial_stmt_py312"
    factor_cache_dir = "D:/wjq/working/citic/codes_wjq(1)/factors/factor_cache_financial_stmt_py312"

    try:
        factor_csv = Path(input_file) if input_file else Path("factors/input/factors.csv")
        factors = load_factors_from_csv(factor_csv)
        logger.info(f"Loaded {len(factors)} factors from {factor_csv}: {[name for name, _ in factors]}")

        config = MultiFactorBacktestConfig(
            provider_uri=provider_uri,
            factors=factors,
            region=region,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            top_quantile=top_quantile,
            transaction_cost=transaction_cost,
            output_dir=Path(output_dir),
        )

        cache_dir = Path(factor_cache_dir)
        logger.info("Starting multi-factor backtest from preloaded cache...")
        result = run_multi_factor_backtest_from_cache(config, cache_dir, output_prefix=output_prefix)
        print(json.dumps(result, indent=2, default=str))
        logger.info("Backtest completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
