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
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.config import MultiFactorBacktestConfig

logger = logging.getLogger(__name__)


def sanitize_factor_name(name: str) -> str:
    """Convert factor name into a filesystem-safe file stem."""
    sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip()]
    file_stem = "".join(sanitized).strip("_")
    return file_stem or "factor"


def load_factors_from_csv(input_file: Path) -> List[Tuple[str, str]]:
    """Load factors from a csv file."""
    if not input_file.exists() or not input_file.is_file():
        raise ValueError(f"input file not found: {input_file}")

    df = pd.read_csv(input_file)
    if df.empty:
        raise ValueError(f"Empty factor file: {input_file}")

    factors: List[Tuple[str, str]] = []
    name_keys = {"name", "factor", "factor_name"}
    expr_keys = {"expression", "expr", "formula"}

    norm_cols = {c.strip().lower(): c for c in df.columns}
    name_col = next((norm_cols[k] for k in name_keys if k in norm_cols), None)
    expr_col = next((norm_cols[k] for k in expr_keys if k in norm_cols), None)

    if name_col is None or expr_col is None:
        raise ValueError(
            f"Invalid columns in {input_file.name}. Need name/factor/factor_name and expression/expr/formula."
        )

    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        expr = str(row[expr_col]).strip()
        if not name or not expr or name.lower() == "nan" or expr.lower() == "nan":
            continue
        factors.append((name, expr))

    if not factors:
        raise ValueError(f"No valid factors parsed from {input_file}")

    return factors


def init_qlib_env(provider_uri: str, region: str = "cn") -> None:
    """Initialize qlib environment with single-kernel threading backend for Windows compatibility."""
    import importlib
    import inspect

    qlib = importlib.import_module("qlib")

    provider_path = Path(provider_uri).expanduser()
    resolved_path = provider_path.resolve() if provider_path.exists() else provider_path
    calendar_file = provider_path / "calendars" / "day.txt"
    if not calendar_file.exists():
        raise FileNotFoundError(
            "Invalid qlib provider data path. Missing calendars/day.txt under "
            f"provider_uri={provider_uri} (resolved={resolved_path})."
        )

    qlib_init = getattr(qlib, "init", None) or getattr(qlib, "auto_init", None)
    if qlib_init is None:
        module_file = getattr(qlib, "__file__", "<unknown>")
        raise RuntimeError(
            "Imported module 'qlib' does not expose init/auto_init. "
            f"Resolved module: {module_file}. "
            "This usually means the wrong package is installed or imported. "
            "Please ensure Microsoft pyqlib is installed in the active interpreter: pip install pyqlib, "
            "and remove conflicting package(s): pip uninstall qlib."
        )

    # Keep compatibility with different pyqlib versions by passing only supported kwargs.
    requested_kwargs = {
        "provider_uri": provider_uri,
        "region": region,
        "expression_cache": None,
        "kernels": 1,
        "joblib_backend": "threading",
    }
    supported_kwargs = set(inspect.signature(qlib_init).parameters.keys())
    init_kwargs = {k: v for k, v in requested_kwargs.items() if k in supported_kwargs}
    qlib_init(**init_kwargs)
    logger.info(f"Qlib initialized: provider_uri={provider_uri}, resolved={resolved_path}, region={region}")


def resolve_effective_date_range(start_date: str, end_date: str) -> Tuple[str, str]:
    """Clip requested date range to qlib calendar range.

    Returns the effective [start_date, end_date] as YYYY-MM-DD strings.
    Raises if there is no overlap.
    """
    from qlib.data import D

    cal = D.calendar(freq="day")
    if len(cal) == 0:
        raise RuntimeError("Qlib calendar is empty. Please check provider data at provider_uri.")

    cal_start = pd.Timestamp(cal[0]).normalize()
    cal_end = pd.Timestamp(cal[-1]).normalize()
    req_start = pd.Timestamp(start_date).normalize()
    req_end = pd.Timestamp(end_date).normalize()

    eff_start = max(req_start, cal_start)
    eff_end = min(req_end, cal_end)

    if eff_start > eff_end:
        raise RuntimeError(
            "Requested date range has no overlap with qlib data range. "
            f"requested=[{req_start.date()}..{req_end.date()}], "
            f"qlib=[{cal_start.date()}..{cal_end.date()}]."
        )

    if eff_start != req_start or eff_end != req_end:
        logger.warning(
            "Date range adjusted to qlib coverage: "
            f"requested=[{req_start.date()}..{req_end.date()}], "
            f"effective=[{eff_start.date()}..{eff_end.date()}], "
            f"qlib=[{cal_start.date()}..{cal_end.date()}]"
        )

    return eff_start.strftime("%Y-%m-%d"), eff_end.strftime("%Y-%m-%d")


def load_available_instruments(provider_uri: str) -> List[str]:
    """Load available instruments from qlib features directory or fallback file.
    
    Args:
        provider_uri: Path to qlib data directory
        
    Returns:
        List of available stock codes (e.g., ['sz000001', 'sz000002', ...])
    """
    # Try to read from cached file first
    fallback_file = Path(__file__).parent / "data" / "available_stocks_in_qlib.txt"
    if fallback_file.exists():
        try:
            with open(fallback_file, 'r') as f:
                stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(stocks)} instruments from cache file: {fallback_file}")
            return stocks
        except Exception as e:
            logger.warning(f"Failed to read cache file {fallback_file}: {e}")
    
    # Fallback: read directly from features directory
    features_dir = Path(provider_uri) / "features"
    if features_dir.exists():
        try:
            stocks = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
            logger.info(f"Loaded {len(stocks)} instruments from features directory")
            return stocks
        except Exception as e:
            logger.warning(f"Failed to read features directory {features_dir}: {e}")
    
    raise RuntimeError(f"Cannot load available instruments from {provider_uri}")


def load_multi_factor_frame(
    factors: List[Tuple[str, str]],
    instruments: str,
    start_date: str,
    end_date: str,
    provider_uri: str = "D:/qlib_data/cn_data",
    cache_dir: Path | None = None,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    """
    Load multiple factor expressions using qlib D.features in a single call.

    Args:
        factors: List of (name, expression) tuples
        instruments: Instruments string (e.g., 'csi300') - used for logging only
        start_date: Start date string
        end_date: End date string
        provider_uri: Path to qlib data directory
        cache_dir: Optional cache directory. If provided, each loaded factor is persisted immediately.

    Returns:
        Dict mapping factor_name -> DataFrame with factor values and labels.
    """
    # Import D after qlib.init() has been called
    from qlib.data import D

    # Resolve instruments: read from features directory, not from D.instruments()
    instruments_obj = load_available_instruments(provider_uri)
    logger.info(f"Instruments resolved: {len(instruments_obj)} stocks (from {instruments})")

    logger.info(f"Loading {len(factors)} factors (robust mode)")

    result: Dict[str, pd.DataFrame] = {}
    errors: List[Dict[str, str]] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for name, expr in factors:
        try:
            raw = D.features(
                instruments_obj,
                [expr, "Ref($close,-1)/$close-1"],
                start_time=start_date,
                end_time=end_date,
                freq="day",
            )
            factor_df = raw.copy()
            factor_df.columns = ["factor", "label"]
            result[name] = factor_df
            logger.info(f"Factor '{name}' loaded: shape={factor_df.shape}, nulls={factor_df.isnull().sum().sum()}")

            if cache_dir is not None:
                cache_file = cache_dir / f"{sanitize_factor_name(name)}.pkl"
                factor_df.to_pickle(cache_file)
                logger.info(f"Factor '{name}' cached immediately -> {cache_file}")
        except Exception as e:
            logger.warning(f"Skip factor '{name}' in load stage: {e}")
            errors.append(
                {
                    "factor": name,
                    "stage": "load",
                    "error": str(e),
                    "expression": expr,
                }
            )

    return result, errors


def save_factor_cache(
    factor_frames: Dict[str, pd.DataFrame],
    factors: List[Tuple[str, str]],
    cache_dir: Path,
    effective_start_date: str,
    effective_end_date: str,
    errors: List[Dict[str, str]],
) -> Dict[str, str]:
    """Persist computed factor frames to cache files."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    factor_file_map: Dict[str, str] = {}

    for factor_name, factor_df in factor_frames.items():
        cache_file = cache_dir / f"{sanitize_factor_name(factor_name)}.pkl"
        factor_df.to_pickle(cache_file)
        factor_file_map[factor_name] = cache_file.name
        logger.info(f"Cached factor '{factor_name}' -> {cache_file}")

    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / "cache_errors.csv", index=False)

    return factor_file_map


def load_factor_cache(
    factors: List[Tuple[str, str]],
    cache_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    """Load cached factor frames from disk."""

    factor_frames: Dict[str, pd.DataFrame] = {}
    errors: List[Dict[str, str]] = []
    for factor_name, expr in factors:
        cache_name = f"{sanitize_factor_name(factor_name)}.pkl"
        cache_file = cache_dir / cache_name
        if not cache_file.exists():
            errors.append(
                {
                    "factor": factor_name,
                    "stage": "cache_load",
                    "error": f"Cache file not found: {cache_file}",
                    "expression": expr,
                }
            )
            continue

        factor_df = pd.read_pickle(cache_file)
        factor_frames[factor_name] = factor_df
        logger.info(f"Loaded cached factor '{factor_name}': shape={factor_df.shape}, file={cache_file}")

    return factor_frames, errors


def infer_effective_date_range_from_frames(factor_frames: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    """Infer cached data coverage from the loaded factor frames."""
    # 只取每个 DataFrame 的 min/max 日期，避免将全部行展开到内存。
    min_dates: List[pd.Timestamp] = []
    max_dates: List[pd.Timestamp] = []
    for factor_df in factor_frames.values():
        if factor_df.empty:
            continue
        index_dates = (
            factor_df.index.get_level_values("datetime")
            if "datetime" in factor_df.index.names
            else factor_df.index.get_level_values(1)
        )
        dt_index = pd.to_datetime(index_dates)
        min_dates.append(dt_index.min())
        max_dates.append(dt_index.max())

    if not min_dates:
        raise RuntimeError("Cannot infer effective date range from empty factor cache.")

    min_date = min(min_dates).normalize()
    max_date = max(max_dates).normalize()
    return min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")


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
    prev_long_holdings = set()
    prev_short_holdings = set()

    for dt, group in factor_df.groupby(level=datetime_level, sort=True):
        try:
            # group is a DataFrame with index level 1 (instrument) and columns [factor, label]
            group_clean = group.dropna()
            if len(group_clean) < 2:
                continue

            # Compute IC for the day
            sp_result: Any = spearmanr(group_clean["factor"].values, group_clean["label"].values)
            ic_raw = sp_result.statistic if hasattr(sp_result, "statistic") else sp_result[0]
            ic_value = float(ic_raw)
            if not np.isnan(ic_value):
                daily_ics.append((dt, ic_value))

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

    return {
        "factor_name": factor_name,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "days": days,
    }


def prepare_factor_cache(config: MultiFactorBacktestConfig, cache_dir: Path) -> Dict[str, Any]:
    """Compute factors with qlib and persist them under factors/factor_cache."""
    config.validate()
    init_qlib_env(config.provider_uri, config.region)

    effective_start_date, effective_end_date = resolve_effective_date_range(config.start_date, config.end_date)
    factor_frames, factor_errors = load_multi_factor_frame(
        config.factors,
        config.instruments,
        effective_start_date,
        effective_end_date,
        provider_uri=config.provider_uri,
        cache_dir=cache_dir,
    )

    non_empty_factor_frames: Dict[str, pd.DataFrame] = {}
    for factor_name, factor_df in factor_frames.items():
        if factor_df.empty:
            expr = next((x[1] for x in config.factors if x[0] == factor_name), "")
            factor_errors.append(
                {
                    "factor": factor_name,
                    "stage": "load",
                    "error": "Empty factor frame in effective date range",
                    "expression": expr,
                }
            )
        else:
            non_empty_factor_frames[factor_name] = factor_df

    if not non_empty_factor_frames:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        error_file = cache_dir / "cache_errors.csv"
        if factor_errors:
            cache_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(factor_errors).to_csv(error_file, index=False)
        raise RuntimeError(
            f"No factor loaded successfully. Errors saved to {error_file if factor_errors else 'N/A'}"
        )

    factor_file_map: Dict[str, str] = {
        factor_name: f"{sanitize_factor_name(factor_name)}.pkl"
        for factor_name in non_empty_factor_frames
    }

    if factor_errors:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(factor_errors).to_csv(cache_dir / "cache_errors.csv", index=False)

    return {
        "effective_start_date": effective_start_date,
        "effective_end_date": effective_end_date,
        "factor_files": factor_file_map,
        "errors": factor_errors,
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

            ic_df = pd.DataFrame(backtest_output["daily_ics"], columns=["datetime", factor_name])
            ic_df = ic_df.set_index("datetime")
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


def run_multi_factor_backtest(
    config: MultiFactorBacktestConfig,
    output_prefix: str = "multi_factor",
    cache_dir: Path | None = None,
) -> Dict:
    """Compute factors first, optionally cache them, then run the backtest."""
    target_cache_dir = cache_dir or Path("factors/factor_cache")
    prepare_factor_cache(config, target_cache_dir)
    return run_multi_factor_backtest_from_cache(config, target_cache_dir, output_prefix=output_prefix)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ====== 写死参数区 ======
    provider_uri = "D:/qlib_data/cn_data"
    region = "cn"
    instruments = "csi300"
    input_file = "D:/wjq/learning/grad_project/data/factors/input/factors_merged_positive_eco.csv"  # 或直接指定csv路径，如 "factors/input/factors.csv"
    start_date = "2023-09-01"
    end_date = "2025-12-31"
    top_quantile = 0.2
    transaction_cost = 0.0015
    output_dir = "factors/output/multi_factor_2023_2025_4"
    output_prefix = "multi_factor_2023_2025_4"
    factor_cache_dir = "factors/factor_cache_2023_2025_4"
    rebuild_factor_cache = True

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
        if rebuild_factor_cache:
            logger.info("Computing and caching factors before backtest...")
            prepare_factor_cache(config, cache_dir)

        logger.info("Starting multi-factor backtest from cached factors...")
        result = run_multi_factor_backtest_from_cache(config, cache_dir, output_prefix=output_prefix)
        print(json.dumps(result, indent=2, default=str))
        logger.info("Backtest completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
