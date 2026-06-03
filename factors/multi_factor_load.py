"""
Multi-factor loading module.

This module is responsible for factor expression loading, qlib initialization,
and cache persistence. Backtest logic is intentionally kept out of this file.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
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
    """Clip requested date range to qlib calendar range."""
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
    """Load available instruments from qlib features directory or fallback file."""
    fallback_file = Path(__file__).parent / "data" / "available_stocks_in_qlib.txt"
    if fallback_file.exists():
        try:
            with open(fallback_file, "r") as f:
                stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(stocks)} instruments from cache file: {fallback_file}")
            return stocks
        except Exception as e:
            logger.warning(f"Failed to read cache file {fallback_file}: {e}")

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
    provider_uri: str,
    cache_dir: Path | None = None,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    """Load multiple factor expressions using qlib D.features."""
    from qlib.data import D

    instruments_obj = D.instruments(instruments)
    logger.info(f"Instruments resolved from qlib: {instruments}")
    logger.info(f"Loading {len(factors)} factors")

    result: Dict[str, pd.DataFrame] = {}
    errors: List[Dict[str, str]] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for name, expr in tqdm(factors, desc="Load factors", unit="factor"):
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

            if cache_dir is not None:
                cache_file = cache_dir / f"{sanitize_factor_name(name)}.pkl"
                factor_df.to_pickle(cache_file)
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
    """Infer cached data coverage from loaded factor frames."""
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


def prepare_factor_cache(config: MultiFactorBacktestConfig, cache_dir: Path) -> Dict[str, Any]:
    """Compute factors with qlib and persist them under cache_dir."""
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
        error_file = cache_dir / "cache_errors.csv"
        if factor_errors:
            cache_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(factor_errors).to_csv(error_file, index=False)
        raise RuntimeError(
            f"No factor loaded successfully. Errors saved to {error_file if factor_errors else 'N/A'}"
        )

    if factor_errors:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(factor_errors).to_csv(cache_dir / "cache_errors.csv", index=False)

    factor_file_map: Dict[str, str] = {
        factor_name: f"{sanitize_factor_name(factor_name)}.pkl"
        for factor_name in non_empty_factor_frames
    }

    return {
        "effective_start_date": effective_start_date,
        "effective_end_date": effective_end_date,
        "factor_files": factor_file_map,
        "errors": factor_errors,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ====== 写死参数区：仅负责因子加载 ======
    provider_uri = "D:/qlib_data/cn_data"
    region = "cn"
    instruments = "csi300"
    input_file = "D:/wjq/working/citic/codes_wjq(1)/factors/input/one_factor_test.csv"
    start_date = "2023-09-01"
    end_date = "2025-12-31"
    top_quantile = 0.2
    transaction_cost = 0.0015
    output_dir = "D:/wjq/working/citic/codes_wjq(1)/factors/output/one_factor_backtest_py312_rerun"
    factor_cache_dir = "D:/wjq/working/citic/codes_wjq(1)/factors/factor_cache_one_factor_py312_rerun"

    try:
        factor_csv = Path(input_file) if input_file else Path("factors/input/factors.csv")
        factors = load_factors_from_csv(factor_csv)
        logger.info(f"Loaded {len(factors)} factors from {factor_csv}")

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
        result = prepare_factor_cache(config, cache_dir)
        logger.info(f"Factor cache built successfully: {cache_dir}")
        logger.info(result)
        return 0
    except Exception as e:
        logger.error(f"Factor loading failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
