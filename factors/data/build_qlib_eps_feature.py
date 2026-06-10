from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HS300_CACHE_DIR = DATA_DIR / "hs300_stock_cache"
FEATURES_CONFIG_FILE = ROOT / "input" / "factors_financial_stmt_raw.csv"

# =========================
# 用户配置区（直接写死参数）
# =========================
QLIB_DIR = Path(r"D:\qlib_data\cn_data")
INDEX_CODE = "000300"
SMOKE_TEST = False
SMOKE_COUNT = 10
START_DATE = "2018-01-01"
END_DATE = ""
FEATURE_SPECS: dict[str, dict[str, str]] = {}


def to_qlib_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code[0] in ("6", "9"):
        return f"sh{code}"
    return f"sz{code}"


def load_symbols_from_cache(index_code: str = "000300") -> list[str]:
    cache_file = DATA_DIR / "index_cache" / f"{index_code}_constituents.csv"
    if cache_file.exists():
        cache_df = pd.read_csv(cache_file)
        if "成分券代码" in cache_df.columns:
            return (
                cache_df["成分券代码"]
                .astype(str)
                .str.strip()
                .str.zfill(6)
                .drop_duplicates()
                .tolist()
            )

    hs300_file = HS300_CACHE_DIR / "hs300_constituents.csv"
    if hs300_file.exists():
        hs300_df = pd.read_csv(hs300_file)
        if "成分券代码" in hs300_df.columns:
            return (
                hs300_df["成分券代码"]
                .astype(str)
                .str.strip()
                .str.zfill(6)
                .drop_duplicates()
                .tolist()
            )

    raise FileNotFoundError("No constituents cache found in index_cache or hs300_stock_cache.")


def load_calendar() -> pd.Series:
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    if not cal_file.exists():
        raise FileNotFoundError(f"Qlib calendar file not found: {cal_file}")
    cal = pd.to_datetime(pd.read_csv(cal_file, header=None).iloc[:, 0], errors="coerce").dropna()
    return cal.reset_index(drop=True)


def load_feature_specs_from_csv() -> dict[str, dict[str, str]]:
    if not FEATURES_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Feature config file not found: {FEATURES_CONFIG_FILE}")

    df = pd.read_csv(FEATURES_CONFIG_FILE)
    required_cols = {"feature_name", "source_file", "value_column", "alignment_column"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {FEATURES_CONFIG_FILE.name}: {sorted(miss)}")

    specs: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        feature_name = str(row["feature_name"]).strip()
        source_file = str(row["source_file"]).strip()
        value_column = str(row["value_column"]).strip()
        alignment_column = str(row["alignment_column"]).strip()
        if not feature_name or not source_file or not value_column or not alignment_column:
            continue
        specs[feature_name] = {
            "source_file": source_file,
            "value_column": value_column,
            "alignment_column": alignment_column,
        }

    if not specs:
        raise ValueError(f"No valid feature specs found in: {FEATURES_CONFIG_FILE}")
    return specs


def load_feature_events(code: str, feature_name: str, spec: dict[str, str]) -> pd.DataFrame:
    fp = DATA_DIR / spec["source_file"] / f"{code}.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["event_date", feature_name])

    df = pd.read_csv(fp)
    value_column = spec["value_column"]
    alignment_column = spec["alignment_column"]
    if value_column not in df.columns or alignment_column not in df.columns:
        return pd.DataFrame(columns=["event_date", feature_name])

    if "币种" in df.columns:
        df = df[df["币种"].astype(str).str.upper().eq("CNY")]

    if "类型" in df.columns:
        df = df[df["类型"].astype(str).str.contains("合并", na=False)]

    out = df[[alignment_column, value_column]].copy()
    out.columns = ["event_date", feature_name]
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out[feature_name] = pd.to_numeric(out[feature_name], errors="coerce")
    if "更新日期" in df.columns:
        out["update_date"] = pd.to_datetime(df["更新日期"], errors="coerce")
        out = out.sort_values(["event_date", "update_date"]).drop_duplicates(subset=["event_date"], keep="last")
        out = out.drop(columns=["update_date"])
    out = out.dropna(subset=["event_date", feature_name]).sort_values("event_date")
    return out


def build_daily_feature(events: pd.DataFrame, calendar: pd.Series, feature_name: str) -> pd.DataFrame:
    if events.empty or calendar.empty:
        return pd.DataFrame(columns=["date", feature_name])

    daily = pd.DataFrame({"date": calendar}).sort_values("date")
    daily = pd.merge_asof(
        daily,
        events,
        left_on="date",
        right_on="event_date",
        direction="backward",
    )
    start_ts = pd.to_datetime(START_DATE) if START_DATE else None
    end_ts = pd.to_datetime(END_DATE) if END_DATE else None
    if start_ts is not None:
        daily.loc[daily["date"] < start_ts, feature_name] = np.nan
    if end_ts is not None:
        daily.loc[daily["date"] > end_ts, feature_name] = np.nan

    daily = daily[["date", feature_name]]
    return daily


def write_qlib_bin(symbol: str, daily: pd.DataFrame, feature_name: str) -> bool:
    if daily.empty:
        return False

    valid_mask = daily[feature_name].notna()
    if not valid_mask.any():
        return False

    first_valid_idx = int(valid_mask.idxmax())
    values = daily.loc[first_valid_idx:, feature_name].astype("float32").to_numpy()

    # qlib day.bin: 第一个值是 calendar offset，后续为逐日特征值。
    arr = np.concatenate(
        [
            np.array([first_valid_idx], dtype="<f4"),
            values.astype("<f4", copy=False),
        ]
    )

    out_dir = QLIB_DIR / "features" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{feature_name}.day.bin"
    arr.tofile(out_file)
    return True


def main() -> None:
    feature_specs = load_feature_specs_from_csv()
    symbols = load_symbols_from_cache(index_code=INDEX_CODE)
    if SMOKE_TEST:
        symbols = symbols[: max(1, SMOKE_COUNT)]

    calendar = load_calendar()
    if calendar.empty:
        raise RuntimeError("Filtered qlib calendar is empty. Check START_DATE / END_DATE.")

    print(f"Symbols total: {len(symbols)}")
    print(f"Features total: {len(feature_specs)}")
    print(f"Feature config: {FEATURES_CONFIG_FILE}")
    for feature_name, spec in feature_specs.items():
        ok = 0
        miss = 0
        for code in symbols:
            symbol = to_qlib_symbol(code)
            events = load_feature_events(code, feature_name, spec)
            daily = build_daily_feature(events, calendar, feature_name)
            written = write_qlib_bin(symbol, daily, feature_name)
            if written:
                ok += 1
            else:
                miss += 1

        print(f"Feature name: {feature_name}")
        print(f"Written: {ok}")
        print(f"Skipped: {miss}")
    print(f"Output root: {QLIB_DIR / 'features'}")


if __name__ == "__main__":
    main()
