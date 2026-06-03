from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HS300_CACHE_DIR = DATA_DIR / "hs300_stock_cache"


# =========================
# 用户配置区（直接写死参数）
# =========================
QLIB_DIR = Path(r"D:\qlib_data\cn_data")
INDEX_CODE = "000300"
SMOKE_TEST = True
SMOKE_COUNT = 3
START_DATE = "2022-01-01"
END_DATE = ""
OUT_FILE = DATA_DIR / "quarterly" / "metadata_daily_debug.csv"
DEBUG_ROWS = 12


# 因子配置
FACTORS_DICT: dict[str, list[str]] = {
	"fund_pe": ["close", "eps"],
}


# 元数据配置（量价字段不需要写在这里，默认从 qlib 获取）
META_DATA_DICT: dict[str, dict[str, str]] = {
	"eps": {
		"source_file": "stock_financial_analysis_indicator",
		"column_name": "加权每股收益(元)",
		"alignment_column": "日期",
	}
}


QLIB_DAILY_FIELDS = {
	"open",
	"high",
	"low",
	"close",
	"volume",
	"vwap",
	"factor",
	"amount",
}


def to_qlib_symbol(code: str) -> str:
	code = str(code).strip().zfill(6)
	if code[0] in ("6", "9"):
		return f"sh{code}"
	return f"sz{code}"


def load_qlib_daily_field(code: str, field_name: str, calendar: pd.Series) -> pd.DataFrame:
	feature_file = QLIB_DIR / "features" / to_qlib_symbol(code) / f"{field_name}.day.bin"
	if not feature_file.exists():
		return pd.DataFrame(columns=["instrument", "date", field_name])

	arr = np.fromfile(feature_file, dtype="<f4")
	if len(arr) <= 1:
		return pd.DataFrame(columns=["instrument", "date", field_name])

	offset = int(arr[0])
	values = arr[1:]
	end = offset + len(values)
	if offset < 0 or end > len(calendar):
		return pd.DataFrame(columns=["instrument", "date", field_name])

	dates = calendar.iloc[offset:end].reset_index(drop=True)
	out = pd.DataFrame({"instrument": code, "date": dates, field_name: values})
	out[field_name] = pd.to_numeric(out[field_name], errors="coerce")
	out = out.dropna(subset=["date", field_name]).sort_values("date")
	return out


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


def build_metadata_daily_panel(codes: list[str], factors_dict: dict[str, list[str]], meta_dict: dict[str, dict[str, str]]) -> pd.DataFrame:
	required_fields = sorted({field for fields in factors_dict.values() for field in fields})
	cal_file = QLIB_DIR / "calendars" / "day.txt"
	if not cal_file.exists():
		raise FileNotFoundError(f"Qlib calendar file not found: {cal_file}")
	calendar = pd.to_datetime(pd.read_csv(cal_file, header=None).iloc[:, 0], errors="coerce")

	all_rows: list[pd.DataFrame] = []
	start_ts = pd.to_datetime(START_DATE) if START_DATE else None
	end_ts = pd.to_datetime(END_DATE) if END_DATE else None

	for code in codes:
		base_daily_df = load_qlib_daily_field(code=code, field_name="close", calendar=calendar)
		if base_daily_df.empty:
			continue

		if start_ts is not None:
			base_daily_df = base_daily_df[base_daily_df["date"] >= start_ts]
		if end_ts is not None:
			base_daily_df = base_daily_df[base_daily_df["date"] <= end_ts]
		if base_daily_df.empty:
			continue

		panel = base_daily_df.sort_values("date").copy()
		panel["qlib_symbol"] = to_qlib_symbol(code)

		for field_name in required_fields:
			if field_name == "close":
				continue

			if field_name in QLIB_DAILY_FIELDS:
				field_df = load_qlib_daily_field(code=code, field_name=field_name, calendar=calendar)
				if not field_df.empty:
					panel = panel.merge(field_df[["date", field_name]], on="date", how="left")
				else:
					panel[field_name] = np.nan
				continue

			if field_name not in meta_dict:
				panel[field_name] = np.nan
				panel[f"{field_name}_alignment_date"] = pd.NaT
				panel[f"{field_name}_source_col"] = pd.NA
				continue

			spec = meta_dict[field_name]
			file_path = HS300_CACHE_DIR / spec["source_file"] / f"{code}.csv"
			if not file_path.exists():
				panel[field_name] = np.nan
				panel[f"{field_name}_alignment_date"] = pd.NaT
				panel[f"{field_name}_source_col"] = pd.NA
				continue

			df = pd.read_csv(file_path)
			value_col = spec["column_name"]
			align_col = spec["alignment_column"]
			if value_col not in df.columns or align_col not in df.columns:
				panel[field_name] = np.nan
				panel[f"{field_name}_alignment_date"] = pd.NaT
				panel[f"{field_name}_source_col"] = pd.NA
				continue

			meta_rows = df[[align_col, value_col]].copy()
			meta_rows.columns = [f"{field_name}_alignment_date", field_name]
			meta_rows[f"{field_name}_alignment_date"] = pd.to_datetime(meta_rows[f"{field_name}_alignment_date"], errors="coerce")
			meta_rows[field_name] = pd.to_numeric(meta_rows[field_name], errors="coerce")
			meta_rows = meta_rows.dropna(subset=[f"{field_name}_alignment_date", field_name]).sort_values(f"{field_name}_alignment_date")
			meta_rows[f"{field_name}_source_col"] = value_col

			if meta_rows.empty:
				panel[field_name] = np.nan
				panel[f"{field_name}_alignment_date"] = pd.NaT
				panel[f"{field_name}_source_col"] = pd.NA
				continue

			panel = pd.merge_asof(
				panel.sort_values("date"),
				meta_rows,
				left_on="date",
				right_on=f"{field_name}_alignment_date",
				direction="backward",
			)

		all_rows.append(panel)

	if not all_rows:
		return pd.DataFrame()
	return pd.concat(all_rows, ignore_index=True)


def calculate_factors(metadata_df: pd.DataFrame, factors_dict: dict[str, list[str]]) -> pd.DataFrame:
	out = metadata_df.copy()
	for factor_name, field_names in factors_dict.items():
		if factor_name == "fund_pe" and {"close", "eps"}.issubset(field_names):
			out[factor_name] = np.where(out["eps"] != 0, out["close"] / out["eps"], np.nan)
			out[factor_name] = out[factor_name].replace([np.inf, -np.inf], np.nan)
	return out


def main() -> None:
	symbols = load_symbols_from_cache(index_code=INDEX_CODE)
	if SMOKE_TEST:
		symbols = symbols[: max(1, SMOKE_COUNT)]

	metadata_daily_df = build_metadata_daily_panel(codes=symbols, factors_dict=FACTORS_DICT, meta_dict=META_DATA_DICT)

	if metadata_daily_df.empty:
		print("No metadata dataframe generated. Please check qlib path and metadata source files.")
		return

	result_df = calculate_factors(metadata_daily_df, FACTORS_DICT)

	OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

	print(f"Processed symbols: {len(symbols)}")
	print(f"Required fields: {sorted({field for fields in FACTORS_DICT.values() for field in fields})}")
	print(f"Output rows: {len(result_df)}")
	print(f"Output file: {OUT_FILE}")
	print("\n[Metadata DataFrame Columns]")
	print(result_df.columns.tolist())
	print("\n[Metadata DataFrame Head]")
	print(result_df.head(DEBUG_ROWS).to_string(index=False))


if __name__ == "__main__":
	main()
