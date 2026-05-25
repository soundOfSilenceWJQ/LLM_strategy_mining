from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from multi_factor_backtest import init_qlib_env, load_available_instruments


PROVIDER_URI = "D:/qlib_data/cn_data"
REGION = "cn"
INPUT_FILE = Path("d:/wjq/learning/grad_project/data/factors/input/factors_merged_positive_eco.csv")
OUTPUT_FILE = Path("d:/wjq/learning/grad_project/data/factors/input/factors_merged_qlib_new.csv")
DROPPED_FILE = Path("d:/wjq/learning/grad_project/data/factors/input/factors_merged_qlib_new_dropped.csv")

# Use a short window for expression validation; we only care if expression parses/executes.
START_DATE = "2024-01-02"
END_DATE = "2024-01-31"
TEST_INSTRUMENT_COUNT = 5


def is_expression_computable(expression: str, instruments: list[str]) -> tuple[bool, str]:
    from qlib.data import D

    try:
        frame = D.features(
            instruments,
            [expression],
            start_time=START_DATE,
            end_time=END_DATE,
            freq="day",
        )
        if not isinstance(frame, pd.DataFrame):
            return False, "non_dataframe_output"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required_cols = {"name", "expression"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input csv must contain columns {required_cols}, got {list(df.columns)}")

    init_qlib_env(PROVIDER_URI, REGION)
    instruments = load_available_instruments(PROVIDER_URI)
    if not instruments:
        raise RuntimeError("No instruments available for expression validation.")

    test_instruments = instruments[:TEST_INSTRUMENT_COUNT]

    keep_rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        expression = str(row["expression"]).strip()
        if not name or not expression or name.lower() == "nan" or expression.lower() == "nan":
            continue

        ok, reason = is_expression_computable(expression, test_instruments)
        if ok:
            keep_rows.append(row.to_dict())
        else:
            dropped = row.to_dict()
            dropped["drop_reason"] = reason
            drop_rows.append(dropped)

    keep_df = pd.DataFrame(keep_rows)
    drop_df = pd.DataFrame(drop_rows)

    keep_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    drop_df.to_csv(DROPPED_FILE, index=False, encoding="utf-8-sig")

    print(f"input factors: {len(df)}")
    print(f"qlib computable factors: {len(keep_df)}")
    print(f"removed factors: {len(drop_df)}")
    print(f"output file: {OUTPUT_FILE}")
    print(f"dropped file: {DROPPED_FILE}")


if __name__ == "__main__":
    main()
