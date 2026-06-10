from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


# =====================
# Hardcoded parameters
# =====================
PROVIDER_URI = "D:/qlib_data/cn_data"
REGION = "cn"
INDEX_NAME = "csi300"
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
SMOKE_COUNT = 0  # 0 means all symbols

PAUSE_SEC = 0.05
API_RETRY = 3
MISSING_RETRY_ROUNDS = 3

REPORT_BALANCE_SHEET = "资产负债表"
REPORT_INCOME_STATEMENT = "利润表"
REPORT_CASH_FLOW = "现金流量表"

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "akshare_cache"
SYMBOLS_FILE = CACHE_ROOT / "symbols_from_qlib.csv"

AKSHARE_TASKS: list[dict[str, Any]] = [
    {
        "func": "stock_financial_analysis_indicator_em",
        "kwargs": {"symbol": "{em_symbol}", "indicator": "按单季度"},
    },
    {
        "func": "stock_financial_report_sina",
        "kwargs": {"stock": "{market_symbol}", "symbol": REPORT_BALANCE_SHEET},
        "output_name": "stock_financial_report_sina_balance_sheet",
    },
    {
        "func": "stock_financial_report_sina",
        "kwargs": {"stock": "{market_symbol}", "symbol": REPORT_INCOME_STATEMENT},
        "output_name": "stock_financial_report_sina_income_statement",
    },
    {
        "func": "stock_financial_report_sina",
        "kwargs": {"stock": "{market_symbol}", "symbol": REPORT_CASH_FLOW},
        "output_name": "stock_financial_report_sina_cash_flow",
    },
]


def safe_call(func: Any, kwargs: dict[str, Any], retries: int = API_RETRY, sleep_sec: float = 0.8) -> pd.DataFrame:
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            return func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i < retries - 1:
                time.sleep(sleep_sec * (i + 1))
    assert last_exc is not None
    raise last_exc


def init_qlib() -> None:
    import qlib

    qlib.init(provider_uri=PROVIDER_URI, region=REGION)


def to_plain_symbol(qlib_symbol: str) -> str:
    m = re.search(r"(\d{6})", str(qlib_symbol))
    if not m:
        raise ValueError(f"Cannot parse 6-digit code from qlib symbol: {qlib_symbol}")
    return m.group(1)


def to_market_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def to_em_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def get_symbols_from_qlib() -> list[str]:
    from qlib.data import D

    instruments = D.instruments(INDEX_NAME)
    frame = D.features(
        instruments,
        ["$close"],
        start_time=START_DATE,
        end_time=END_DATE,
        freq="day",
    )
    if frame.empty:
        raise RuntimeError("No instrument data returned from qlib. Check index/time range/provider_uri.")

    raw_symbols = frame.index.get_level_values("instrument").astype(str).unique().tolist()
    symbols = sorted({to_plain_symbol(s) for s in raw_symbols})
    if SMOKE_COUNT > 0:
        symbols = symbols[:SMOKE_COUNT]

    pd.DataFrame({"symbol": symbols}).to_csv(SYMBOLS_FILE, index=False, encoding="utf-8-sig")
    return symbols


def render_kwargs(template_kwargs: dict[str, Any], symbol: str) -> dict[str, Any]:
    market_symbol = to_market_symbol(symbol)
    em_symbol = to_em_symbol(symbol)
    rendered: dict[str, Any] = {}
    for k, v in template_kwargs.items():
        if isinstance(v, str):
            rendered[k] = (
                v.replace("{symbol}", symbol)
                .replace("{market_symbol}", market_symbol)
                .replace("{em_symbol}", em_symbol)
            )
        else:
            rendered[k] = v
    return rendered


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_task(symbols: list[str], task: dict[str, Any]) -> None:
    func_name = task["func"]
    out_name = task.get("output_name", func_name)
    out_dir = CACHE_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(ak, func_name):
        raise AttributeError(f"akshare has no function: {func_name}")
    func = getattr(ak, func_name)

    print(f"\n[{out_name}] start, symbols={len(symbols)}")

    for round_idx in range(1, MISSING_RETRY_ROUNDS + 1):
        missing = [s for s in symbols if not file_ok(out_dir / f"{s}.csv")]
        if not missing:
            print(f"[{out_name}] done before round {round_idx}")
            return

        ok = 0
        failed = 0
        print(f"[{out_name}] round {round_idx}, missing={len(missing)}")
        for idx, symbol in enumerate(missing, start=1):
            out_file = out_dir / f"{symbol}.csv"
            kwargs = render_kwargs(task["kwargs"], symbol)
            try:
                df = safe_call(func, kwargs)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df.to_csv(out_file, index=False, encoding="utf-8-sig")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print(f"[{out_name}] fail {symbol}: {exc}")

            if PAUSE_SEC > 0 and idx < len(missing):
                time.sleep(PAUSE_SEC)

        print(f"[{out_name}] round {round_idx} finished: ok={ok}, failed={failed}")

    still_missing = [s for s in symbols if not file_ok(out_dir / f"{s}.csv")]
    print(f"[{out_name}] final missing after retries: {len(still_missing)}")


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    init_qlib()
    symbols = get_symbols_from_qlib()
    print(f"INDEX={INDEX_NAME}, time=[{START_DATE}..{END_DATE}], symbols={len(symbols)}")
    print(f"symbols saved: {SYMBOLS_FILE}")

    for task in AKSHARE_TASKS:
        run_task(symbols, task)


if __name__ == "__main__":
    main()
