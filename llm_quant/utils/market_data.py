"""
市场数据获取模块（基于 akshare）
支持：A股日频OHLCV、CSI300成分股列表
"""

from __future__ import annotations
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")


# ── CSI300 成分股 ─────────────────────────────────────────
@lru_cache(maxsize=1)
def get_csi300_stocks() -> list[str]:
    """获取沪深300成分股代码列表（带市场前缀：sh/sz）。"""
    try:
        df = ak.index_stock_cons(symbol="000300")
        codes = df["品种代码"].tolist()
        # akshare 返回的是6位代码，加前缀
        prefixed = []
        for c in codes:
            prefixed.append(("sh" if c.startswith("6") else "sz") + c)
        return prefixed[:50]   # 简化：取前50支
    except Exception as e:
        print(f"[MarketData] 获取CSI300成分股失败: {e}")
        # 降级：返回一组代表性大盘蓝筹股
        return [
            "sh600519","sh601318","sh600036","sh601166","sh600900",
            "sh601398","sh600276","sh601288","sh600030","sh601628",
            "sh600104","sh601088","sh600050","sh600809","sh600887",
            "sz000858","sz000333","sz000001","sz002415","sz000002",
        ]


# ── 单股 OHLCV ────────────────────────────────────────────
def get_stock_daily(
    symbol: str,
    start_date: str = "20230901",
    end_date: str = "20251231",
    adjust: str = "hfq",          # 后复权
) -> pd.DataFrame:
    """
    获取单支股票日频OHLCV数据。
    返回列：date, open, high, low, close, volume
    """
    # akshare 格式：sh600519 → 600519, 市场前缀去掉
    code = symbol.replace("sh", "").replace("sz", "")
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["symbol"] = symbol
        return df[["open", "high", "low", "close", "volume", "symbol"]]
    except Exception as e:
        print(f"[MarketData] 获取 {symbol} 数据失败: {e}")
        return pd.DataFrame()


# ── 多股票数据面板 ────────────────────────────────────────
def get_panel_data(
    symbols: list[str] | None = None,
    start_date: str = "20230901",
    end_date: str = "20251231",
) -> pd.DataFrame:
    """
    获取多支股票的日频面板数据。
    返回 MultiIndex DataFrame: (date, symbol) × [open, high, low, close, volume]
    """
    if symbols is None:
        symbols = get_csi300_stocks()

    frames = []
    for sym in symbols:
        df = get_stock_daily(sym, start_date, end_date)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel = panel.reset_index()
    panel = panel.set_index(["date", "symbol"]).sort_index()
    return panel


# ── 计算收益率 ────────────────────────────────────────────
def compute_returns(
    panel: pd.DataFrame,
    horizon: int = 1,               # 持有期（交易日）
) -> pd.Series:
    """
    计算面板数据中每支股票的未来N日收益率。
    返回 Series with MultiIndex (date, symbol)
    """
    close = panel["close"].unstack("symbol")          # date × symbol
    fwd_ret = close.shift(-horizon) / close - 1       # 未来N日收益率
    fwd_ret = fwd_ret.stack()
    fwd_ret.name = f"ret_{horizon}d"
    return fwd_ret
