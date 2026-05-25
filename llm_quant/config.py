"""
系统配置文件
System configuration for LLM Quantitative Strategy Mining System
"""

import os
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "store"
DATA_DIR.mkdir(exist_ok=True)

INFO_DB_PATH = DATA_DIR / "info_store.json"      # 非结构化信息库
PAR_DB_PATH  = DATA_DIR / "par_store.json"       # 原始阿尔法因子库
IAR_DB_PATH  = DATA_DIR / "iar_store.json"       # 投资级阿尔法因子库

# ── LLM 配置 ──────────────────────────────────────────────
# 从环境变量读取；也可在此直接填写（不建议提交到版本库）
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-4-5"         # Claude 系列旗舰模型
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3                    # 较低温度保证金融逻辑严谨性

# ── 市场数据配置 ──────────────────────────────────────────
DEFAULT_UNIVERSE = "CSI300"              # 默认标的池
BACKTEST_START = "2023-09-01"
BACKTEST_END   = "2025-12-31"
RISK_FREE_RATE = 0.015                   # 1年期定存基准利率 1.5%
TRADING_DAYS_PER_YEAR = 250

# ── 因子评价阈值 ──────────────────────────────────────────
IC_THRESHOLD   = 0.02                   # IC均值最低阈值（入库PAR）
ICIR_THRESHOLD = 0.3                    # ICIR最低阈值
IC_WIN_RATE_THRESHOLD = 0.52            # IC月胜率最低阈值

# ── IAR 配置 ──────────────────────────────────────────────
IAR_MAX_FACTORS = 30                    # IAR最大容纳因子数（简化版）
PAR_TOP_RATIO   = 0.3                   # PAR前30%进入IAR

# ── 爬虫配置 ──────────────────────────────────────────────
CRAWL_TIMEOUT   = 10                    # 请求超时秒数
CRAWL_MAX_ITEMS = 20                    # 每次最多抓取条数

# ── 信息库 RAG 配置 ──────────────────────────────────────
RAG_TOP_K = 5                           # RAG检索返回Top-K条信息

# ── 因子代码规范 ──────────────────────────────────────────
ALLOWED_OPERATORS = [
    "CLOSE", "OPEN", "HIGH", "LOW", "VOLUME", "VWAP",
    "MA", "SMA", "EMA", "STD", "VAR", "MAX", "MIN",
    "DELAY", "DELTA", "RANK", "ZSCORE", "SIGN",
    "ABS", "LOG", "CORR", "COV", "SUM", "PROD",
]
