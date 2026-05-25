"""
LLM 因子选择的辅助工具函数
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_sample_market_info(output_file: Path | str = "market_summary.txt") -> str:
    """
    创建示例市场信息文件

    Returns:
        市场信息文本内容
    """
    market_info = """【2026年5月20日 市场信息汇总】

【宏观经济背景】
- GDP 增速: 5.2% (同比)
- CPI 涨幅: 2.3% (同比)  
- PPI 涨幅: 1.5% (同比)
- 货币供应量 M2: 12.5% (同比增长)
- 社会融资规模: 3.2 万亿 (新增)

【政策动向】
- 央行连续 3 周维持中期借贷便利(MLF)利率不变，当前为 2.45%
- 国务院最近强调稳增长和促进消费，有望释放更多积极政策
- 新能源产业支持力度持续，新增补贴 150 亿元

【股市情况】
- 上证指数: 3,245 点 (较年初上涨 8.5%)
- 深证成指: 10,850 点
- 沪深 300: 4,150 点
- 创业板指: 2,150 点
- 技术股表现强势，半导体与新能源领先
- 消费股近期表现滞后，估值处于历史低位

【板块热点】
- 新能源汽车: 产销数据超预期，政策持续支持
- 人工智能: 大模型应用落地加速，相关公司业绩向好
- 芯片半导体: 国产替代加速，行业景气度高企
- 房地产: 调控政策边际松动，部分龙头股价回升
- 消费: 电商平台流量增长放缓，竞争激烈

【市场风格】
- 当前为增长风格主导 (成长> 价值)
- 科技/创新类表现强于周期/防守类
- 高估值、高成长的品种更受资金追捧
- 机构持仓集中度较高，回撤时可能出现集中平仓

【风险提示】
- 地缘政治局势仍存不确定性
- 美联储加息预期反复，美元汇率承压
- 流动性总体宽松但结构分化
- 部分行业估值过高，调整风险增加

【海外市场】
- 美股三大指数近期震荡上行
- 欧洲经济增速放缓，利率政策分化
- 新兴市场波动较大，资金流向不稳定

【行业预期】
- 半导体行业下半年景气持续，订单能见度良好
- 新能源产业面临产能过剩风险，竞争加剧
- 消费升级趋势不变，但增速可能放缓
- 医药生物通过创新药上市，有望带动增长
"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(market_info)
    
    return market_info


def create_sample_factors_list(output_file: Path | str = "factors_list.json") -> list[dict[str, Any]]:
    """
    创建示例因子列表

    Returns:
        因子列表数据
    """
    factors = [
        {
            "name": "momentum_10",
            "category": "动量因子",
            "description": "10日动量因子，衡量最近短期上升趋势",
            "formula": "close[-1] / close[-10] - 1",
            "ic": 0.045,
            "type": "price_based",
            "lookback": 10,
        },
        {
            "name": "fund_pb",
            "category": "价值因子",
            "description": "股价净值比，衡量估值水平",
            "formula": "close / book_value_per_share",
            "ic": 0.032,
            "type": "fundamental",
            "frequency": "quarterly",
        },
        {
            "name": "growth_eps",
            "category": "成长因子",
            "description": "EPS 环比增长率",
            "formula": "eps[0] / eps[-1] - 1",
            "ic": 0.028,
            "type": "fundamental",
            "frequency": "quarterly",
        },
        {
            "name": "macro_gdp",
            "category": "宏观因子",
            "description": "GDP 同比增速对股价的影响",
            "formula": "gdp_growth_rate",
            "ic": -0.005,
            "type": "macro",
            "frequency": "quarterly",
        },
        {
            "name": "quality_roe",
            "category": "质量因子",
            "description": "净资产收益率，衡量企业盈利能力",
            "formula": "net_profit / equity",
            "ic": 0.018,
            "type": "fundamental",
            "frequency": "quarterly",
        },
        {
            "name": "size_factor",
            "category": "规模因子",
            "description": "市值规模，小盘股相对表现",
            "formula": "log(market_cap)",
            "ic": 0.012,
            "type": "style",
            "frequency": "daily",
        },
        {
            "name": "volatility_reverse",
            "category": "风险因子",
            "description": "波动率反转，低波动股票相对表现",
            "formula": "1 / std_dev",
            "ic": 0.022,
            "type": "risk",
            "frequency": "daily",
        },
        {
            "name": "earnings_surprise",
            "category": "盈利因子",
            "description": "盈利意外度，实际EPS相对预期的偏差",
            "formula": "(actual_eps - expected_eps) / expected_eps",
            "ic": 0.055,
            "type": "fundamental",
            "frequency": "quarterly",
        },
        {
            "name": "cash_flow_growth",
            "category": "现金流因子",
            "description": "经营现金流同比增速",
            "formula": "ocf[0] / ocf[-4] - 1",
            "ic": 0.038,
            "type": "fundamental",
            "frequency": "quarterly",
        },
        {
            "name": "sentiment_factor",
            "category": "情绪因子",
            "description": "基于研报数量和新闻情绪的因子",
            "formula": "report_count * sentiment_score",
            "ic": 0.025,
            "type": "alternative",
            "frequency": "daily",
        },
    ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(factors, f, ensure_ascii=False, indent=2)
    
    return factors


def create_sample_factor_performance(output_file: Path | str = "factor_performance.json") -> dict[str, Any]:
    """
    创建示例因子表现数据

    Returns:
        因子表现数据
    """
    performance = {
        "top_factors": [
            {
                "name": "earnings_surprise",
                "ic": 0.055,
                "annual_return": 0.185,
                "sharpe": 0.92,
                "max_drawdown": -0.22,
                "recent_ic_trend": "上升",
            },
            {
                "name": "momentum_10",
                "ic": 0.045,
                "annual_return": 0.152,
                "sharpe": 0.85,
                "max_drawdown": -0.18,
                "recent_ic_trend": "稳定",
            },
            {
                "name": "cash_flow_growth",
                "ic": 0.038,
                "annual_return": 0.128,
                "sharpe": 0.75,
                "max_drawdown": -0.25,
                "recent_ic_trend": "上升",
            },
        ],
        "bottom_factors": [
            {
                "name": "macro_gdp",
                "ic": -0.005,
                "annual_return": 0.012,
                "sharpe": 0.08,
                "max_drawdown": -0.45,
                "recent_ic_trend": "下降",
            },
            {
                "name": "size_factor",
                "ic": 0.012,
                "annual_return": 0.045,
                "sharpe": 0.32,
                "max_drawdown": -0.35,
                "recent_ic_trend": "下降",
            },
        ],
        "market_style": "增长风格主导，小盘风格相对受压",
        "factor_correlation": {
            "momentum_10": {"earnings_surprise": 0.42, "fund_pb": -0.35},
            "earnings_surprise": {"cash_flow_growth": 0.58, "quality_roe": 0.48},
            "fund_pb": {"macro_gdp": -0.28},
        },
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(performance, f, ensure_ascii=False, indent=2)
    
    return performance


def setup_sample_data(base_dir: Path | str = None):
    """
    设置示例数据目录结构和文件

    Args:
        base_dir: 基础目录
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data"
    else:
        base_dir = Path(base_dir)

    # 创建子目录
    market_info_dir = base_dir / "market_info"
    factors_dir = base_dir / "factors"
    
    market_info_dir.mkdir(parents=True, exist_ok=True)
    factors_dir.mkdir(parents=True, exist_ok=True)

    # 创建示例文件
    create_sample_market_info(market_info_dir / "market_summary.txt")
    create_sample_factors_list(factors_dir / "factors_list.json")
    create_sample_factor_performance(factors_dir / "factor_performance.json")

    print(f"✓ 示例数据已创建在: {base_dir}")
    print(f"  - {market_info_dir}/market_summary.txt")
    print(f"  - {factors_dir}/factors_list.json")
    print(f"  - {factors_dir}/factor_performance.json")

    return base_dir
