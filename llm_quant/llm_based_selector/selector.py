"""
LLM 因子选择框架

管理因子数据加载、市场信息聚合和选择推荐流程
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from llm_quant.llm_based_selector.agent import FactorSelectorAgent


class FactorSelectorFramework:
    """因子选择框架的主类"""

    def __init__(self, base_dir: str | Path = None):
        """
        初始化因子选择框架

        Args:
            base_dir: 数据基础目录，默认为当前脚本所在目录
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "data"
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.market_info_dir = base_dir / "market_info"
        self.factors_dir = base_dir / "factors"
        self.output_dir = base_dir / "recommendations"

        # 创建必要的目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.agent = FactorSelectorAgent()

    def load_market_info(self, filename: str = "market_summary.txt") -> str:
        """
        加载大盘信息文件

        Args:
            filename: 市场信息文件名

        Returns:
            市场信息文本内容
        """
        info_file = self.market_info_dir / filename
        if not info_file.exists():
            raise FileNotFoundError(f"市场信息文件不存在: {info_file}")

        with open(info_file, "r", encoding="utf-8") as f:
            return f.read()

    def load_factors_list(self, filename: str = "factors_list.json") -> list[dict[str, Any]]:
        """
        加载因子列表

        Args:
            filename: 因子列表文件名

        Returns:
            因子列表 (JSON 格式)
        """
        factors_file = self.factors_dir / filename
        if not factors_file.exists():
            raise FileNotFoundError(f"因子列表文件不存在: {factors_file}")

        with open(factors_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_factor_performance(
        self, filename: str = "factor_performance.json"
    ) -> dict[str, Any]:
        """
        加载因子表现数据

        Args:
            filename: 因子表现文件名

        Returns:
            因子表现数据 (JSON 格式)
        """
        perf_file = self.factors_dir / filename
        if not perf_file.exists():
            return None

        with open(perf_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def select_factors(
        self,
        date: str | None = None,
        market_info_file: str = "market_summary.txt",
        factors_list_file: str = "factors_list.json",
        performance_file: str = "factor_performance.json",
    ) -> dict[str, Any]:
        """
        执行因子选择

        Args:
            date: 当前日期，默认为今天
            market_info_file: 市场信息文件名
            factors_list_file: 因子列表文件名
            performance_file: 因子表现文件名

        Returns:
            选择结果 (包含推荐因子和分析)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 加载数据
        market_info = self.load_market_info(market_info_file)
        factors_list = self.load_factors_list(factors_list_file)
        performance = self.load_factor_performance(performance_file)

        # 执行选择
        result = self.agent.select_factors(
            date=date,
            market_info=market_info,
            factors_list=factors_list,
            factor_performance=performance,
        )

        return result

    def save_recommendation(
        self,
        result: dict[str, Any],
        output_file: str | None = None,
    ) -> Path:
        """
        保存推荐结果

        Args:
            result: 选择结果
            output_file: 输出文件名，默认为日期+时间戳

        Returns:
            保存的文件路径
        """
        if output_file is None:
            date = result.get("date", "unknown")
            timestamp = result.get("processed_at", "").split("T")[-1].replace(":", "")
            output_file = f"recommendation_{date}_{timestamp}.json"

        output_path = self.output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_path

    def run_pipeline(
        self,
        date: str | None = None,
        market_info_file: str = "market_summary.txt",
        factors_list_file: str = "factors_list.json",
        performance_file: str = "factor_performance.json",
        save_output: bool = True,
    ) -> dict[str, Any]:
        """
        运行完整的因子选择管道

        Args:
            date: 当前日期
            market_info_file: 市场信息文件名
            factors_list_file: 因子列表文件名
            performance_file: 因子表现文件名
            save_output: 是否保存结果

        Returns:
            完整的选择结果
        """
        result = self.select_factors(
            date=date,
            market_info_file=market_info_file,
            factors_list_file=factors_list_file,
            performance_file=performance_file,
        )

        if save_output:
            output_path = self.save_recommendation(result)
            result["output_file"] = str(output_path)

        return result

    @staticmethod
    def print_recommendation(result: dict[str, Any]):
        """打印推荐结果到控制台"""
        print("\n" + "=" * 80)
        print(f"因子选择推荐 - {result.get('date', 'N/A')}")
        print("=" * 80)

        recommendation = result.get("recommendation", {})

        # 市场分析
        market = recommendation.get("market_analysis", {})
        print(f"\n【市场分析】")
        print(f"当前制度: {market.get('current_regime', 'N/A')}")
        print(f"宏观展望: {market.get('macroeconomic_outlook', 'N/A')}")
        if market.get("dominant_factors"):
            print(f"主导因子: {', '.join(market.get('dominant_factors', []))}")
        if market.get("risk_factors"):
            print(f"风险因子: {', '.join(market.get('risk_factors', []))}")

        # 推荐因子
        recommended = recommendation.get("recommended_factors", [])
        print(f"\n【推荐因子】(共 {len(recommended)} 个)")
        for idx, factor in enumerate(recommended, 1):
            print(f"\n  {idx}. {factor.get('factor_name', 'N/A')}")
            print(f"     权重: {factor.get('weight', 0):.2%}")
            print(f"     风险: {factor.get('risk_level', 'N/A')}")
            print(f"     历史IC: {factor.get('historical_ic', 0):.4f}")
            print(f"     理由: {factor.get('rationale', 'N/A')}")

        # 需要回避的因子
        avoid = recommendation.get("factors_to_avoid", [])
        if avoid:
            print(f"\n【需要回避的因子】")
            for factor in avoid:
                print(f"  - {factor.get('factor_name', 'N/A')}: {factor.get('reason', 'N/A')}")

        # 实施建议
        impl = recommendation.get("implementation_suggestions", {})
        print(f"\n【实施建议】")
        print(f"  调整周期: {impl.get('rebalance_frequency', 'N/A')}")
        print(f"  入场建议: {impl.get('entry_timing', 'N/A')}")
        if impl.get("exit_signals"):
            print(f"  退出信号: {', '.join(impl.get('exit_signals', []))}")

        # 不确定性
        print(f"\n【信心度】{recommendation.get('confidence_level', 'N/A')}")
        if recommendation.get("uncertainty_sources"):
            print(f"  不确定来源: {', '.join(recommendation.get('uncertainty_sources', []))}")

        print("\n" + "=" * 80)
