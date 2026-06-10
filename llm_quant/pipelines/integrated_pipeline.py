from __future__ import annotations

from datetime import datetime
from typing import Any

from llm_quant.info_based_adviser import InfoBasedAdviserAgent
from llm_quant.info_based_summarize import InfoBasedSummarizerAgent
from llm_quant.llm_based_selector import FactorSelectorAgent


class IntegratedSignalPipeline:
    """整合单条解读、日度综述和因子推荐的统一入口。"""

    def __init__(
        self,
        adviser: InfoBasedAdviserAgent | None = None,
        summarizer: InfoBasedSummarizerAgent | None = None,
        selector: FactorSelectorAgent | None = None,
    ):
        self.adviser = adviser or InfoBasedAdviserAgent()
        self.summarizer = summarizer or InfoBasedSummarizerAgent()
        self.selector = selector or FactorSelectorAgent()

    def run_daily(
        self,
        date: str,
        raw_records: list[dict[str, Any]],
        factors_list: list[dict[str, Any]],
        factor_performance: dict[str, Any] | None = None,
        market_info: str = "",
    ) -> dict[str, Any]:
        """
        统一执行：单条解读 -> 日度综述 -> 因子推荐。
        """
        analyzed_rows = [self.adviser.analyze_record(x) for x in raw_records]
        summary = self.summarizer.summarize_day(date=date, analyzed_rows=analyzed_rows)

        if not market_info:
            market_info = summary.get("summary", {}).get("market_overview", "")

        selection = self.selector.select_factors(
            date=date,
            market_info=market_info,
            factors_list=factors_list,
            factor_performance=factor_performance,
        )

        return {
            "date": date,
            "counts": {
                "raw_records": len(raw_records),
                "analyzed_rows": len(analyzed_rows),
                "factors": len(factors_list),
            },
            "analyzed_rows": analyzed_rows,
            "daily_summary": summary,
            "factor_selection": selection,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
