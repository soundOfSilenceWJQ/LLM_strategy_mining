from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_quant.agents.strategy_generator import StrategyGenerator
from llm_quant.agents.text_processor import TextProcessor
from llm_quant.info_based_adviser import InfoBasedAdviserAgent
from llm_quant.info_based_summarize import InfoBasedSummarizerAgent, InfoParsingAgent
from llm_quant.llm_based_selector import FactorSelectorAgent
from llm_quant.pipelines import IntegratedSignalPipeline
from llm_quant.utils.llm_client import LLMClient


HARDCODED_API_KEY = "ark-bd1a4194-8dc8-4446-850f-cf4934797f1c-9389a"

class DummyInfoStore:
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "id": "rec1",
                "title": "政策支持科技",
                "logic": "政策推动科技景气",
                "summary": "科技景气",
                "key_drivers": ["政策"],
                "time_horizon": "中期",
            }
        ]


def _dump_debug(intermediates: dict[str, Any], date_tag: str) -> str:
    out_dir = Path(__file__).resolve().parent / "store" / "smoke_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = out_dir / f"smoke_debug_{date_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, str] = {}
    for key, value in intermediates.items():
        file_name = f"{key}.txt"
        file_path = run_dir / file_name
        file_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        file_map[key] = str(file_path)

    index_path = run_dir / "index.txt"
    index_path.write_text(json.dumps(file_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(index_path)


def run_real_smoke_test(print_intermediate: bool = True) -> dict[str, Any]:
    """真实LLM调用场景下的最小冒烟测试（尽量减少调用次数与上下文长度）。"""
    now = datetime.now().strftime("%Y-%m-%d")
    llm = LLMClient(api_key=HARDCODED_API_KEY)

    raw_record = {
        "source_type": "news",
        "section": "宏观",
        "title": "政策支持科技创新",
        "publish_time": now,
        "url": "https://example.com/news/1",
        "content": "监管层强调支持科技创新与产业升级。",
        "source": "测试源",
        "date": now,
    }

    adviser = InfoBasedAdviserAgent(llm=llm)
    adviser_out = adviser.analyze_record(raw_record)

    parser = InfoParsingAgent(llm=llm)
    parse_realtime_out = parser.analyze_realtime_text(
        raw_text="政策支持科技创新，行业景气改善。",
        metadata={"source": "测试源", "publish_date": now, "title": "测试标题", "url": ""},
        max_chars=80,
    )

    summarizer = InfoBasedSummarizerAgent(llm=llm)
    summary_out = summarizer.summarize_day(date=now, analyzed_rows=[adviser_out])

    strategy = StrategyGenerator(llm=llm, info_store=cast(Any, DummyInfoStore()))
    factor_out = strategy.generate_factor_from_literature(
        factor_description="EPS一致预期30日修正幅度",
        economic_logic="预期上修驱动未来超额收益",
        other_info="中证500样本",
        source_meta={"literature_source": "test-paper", "rebalance_cycle": "月频"},
        n_context=1,
    )
    repaired_out = strategy.repair_factor_with_history(
        current_factor=factor_out or {},
        error_info="NameError: eps not defined",
        correct_advice="改为使用可用字段并去除未来函数",
        conversation_history=[
            {"role": "user", "content": "之前公式运行失败"},
            {"role": "assistant", "content": "建议检查字段名和时序方向"},
        ],
    )

    selector_agent = FactorSelectorAgent(llm=llm)
    selector_out = selector_agent.select_factors(
        date=now,
        market_info="市场震荡，成长风格偏强。",
        factors_list=[
            {"name": "eps_revision_30d", "description": "盈利预期修正", "category": "fundamental"},
            {"name": "momentum_20d", "description": "20日动量", "category": "momentum"},
        ],
        factor_performance=None,
    )

    text_processor = TextProcessor(llm=llm)
    text_proc_out = text_processor.process(raw_record)

    integrated = IntegratedSignalPipeline(adviser=adviser, summarizer=summarizer, selector=selector_agent)
    integrated_out = integrated.run_daily(
        date=now,
        raw_records=[raw_record],
        factors_list=[
            {"name": "eps_revision_30d", "description": "盈利预期修正", "category": "fundamental"}
        ],
        factor_performance=None,
        market_info="",
    )

    intermediates = {
        "adviser_out": adviser_out,
        "parse_realtime_out": parse_realtime_out,
        "summary_out": summary_out,
        "factor_out": factor_out,
        "repaired_out": repaired_out,
        "selector_out": selector_out,
        "text_proc_out": text_proc_out,
        "integrated_out": integrated_out,
    }

    if print_intermediate:
        print("\n===== Intermediate Outputs =====")
        for key, value in intermediates.items():
            print(f"\n[{key}]")
            print(json.dumps(value, ensure_ascii=False, indent=2)[:4000])

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = _dump_debug(intermediates=intermediates, date_tag=date_tag)

    checks = {
        "adviser_ok": bool(adviser_out.get("analysis", {}).get("market_impact")),
        "parser_ok": bool(parse_realtime_out.get("refined_summary")),
        "summarizer_ok": bool(summary_out.get("summary", {}).get("investor_advice")),
        "strategy_generate_ok": bool((factor_out or {}).get("code")),
        "strategy_repair_ok": bool((repaired_out or {}).get("repair_log")),
        "selector_ok": bool(selector_out.get("recommendation", {}).get("recommended_factors")),
        "text_processor_ok": bool(text_proc_out.get("traceability")),
        "integrated_ok": bool(integrated_out.get("factor_selection")),
    }

    return {
        "mode": "real",
        "checks": checks,
        "all_passed": all(checks.values()),
        "debug_file": debug_file,
        "sample": {
            "factor_name": (factor_out or {}).get("name"),
            "recommended_count": len(selector_out.get("recommendation", {}).get("recommended_factors", [])),
            "summary_stance": summary_out.get("summary", {}).get("investor_advice", {}).get("stance", ""),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for llm_quant agents")
    parser.add_argument(
        "--mode",
        choices=["real"],
        default="real",
        help="real: call live LLM with hardcoded api key",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable printing intermediate outputs to console",
    )
    args = parser.parse_args()

    out = run_real_smoke_test(print_intermediate=not args.quiet)
    print(json.dumps(out, ensure_ascii=False, indent=2))
