#!/usr/bin/env python3
"""
Merge and validate results from two factor computation pipelines:
1. qlib price factors (2018-2020)
2. AkShare financial/macro factors (2018-2023)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def load_summary_json(json_path: Path) -> Dict[str, Any]:
    """Load summary JSON from factor computation."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    output_dir = Path("factors/output")
    
    # Load both results
    qlib_summary = load_summary_json(output_dir / "qlib_price_factors_2018_2020_summary.json")
    akshare_summary = load_summary_json(output_dir / "akshare_financial_macro_2018_2023_summary.json")
    
    # Merge summaries
    all_factors = []
    all_factors.extend(qlib_summary.get("summary", []))
    all_factors.extend(akshare_summary.get("summary", []))
    
    # Compute statistics
    total_factors = len(all_factors)
    avg_ic_mean = sum(f.get("ic_mean", 0) for f in all_factors) / total_factors if total_factors > 0 else 0
    avg_sharpe = sum(f.get("sharpe", 0) for f in all_factors) / total_factors if total_factors > 0 else 0
    
    # Count by data source
    qlib_count = len(qlib_summary.get("summary", []))
    akshare_count = len(akshare_summary.get("summary", []))
    
    # Generate report
    report = {
        "computation_summary": {
            "total_factors": total_factors,
            "qlib_factors": qlib_count,
            "akshare_factors": akshare_count,
            "qlib_date_range": "2018-01-01 to 2020-09-25",
            "akshare_date_range": "2018-01-01 to 2023-12-31",
            "qlib_stocks": qlib_summary.get("config", {}).get("stock_count", 0),
            "akshare_stocks": akshare_summary.get("config", {}).get("stock_count", 0)
        },
        "performance_statistics": {
            "avg_ic_mean": float(avg_ic_mean),
            "avg_sharpe": float(avg_sharpe),
            "median_annual_return": float(pd.Series([f.get("annual_return", 0) for f in all_factors]).median()),
            "median_annual_volatility": float(pd.Series([f.get("annual_volatility", 0) for f in all_factors]).median()),
            "median_max_drawdown": float(pd.Series([f.get("max_drawdown", 0) for f in all_factors]).median()),
        },
        "qlib_factors": qlib_summary.get("summary", []),
        "akshare_factors": akshare_summary.get("summary", []),
        "all_factors": sorted(all_factors, key=lambda x: x.get("factor_name", "")),
        "output_files": {
            "qlib": qlib_summary.get("files", {}),
            "akshare": akshare_summary.get("files", {}),
            "merged_report": "factors/output/comprehensive_factor_report_2018_2026.json"
        }
    }
    
    # Save comprehensive report
    report_path = output_dir / "comprehensive_factor_report_2018_2026.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE FACTOR COMPUTATION REPORT")
    print("="*80)
    print(f"\n总因子数: {total_factors}")
    print(f"  - qlib纯价格因子(2018-2020): {qlib_count}")
    print(f"  - AkShare财务+宏观因子(2018-2023): {akshare_count}")
    print(f"\n计算统计:")
    print(f"  - 平均IC均值: {avg_ic_mean:.6f}")
    print(f"  - 平均Sharpe比率: {avg_sharpe:.6f}")
    print(f"  - 中位年回报率: {report['performance_statistics']['median_annual_return']:.6f}")
    print(f"  - 中位年波动率: {report['performance_statistics']['median_annual_volatility']:.6f}")
    print(f"  - 中位最大回撤: {report['performance_statistics']['median_max_drawdown']:.6f}")
    
    print(f"\nqlib因子列表 ({qlib_count}):")
    for i, factor in enumerate(qlib_summary.get("summary", []), 1):
        name = factor.get("factor_name", "")
        ic_mean = factor.get("ic_mean", 0)
        sharpe = factor.get("sharpe", 0)
        print(f"  {i:2d}. {name:30s} IC={ic_mean:8.4f}  Sharpe={sharpe:8.4f}")
    
    print(f"\nAkShare因子列表 ({akshare_count}):")
    for i, factor in enumerate(akshare_summary.get("summary", []), 1):
        name = factor.get("factor_name", "")
        ic_mean = factor.get("ic_mean", 0)
        sharpe = factor.get("sharpe", 0)
        coverage = factor.get("coverage", 0) * 100
        print(f"  {i:2d}. {name:30s} IC={ic_mean:8.4f}  Sharpe={sharpe:8.4f}  Coverage={coverage:.1f}%")
    
    print(f"\n报告已保存到: {report_path}")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
