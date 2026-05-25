"""
LLM 因子选择框架 - 演示脚本

演示如何使用 LLM 进行智能因子选择
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_quant.llm_based_selector.selector import FactorSelectorFramework
from llm_quant.llm_based_selector.utils import setup_sample_data


def main():
    """
    演示主函数
    
    工作流程:
    1. 设置示例数据
    2. 初始化因子选择框架
    3. 加载数据
    4. 执行因子选择
    5. 显示和保存结果
    """
    
    print("\n" + "=" * 80)
    print("LLM 因子选择框架 - 演示程序")
    print("=" * 80)
    
    # Step 1: 设置示例数据
    print("\n【Step 1】准备示例数据...")
    data_dir = Path(__file__).parent.parent / "data"
    setup_sample_data(data_dir)
    
    # Step 2: 初始化框架
    print("\n【Step 2】初始化因子选择框架...")
    try:
        framework = FactorSelectorFramework(base_dir=data_dir)
        print("✓ 框架初始化成功")
    except Exception as e:
        print(f"✗ 框架初始化失败: {e}")
        return
    
    # Step 3: 加载数据
    print("\n【Step 3】加载数据...")
    try:
        market_info = framework.load_market_info("market_summary.txt")
        print(f"✓ 市场信息已加载 ({len(market_info)} 字符)")
        
        factors_list = framework.load_factors_list("factors_list.json")
        print(f"✓ 因子列表已加载 ({len(factors_list)} 个因子)")
        
        performance = framework.load_factor_performance("factor_performance.json")
        print(f"✓ 因子表现已加载")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # Step 4: 执行因子选择
    print("\n【Step 4】执行因子选择...")
    print("  调用 LLM 进行因子推荐中...")
    try:
        result = framework.run_pipeline(
            date=None,  # 使用当前日期
            market_info_file="market_summary.txt",
            factors_list_file="factors_list.json",
            performance_file="factor_performance.json",
            save_output=True,
        )
        print("✓ 因子选择完成")
    except Exception as e:
        print(f"✗ 因子选择失败: {e}")
        print("  提示: 请检查 LLM 是否可用（需要配置 API Key）")
        return
    
    # Step 5: 显示结果
    print("\n【Step 5】显示推荐结果...")
    FactorSelectorFramework.print_recommendation(result)
    
    # 保存结果的文件路径
    output_file = result.get("output_file", "N/A")
    print(f"\n✓ 结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    
    # 打印推荐因子摘要
    recommendation = result.get("recommendation", {})
    recommended_factors = recommendation.get("recommended_factors", [])
    
    print(f"\n【推荐因子摘要】")
    print(f"推荐 {len(recommended_factors)} 个因子:")
    for factor in recommended_factors:
        name = factor.get("factor_name", "N/A")
        weight = factor.get("weight", 0)
        print(f"  - {name}: {weight:.2%}")
    
    print("\n" + "=" * 80)


def demo_without_llm():
    """
    演示不依赖 LLM 的版本（使用备用推荐）
    
    这个版本不需要配置 LLM API Key，可以直接运行
    """
    
    print("\n" + "=" * 80)
    print("LLM 因子选择框架 - 离线演示 (不需要 LLM)")
    print("=" * 80)
    
    from llm_quant.llm_based_selector.agent import FactorSelectorAgent
    from llm_quant.llm_based_selector.utils import (
        create_sample_market_info,
        create_sample_factors_list,
        create_sample_factor_performance,
    )
    
    # 准备数据
    print("\n【准备数据】")
    market_info = create_sample_market_info()
    factors_list = create_sample_factors_list()
    performance = create_sample_factor_performance()
    
    print(f"✓ 市场信息样本: {len(market_info)} 字符")
    print(f"✓ 因子样本: {len(factors_list)} 个")
    print(f"✓ 表现数据: 包含 {len(performance.get('top_factors', []))} 个表现最好的因子")
    
    # 创建 Agent (LLM 不可用时使用备用方案)
    print("\n【执行因子选择】")
    agent = FactorSelectorAgent(llm=None)  # 显式指定 LLM 不可用
    
    result = agent.select_factors(
        date="2026-05-20",
        market_info=market_info,
        factors_list=factors_list,
        factor_performance=performance,
    )
    
    print("✓ 选择完成（使用备用方案）")
    
    # 显示结果
    print("\n【推荐结果】")
    FactorSelectorFramework.print_recommendation(result)
    
    print("\n" + "=" * 80)
    print("离线演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 因子选择框架演示")
    parser.add_argument(
        "--mode",
        choices=["full", "offline"],
        default="offline",
        help="演示模式: full 需要 LLM API，offline 使用备用方案",
    )
    
    args = parser.parse_args()
    
    if args.mode == "offline":
        # 离线模式 - 无需 LLM，直接运行
        demo_without_llm()
    else:
        # 完整模式 - 需要 LLM
        main()
