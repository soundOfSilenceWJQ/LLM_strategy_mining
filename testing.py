
import os
import datetime
import numpy as np
import pandas as pd
from qlib.data.dataset.loader import QlibDataLoader
from factor_miner import FactorMiner, CONFIG


def test_factors_metrics(
    factor_dict, output_file
):
    """
    批量测试因子表达式列表并保存结果到CSV文件

    Parameters:
    factor_dict: dict, 因子表达式字典，键为因子名称，值为因子表达式
    output_file: str, 输出CSV文件路径，默认为None时自动生成文件名
    factor_name_prefix: str, 因子名称前缀，默认为"FACTOR"
    rebalance_freq: int, 调仓频率（天数），1表示每日调仓，21表示每21天调仓一次

    Returns:
    pd.DataFrame: 包含所有因子测试结果的DataFrame
    """
    rebalance_freq = CONFIG['REBALANCE_FREQ']
    print("🚀 开始批量因子测试...")
    print(f"📝 待测试因子数量: {len(factor_dict)}")
    print(f"🔄 调仓频率: {rebalance_freq}天")

    # 初始化优化器
    optimizer = FactorMiner()

    # 存储结果的列表
    results = []

    for factor_name, expression in factor_dict.items():
        print(f"\n{'='*60}")
        print(f"🔍 测试因子: {factor_name}")
        print(f"📝 表达式: {expression}")
        print(f"🔄 调仓频率: {rebalance_freq}天")
        print(f"{'='*60}")

        try:
            # 回测因子
            evaluation = optimizer.evaluate_factor(expression, factor_name)

            # 整理结果
            result_dict = {
                "factor_name": factor_name,
                "expression": expression,
                "rebalance_freq": rebalance_freq,
                "ic_mean": evaluation.ic_mean,
                "ic_ir": evaluation.ic_ir,
                "rank_ic_mean": evaluation.rank_ic_mean,
                "rank_ic_ir": evaluation.rank_ic_ir,
                "status": "SUCCESS",
            }

            # 打印关键指标
            print(f"✅ 测试成功")
            print(f"📊 关键指标:")
            print(f"   - IC均值: {evaluation.ic_mean:.6f}")
            print(f"   - IC信息比率: {evaluation.ic_ir:.6f}")
            print(f"   - 排序IC均值: {evaluation.rank_ic_mean:.6f}")
            print(f"   - 排序IC信息比率: {evaluation.rank_ic_ir:.6f}")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 测试失败: {error_msg}")

            # 记录错误结果
            result_dict = {
                "factor_name": factor_name,
                "expression": expression,
                "rebalance_freq": rebalance_freq,
                "ic_mean": np.nan,
                "ic_ir": np.nan,
                "rank_ic_mean": np.nan,
                "rank_ic_ir": np.nan,
                "status": f"ERROR: {error_msg}",
            }

        results.append(result_dict)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"factor_test_results_freq{rebalance_freq}_{timestamp}.csv"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    try:
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n✅ 测试结果已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️ 保存文件失败: {e}")
        # 尝试保存到当前目录
        fallback_file = f"factor_test_results_freq{rebalance_freq}_fallback_{datetime.datetime.now().strftime('%H%M%S')}.csv"
        try:
            results_df.to_csv(fallback_file, index=False, encoding="utf-8-sig")
            print(f"✅ 已保存到备用文件: {fallback_file}")
        except Exception as e2:
            print(f"❌ 备用保存也失败: {e2}")

    # 打印汇总统计
    print(f"\n📊 测试汇总 (调仓频率: {rebalance_freq}天):")
    print(f"   - 总因子数: {len(factor_dict)}")

    success_count = len(results_df[results_df["status"] == "SUCCESS"])
    error_count = len(results_df[results_df["status"] != "SUCCESS"])

    print(f"   - 成功测试: {success_count}")
    print(f"   - 失败测试: {error_count}")

    if success_count > 0:
        # 统计成功因子的表现
        success_df = results_df[results_df["status"] == "SUCCESS"]

        print(f"\n🏆 表现最佳因子 (按Rank IC绝对值排序):")
        best_factors = success_df.loc[success_df["rank_ic_mean"].abs().nlargest(3).index]

        for idx, row in best_factors.iterrows():
            print(
                f"   {row['factor_name']}: Rank IC={row['rank_ic_mean']:.6f}, Rank IR={row['rank_ic_ir']:.6f}"
            )

        print(f"\n📈 整体统计 (仅成功因子):")
        print(
            f"   - IC均值范围: [{success_df['ic_mean'].min():.6f}, {success_df['ic_mean'].max():.6f}]"
        )
        print(f"   - IC绝对值平均: {success_df['ic_mean'].abs().mean():.6f}")
        print(
            f"   - ICIR绝对值平均: {success_df['ic_ir'].abs().mean():.6f}"
        )
        print(f"   - rank IC绝对值平均: {success_df['rank_ic_mean'].abs().mean():.6f}")
        

    print(f"\n🎉 批量测试完成!")

    return results_df

def calc_corr(exp1: str, exp2: str) -> float:
    """
    计算两个因子表达式之间的相关度
    
    Parameters:
    exp1: str, 第一个因子表达式
    exp2: str, 第二个因子表达式
    
    Returns:
    float: 两个因子的相关系数，失败时返回NaN
    """
    try:
        print(f"🔍 计算因子相关度:")
        print(f"   因子1: {exp1}")
        print(f"   因子2: {exp2}")
        
        # 配置数据加载器
        fields = [exp1, exp2]
        names = ["FACTOR1", "FACTOR2"]
        data_loader_config = {
            "feature": (fields, names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)  # type: ignore
        
        # 加载数据
        df = data_loader.load(
            instruments=CONFIG["UNIVERSE"],
            start_time=CONFIG["START_DATE"],
            end_time=CONFIG["END_DATE"],
        )
        
        # 确定列名
        factor1_col = ("feature", "FACTOR1")
        factor2_col = ("feature", "FACTOR2")
        
        # 计算整体相关度
        factor1_data = df[factor1_col].dropna()
        factor2_data = df[factor2_col].dropna()
        
        # 确保两个因子数据长度一致
        common_index = factor1_data.index.intersection(factor2_data.index)
        factor1_aligned = factor1_data.reindex(common_index)
        factor2_aligned = factor2_data.reindex(common_index)
        
        if len(factor1_aligned) == 0 or len(factor2_aligned) == 0:
            print("⚠️ 没有有效的重叠数据点")
            return float('nan')
        
        # 计算皮尔逊相关系数
        print(f"📊 开始计算相关系数...")
        
        # 检查数据中的特殊值
        factor1_finite = factor1_aligned[np.isfinite(factor1_aligned)]
        factor2_finite = factor2_aligned[np.isfinite(factor2_aligned)]
        
        print(f"   - factor1中有限值数量: {len(factor1_finite)}")
        print(f"   - factor2中有限值数量: {len(factor2_finite)}")
        print(f"   - factor1中无穷值数量: {np.isinf(factor1_aligned).sum()}")
        print(f"   - factor2中无穷值数量: {np.isinf(factor2_aligned).sum()}")
        print(f"   - factor1中NaN数量: {factor1_aligned.isna().sum()}")
        print(f"   - factor2中NaN数量: {factor2_aligned.isna().sum()}")
        
        # 显示一些样本值
        print(f"   - factor1前5个值: {factor1_aligned.head().tolist()}")
        print(f"   - factor2前5个值: {factor2_aligned.head().tolist()}")
        
        # 检查方差
        factor1_var = factor1_aligned.var()
        factor2_var = factor2_aligned.var()
        print(f"   - factor1方差: {factor1_var}")
        print(f"   - factor2方差: {factor2_var}")
        
        correlation = factor1_aligned.corr(factor2_aligned, method="pearson")
        
        print(f"📊 相关度计算结果:")
        print(f"   - 有效数据点: {len(factor1_aligned)}")
        print(f"   - 皮尔逊相关系数: {correlation:.6f}")
        
        return correlation
        
    except Exception as e:
        error_msg = f"因子相关度计算失败: {str(e)}"
        print(f"❌ {error_msg}")
        return float('nan')