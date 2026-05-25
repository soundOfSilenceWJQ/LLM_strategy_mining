from utils import load_factors_from_csv
import pandas as pd
from factor_miner import FactorMiner, CONFIG
from testing import test_factors_metrics, calc_corr
from enum import Enum

# 构造二选一枚举类型
class FactorSource(Enum):
    CSV = "CSV"
    TEXT = "TEXT"

class Workflow(Enum):
    OPTIMIZE = "OPTIMIZE"
    RESOLVE = "RESOLVE"
    TEST_METRICS = "TEST_METRICS"
    CORRELATION = "CORRELATION"

factor_src = FactorSource.TEXT
workflow = Workflow.TEST_METRICS

if __name__ == "__main__":
    if factor_src == FactorSource.CSV:
        # 从CSV文件加载因子表达式
        # csv_file_path = r"D:\summer2025\AlphaForge\AlphaForge-master\AlphaForge-master\out\test_csi300_2020_0\csv_zoo_final_with_qlib.csv"
        csv_file_path = "csv_zoo_with_qlib.csv"
        exp_list = load_factors_from_csv(csv_file_path, qlib_exp_column="qlib_exprs")
        if len(exp_list) > 0:
            # 构造名称列表
            factor_names = [f"FACTOR_{i+1}" for i in range(len(exp_list))]
            factor_dict = dict(zip(factor_names, exp_list))

    if factor_src == FactorSource.TEXT:
        # 从配置中加载初始因子
        factor_dict = CONFIG["INITIAL_FACTORS"]

    if workflow == Workflow.OPTIMIZE:
        # 优化因子
        print(f"🚀 开始批量优化...")
        best_factor_df = pd.DataFrame(columns=["name", "expression", "ic_mean", "ic_ir", 
                                               "rank_ic_mean", "rank_ic_ir"
                                               ])
        
        for item in factor_dict.items():
            optimizer = FactorMiner()
            factor_name, factor_expression = item
            if factor_expression:
                opti_result = optimizer.optimize_factor(
                    initial_factor_name=factor_name,
                    initial_expression=factor_expression,
                )

                best_eval = opti_result.best_factor.evaluation
                # 将结果添加到DataFrame
                new_row = pd.DataFrame([{
                    "name": factor_name,
                    "expression": factor_expression,
                    "ic_mean": best_eval.ic_mean,
                    "ic_ir": best_eval.ic_ir,
                    "rank_ic_mean": best_eval.rank_ic_mean,
                    "rank_ic_ir": best_eval.rank_ic_ir,
                }])
                best_factor_df = pd.concat([best_factor_df, new_row], ignore_index=True)

        best_factor_df.to_csv("best_factors.csv", index=False)

    if workflow == Workflow.RESOLVE:
        # 解析表达式
        resolved_df = pd.DataFrame(columns=["No.", "expression", "reason"])
        resolver = FactorMiner()
        
        cnt = 0
        total_factors = len(factor_dict)
        
        for factor_name, factor_expression in factor_dict.items():
            print(f"\n[{cnt+1}/{total_factors}] 解析因子: {factor_name}")
            
            resolved_expression, resolved_reason = resolver.resolve_factor(factor_expression)

            # 保存解析结果到CSV文件
            resolved_df_single = pd.DataFrame.from_dict({
                "No.": [cnt],
                "expression": [resolved_expression],
                "reason": [resolved_reason]
            })
            cnt += 1
            resolved_df = pd.concat([resolved_df, resolved_df_single], ignore_index=True)
        
        
        output_file = "resolved_factors_new3.csv"
        resolved_df.to_csv(output_file, index=False)
        print(f"   - 结果保存到: {output_file}")
        print(f"{'='*80}")
        

    if workflow == Workflow.TEST_METRICS:
        # 回测因子
        print(f"🚀 开始批量测试...")
        test_factors_metrics(factor_dict, 'alpha158(all)_res.csv')

    if workflow == Workflow.CORRELATION:
        resolved_df = pd.read_csv("Your CSV File Path with Resolved Factors")
        original_df = pd.read_csv("Your CSV File Path with Original Factors")
        merged_df = pd.merge(original_df, resolved_df, left_on="No.", right_on="No.", how='right')
        print(merged_df)
        
        # 计算qlib_exprs和expression列之间的相关性
        print("\n" + "="*80)
        print("🔍 开始计算因子表达式相关性分析")
        print("="*80)
        
        correlation_results = []
        
        for index, row in merged_df.iterrows():
            try:
                qlib_expr = row['qlib_exprs']
                expression = row['expression']
                no = row['No.']
                
                # 跳过空值
                if pd.isna(qlib_expr) or pd.isna(expression):
                    print(f"⚠️ No.{no}: 跳过空值表达式")
                    continue
                
                print(f"\n📊 No.{no} 相关性分析:")
                print(f"   原始表达式: {qlib_expr}")
                print(f"   解析表达式: {expression}")
                
                # 计算相关性
                correlation = calc_corr(str(qlib_expr), str(expression))
                
                correlation_results.append({
                    'No.': no,
                    'qlib_exprs': qlib_expr,
                    'expression': expression,
                    'correlation': correlation
                })
                
                if not pd.isna(correlation):
                    print(f"   🎯 相关系数: {correlation:.6f}")
                    if abs(correlation) > 0.9:
                        print(f"   ✅ 高度相关 (|r| > 0.9)")
                    elif abs(correlation) > 0.7:
                        print(f"   📈 中度相关 (|r| > 0.7)")
                    elif abs(correlation) > 0.3:
                        print(f"   📉 低度相关 (|r| > 0.3)")
                    else:
                        print(f"   ❌ 相关性较低 (|r| ≤ 0.3)")
                else:
                    print(f"   ❌ 计算失败")
                    
            except Exception as e:
                print(f"❌ No.{no} 计算失败: {str(e)}")
                correlation_results.append({
                    'No.': no,
                    'qlib_exprs': qlib_expr if 'qlib_expr' in locals() else 'N/A',
                    'expression': expression if 'expression' in locals() else 'N/A',
                    'correlation': float('nan')
                })
        
        # 保存相关性分析结果
        correlation_df = pd.DataFrame(correlation_results)
        output_file = "factor_correlation_analysis.csv"
        correlation_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n📊 相关性分析汇总:")
        print(f"   - 总因子对数: {len(correlation_results)}")
        
        valid_correlations = correlation_df[~pd.isna(correlation_df['correlation'])]
        if len(valid_correlations) > 0:
            print(f"   - 成功计算: {len(valid_correlations)}")
            print(f"   - 失败计算: {len(correlation_results) - len(valid_correlations)}")
            print(f"   - 平均相关系数: {valid_correlations['correlation'].mean():.6f}")
            print(f"   - 最高相关系数: {valid_correlations['correlation'].max():.6f}")
            print(f"   - 最低相关系数: {valid_correlations['correlation'].min():.6f}")
            
            high_corr = valid_correlations[valid_correlations['correlation'].abs() > 0.9]
            print(f"   - 高度相关对数 (|r|>0.9): {len(high_corr)}")
        
        print(f"   - 结果已保存到: {output_file}")
        print("="*80)
