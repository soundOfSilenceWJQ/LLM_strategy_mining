import os
import pandas as pd
import numpy as np
import json
import hashlib
import re
from typing import Optional
from langchain.schema import HumanMessage, SystemMessage
from optimizer.config import FACTOR_OPTIMIZATION_CONFIG as CONFIG

def generate_factor_identifier(expression: str) -> str:
    """
    从因子表达式生成简短的标识符用于文件名
    
    Args:
        expression: 因子表达式
        
    Returns:
        简化的因子标识符
    """
    # 移除特殊字符，保留字母数字和常见操作符
    clean_exp = "".join(c for c in expression if c.isalnum() or c in ['_', '-'])
    
    # 如果清理后的表达式太长，使用前20个字符加哈希
    if len(clean_exp) > 20:
        # 取前20个字符加上表达式的MD5哈希前8位
        hash_suffix = hashlib.md5(expression.encode()).hexdigest()[:8]
        return f"{clean_exp[:20]}_{hash_suffix}"
    elif len(clean_exp) < 5:
        # 如果太短，使用哈希
        hash_suffix = hashlib.md5(expression.encode()).hexdigest()[:12]
        return f"EXPR_{hash_suffix}"
    else:
        return clean_exp


def parse_factor_score(analysis_text: str) -> Optional[int]:
    """
    从LLM分析结果中提取因子评分

    Args:
        analysis_text: LLM分析结果文本

    Returns:
        评分数值，如果提取失败返回None
    """
    # 使用正则表达式提取评分

    # 主要匹配模式：汉字"评分"后的第一个冒号后的第一个数字
    # 支持 **评分**：50/100、评分：80分、因子评分：90 等格式
    primary_pattern = r"[*]*评分[*]*[：:]\s*(\d+)"
    match = re.search(primary_pattern, analysis_text)

    if match:
        score = int(match.group(1))
        # 验证评分范围
        if 0 <= score <= 100:
            return score

    # 备用匹配模式
    backup_patterns = [
        r"【评分[：:]\s*(\d+)\s*分】",  # 【评分：XX分】格式
        r"得分[：:]\s*(\d+)",          # 得分：XX
        r"综合评分[：:]\s*(\d+)",      # 综合评分：XX
        r"分数[：:]\s*(\d+)",          # 分数：XX
        r"评价[：:]\s*(\d+)",          # 评价：XX
        r"(\d+)\s*分",                 # XX分
        r"(\d+)\s*/\s*100",            # XX/100
    ]

    for pattern in backup_patterns:
        match = re.search(pattern, analysis_text)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score

    return None


def parse_llm_resolve_response(response: str) -> tuple[str, str]:
    """
    解析LLM的因子解析响应，提取格式为 FACTOR_EXPRESSION|FACTOR_REASON 的内容
    
    Args:
        response: LLM的响应文本
        
    Returns:
        tuple[str, str]: (因子表达式, 有效性理由)
    """
    # 首先查找"最终结果"部分
    lines = response.strip().split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
            
        # 若有|，那么匹配|之前的部分和|之后的部分
        # 使用正则表达式匹配管道符前后的内容
        match = re.match(r"^(.*)\|(.*)$", line)
        if match:
            factor_expression = match.group(1).strip()
            factor_reason = match.group(2).strip()
            # 验证表达式不为空且包含基本的qlib语法
            if factor_expression and ('$' in factor_expression or '(' in factor_expression):
                if factor_reason and len(factor_reason) > 10:
                    return factor_expression, factor_reason
    
    # 如果没有找到有效格式，尝试更宽松的正则匹配
    pattern = r'([^|]+?)\s*\|\s*(.+?)(?:\n|$)'
    match = re.search(pattern, response, re.MULTILINE)
    
    if match:
        factor_expression = match.group(1).strip()
        factor_reason = match.group(2).strip()
        
        # 进一步验证表达式的有效性
        if '$' in factor_expression or '(' in factor_expression:
            return factor_expression, factor_reason
    
    # 如果解析失败，返回默认值
    return "Invalid", "解析失败，未找到有效的因子信息格式"


def load_factors_from_csv(csv_file_path, qlib_exp_column="qlib_exp"):
    """
    从CSV文件中读取qlib因子表达式列表

    Parameters:
    csv_file_path: str, CSV文件路径
    qlib_exp_column: str, 包含qlib表达式的列名，默认为'qlib_exp'

    Returns:
    list: qlib因子表达式列表
    """
    try:
        print(f"📂 正在读取CSV文件: {csv_file_path}")

        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"❌ 文件不存在: {csv_file_path}")
            return []

        # 读取CSV文件
        df = pd.read_csv(csv_file_path, encoding="utf-8")
        print(f"✅ 成功读取CSV文件，形状: {df.shape}")

        # 检查列是否存在
        if qlib_exp_column not in df.columns:
            print(f"❌ 列 '{qlib_exp_column}' 不存在于CSV文件中")
            print(f"📋 可用列名: {list(df.columns)}")
            return []

        # 提取qlib表达式列
        exp_series = df[qlib_exp_column]

        # 去除空值
        exp_series = exp_series.dropna()

        # 转换为列表
        exp_list = exp_series.tolist()

        print(f"📊 读取统计:")
        print(f"   - 总行数: {len(df)}")
        print(f"   - 有效表达式数: {len(exp_list)}")
        print(f"   - 空值数: {len(df) - len(exp_list)}")

        if len(exp_list) > 0:
            print(f"📝 前3个表达式预览:")
            for i, exp in enumerate(exp_list[:3]):
                print(f"   {i+1}. {exp}")

            if len(exp_list) > 3:
                print(f"   ... 还有 {len(exp_list) - 3} 个表达式")

        return exp_list

    except Exception as e:
        print(f"❌ 读取CSV文件失败: {str(e)}")
        return []


def ensure_directories():
    """确保必要的目录存在"""
    for dir_key, dir_path in CONFIG["PATHS"].items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 创建目录: {dir_path}")

