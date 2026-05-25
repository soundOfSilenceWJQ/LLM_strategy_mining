# LLM 因子选择框架

基于大语言模型的智能因子选择系统，能够根据市场信息、因子特征和历史表现，推荐最适合当前环境的因子组合。

## 核心组件

### 1. `agent.py` - FactorSelectorAgent
LLM 因子选择 Agent，负责与 LLM 交互和推荐逻辑。

**主要功能:**
- 接收市场信息、因子列表、历史表现数据
- 调用 LLM 进行智能分析
- 返回结构化的因子推荐结果
- 在 LLM 不可用时提供备用推荐

**核心方法:**
```python
agent.select_factors(
    date="2026-05-20",
    market_info="市场信息文本",
    factors_list=[{...}, {...}],
    factor_performance={...}
) -> dict
```

### 2. `selector.py` - FactorSelectorFramework
因子选择框架主类，管理整个流程。

**主要功能:**
- 加载市场信息和因子数据
- 协调 Agent 进行推荐
- 保存和展示结果
- 提供端到端的管道

**核心方法:**
```python
framework = FactorSelectorFramework(base_dir="data")

# 方法1: 分步执行
market_info = framework.load_market_info()
factors_list = framework.load_factors_list()
result = framework.select_factors()

# 方法2: 一次性执行管道
result = framework.run_pipeline(save_output=True)

# 显示结果
FactorSelectorFramework.print_recommendation(result)
```

### 3. `utils.py` - 辅助工具
提供示例数据生成和初始化功能。

**主要函数:**
- `create_sample_market_info()` - 创建示例市场信息
- `create_sample_factors_list()` - 创建示例因子列表
- `create_sample_factor_performance()` - 创建示例表现数据
- `setup_sample_data()` - 一次性设置所有示例数据

### 4. `demo.py` - 演示脚本
可以直接运行的演示程序。

**两种模式:**
- `--mode offline`: 离线模式，不需要 LLM (推荐用于测试)
- `--mode full`: 完整模式，需要配置 LLM API

## 目录结构

```
llm_quant/
  llm_based_selector/
    __init__.py          # 模块入口
    agent.py             # LLM Agent 类
    selector.py          # 主框架类
    utils.py             # 辅助工具
    demo.py              # 演示脚本
    README.md            # 本文件
  data/
    market_info/
      market_summary.txt # 市场信息文本
    factors/
      factors_list.json  # 因子列表
      factor_performance.json  # 因子表现数据
    recommendations/
      recommendation_YYYY-MM-DD_HHMMSS.json  # 推荐结果
```

## 使用方式

### 快速开始 (离线模式，无需 LLM)

```bash
cd llm_quant/llm_based_selector
python demo.py --mode offline
```

这会:
1. 自动生成示例数据
2. 使用备用推荐算法进行因子选择 (无需 LLM)
3. 显示推荐结果
4. 保存为 JSON 文件

### 完整模式 (需要 LLM API)

```bash
python demo.py --mode full
```

需要先配置 LLM:
- 设置环境变量 `ANTHROPIC_API_KEY` 或 `LLM_API_KEY`
- 或修改 `config.py` 中的 API Key 配置

### 在代码中使用

```python
from llm_quant.llm_based_selector import FactorSelectorFramework

# 初始化框架
framework = FactorSelectorFramework(base_dir="data")

# 执行因子选择
result = framework.run_pipeline(
    date="2026-05-20",
    market_info_file="market_summary.txt",
    factors_list_file="factors_list.json",
    performance_file="factor_performance.json",
    save_output=True
)

# 显示结果
FactorSelectorFramework.print_recommendation(result)

# 输出结果已保存到: data/recommendations/recommendation_2026-05-20_HHMMSS.json
```

## 数据格式

### 市场信息文件 (market_summary.txt)

纯文本格式，包含:
- 宏观经济指标 (GDP、CPI、M2 等)
- 政策动向
- 股市情况
- 板块热点
- 市场风格
- 风险提示

示例:
```
【2026年5月20日 市场信息汇总】

【宏观经济背景】
- GDP 增速: 5.2% (同比)
- CPI 涨幅: 2.3% (同比)
...

【市场风格】
- 当前为增长风格主导 (成长 > 价值)
...
```

### 因子列表 (factors_list.json)

JSON 数组格式:
```json
[
  {
    "name": "momentum_10",
    "category": "动量因子",
    "description": "10日动量因子",
    "formula": "close[-1] / close[-10] - 1",
    "ic": 0.045,
    "type": "price_based",
    "lookback": 10
  },
  ...
]
```

**字段说明:**
- `name`: 因子名称 (唯一标识)
- `category`: 因子分类 (动量/价值/成长/宏观等)
- `description`: 因子描述
- `formula`: 因子公式 (可选)
- `ic`: 历史 IC 值 (可选)
- `type`: 因子类型
- `frequency`: 更新频率 (daily/weekly/monthly/quarterly)

### 因子表现数据 (factor_performance.json)

JSON 对象格式:
```json
{
  "top_factors": [
    {
      "name": "earnings_surprise",
      "ic": 0.055,
      "annual_return": 0.185,
      "sharpe": 0.92,
      "max_drawdown": -0.22,
      "recent_ic_trend": "上升"
    },
    ...
  ],
  "bottom_factors": [...],
  "market_style": "增长风格主导",
  "factor_correlation": {...}
}
```

### 推荐结果 (recommendation_*.json)

JSON 对象格式:
```json
{
  "date": "2026-05-20",
  "recommendation": {
    "market_analysis": {
      "current_regime": "增长主导",
      "dominant_factors": [...],
      "risk_factors": [...],
      "macroeconomic_outlook": "..."
    },
    "recommended_factors": [
      {
        "factor_name": "earnings_surprise",
        "rationale": "推荐原因",
        "historical_ic": 0.055,
        "weight": 0.25,
        "risk_level": "低"
      },
      ...
    ],
    "portfolio_composition": {...},
    "factors_to_avoid": [...],
    "implementation_suggestions": {...},
    "confidence_level": "高",
    "uncertainty_sources": [...],
    "next_review_date": "2026-06-20"
  },
  "processed_at": "2026-05-20T14:30:45"
}
```

## 工作流程

```
┌─────────────────────────────────────────┐
│ 1. 准备输入数据                          │
│    ├─ market_summary.txt                │
│    ├─ factors_list.json                 │
│    └─ factor_performance.json (可选)    │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 2. 初始化 FactorSelectorFramework       │
│    framework = FactorSelectorFramework() │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 3. 加载数据                              │
│    ├─ load_market_info()               │
│    ├─ load_factors_list()              │
│    └─ load_factor_performance()        │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 4. 调用 FactorSelectorAgent             │
│    agent.select_factors(...)            │
└─────────────────┬───────────────────────┘
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
   ┌─────────┐      ┌─────────────────┐
   │ LLM可用 │      │ LLM不可用       │
   └────┬────┘      └────────┬────────┘
        ↓                    ↓
   ┌──────────┐      ┌──────────────┐
   │LLM分析   │      │备用推荐      │
   └────┬─────┘      └────┬─────────┘
        └──────┬─────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. 返回推荐结果                         │
│    {                                    │
│      "date": "2026-05-20",             │
│      "recommendation": {...},           │
│      "processed_at": "..."             │
│    }                                    │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 6. 保存和显示结果                       │
│    ├─ save_recommendation()             │
│    ├─ print_recommendation()            │
│    └─ 输出到 recommendations/           │
└─────────────────────────────────────────┘
```

## LLM 提示词设计

系统提示词强调:
- 身份: 资深量化因子研究员和宏观策略分析师
- 任务: 基于市场信息和因子特征进行推荐
- 方法: 逻辑、数据驱动、市场常识
- 格式: 必须输出有效 JSON

用户提示词包含:
- 当前日期和市场情况摘要
- 样本统计 (因子数量等)
- 完整的大盘信息
- 所有可用因子列表
- 历史表现数据 (Top/Bottom 因子)
- 输出要求和 JSON Schema

## 扩展建议

### 1. 集成更多数据源
- 实时市场数据 API
- 大盘指数走势
- 宏观经济数据库
- 基金持仓变化

### 2. 增强 LLM 提示词
- 添加市场风格指标
- 包含因子间相关性分析
- 融入风险管理规则
- 参考历史表现模式

### 3. 评估和反馈
- 追踪推荐因子的实际表现
- 计算推荐的 IC 和收益
- 定期审视推荐准确度
- 优化提示词和算法

### 4. 自动化集成
- 定时任务自动生成推荐
- 邮件/通知推送
- 可视化仪表板
- 与交易系统集成

## 常见问题

**Q: 没有 LLM API 如何使用?**  
A: 使用 `--mode offline` 运行演示或在代码中传入 `llm=None`，系统会使用备用推荐算法。

**Q: 市场信息文件如何获取?**  
A: 可以手动编写，或集成新闻 API、宏观数据库等自动生成。

**Q: 因子列表的字段是否可以自定义?**  
A: 可以，系统会提取相关字段进行分析，其他字段会被忽略。

**Q: 如何改进推荐质量?**  
A: 增加更多历史表现数据、细化市场信息、优化 LLM 提示词。

**Q: 能否用于实盘交易?**  
A: 当前版本用于研究和决策支持，实盘应该谨慎验证和风险控制。

## 联系方式

如有问题或建议，欢迎反馈。
