"""
因子优化系统的提示词模板
使用LangChain的PromptTemplate进行管理
"""

from langchain.prompts import PromptTemplate

# 因子分析提示词模板
FACTOR_ANALYSIS_TEMPLATE = """你是一位专业的量化投资因子研究专家。请对以下因子的回测指标进行深度分析和评价。

📊 当前因子基本信息：
- 因子名称: {factor_name}
- 因子表达式: {factor_expression}

⏰ 回测区间：{start_date} 至 {end_date}
- 调仓周期：{rebalance_freq}天
🎯 回测股票池：{universe}成分股

📈 回测指标详情：
- IC均值: {ic_mean:.6f}
- IC信息比率: {ic_ir:.6f}
- 排序IC均值: {rank_ic_mean:.6f}
- 排序IC信息比率: {rank_ic_ir:.6f}

请从以下维度进行专业分析：

1. **预测能力评估**
   - 上述指标表现如何？是否显示出稳定的预测能力？
   - 因子是否达到可投资水平？

2. **因子表达式分析**
   - 表达式的逻辑是否合理？
   - 是否有进一步优化的空间？

3. **综合评价**
   - 该因子的整体质量如何？
   - 在实际投资中的应用前景如何？
   - 存在哪些风险和局限性？
   - 结合因子的综合表现以及你对该因子的理解给一个评分（在0-100分之间）

4. **改进建议**
   - 针对当前表现，有哪些具体的优化方向？
   - 可以考虑哪些技术手段来提升因子效果？

请提供专业、客观的分析，并给出具体的改进建议，评分在最后一行以如下格式给出：评分: [x]。（x为你的打分）"""

# 因子优化提示词模板
FACTOR_OPTIMIZATION_TEMPLATE = """你是一位专业的量化投资因子研究专家。现在需要你优化一个股票量化因子。

⏰ 回测设置：
- 回测区间：{start_date} 至 {end_date}
- 调仓周期：{rebalance_freq}天
- 股票池：{universe}成分股

🔍 LLM对当前最佳因子的专业分析评价：
================================
{factor_analysis}
================================

请基于以上专业分析，结合当前最佳因子的优势和不足，制定针对性的优化策略。

当前最佳因子信息：
- 因子名称: {best_factor}
- 因子表达式: {best_expression}
- IC均值: {ic_mean:.6f}
- IC信息比率: {ic_ir:.6f}
- 排序IC均值: {rank_ic_mean:.6f}
- 排序IC信息比率: {rank_ic_ir:.6f}

🎯 优化目标和约束条件：
1. 主要目标：结合你对股票市场的经验，和金融背景知识，提高Rank IC均值和Rank ICIR的绝对值（统一使用绝对值进行比较）
2. 重要约束：Rank IC和Rank ICIR的符号不能改变（为了保证优化后因子和原始因子保持同样的逻辑，若原来是正向因子，则优化后也要是正向因子，反之亦然）
3. 因子性质：不能发生本质改变，应保持原有的经济逻辑
4. 因子可解释性：优化后的因子应具有合理的经济意义

🚨 严格语法约束 - 必须严格遵守！：
1. **只能使用以下基础数据字段**：
   - $open, $close, $high, $low, $volume, $vwap, $factor
   - 禁止使用：$price, $ret, $return, $turnover 等未定义字段

2. **只能使用以下qlib算子（参数格式必须正确）**：
   时间序列：Ref($field, n), Delta($field, n), Delay($field, n)
   统计算子：Mean($field, n), Std($field, n), Max($field, n), Min($field, n), Sum($field, n), Mad($field, n), Med($field, n), Count($field, n), Skew($field, n), Kurt($field, n)
   移动平均：EMA($field, n), WMA($field, n)
   双变量：Corr($field1, $field2, n), Cov($field1, $field2, n)
   排序索引：Rank($field, n), IdxMax($field, n), IdxMin($field, n)
   数学函数：Abs($field), Sign($field), Log($field)
   回归分析：Slope($field, n), Rsquare($field, n), Resi($field, n)

   **新增统计算子详细说明**：
   - Mad($field, n): 计算n个窗口期内的平均绝对偏差(Mean Absolute Deviation)
   - Med($field, n): 计算n个窗口期内的中位数(Median)
   - Count($field, n): 计算n个窗口期内非空值的数量
   - Skew($field, n): 计算n个窗口期内的偏度(Skewness)，衡量数据分布的非对称性
   - Kurt($field, n): 计算n个窗口期内的峰度(Kurtosis)，衡量数据分布的尖锐程度

3. **语法检查清单**：
   ✅ 所有括号必须成对匹配：( 和 ), 检查每个左括号都有对应右括号
   ✅ 参数个数必须正确：如Mean($close, 20)需要2个参数，不能是1个或3个
   ✅ 参数类型必须正确：字段名以$开头，数字参数为正整数
   ✅ 算子名称大小写必须正确：Mean不是mean，Std不是std
   ✅ 字段名必须完全正确：$close不是$Close，$volume不是$Volume

4. **表达式构建步骤**：
   步骤1：列出要用的所有字段和算子
   步骤2：检查每个算子的参数要求
   步骤3：逐步构建表达式，确保每步语法正确
   步骤4：最终检查整个表达式的括号匹配

⚠️ 关键要求：
- 只提供一种最优的优化方案，不要给出多个选择
- 详细解释优化逻辑：为什么这样改进，每一步的技术原理
- 分析预期改进效果并说明理论依据
- **必须进行语法自检**：在给出表达式前，按照上述清单逐项检查

🚫 复杂度限制：
- 表达式不能过于复杂，最多包含3-4个操作符
- 表达式长度不应超过80个字符
- 时间窗口参数建议范围：5-60天

📝 输出格式要求：
请按以下格式输出：

1. **语法自检过程**：
   - 使用的字段：[列出所有$字段]
   - 使用的算子：[列出所有算子及参数个数]
   - 括号检查：[确认左右括号数量相等]
   - 参数检查：[确认每个算子参数正确]

2. **优化分析**：
   - 对原有因子和新因子的对比分析
   - 优化逻辑和技术原理说明
   - 预期改进效果

3. **最终表达式**：
   新因子名称|新因子表达式

❌ 常见错误示例（禁止出现）：
- Mean($close)  # 缺少窗口参数
- mean($close, 20)  # 算子名称小写错误
- Mean($price, 20)  # 使用了不存在的字段
- Mean($close, 20  # 缺少右括号
- Mean($close, 20))  # 多余的右括号

注意，不要生成和原来因子完全相同的因子！请检查新生成的因子和原因子是否完全相同，若相同请重新优化。

"""

FACTOR_OPTIMIZATION_RAG_TEMPLATE = """你是一位专业的量化投资因子研究专家，擅长基于现有知识和研究成果来分析和优化股票因子。

📚 **知识库检索内容**：
以下是从专业知识库中检索到的相关技术和方法：
================================
{context}
================================

⏰ **回测设置**：
- 回测区间：{start_date} 至 {end_date}
- 调仓周期：{rebalance_freq}天
- 股票池：{universe}成分股

🔍 **LLM对当前最佳因子的专业分析评价**：
================================
{factor_analysis}
================================

请基于知识库内容和专业分析，结合当前最佳因子的优势和不足，制定针对性的优化策略。

📊 **当前最佳因子详情**：
- 因子名称: {best_factor}
- 因子表达式: {best_expression}
- IC均值: {ic_mean:.6f}
- IC信息比率: {ic_ir:.6f}
- 排序IC均值: {rank_ic_mean:.6f}
- 排序IC信息比率: {rank_ic_ir:.6f}

🎯 **优化目标和策略指导**：
请充分利用知识库中的相关技术和方法，结合当前因子的表现和分析结果，制定针对性的优化策略：

1. **主要优化目标**：结合知识库内容和你对股票市场的经验，提高Rank IC均值和Rank ICIR的绝对值（统一使用绝对值进行比较）
2. **知识库应用要求**：充分借鉴知识库中提到的优化技术、理论方法和实践经验，进行创新性应用
3. **重要约束**：Rank IC和Rank ICIR的符号不能改变（为了保证优化后因子和原始因子保持同样的逻辑）
4. **因子性质**：不能发生本质改变，应保持原有的经济逻辑和知识库中提到的理论基础
5. **因子可解释性**：优化后的因子应具有合理的经济意义，符合知识库中的理论框架

🚨 **严格语法约束 - 必须严格遵守！**：
1. **只能使用以下基础数据字段**：
   - $open, $close, $high, $low, $volume, $vwap, $factor
   - 禁止使用：$price, $ret, $return, $turnover 等未定义字段

2. **只能使用以下qlib算子（参数格式必须正确）**：
   时间序列：Ref($field, n), Delta($field, n), Delay($field, n)
   统计算子：Mean($field, n), Std($field, n), Max($field, n), Min($field, n), Sum($field, n), Mad($field, n), Med($field, n), Count($field, n), Skew($field, n), Kurt($field, n)
   移动平均：EMA($field, n), WMA($field, n)
   双变量：Corr($field1, $field2, n), Cov($field1, $field2, n)
   排序索引：Rank($field, n), IdxMax($field, n), IdxMin($field, n)
   数学函数：Abs($field), Sign($field), Log($field)
   回归分析：Slope($field, n), Rsquare($field, n), Resi($field, n)

   **新增统计算子详细说明**：
   - Mad($field, n): 计算n个窗口期内的平均绝对偏差(Mean Absolute Deviation)
   - Med($field, n): 计算n个窗口期内的中位数(Median)
   - Count($field, n): 计算n个窗口期内非空值的数量
   - Skew($field, n): 计算n个窗口期内的偏度(Skewness)，衡量数据分布的非对称性
   - Kurt($field, n): 计算n个窗口期内的峰度(Kurtosis)，衡量数据分布的尖锐程度

3. **语法检查清单**：
   ✅ 所有括号必须成对匹配：( 和 ), 检查每个左括号都有对应右括号
   ✅ 参数个数必须正确：如Mean($close, 20)需要2个参数，不能是1个或3个
   ✅ 参数类型必须正确：字段名以$开头，数字参数为正整数
   ✅ 算子名称大小写必须正确：Mean不是mean，Std不是std
   ✅ 字段名必须完全正确：$close不是$Close，$volume不是$Volume

4. **表达式构建步骤**：
   步骤1：列出要用的所有字段和算子
   步骤2：检查每个算子的参数要求
   步骤3：逐步构建表达式，确保每步语法正确
   步骤4：最终检查整个表达式的括号匹配

⚠️ **关键要求**：
- **知识驱动优化**：基于知识库内容进行有针对性的改进，将理论知识与实际应用相结合
- **创新性应用**：在遵循约束的前提下，创新性地应用知识库中的方法和理论
- **详细解释逻辑**：说明如何运用知识库内容，每一步的技术原理和理论依据
- **预期改进效果**：基于知识库理论分析预期的改进效果
- **必须进行语法自检**：在给出表达式前，按照上述清单逐项检查

🚫 **复杂度限制**：
- 表达式不能过于复杂，最多包含4-5个核心操作符
- 表达式长度不应超过100个字符
- 时间窗口参数建议范围：3-60天

📝 **输出格式要求**：
请按以下格式输出：

1. **知识库应用分析**：
   - 从知识库中提取的关键技术或理论方法
   - 这些方法如何适用于当前因子优化
   - 预期的改进机制和理论依据

2. **语法自检过程**：
   - 使用的字段：[列出所有$字段]
   - 使用的算子：[列出所有算子及参数个数]
   - 括号检查：[确认左右括号数量相等]
   - 参数检查：[确认每个算子参数正确]

3. **优化策略详解**：
   - 基于知识库和分析的具体优化思路
   - 对原有因子和新因子的对比分析
   - 优化逻辑和技术原理说明
   - 结合知识库理论的预期改进效果

4. **最终表达式**：
   新因子名称|新因子表达式

❌ **常见错误示例（禁止出现）**：
- Mean($close)  # 缺少窗口参数
- mean($close, 20)  # 算子名称小写错误
- Mean($price, 20)  # 使用了不存在的字段
- Mean($close, 20  # 缺少右括号
- Mean($close, 20))  # 多余的右括号

注意，不要生成和原来因子完全相同的因子。
"""

# 因子解析提示词模板
FACTOR_RESOLVE_TEMPLATE = """你是一位资深的量化投资因子研究专家，精通qlib因子表达式的构建和解释。请对以下因子表达式进行深入分析和解析。

📊 待分析的因子表达式：
{expression}

🎯 **核心目标**：
- **提取本质**：要能提取出因子表达式的核心逻辑和特征，将无关紧要的部分进行简化和删除
- **保持较高相关性**：解析后的表达式与原始表达式的相关性应尽可能高（尽量>0.5）
- **在保证核心原则的前提下尽可能对表达式进行简化**，核心原则是：操作数（$close, $volume等）确定时因子值的序尽可能不变
- **确保语法正确**：严格按照qlib语法规则，避免任何语法错误
- **使得因子的计算值在合理的区间**：不能出现过大、过小的情况（在1e-6到1e6之间）


🚨 严格语法约束 - 必须严格遵守！：
1. **只能使用以下基础数据字段**：
   - $open, $close, $high, $low, $volume, $vwap, $factor
   - 禁止使用：$price, $ret, $return, $turnover 等未定义字段

2. **只能使用以下qlib算子（参数格式必须正确）**：
   时间序列：Ref($field, n), Delta($field, n), Delay($field, n)
   统计算子：Mean($field, n), Std($field, n), Max($field, n), Min($field, n), Sum($field, n), Mad($field, n), Med($field, n), Count($field, n), Skew($field, n), Kurt($field, n)
   移动平均：EMA($field, n), WMA($field, n)
   双变量：Corr($field1, $field2, n), Cov($field1, $field2, n)
   排序索引：Rank($field, n), IdxMax($field, n), IdxMin($field, n)
   数学函数：Abs($field), Sign($field), Log($field)
   回归分析：Slope($field, n), Rsquare($field, n), Resi($field, n)
   其他：
   - Mad($field, n): 计算n个窗口期内的平均绝对偏差，用于衡量数据离散程度，比标准差更稳健
   - Med($field, n): 计算n个窗口期内的中位数，抗异常值，适合捕捉稳定的价格水平
   - Count($field, n): 计算n个窗口期内非空值的数量，可用于检测数据完整性或交易活跃度
   - Skew($field, n): 计算偏度，正值表示右偏(长尾在右)，负值表示左偏，可捕捉价格分布的非对称性
   - Kurt($field, n): 计算峰度，数值越大表示分布越尖锐，可识别价格波动的极端情况

3. **语法检查清单**：
   ✅ 所有括号必须成对匹配：( 和 ), 检查每个左括号都有对应右括号
   ✅ 参数个数必须正确：如Mean($close, 20)需要2个参数，不能是1个或3个
   ✅ 参数类型必须正确：字段名以$开头，数字参数为正整数
   ✅ 算子名称大小写必须正确：Mean不是mean，Std不是std
   ✅ 字段名必须完全正确：$close不是$Close，$volume不是$Volume

4. **表达式构建步骤**：
   步骤1：列出要用的所有字段和算子
   步骤2：检查每个算子的参数要求
   步骤3：逐步构建表达式，确保每步语法正确
   步骤4：最终检查整个表达式的括号匹配

📋 **分析任务**：

1. **表达式结构分析**：
   - 逐步分解表达式的每个组成部分
   - 识别主要算子、数据字段和参数
   - 理解表达式的计算逻辑和数据流

2. **适中优化**：
   - **要求表达式保持等价性**：操作数（$close, $volume等）确定时因子值的序不变
   - **必要时保持原有的复杂度和结构**，避免过度简化


3. **经济含义保持**：
   - 保持因子的核心特征和预测能力
   - 维持原有的投资逻辑和理论基础

⚠️ **关键要求**：
- **优先保持相关性**：解析结果必须与原始表达式高度相关
- **语法准确性**：确保表达式完全符合qlib语法规范


🚫 **复杂度要求**：
- **不设置复杂度限制**：必要时保持原表达式的复杂度

📝 **严格输出格式要求**：
请严格按照以下格式输出：

1. **表达式分析**：
   - 原表达式结构：[描述主要组成部分]

2. **语法自检过程**：
   - 使用的字段：[列出所有$字段]
   - 使用的算子：[列出所有算子及参数个数]
   - 括号检查：[确认左右括号数量相等]
   - 参数检查：[确认每个算子参数正确]

3. **因子解释**：
   - 怎样保持输出的简化因子和原有因子相关性（尽量大于0.5）
   - 因子类型和经济含义（简要说明）
   - 有效性理论依据（简要说明）

4. **最终结果**：
   FACTOR_EXPRESSION|FACTOR_REASON（注意，你的回答有且只能有一个管道符“|”，在这个地方）

其中：
- FACTOR_EXPRESSION: 简化后的因子表达式
- FACTOR_REASON: 因子有效性的理论依据（200-250字）

❌ 常见错误示例（禁止出现）：
- Mean($close)  # 缺少窗口参数
- mean($close, 20)  # 算子名称小写错误
- Mean($price, 20)  # 使用了不存在的字段
- Mean($close, 20  # 缺少右括号
- Mean($close, 20))  # 多余的右括号

✅ 正确示例：
- Mean($close, 20)
- Std($volume, 30) / Mean($volume, 30)
- Delta($close, 1) / Ref($close, 1)
- Mad($close, 20) / Mean($close, 20)  # 稳健的离散度指标
- Med($close, 10) / $close  # 价格相对位置
- Skew($volume, 30)  # 成交量分布偏度
- Kurt($close, 20) / Std($close, 20)  # 标准化峰度

✅ **处理示例**：

**原表达式**：((1.0 - Ref(Corr($high, (($volume - -0.5) + -0.5), 30), 1)) - -0.01)
**分析**：发现($volume - -0.5) + -0.5实际等于$volume，- -0.01和常数项1不影响因子值序的关系
**修正结果**：- Ref(Corr($high, $volume, 30), 1)
**说明**：保证了表达式的等价性，删掉这些常数项不改变因子值序的关系

+-和-+简化为-，--和++简化为+，常数项删除，尽可能运用你的数学直觉和经济学直觉进行表达式化简。
输出示例：
**原表达式**：((1.0 - Ref(Corr($high, (($volume - -0.5) + -0.5), 30), 1)) - -0.01)

1. **表达式分析**：
   - 主要组成部分包括：Ref、Corr、$high、$volume、常数项运算（-0.5、+ -0.5、- -0.01）、外层减法和加法。
   - ($volume - -0.5) + -0.5 实际等价于 $volume，- -0.01 等价于 +0.01，常数项对因子排序无实质影响。

2. **语法自检过程**：
   - 使用的字段：[$high, $volume]
   - 使用的算子：[Corr(3), Ref(2)]
   - 括号检查：[左右括号数量相等]
   - 参数检查：[Corr($high, $volume, 30)参数正确，Ref(Corr(...), 1)参数正确]

3. **因子解释**：
   - 简化后表达式与原表达式高度相关，去除无关常数项不会改变因子值序。
   - 因子类型为量价相关性因子，衡量高价与成交量在30日窗口内的相关性变化。
   - 有效性理论依据：高价与成交量的相关性反映市场活跃度和资金推动效应，相关性变化可捕捉资金流向与价格波动的联动，具有一定的预测能力。

4. **最终结果**：
   -Ref(Corr($high, $volume, 30), 1)|该因子通过考察高价与成交量在30日窗口内的相关性，并对相关性进行滞后处理，能够捕捉市场资金推动价格的变化趋势。高价与成交量的联动反映了市场活跃度和主力资金行为，相关性变化往往预示着价格波动的潜在方向。因子简化后保留了核心逻辑，去除无关常数项，保证了与原表达式的高度相关性（序列排序基本不变），理论上可用于量化选股和市场趋势判断，具备较强的经济解释力和实用性。
"""

# 创建PromptTemplate对象
factor_analysis_prompt = PromptTemplate(
    input_variables=[
        "factor_name", "factor_expression", "start_date", "end_date", "rebalance_freq",
        "universe", "ic_mean", "ic_ir", "rank_ic_mean", "rank_ic_ir"
    ],
    template=FACTOR_ANALYSIS_TEMPLATE
)

factor_optimization_prompt = PromptTemplate(
    input_variables=[
        "start_date", "end_date", "universe", "factor_analysis", "rebalance_freq",
        "best_factor", "best_expression", "ic_mean", "ic_ir",
        "rank_ic_mean", "rank_ic_ir"
    ],
    template=FACTOR_OPTIMIZATION_TEMPLATE
)

factor_optimization_rag_prompt = PromptTemplate(
    input_variables=[
        "context", "start_date", "end_date", "universe", "factor_analysis", "rebalance_freq",
        "best_factor", "best_expression", "ic_mean", "ic_ir",
        "rank_ic_mean", "rank_ic_ir"
    ],
    template=FACTOR_OPTIMIZATION_RAG_TEMPLATE
)

factor_resolve_prompt = PromptTemplate(
    input_variables=["expression"],
    template=FACTOR_RESOLVE_TEMPLATE
)