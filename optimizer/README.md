# AI agent智能因子挖掘系统 (Factor Optimization System)

这是一个基于 AI 和量化金融的因子挖掘系统，结合了 qlib 量化框架、LangChain LLM 集成和 RAG（检索增强生成）技术，用于自动化因子挖掘和优化。

## 系统特性

- **智能因子优化**: 基于 DeepSeek 等大语言模型的因子表达式优化
- **RAG 知识增强**: 集成检索增强生成技术，利用历史知识库优化因子
- **回测验证**: 基于 qlib 框架的专业量化回测
- **工作流管理**: 使用 LangGraph 的状态图工作流
- **多模式运行**: 支持自动模式和交互模式
- **完整日志**: 详细的优化历史和聊天记录

## 目录结构

```
subscribe/
├── main.py                 # 主程序入口
├── factor_miner.py        # 核心因子挖掘引擎
├── config.py              # 系统配置文件
├── prompts.py             # LLM 提示词模板
├── utils.py               # 工具函数
├── chroma_db/             # RAG 知识库向量数据库
└── README.md              # 项目说明文档
```

## 安装配置

### 1. 环境要求

- Python 3.8+
- Windows/Linux/macOS

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 系统依赖

必需的 Python 包（已包含在 requirements.txt 中）：
- `qlib`: 量化金融框架
- `langchain`: LLM 集成框架
- `langgraph`: 工作流状态图
- `pandas`, `numpy`: 数据处理
- `scikit-learn`: 机器学习工具
- `chromadb`: 向量数据库（RAG 功能）

## 配置说明

### 主要配置文件：`config.py`

#### 1. 回测配置
```python
FACTOR_OPTIMIZATION_CONFIG = {
    # 回测时间范围
    "START_DATE": "2020-01-01",
    "END_DATE": "2020-12-31",
    "REBALANCE_FREQ": 20,
    
    # 优化参数
    "MAX_ITERATION": 15,  # 最大优化轮数
    
    # 股票池配置
    "UNIVERSE": "csi300",  # 可选: csi300, csi500, csi800
}
```

#### 2. 数据路径配置
```python
"PATHS": {
    "DATA_SRC_PATH": r"D:\path\to\qlib_data\cn_data_rolling",  # ⚠️ 需要配置实际路径
    "CHAT_LOGS_DIR": "chats",
    "OPTIMIZATION_HISTORIES_DIR": "optimization_histories",
    "RESULTS_DIR": ".",
}
```

**重要**: 必须配置正确的 `DATA_SRC_PATH` 指向您的 qlib 中国数据目录。

#### 3. LLM 配置
```python
"LLM": {
    "API_KEY": "sk-your-api-key-here",          # ⚠️ 需要配置实际 API Key
    "BASE_URL": "https://api.deepseek.com/v1",
    "MODEL_NAME": "deepseek-chat",
    "TEMPERATURE": 0.3,
}
```

**重要**: 需要获取并配置 DeepSeek API Key。

#### 4. RAG 功能配置
```python
# RAG模式是否开启
"RAG_ENABLE": False,  # 设为 True 启用 RAG 功能
```

#### 5. 运行模式配置
```python
"RUNTIME": {
    "AUTO_MODE": True,  # True: 自动模式, False: 交互模式
}
```

#### 6. 初始因子配置
```python
"INITIAL_FACTORS": {
    "STD": "Std($close, 20)/$close",
    # 可添加更多 Alpha158 因子
}
```

## 使用指南

### 1. 基本运行

```bash
cd subscribe
python main.py
```

### 2. 运行模式选择

程序支持多种工作流模式：

#### Workflow 枚举：
- `OPTIMIZE`: 因子优化模式（主要功能）
- `RESOLVE`: 因子解析模式
- `TEST_METRICS`: 指标测试模式
- `CORRELATION`: 相关性分析模式

#### FactorSource 枚举：
- `CONFIG`: 使用配置文件中的初始因子
- `CSV`: 从 CSV 文件读取因子
- `MANUAL`: 手动输入因子

### 3. 配置工作流

在 `main.py` 中修改：
```python
if __name__ == "__main__":
    workflow_type = Workflow.OPTIMIZE      # 选择工作流类型
    factor_source = FactorSource.CONFIG    # 选择因子来源
```

### 4. RAG 功能使用

#### 启用 RAG：
1. 在 `config.py` 中设置 `"RAG_ENABLE": True`
2. 确保 `chroma_db/` 目录存在且包含知识库数据
3. 运行程序时将自动初始化 RAG 系统

#### RAG 知识库：
- 系统会自动加载文档到向量数据库
- 支持 PDF、文本等格式的文档
- 使用 TF-IDF 嵌入进行语义检索

### 5. 输出文件

运行后会生成以下文件：
- `chats/chat_*.txt`: 聊天记录
- `optimization_histories/*.json`: 优化历史
- `*.csv`: 分析结果
- `*.png`: 可视化图表

## 核心组件说明

### 1. Factor 类
```python
@dataclass
class Factor:
    name: str                    # 因子名称
    expression: str             # 因子表达式
    evaluation: FactorEvaluation # 评估结果
    analysis: str               # LLM 分析结果
```

### 2. 工作流状态 (MiningState)
- `factor`: 当前优化的因子对象
- `best_factor`: 历史最佳因子
- `iteration`: 当前迭代次数
- `chat_history`: 对话历史
- `rag`: RAG 检索器实例

### 3. 关键方法
- `_evaluate_factor()`: 因子回测评估
- `_optimize_factor()`: 基础因子优化
- `_optimize_factor_rag()`: RAG 增强因子优化
- `_initialize_rag()`: RAG 系统初始化

## 故障排除

### 常见问题

1. **数据路径错误**
   ```
   错误: qlib 数据加载失败
   解决: 检查并修正 config.py 中的 DATA_SRC_PATH
   ```

2. **API Key 无效**
   ```
   错误: LLM API 调用失败
   解决: 确保 config.py 中配置了有效的 API_KEY
   ```

3. **RAG 初始化失败**
   ```
   错误: 向量数据库加载失败
   解决: 检查 chroma_db/ 目录和文档文件
   ```

4. **依赖包缺失**
   ```bash
   pip install -r requirements.txt
   ```

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的优化策略

1. 在 `factor_miner.py` 中添加新的节点方法
2. 在 `prompts.py` 中定义对应的提示词模板
3. 在状态图中连接新节点

### 自定义因子评估

修改 `_evaluate_factor()` 方法，添加自定义的评估指标。

### 集成新的 LLM

在 `config.py` 中配置新的 LLM 设置，并在相应方法中适配 API 调用。
