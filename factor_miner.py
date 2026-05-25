import math
import os
import qlib
from qlib.data.dataset.loader import QlibDataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # 提供替代的简单工作流
    class SimpleWorkflow:
        def __init__(self, func):
            self.func = func
        def invoke(self, state, config=None):
            return self.func(state)
import warnings
from config import FACTOR_OPTIMIZATION_CONFIG as CONFIG
from call_ollama import call_ollama
from prompts import (
    factor_analysis_prompt,
    factor_optimization_prompt,
    factor_optimization_rag_prompt,
    factor_resolve_prompt,
)
from utils import (
    generate_factor_identifier,
    parse_factor_score,
    parse_llm_resolve_response,
    ensure_directories,
)
import re

# RAG相关导入
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

import logging

# 设置根日志级别
logging.basicConfig(level=logging.WARNING)


# 获取当前脚本的目录
data_path = CONFIG["PATHS"]["DATA_SRC_PATH"]

# 初始化qlib
qlib.init(provider_uri=data_path, logging_level=logging.WARNING)


class Factor:
    def __init__(self, name: str, expression: str, evaluation: Optional['FactorEvaluation'] = None):
        self.name = name
        self.expression = expression
        self.meaning = ""
        self.evaluation = evaluation or FactorEvaluation(ic_mean=0.0, ic_ir=0.0, rank_ic_mean=0.0, rank_ic_ir=0.0)
        self.analysis = ""

    def __str__(self) -> str:
        return f"Factor(name={self.name}, expression={self.expression})"
        
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class FactorEvaluation:
    """因子评价结果"""
    ic_mean: float
    ic_ir: float
    rank_ic_mean: float
    rank_ic_ir: float


@dataclass
class MiningState:
    current_factor: Factor
    best_factor: Factor
    iteration_count: int
    optimization_history: List[Dict] = field(default_factory=list)  # 优化历史记录
    last_error: str = ""  # 记录最后一次计算错误信息
    constraint_violated: bool = False  # 是否违反约束条件


class Logger:
    def __init__(self, log_dir: str):
        self.log_dir: str = log_dir
        self._message_buffer: List[str] = []
        self._buffer_size_limit = 5  # 缓冲区大小限制
    
    def __del__(self):
        """析构函数，确保程序结束时缓冲区内容被写入"""
        try:
            self.flush_buffer()
        except:
            pass  # 忽略析构时的异常
    
    def flush_buffer(self):
        """
        将缓冲区中的消息批量写入文件
        """
        try:
            with open(self.chat_log_file, "a", encoding="utf-8") as f:
                for msg in self._message_buffer:
                    f.write(msg + "\n")
            self._message_buffer.clear()
        except Exception as e:
            # 只在第一次失败时打印警告，避免大量错误消息
            if not hasattr(self, "_write_error_logged"):
                print(f"⚠️ 写入聊天记录失败: {str(e)}")
                self._write_error_logged = True
            self._message_buffer.clear()  # 清空缓冲区避免内存泄漏

    def initialize_log_file(self, initial_factor_name: str):
        """初始化聊天记录文件，文件名包含初始因子名称"""
        # 生成时间戳
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.optimization_timestamp = timestamp

        # 清理因子名称，确保可以用作文件名
        safe_factor_name = "".join(
            c for c in initial_factor_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_factor_name = safe_factor_name.replace(" ", "_")

        # 设置聊天记录文件路径，包含因子名称
        chat_filename = f"chat_{timestamp}_{safe_factor_name}.txt"
        self.chat_log_file = os.path.join(
            self.log_dir, chat_filename
        )

        with open(self.chat_log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AI agent因子挖掘记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"开始时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初始因子: {initial_factor_name}\n")
            if not CONFIG["USE_OLLAMA"]:
                f.write(f"模型: {CONFIG['LLM']['MODEL_NAME']}\n")
            else:
                f.write(f"模型: ollama {CONFIG['OLLAMA']['MODEL_NAME']}\n")
            f.write("=" * 80 + "\n\n")
        print(f"📝 聊天记录将保存到: {self.chat_log_file}")
        
    def log(self, msg: str):
        """
        打印消息并批量写入聊天记录文件

        Args:
            msg: 要输出的消息字符串
        """
        # 打印消息（截断长消息）
        if len(msg) > 300:
            print(msg[:300] + "...")
        else:
            print(msg)

        # 将消息添加到缓冲区
        if hasattr(self, 'chat_log_file') and self.chat_log_file:
            self._message_buffer.append(msg)

            # 当缓冲区满时，批量写入文件
            if len(self._message_buffer) >= self._buffer_size_limit:
                self.flush_buffer()


class FactorMiner:
    def __init__(self):

        self.llm = ChatOpenAI(
            api_key=CONFIG["LLM"]["API_KEY"],
            base_url=CONFIG["LLM"]["BASE_URL"],
            model=CONFIG["LLM"]["MODEL_NAME"],
            temperature=CONFIG["LLM"]["TEMPERATURE"],
        )

        # 用vwap计算换仓收益
        self.labels = [
            "Ref($vwap, -{rebalance_freq}-1)/Ref($vwap, -1) - 1".format(
                rebalance_freq=CONFIG["REBALANCE_FREQ"]
            )
        ]
        self.label_names = ["LABEL"]

        # 确保必要的目录存在
        ensure_directories()

        self.optimization_timestamp: str = ""

        # 构建langgraph工作流
        self.workflow = self._build_workflow()
        self.chat1 = Logger(log_dir=CONFIG["PATHS"]["CHAT_LOGS_DIR"])
        if CONFIG["SUMMARY_ENABLE"]:
            self.chat2 = Logger(log_dir=CONFIG["PATHS"]["SUMMARY_LOGS_DIR"])
        
        if CONFIG["RAG_ENABLE"]:
            # 初始化RAG组件
            self._initialize_rag()


    def __del__(self):
        """
        析构函数，确保缓冲区被清空
        """
        if hasattr(self, 'chat1'):
            self.chat1.flush_buffer()
        if CONFIG["SUMMARY_ENABLE"] and hasattr(self, 'chat2'):
            self.chat2.flush_buffer()


    def _initialize_rag(self):
        """初始化RAG知识库检索系统 - 简化版本"""
        if not RAG_AVAILABLE:
            print("⚠️ RAG 组件不可用，跳过初始化")
            self.rag_retriever = None
            return
            
        try:
            knowledge_path = CONFIG["PATHS"].get("KNOWLEDGE_PATH")
            
            if not os.path.exists(knowledge_path):
                print(f"⚠️ 知识库路径不存在: {knowledge_path}")
                self.rag_retriever = None
                return
            
            # 1. 加载文档
            loader = DirectoryLoader(
                knowledge_path,
                glob="**/*.txt", 
                loader_cls=TextLoader, 
                loader_kwargs={"encoding": "utf-8"}
            )
            raw_docs = loader.load()
            
            if not raw_docs:
                print("⚠️ 知识库中没有找到文档")
                self.rag_retriever = None
                return
            
            # 2. 分块
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = text_splitter.split_documents(raw_docs)
            
            # 简化的 RAG 实现 - 基于 TF-IDF 相似度搜索
            self.rag_docs = docs
            self.rag_texts = [doc.page_content for doc in docs]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.rag_texts)
            
            # 简单的检索器
            class SimpleTfidfRetriever:
                def __init__(self, texts, tfidf_matrix, vectorizer, top_k=3):
                    self.texts = texts
                    self.tfidf_matrix = tfidf_matrix
                    self.vectorizer = vectorizer
                    self.top_k = top_k
                
                def invoke(self, query):
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np
                    
                    # 查询向量化
                    query_vec = self.vectorizer.transform([query])
                    
                    # 计算相似度
                    similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                    
                    # 获取最相似的文档
                    top_indices = np.argsort(similarities)[-self.top_k:][::-1]
                    
                    # 返回结果（模拟 langchain 的格式）
                    results = []
                    for idx in top_indices:
                        results.append(type('Doc', (), {'page_content': self.texts[idx]})())
                    return results
            
            self.rag_retriever = SimpleTfidfRetriever(
                self.rag_texts, self.tfidf_matrix, self.tfidf_vectorizer
            )
            
            # 4. 文档格式化函数
            def format_docs(docs):
                return "\n".join([doc.page_content for doc in docs])
            
            self.format_docs = format_docs
            
            print(f"✅ 简化版 RAG 知识库初始化成功，加载了 {len(docs)} 个文档片段")
            
        except Exception as e:
            print(f"❌ RAG初始化失败: {str(e)}")
            self.rag_retriever = None
    

    def call_llm(self, prompt, sys_msg):
        """调用LLM获取响应"""
        if not CONFIG["USE_OLLAMA"]:
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            # 确保response.content是字符串类型
            return str(response.content) if response.content else "响应内容为空"
        else:
            return call_ollama(sys_msg, prompt)


    def evaluate_factor(
        self,
        factor_expression: str,
        factor_name: str,
        state: Optional[MiningState] = None,
    ) -> FactorEvaluation:
        """
        对因子进行计算和回测
        """
        try:
            # 配置数据加载器
            fields = [factor_expression]
            names = [factor_name]
            data_loader_config = {
                "feature": (fields, names),
                "label": (self.labels, self.label_names),
            }
            data_loader = QlibDataLoader(config=data_loader_config)  # type: ignore

            # 加载数据
            df = data_loader.load(
                instruments=CONFIG["UNIVERSE"],
                start_time=CONFIG["START_DATE"],
                end_time=CONFIG["END_DATE"],
            )

            # 确定列名
            feature_col = ("feature", factor_name)
            label_col = ("label", "LABEL")

            # 计算IC指标
            daily_ic = (
                df.groupby(level="datetime")
                .apply(lambda x: x[feature_col].corr(x[label_col], method="pearson"))
                .dropna()
            )

            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std()
            ic_ir = ic_mean / ic_std if ic_std != 0 else 0

            daily_rank_ic = (
                df.groupby(level="datetime")
                .apply(lambda x: x[feature_col].corr(x[label_col], method="spearman"))
                .dropna()
            )
            rank_ic_mean = daily_rank_ic.mean()
            rank_ic_ir = (
                rank_ic_mean / daily_rank_ic.std() if daily_rank_ic.std() != 0 else 0
            )

            # 清空错误信息，表示计算成功
            if state is not None:
                state.last_error = ""

            return FactorEvaluation(
                ic_mean=ic_mean,
                ic_ir=ic_ir,
                rank_ic_mean=rank_ic_mean,
                rank_ic_ir=rank_ic_ir,
            )

        except Exception as e:
            error_msg = f"因子 {factor_name} 回测时出错: {str(e)}"
            # 打印最后100个字符
            print(f"⚠️ {error_msg[-100:]}")
            # 保存错误信息到状态中
            if state is not None:
                state.last_error = f"因子表达式计算错误: {str(e)}"

            return FactorEvaluation(
                ic_mean=0.0,
                ic_ir=0.0,
                rank_ic_mean=0.0,
                rank_ic_ir=0.0,
            )


    def generate_optimization_prompt(self, state: MiningState) -> str:
        """
        生成优化提示词，使用模板化方法
        """
        # 使用最佳因子的信息来生成优化提示词
        best_evaluation = state.best_factor.evaluation
        best_factor_analysis = state.best_factor.analysis or ""

        # 使用模板化的提示词
        prompt = factor_optimization_prompt.format(
            start_date=CONFIG["START_DATE"],
            end_date=CONFIG["END_DATE"],
            rebalance_freq=CONFIG["REBALANCE_FREQ"],
            universe=CONFIG["UNIVERSE"],
            factor_analysis=best_factor_analysis,
            best_factor=state.best_factor.name,
            best_expression=state.best_factor.expression,
            ic_mean=best_evaluation.ic_mean,
            ic_ir=best_evaluation.ic_ir,
            rank_ic_mean=best_evaluation.rank_ic_mean,
            rank_ic_ir=best_evaluation.rank_ic_ir,
        )

        return prompt

    def generate_optimization_prompt_rag(self, state: MiningState) -> str:
        if self.rag_retriever is not None:
            # 构造查询问题用于RAG检索
            query = f"""如何优化因子 {state.best_factor.name}？
    当前因子表达式: {state.best_factor.expression}
    因子含义：{state.best_factor.meaning}，因子分析：{state.best_factor.analysis}
    当前表现: IC均值={state.best_factor.evaluation.ic_mean:.6f}, 排序IC均值={state.best_factor.evaluation.rank_ic_mean:.6f}
    需要提升因子的有效性和预测能力。"""
            
            try:
                # 从知识库检索相关内容
                retrieved_docs = self.rag_retriever.invoke(query)
                context = self.format_docs(retrieved_docs)
                
                self.chat1.log(f"📚 从知识库检索到 {len(retrieved_docs)} 个相关文档片段")
                
                # 构建增强的提示词，结合RAG检索的内容
                rag_prompt_template = factor_optimization_rag_prompt
                
                # 使用最佳因子的信息来生成RAG优化提示词
                best_evaluation = state.best_factor.evaluation
                best_factor_analysis = state.best_factor.analysis or ""
                
                prompt = rag_prompt_template.format(
                    context=context,
                    start_date=CONFIG["START_DATE"],
                    end_date=CONFIG["END_DATE"],
                    universe=CONFIG["UNIVERSE"],
                    factor_analysis=best_factor_analysis,
                    rebalance_freq=CONFIG["REBALANCE_FREQ"],
                    best_factor=state.best_factor.name,
                    best_expression=state.best_factor.expression,
                    ic_mean=best_evaluation.ic_mean,
                    ic_ir=best_evaluation.ic_ir,
                    rank_ic_mean=best_evaluation.rank_ic_mean,
                    rank_ic_ir=best_evaluation.rank_ic_ir
                )
                return prompt
                
            except Exception as e:
                self.chat1.log(f"❌ RAG检索失败: {str(e)}，使用普通优化方法")
        # 如果rag_retriever为None，则返回普通优化提示词
        return self.generate_optimization_prompt(state)


    def resolve_factor(self, exp: str) -> tuple[str, str]:
        """
        解析因子表达式，提取简化表达式和有效性理由
        """
        # 从表达式中提取简单的因子标识用于文件名
        factor_identifier = generate_factor_identifier(exp)
        self.chat1.initialize_log_file(f"FACTOR_RESOLVE_{factor_identifier}")
        self.chat1.log(f"📁 为单个因子解析初始化聊天记录文件")

        # 使用模板化的提示词
        resolve_prompt = factor_resolve_prompt.format(expression=exp)

        try:
            self.chat1.log(f"🤖 开始调用LLM进行因子解析...")

            sys_message = "你是一位资深的量化投资因子研究专家，精通qlib表达式分析，擅长从金融理论和市场实践角度解释因子的构建逻辑和有效性原理。"
            llm_resolve_result = self.call_llm(resolve_prompt, sys_message)

            # 解析响应，提取因子表达式和理由
            self.chat1.log(f"🔧 开始解析响应格式...")
            factor_expression, factor_reason = parse_llm_resolve_response(
                llm_resolve_result
            )

            self.chat1.log(f"✅ 因子解析完成")
            self.chat1.log(f"📝 解析后简化表达式: {factor_expression}")
            self.chat1.log(f"💡 有效性理由: {factor_reason}")
            self.chat1.log(f"📄 完整响应内容: {llm_resolve_result}")
            self.chat1.log(f"{'='*60}\n")

            return factor_expression, factor_reason

        except Exception as e:
            error_msg = f"因子解析调用失败: {str(e)}"
            self.chat1.log(f"❌ {error_msg}")
            self.chat1.log(f"{'='*60}\n")
            return exp, f"解析失败: {error_msg}"


    def _add_to_optimization_history(
        self, state: MiningState, optimization_result: dict
    ) -> None:
        """添加优化结果到历史记录的通用方法"""
        attempt_count = len(state.optimization_history) + 1
        state.optimization_history.append(
            {
                "attempt": attempt_count,
                "successful_iteration": state.iteration_count,
                "factor_name": state.current_factor.name,
                "expression": state.current_factor.expression,
                "evaluation": state.current_factor.evaluation,
                "optimization_result": optimization_result,
            }
        )


    def _evaluate_factor_and_update(self, state: MiningState) -> MiningState:
        """
        Args:
            state: 优化状态
        """
        self.chat1.log(f"\n=== 回测因子: {state.current_factor.name} ===")

        evaluation = self.evaluate_factor(
            state.current_factor.expression, state.current_factor.name, state
        )
        state.current_factor.evaluation = evaluation

        state.constraint_violated = False
        if state.last_error != "":
            state.constraint_violated = True
            self.chat1.log("❌ 因子计算出错")
            return state

        self.chat1.log(
            f"""
📈 因子评价结果:
- IC均值: {evaluation.ic_mean:.6f}
- IC信息比率: {evaluation.ic_ir:.6f} 
- 排序IC均值: {evaluation.rank_ic_mean:.6f}
- 排序IC信息比率: {evaluation.rank_ic_ir:.6f}"""
        )

        if state.iteration_count > 0:
            # # 检查Rank IC符号是否发生变化
            # if state.best_factor.evaluation.rank_ic_mean * evaluation.rank_ic_mean < 0:
            #     self.chat1.log(f"→ 因子Rank IC符号发生变化")
            #     self.chat1.log(f"   之前: {state.best_factor.evaluation.rank_ic_mean:.6f}, 现在: {evaluation.rank_ic_mean:.6f}")
            #     state.constraint_violated = True
            #     return state

            # 检查Rank IC绝对值是否提升
            if abs(state.best_factor.evaluation.rank_ic_mean) >= abs(evaluation.rank_ic_mean):
                self.chat1.log(f"↓ 因子Rank IC绝对值未提升")
                self.chat1.log(
                    f"   之前: {state.best_factor.evaluation.rank_ic_mean:.6f}, 现在: {evaluation.rank_ic_mean:.6f}"
                )
                state.iteration_count += 1
                state.constraint_violated = True
                return state

        # 更新最佳因子
        if (
            abs(evaluation.rank_ic_mean) > abs(state.best_factor.evaluation.rank_ic_mean)
            or state.iteration_count == 0
        ):
            state.best_factor = Factor(
                name=state.current_factor.name,
                expression=state.current_factor.expression,
                evaluation=evaluation
            )
            if state.iteration_count > 0:
                self.chat1.log(f"🎉 发现更好的因子!")
            if state.iteration_count == 0:
                init_factor_info = {"name": state.current_factor.name, "expression": state.current_factor.expression}
                if CONFIG["SUMMARY_ENABLE"]:
                    self.chat2.log(f"初始因子信息: {init_factor_info}")
        state.iteration_count += 1

        return state


    def _analyze_factor(
        self, state: MiningState, update_best: bool = True
    ) -> MiningState:
        """
        Args:
            state: 优化状态
            update_best: 是否更新最佳因子分析
        """
        self.chat1.log(f"\n=== LLM因子分析 ===")

        if state.constraint_violated:
            self.chat1.log("🚫 因子违反条件，跳过分析")
            state.current_factor.analysis = "因子违反条件，跳过分析。"
            return state

        evaluation = state.current_factor.evaluation

        # 使用模板化的提示词
        analysis_prompt = factor_analysis_prompt.format(
            factor_name=state.current_factor.name,
            factor_expression=state.current_factor.expression,
            start_date=CONFIG["START_DATE"],
            end_date=CONFIG["END_DATE"],
            rebalance_freq=CONFIG["REBALANCE_FREQ"],
            universe=CONFIG["UNIVERSE"],
            ic_mean=evaluation.ic_mean,
            ic_ir=evaluation.ic_ir,
            rank_ic_mean=evaluation.rank_ic_mean,
            rank_ic_ir=evaluation.rank_ic_ir,
        )

        sys_message = "你是一位资深的量化投资因子研究专家，擅长深度分析因子回测指标，提供专业的投资建议。"
        llm_analysis_result = self.call_llm(analysis_prompt, sys_message)

        # 提取并打印因子评分
        factor_score = parse_factor_score(llm_analysis_result)
        if factor_score is not None:
            self.chat1.log(f"📊 因子评分: {factor_score}分 (满分100分)")
        else:
            self.chat1.log(f"⚠️ 未能从分析结果中提取到有效评分")
        self.chat1.log(f"🔍 分析结果: {llm_analysis_result}")
        state.current_factor.analysis = llm_analysis_result

        # 根据参数决定是否更新最佳因子分析
        if update_best and (
            state.current_factor.name == state.best_factor.name
            and state.current_factor.expression == state.best_factor.expression
        ):
            state.best_factor.analysis = llm_analysis_result

        return state


    def _optimize_factor(self, state: MiningState) -> MiningState:
        """
        Args:
            state: 优化状态
        """
        attempt_count = len(state.optimization_history) + 1
        self.chat1.log(
            f"\n=== 优化方案 (第{attempt_count}次尝试，已成功{state.iteration_count}次) ==="
        )
        
        if CONFIG["RAG_ENABLE"]:
            prompt = self.generate_optimization_prompt_rag(state)
        else:
            prompt = self.generate_optimization_prompt(state)

        sys_message = "你是一位专业的量化投资因子研究专家，擅长分析和优化股票因子。请注意：每次只提供一种最优的因子优化方案，充分解释优化逻辑和技术原理，确保表达式语法完全正确（特别注意算子参数填写完整、括号匹配等问题）。"
        llm_opti_result = self.call_llm(prompt, sys_message)

        lines = llm_opti_result.strip().split("\n")
        factor_info = None
        
        # 初始化 optimization_result，避免 UnboundLocalError
        optimization_result = None

        for line in lines:
            if "|" in line and any(char in line for char in ["$", "(", ")"]):
                factor_info = line.strip()
                break

        if factor_info and "|" in factor_info:
            parts = factor_info.split("|")
            if len(parts) >= 2:
                new_factor_name = parts[0].strip()
                new_expression = parts[1].strip()

                optimization_result = {
                    "factor_name": new_factor_name,
                    "expression": new_expression,
                    "reasoning": llm_opti_result,
                }
        else:
            # 如果解析失败，使用当前因子的信息作为备用
            self.chat1.log(f"⚠️ 无法解析LLM优化结果，使用当前因子作为备用")
            optimization_result = {
                "factor_name": state.current_factor.name,
                "expression": state.current_factor.expression,
                "reasoning": llm_opti_result,
            }

        # 添加到历史记录
        self._add_to_optimization_history(state, optimization_result)

        # 更新当前因子
        state.current_factor = Factor(
            name=optimization_result["factor_name"],
            expression=optimization_result["expression"]
        )

        self.chat1.log(f"💡 新因子: {state.current_factor.name}")
        self.chat1.log(f"📝 表达式: {state.current_factor.expression}")
        self.chat1.log(f"🤔 推理: {optimization_result['reasoning']}...")

        return state
       


    def _check_termination(self, state: MiningState) -> str:
        """
        Args:
            state: 优化状态
            min_iteration: 最小迭代次数，在此之前不询问用户
        """
        if state.iteration_count > CONFIG["MAX_ITERATION"]:
            self.chat1.log(
                f"\n⏰ 迭代次数{state.iteration_count}，达到最大迭代次数 ({CONFIG['MAX_ITERATION']})"
            )
            return "end"

        # 在最小迭代次数后询问用户是否继续
        if CONFIG.get("RUNTIME", {}).get("AUTO_MODE", False):
            self.chat1.log("🤖 自动运行模式：继续下一轮优化...")
        else:
            user_input = input("\n继续优化? (y/n, 默认y): ").strip().lower()
            if user_input in ["n", "no"]:
                return "end"

        return "continue"


    def optimize_factor(
        self,
        initial_factor_name: str,
        initial_expression: str,
    ) -> dict:
        """
        Returns:
            最终的优化状态
        """
        self.chat1.log(f"🚀 开始优化因子: {initial_factor_name}")
        self.chat1.log(f"📝 初始表达式: {initial_expression}")
        self.chat1.log(f"🔄 最大迭代次数: {CONFIG['MAX_ITERATION']}")

        # 初始化聊天记录文件，文件名包含因子名称
        self.chat1.initialize_log_file(initial_factor_name)

        # 初始化状态
        initial_evaluation = FactorEvaluation(
            ic_mean=0.0,
            ic_ir=0.0,
            rank_ic_mean=0.0,
            rank_ic_ir=0.0,
        )

        initial_factor = Factor(
            name=initial_factor_name,
            expression=initial_expression,
            evaluation=initial_evaluation
        )

        state = MiningState(
            current_factor=initial_factor,
            best_factor=Factor(
                name=initial_factor_name,
                expression=initial_expression,
                evaluation=initial_evaluation
            ),
            iteration_count=0,
            last_error="",
        )

        # 运行工作流，设置递归限制
        result = self.workflow.invoke(state, config={"recursion_limit": 10000})
        self.chat1.log("✅ 优化流程完成")
        # 提取并打印最佳因子信息
        self.chat1.log("\n" + "="*80)
        self.chat1.log("🏆 优化完成！最佳因子详情：")
        self.chat1.log("="*80)
        self.chat1.log(f"📝 初始因子名称: {initial_factor.name}")
        self.chat1.log(f"📝 初始因子表达式: {initial_factor.expression}")
        self.chat1.log(f"🎯 最佳因子名称: {result.best_factor.name}")
        self.chat1.log(f"🎯 最佳因子表达式: {result.best_factor.expression}")
        self.chat1.log(f"🎯 最佳因子评价: {result.best_factor.evaluation}")
        # 确保所有消息都被写入文件
        self.chat1.flush_buffer()
        return result
    
    
    def _build_workflow(self):
        """
        构建工作流 - 兼容版本
        """
        if LANGGRAPH_AVAILABLE:
            # 使用完整的 langgraph
            workflow = StateGraph(MiningState)
            workflow.add_node("evaluate", self._evaluate_factor_and_update)
            workflow.add_node("analyze", self._analyze_factor)
            workflow.add_node("optimize", self._optimize_factor)
            workflow.set_entry_point("evaluate")
            workflow.add_edge("evaluate", "analyze")
            workflow.add_edge("optimize", "evaluate")
            workflow.add_conditional_edges(
                "analyze", self._check_termination, {"continue": "optimize", "end": END}
            )
            compiled_workflow = workflow.compile()
            recursion_limit = CONFIG.get("RUNTIME", {}).get("RECURSION_LIMIT", 1000)
            compiled_workflow.config = {"recursion_limit": recursion_limit}
            return compiled_workflow
        else:
            # 简化的工作流实现
            def simple_workflow(state):
                max_iterations = CONFIG["MAX_ITERATION"]
                while state.iteration_count <= max_iterations:
                    # 评估因子
                    state = self._evaluate_factor_and_update(state)
                    # 分析因子
                    state = self._analyze_factor(state)
                    # 检查终止条件
                    if self._check_termination(state) == "end":
                        break
                    # 优化因子
                    state = self._optimize_factor(state)
                return state
            
            return SimpleWorkflow(simple_workflow)

