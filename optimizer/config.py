# ===== 因子优化框架配置 =====
# 回测时间配置
START_DATE1 = "2010-01-01"
END_DATE1 = "2020-12-31"
START_DATE2 = "2021-01-01"
END_DATE2 = "2023-12-31"

FACTOR_OPTIMIZATION_CONFIG = {
    # 回测时间范围
    "START_DATE": START_DATE1,
    "END_DATE": END_DATE1,
    "REBALANCE_FREQ": 20,
    # 优化参数
    "MAX_ITERATION": 2,  # 最大优化轮数
    # 文件存储路径配置
    "PATHS": {
        'DATA_SRC_PATH': r"C:\\Users\\wangj\\.qlib\\qlib_data\\cn_data",
        "KNOWLEDGE_PATH": ".",  # RAG知识库路径
        "CHAT_LOGS_DIR": r"D:\wjq\working\citic\codes_wjq(1)\chats",  # 记录存储目录
        "SUMMARY_LOGS_DIR": ".",  # 总结存储目录
        "OPTIMIZATION_HISTORIES_DIR": r"D:\wjq\working\citic\codes_wjq(1)\histories",  # 优化历史记录存储目录
        "RESULTS_DIR": r"D:\wjq\working\citic\codes_wjq(1)\results",  # 分析结果存储目录
        "PLOTS_DIR": ".",  # 图表存储目录
    },
    # 文件名模板
    "FILE_TEMPLATES": {
        "CHAT_LOG": "chat_{timestamp}.txt",  # 聊天记录文件名模板
        "OPTIMIZATION_HISTORY": "{factor_name}_{timestamp}.json",  # 优化历史文件名模板
        "OPTIMIZATION_RESULTS": "optimization_results.json",  # 优化结果文件名
        "EVALUATION_RESULTS": "factor_evaluation_results.json",  # 评价结果文件名
        "EVALUATION_CSV": "factor_evaluation_results.csv",  # 评价结果CSV文件名
        "IC_COMPARISON_PLOT": "factor_ic_comparison.png",  # IC对比图文件名
        "OPTIMIZATION_PROGRESS_PLOT": "optimization_progress.png",  # 优化进度图文件名
    },
    # 股票池配置
    "UNIVERSE": "csi500",  # 股票池：csi300, csi500, csi800等

    "USE_OLLAMA": False,
    # LLM分析配置
    "LLM": {
        "API_KEY": "your_api_key_here",
        "BASE_URL": "https://api.deepseek.com/v1",
        "MODEL_NAME": "deepseek-chat",
        "TEMPERATURE": 0.3,  
    },
    # OLLAMA LLM配置
    "OLLAMA": {
        "API_KEY": "your_ollama_api_key_here",
        "BASE_URL": "http://localhost:11434/api/chat",
        "MODEL_NAME": "gemma:2b",
        "TEMPERATURE": 0.3,  
    },
    # 运行模式配置
    "RUNTIME": {
        "AUTO_MODE": True,  # 自动运行模式，不需要用户交互，False需要用户确认是否继续
    },
    # 总结生成
    "SUMMARY_ENABLE": False,
    
    # RAG模式是否开启
    "RAG_ENABLE": False,

    # 初始因子（Alpha158因子，%d取20）
    "INITIAL_FACTORS": {
        # "FACTOR_1": "Corr($high, $volume, 30)",
        # "FACTOR_2": "WMA(($low * $open) / $close, 20)",
        # "FACTOR_3": "Ref(Max($high, 50), 50)"
        "STD": "Std($close, 20)/$close",
        # "BETA": "Slope($close, 20)/$close",
        # "CNTD": "Mean($close>Ref($close, 1), 20)-Mean($close<Ref($close, 1), 20)",
        # "CNTN": "Mean($close<Ref($close, 1), 20)",
        # "CNTP": "Mean($close>Ref($close, 1), 20)",
        # "IMAX": "IdxMax($high, 20)/20",
        # "IMIN": "IdxMin($low, 20)/20",
        # "IMXD": "(IdxMax($high, 20)-IdxMin($low, 20))/20",
        # "MA": "Mean($close, 20)/$close",
        # "MAX": "Max($high, 20)/$close",
        # "MIN": "Min($low, 20)/$close",
        # "QTLD": "Quantile($close, 20, 0.2)/$close",
        # "QTLU": "Quantile($close, 20, 0.8)/$close",
        # "RANK": "Rank($close, 20)",
        # "RESI": "Resi($close, 20)/$close",
        # "ROC": "Ref($close, 20)/$close",
        # "RSQ": "Rsquare($close, 20)",
        # "RSV": "($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)",
        # "SUMD": "(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
        # "SUMM": "Sum(Greater(Ref($close, 1)-$close, 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
        # "SUMP": "Sum(Greater($close-Ref($close, 1), 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
        # "VMA": "Mean($volume, 20)/($volume+1e-12)",
        # "VSTD": "Std($volume, 20)/($volume+1e-12)",
        # "VSUMD": "(Sum(Greater($volume-Ref($volume, 1), 0), 20)-Sum(Greater(Ref($volume, 1)-$volume, 0), 20))/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
        # "VSUMN": "Sum(Greater(Ref($volume, 1)-$volume, 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
        # "VSUMP": "Sum(Greater($volume-Ref($volume, 1), 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)",
        # "WVMA": "Std(Abs($close/Ref($close, 1)-1)*$volume, 20)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 20)+1e-12)",
        # "CORRD": "Corr($close/Ref($close, 1), Log($volume/Ref($volume, 1)+1), 20)",
        # "CORR": "Corr($close, Log($volume+1), 20)",
    },
}
