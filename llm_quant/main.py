"""
主流程编排（Main Orchestration）
实现论文中描述的五层智能体系统全闭环运行

使用方式：
  python main.py                    # 运行完整流程
  python main.py --mode extract     # 仅信息提取
  python main.py --mode generate    # 仅因子生成（需要已有信息库）
  python main.py --mode validate    # 仅验证分析
  python main.py --mode rotate      # 仅IAR轮动
  python main.py --mode demo        # 演示模式（手动输入信息，跳过爬虫）
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

# 加载路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_quant.config import ANTHROPIC_API_KEY
from llm_quant.utils.llm_client import LLMClient
from llm_quant.data.info_store import InfoStore
from llm_quant.data.factor_store import FactorStore
from llm_quant.agents.info_extractor import InfoExtractor
from llm_quant.agents.text_processor import TextProcessor
from llm_quant.agents.strategy_generator import StrategyGenerator
from llm_quant.agents.validator import Validator
from llm_quant.agents.selector import Selector


# ── 演示用预设信息（当无法联网爬虫时使用）──────────────────
DEMO_ARTICLES = [
    {
        "title": "央行宣布降准0.5个百分点，释放长期资金约1万亿元",
        "content": (
            "中国人民银行宣布，自2025年2月5日起，下调金融机构存款准备金率0.5个百分点（不含已执行5%存款准备金率的金融机构），"
            "本次下调后，金融机构加权平均存款准备金率约为6.6%。此次降准预计释放长期资金约1万亿元，"
            "有助于降低银行资金成本，支持实体经济发展，对资本市场形成利好，银行、地产、基建板块有望受益。"
        ),
        "source": "新华社",
    },
    {
        "title": "A股量化资金规模突破3万亿，因子拥挤度加剧",
        "content": (
            "据中国证券投资基金业协会数据，国内量化私募证券基金管理规模突破3.1万亿元，"
            "主流价量因子拥挤度持续提升，动量、反转等传统因子有效周期从1-2年缩短至3-6个月。"
            "机构建议关注低拥挤度另类因子，包括：分析师预期修正因子、市场情绪因子、政策敏感性因子等，"
            "以获取更稳定的超额收益。"
        ),
        "source": "中国证券报",
    },
    {
        "title": "人工智能算力需求爆发，半导体板块迎来政策+业绩双催化",
        "content": (
            "证监会发布《人工智能产业金融支持指导意见》，对AI大模型、半导体材料、算力基础设施企业"
            "给予融资支持，对研发投入占比超10%的AI企业实施税收优惠。"
            "AI算力需求爆发带动半导体设备、存储芯片景气度持续上行，"
            "相关个股EPS预测持续上调，科技成长风格有望延续强势。"
            "建议关注：研发投入比例高、受益政策催化、AI产业链核心受益标的。"
        ),
        "source": "华泰证券研报",
    },
    {
        "title": "房地产政策组合拳落地，核心城市销量回暖，价值板块修复可期",
        "content": (
            "多个一线城市相继取消限购，叠加央行下调房贷利率，"
            "核心城市新房成交量环比提升15%-20%。"
            "地产链条（建材、家电、家居）景气度边际改善，"
            "高股息、低估值的地产股及银行股获机构增配。"
            "当前市场呈现价值风格占优态势，PB低于1倍的银行股安全边际凸显。"
        ),
        "source": "东方财富研究院",
    },
    {
        "title": "华泰证券量化研报：分析师一致预期修正因子的超额收益挖掘",
        "content": (
            "基于2018-2024年A股数据的实证研究显示，分析师盈利预期上调幅度与股票未来1-3个月收益显著正相关，"
            "上调幅度超行业均值20%的个股，未来3个月平均超额收益达8.7%。"
            "因子构建：EPS一致预期修正幅度 = (EPSt - EPSt-30) / |EPSt-30|，"
            "结合股价近30天涨幅小于5%的过滤条件，可有效规避已被市场定价的预期改善。"
            "该因子在中证500池中IC均值达0.032，ICIR为0.91，月胜率67%。"
        ),
        "source": "华泰证券量化团队",
    },
]


class LLMQuantSystem:
    """
    非结构化实时信息驱动的LLM量化策略挖掘系统
    五层智能体协同架构
    """

    def __init__(self, require_llm: bool = True):
        print("="*60)
        print("【LLM量化策略挖掘系统】初始化中...")

        self.require_llm = require_llm

        # 检查 API Key
        if require_llm and not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "未设置 ANTHROPIC_API_KEY！\n"
                "请执行: $env:ANTHROPIC_API_KEY='your-api-key'"
            )

        # 初始化共享组件
        self.llm          = LLMClient() if require_llm else None
        self.info_store   = InfoStore()
        self.factor_store = FactorStore()

        # 初始化五层智能体
        self.layer1 = InfoExtractor()
        self.layer2 = TextProcessor(llm=self.llm, info_store=self.info_store) if self.llm else None
        self.layer3 = StrategyGenerator(llm=self.llm, info_store=self.info_store) if self.llm else None
        self.layer4 = Validator(factor_store=self.factor_store, demo_mode=not require_llm)
        self.layer5 = Selector(
            factor_store=self.factor_store,
            info_store=self.info_store,
            llm=self.llm,
        )

        print(f"  信息库: {len(self.info_store)} 条记录")
        print(f"  PAR: {self.factor_store.par_stats()['total']} 个因子")
        print(f"  IAR: {self.factor_store.iar_stats()['total']} 个因子")
        print("="*60)

    # ── 完整流程 ──────────────────────────────────────────
    def run_full_pipeline(
        self,
        n_factors: int = 3,
        sources: list[str] | None = None,
        use_demo_data: bool = False,
    ):
        """运行完整的五层闭环流程。"""
        print("\n" + "="*60)
        print("【启动完整流程】")

        # ─ 第1层：信息提取 ─
        print("\n>>> 第1层：信息提取层")
        if use_demo_data:
            raw_records = DEMO_ARTICLES
            print(f"  使用演示数据 {len(raw_records)} 条")
        else:
            raw_records = self.layer1.extract(sources=sources)

        if not raw_records:
            print("  未获取到任何信息，终止流程。")
            return

        # ─ 第2层：解析加工 ─
        print("\n>>> 第2层：解析加工层")
        if self.layer2 is not None:
            self.layer2.process_and_store(raw_records[:5])  # 最多处理5条（节省API调用）
        else:
            self._demo_process_and_store(raw_records[:5])

        # 更新时间衰减权重
        self.info_store.decay_weights()

        # ─ 第3层：策略生成 ─
        print("\n>>> 第3层：策略生成层")
        if self.layer3 is not None:
            factors = self.layer3.generate_batch(n_factors=n_factors)
        else:
            factors = self._demo_generate_factors(n_factors=n_factors)
        print(f"  生成候选因子 {len(factors)} 个")

        # ─ 第4层：验证分析 ─
        print("\n>>> 第4层：验证分析层")
        results = self.layer4.validate_batch(factors)
        passed  = sum(1 for r in results if r.get("passed"))
        print(f"\n  验证结果: {passed}/{len(results)} 个因子通过，入库PAR")

        # ─ 第5层：策略筛选 ─
        print("\n>>> 第5层：策略筛选层")
        self.layer5.retire_weak_factors()
        rotation_summary = self.layer5.rotate_iar(use_llm_assessment=False)
        self.layer5.print_summary()

        print("\n【完整流程执行完毕】")
        return rotation_summary

    # ── 单独模式 ──────────────────────────────────────────
    def run_extract_only(self, use_demo: bool = False):
        """仅运行信息提取和解析。"""
        if use_demo:
            records = DEMO_ARTICLES
        else:
            records = self.layer1.extract()
        if self.layer2 is not None:
            self.layer2.process_and_store(records)
        else:
            self._demo_process_and_store(records)
        print(f"\n信息库当前状态: {self.info_store.stats()}")

    def run_generate_only(self, n: int = 3):
        """仅运行因子生成。"""
        factors = self.layer3.generate_batch(n_factors=n)
        for f in factors:
            print(f"\n因子: {f.get('name','?')}")
            print(f"  类别: {f.get('category','?')}")
            print(f"  逻辑: {f.get('logic','?')[:100]}...")
            print(f"  公式: {f.get('expression','?')}")
        return factors

    def run_validate_only(self):
        """重新验证PAR中所有未验证的因子。"""
        par = self.factor_store.get_par(active_only=True)
        unvalidated = [f for f in par if not f.get("validated")]
        print(f"PAR中有 {len(unvalidated)} 个未验证因子")
        self.layer4.validate_batch(unvalidated)

    def run_rotate_only(self):
        """仅执行IAR轮动。"""
        self.layer5.rotate_iar(use_llm_assessment=bool(self.llm))
        self.layer5.print_summary()

    def add_custom_info(self, title: str, content: str):
        """手动添加自定义信息并解析。"""
        record  = self.layer1.add_manual(title, content)
        if self.layer2 is not None:
            enhanced = self.layer2.process(record)
        else:
            enhanced = self._demo_enhance_record(record)
        rec_id  = self.info_store.add(enhanced)
        print(f"已添加信息: {rec_id}")
        return rec_id

    # ── Demo 本地路径（无 LLM） ──────────────────────────
    def _demo_enhance_record(self, record: dict) -> dict:
        title = record.get("title", "")
        content = record.get("content", "")
        combined = f"{title} {content}".lower()

        tags = []
        if any(key in combined for key in ["降准", "降息", "利率", "流动性"]):
            tags.extend(["宏观", "流动性"])
        if any(key in combined for key in ["ai", "人工智能", "半导体", "算力"]):
            tags.extend(["科技", "成长"])
        if any(key in combined for key in ["地产", "房地产", "房贷"]):
            tags.extend(["地产", "价值"])
        if any(key in combined for key in ["量化", "因子", "动量", "反转"]):
            tags.extend(["量化", "因子"])

        if not tags:
            tags = ["新闻", "市场"]

        return {
            **record,
            "summary": title[:50],
            "logic": "基于标题与正文关键词的本地规则提取，未调用LLM。",
            "key_drivers": tags[:3],
            "affected_sectors": tags[:2],
            "pricing_factors": ["动量", "估值", "流动性"],
            "time_horizon": "短期1-4周",
            "sentiment": "positive" if any(key in combined for key in ["利好", "受益", "回暖", "催化"]) else "neutral",
            "tags": list(dict.fromkeys(tags))[:5],
            "risk_warnings": "本地demo规则生成，结果仅用于连通性测试。",
        }

    def _demo_process_and_store(self, raw_records: list[dict]) -> list[str]:
        ids = []
        for i, rec in enumerate(raw_records):
            print(f"[Layer2-demo] 处理 {i+1}/{len(raw_records)}: {rec.get('title','')[:40]}...")
            enhanced = self._demo_enhance_record(rec)
            ids.append(self.info_store.add(enhanced))
        print(f"[Layer2-demo] 本地解析完成，入库 {len(ids)} 条。")
        return ids

    def _demo_generate_factors(self, n_factors: int = 3) -> list[dict]:
        templates = [
            {
                "name": "demo_momentum_14d",
                "category": "momentum",
                "expression": "14日价格动量",
                "logic": "价格延续效应的简化实现，适用于风险偏好改善阶段。",
                "source_insight": "demo本地规则",
                "risk_warnings": "仅用于演示，不代表真实可交易信号。",
                "code": """def compute_factor(df):\n    factor = df.iloc[-1] / df.iloc[-15] - 1\n    return factor""",
            },
            {
                "name": "demo_mean_reversion_5d",
                "category": "reversal",
                "expression": "5日反转因子",
                "logic": "短期超跌后的均值回归信号。",
                "source_insight": "demo本地规则",
                "risk_warnings": "仅用于演示，不代表真实可交易信号。",
                "code": """def compute_factor(df):\n    factor = -(df.iloc[-1] / df.iloc[-6] - 1)\n    return factor""",
            },
            {
                "name": "demo_volatility_20d",
                "category": "volatility",
                "expression": "20日价格波动率因子",
                "logic": "价格波动反映风险偏好和预期不确定性。",
                "source_insight": "demo本地规则",
                "risk_warnings": "仅用于演示，不代表真实可交易信号。",
                "code": """def compute_factor(df):\n    factor = df.pct_change().iloc[-20:].std()\n    return factor""",
            },
        ]
        return [dict(templates[i]) for i in range(min(n_factors, len(templates)))]


# ── CLI 入口 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM量化策略挖掘系统（论文架构实现）"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "extract", "generate", "validate", "rotate", "demo"],
        default="demo",
        help="运行模式（默认: demo）",
    )
    parser.add_argument(
        "--n-factors", type=int, default=3,
        help="生成因子数量（默认: 3）",
    )
    args = parser.parse_args()

    system = LLMQuantSystem(require_llm=(args.mode != "demo"))

    if args.mode == "demo":
        print("\n【演示模式】使用预设金融文本，跳过网络爬虫")
        system.run_full_pipeline(
            n_factors=args.n_factors,
            use_demo_data=True,
        )

    elif args.mode == "full":
        system.run_full_pipeline(
            n_factors=args.n_factors,
            use_demo_data=False,
        )

    elif args.mode == "extract":
        system.run_extract_only(use_demo=False)

    elif args.mode == "generate":
        system.run_generate_only(n=args.n_factors)

    elif args.mode == "validate":
        system.run_validate_only()

    elif args.mode == "rotate":
        system.run_rotate_only()


if __name__ == "__main__":
    main()
