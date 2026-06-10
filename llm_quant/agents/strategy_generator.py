"""
第3层：策略生成层（Strategy Generation Layer）
功能：
    - 将文献/研报中的因子定义复现为可执行代码
    - 生成标准化元数据，保证可解释与可追溯
    - 在因子出错时结合历史对话进行增量修复
"""

from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.utils.llm_client import LLMClient
from llm_quant.data.info_store import InfoStore
from llm_quant.config import ALLOWED_OPERATORS, RAG_TOP_K
from llm_quant.prompts import (
        DEFAULT_PLATFORM_SPEC,
        FACTOR_GENERATOR_SYSTEM_PROMPT,
        render_factor_generation_prompt,
        render_factor_repair_prompt,
)


class StrategyGenerator:
    """
    第3层：策略生成层。
    使用RAG检索相关信息，驱动LLM生成Alpha因子。
    """

    def __init__(self, llm: LLMClient | None = None, info_store: InfoStore | None = None):
        self.llm = llm or LLMClient()
        self.info_store = info_store or InfoStore()

    def generate_factor_from_literature(
        self,
        factor_description: str,
        economic_logic: str,
        other_info: str = "",
        platform_spec: str = "",
        source_meta: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        query: str = "A股市场当前有效的超额收益来源",
        n_context: int = RAG_TOP_K,
    ) -> dict | None:
        """按文献描述复现单个因子，并绑定标准化元数据。"""
        source_meta = source_meta or {}
        context = self._build_rag_context(query=query, n_context=n_context)
        history_context = self._format_history(conversation_history)
        merged_other_info = (other_info or "").strip()
        if history_context:
            merged_other_info = (merged_other_info + "\n\n历史对话摘要:\n" + history_context).strip()

        prompt = render_factor_generation_prompt(
            factor_description=(factor_description or "").strip(),
            economic_logic=(economic_logic or "").strip(),
            other_info=merged_other_info,
            platform_spec=(platform_spec or DEFAULT_PLATFORM_SPEC).strip(),
            operators=", ".join(ALLOWED_OPERATORS),
            context=context[:3000],
        )

        try:
            response = self.llm.chat(user_message=prompt, system_prompt=FACTOR_GENERATOR_SYSTEM_PROMPT)
            factor = self._parse_json(response)
            factor = self._normalize_factor_output(
                factor=factor,
                source_meta=source_meta,
                source_query=query,
                conversation_history=conversation_history,
            )
            print(f"[Layer3] 文献复现因子：{factor.get('name','unknown')} ({factor.get('category','')})")
            return factor
        except Exception as e:
            print(f"[Layer3] 因子复现失败: {e}")
            return None

    def repair_factor_with_history(
        self,
        current_factor: dict[str, Any],
        error_info: str,
        correct_advice: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict | None:
        """结合历史对话对错误因子进行增量修复。"""
        history_context = self._format_history(conversation_history)
        prompt = render_factor_repair_prompt(
            error_info=(error_info or "").strip(),
            correct_advice=(correct_advice or "").strip(),
            current_factor_json=json.dumps(current_factor, ensure_ascii=False, indent=2),
            history_context=history_context or "（无）",
        )

        try:
            response = self.llm.chat(user_message=prompt, system_prompt=FACTOR_GENERATOR_SYSTEM_PROMPT)
            fixed = self._parse_json(response)
            fixed = self._normalize_factor_output(
                factor=fixed,
                source_meta=current_factor.get("source_meta", {}),
                source_query=current_factor.get("source_query", ""),
                conversation_history=conversation_history,
            )
            repair_log = list(current_factor.get("repair_log", []))
            repair_log.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "error_info": error_info,
                    "correct_advice": correct_advice,
                }
            )
            fixed["repair_log"] = repair_log
            return fixed
        except Exception as e:
            print(f"[Layer3] 因子修复失败: {e}")
            return None

    def generate_factor(
        self,
        query: str = "A股市场当前有效的超额收益来源",
        n_context: int = RAG_TOP_K,
    ) -> dict | None:
        """
        生成单个Alpha因子。
        1. RAG检索相关信息
        2. 构建 context
        3. LLM生成因子
        """
        # 兼容旧接口：基于检索上下文生成“通用模板”因子。
        return self.generate_factor_from_literature(
            factor_description="基于近期市场信息提炼可复现因子定义",
            economic_logic="围绕可验证的风险补偿与行为偏差路径构建因子",
            other_info="由系统自动检索上下文辅助复现。",
            query=query,
            n_context=n_context,
        )

    def generate_batch(
        self,
        queries: list[str] | None = None,
        n_factors: int = 5,
    ) -> list[dict]:
        """批量生成多个因子。"""
        if queries is None:
            queries = [
                "A股市场价量动量效应",
                "A股市场均值回归与反转策略",
                "A股市场流动性溢价因子",
                "A股市场估值与盈利因子",
                "A股市场政策敏感性与行业轮动",
                "A股市场波动率与风险因子",
                "A股市场财务质量因子",
            ]

        factors = []
        for i in range(min(n_factors, len(queries))):
            print(f"\n[Layer3] 生成第 {i+1}/{n_factors} 个因子...")
            f = self.generate_factor(query=queries[i % len(queries)])
            if f:
                factors.append(f)
        return factors

    def generate_batch_from_literature(self, factors_input: list[dict[str, Any]]) -> list[dict]:
        """批量按文献字段复现因子。"""
        out: list[dict] = []
        for i, item in enumerate(factors_input):
            print(f"\n[Layer3] 文献复现 {i+1}/{len(factors_input)}")
            fac = self.generate_factor_from_literature(
                factor_description=str(item.get("factor_description", "")),
                economic_logic=str(item.get("economic_logic", "")),
                other_info=str(item.get("other_info", "")),
                platform_spec=str(item.get("platform_spec", "")),
                source_meta=item.get("source_meta", {}),
                conversation_history=item.get("conversation_history", []),
                query=str(item.get("query", "A股市场当前有效的超额收益来源")),
                n_context=int(item.get("n_context", RAG_TOP_K)),
            )
            if fac:
                out.append(fac)
        return out

    # ── 辅助：JSON 解析容错 ───────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)

    def _build_rag_context(self, query: str, n_context: int) -> str:
        related_records = self.info_store.search(query, top_k=n_context)
        if not related_records:
            return "（暂无实时信息，基于通用金融市场知识）"
        return "\n\n".join(
            [
                f"[{i+1}] 标题：{r.get('title','')}\n"
                f"  投资逻辑：{r.get('logic','') or r.get('summary','')}\n"
                f"  关键驱动：{', '.join(r.get('key_drivers',[]))}\n"
                f"  时效：{r.get('time_horizon','')}"
                for i, r in enumerate(related_records)
            ]
        )

    @staticmethod
    def _format_history(conversation_history: list[dict[str, str]] | None) -> str:
        if not conversation_history:
            return ""
        lines: list[str] = []
        for i, msg in enumerate(conversation_history, start=1):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)

    def _normalize_factor_output(
        self,
        factor: dict[str, Any],
        source_meta: dict[str, Any],
        source_query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        factor.setdefault("name", "unnamed_factor")
        factor.setdefault("category", "fundamental")
        factor.setdefault("expression", "")
        factor.setdefault("logic", "")
        factor.setdefault("source_insight", "")
        factor.setdefault("risk_warnings", "")
        factor.setdefault("code", "")

        metadata = factor.get("metadata", {}) if isinstance(factor.get("metadata", {}), dict) else {}
        metadata.setdefault("economic_attribution", "")
        metadata.setdefault("applicable_market_environment", [])
        metadata.setdefault("literature_source", source_meta.get("literature_source", ""))
        metadata.setdefault("factor_type", factor.get("category", ""))
        metadata.setdefault("rebalance_cycle", source_meta.get("rebalance_cycle", "日频"))
        metadata.setdefault("data_requirements", ["CLOSE", "OPEN", "HIGH", "LOW", "VOLUME", "VWAP"])
        factor["metadata"] = metadata

        directional = factor.get("directional_constraint", {})
        if not isinstance(directional, dict):
            directional = {}
        directional.setdefault("higher_is_better", True)
        directional.setdefault("original_direction", "unknown")
        directional.setdefault("transformation", "none")
        directional.setdefault("note", "")

        transform = str(directional.get("transformation", "none")).strip().lower()
        if transform not in {"none", "negate"}:
            transform = "none"
            directional["transformation"] = "none"

        if transform == "negate":
            factor["code"] = self._insert_negate_before_return(str(factor.get("code", "")))
            if not directional.get("note"):
                directional["note"] = "该因子原始方向为反向，已按约束进行取负转化。"

        factor["directional_constraint"] = directional
        factor["source_query"] = source_query
        factor["source_meta"] = source_meta
        factor["conversation_history"] = conversation_history or []
        factor["created_at"] = datetime.now().isoformat(timespec="seconds")
        return factor

    @staticmethod
    def _insert_negate_before_return(code: str) -> str:
        lines = code.splitlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("return "):
                indent = line[: len(line) - len(stripped)]
                if i > 0 and lines[i - 1].strip() == "factor = -factor":
                    return code
                lines.insert(i, f"{indent}factor = -factor")
                return "\n".join(lines)
        return code
