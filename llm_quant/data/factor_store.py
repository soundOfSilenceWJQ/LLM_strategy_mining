"""
因子库（Factor Store）
管理 PAR（原始阿尔法因子库）和 IAR（投资级阿尔法因子库）
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.config import PAR_DB_PATH, IAR_DB_PATH, PAR_TOP_RATIO, IAR_MAX_FACTORS


_FACTOR_SCHEMA = {
    "id":            str,    # 因子唯一ID
    "name":          str,    # 因子名称
    "category":      str,    # 类别：momentum/reversal/volatility/fundamental/liquidity等
    "expression":    str,    # 因子公式（文字描述或简写）
    "code":          str,    # 可执行Python代码
    "logic":         str,    # 经济学逻辑说明
    "source_info":   str,    # 来源信息ID
    "created_at":    str,    # 创建时间
    "validated":     bool,   # 是否通过代码验证
    "ic_mean":       float,
    "icir":          float,
    "ic_win_rate":   float,
    "annual_return": float,
    "sharpe":        float,
    "max_drawdown":  float,
    "score":         float,  # 综合评分
    "in_iar":        bool,   # 是否在IAR中
    "active":        bool,   # 是否有效（未被淘汰）
}


class FactorStore:
    """
    因子库，同时管理 PAR 和 IAR。
    PAR：原始因子库，验证通过即可入库。
    IAR：投资级因子库，从PAR中按综合评分选Top30%。
    """

    def __init__(self, par_path: Path = PAR_DB_PATH, iar_path: Path = IAR_DB_PATH):
        self.par_path = par_path
        self.iar_path = iar_path
        self._par: list[dict] = []
        self._iar: list[dict] = []
        self._load()

    # ── 持久化 ────────────────────────────────────────────
    def _load(self):
        if self.par_path.exists():
            with open(self.par_path, "r", encoding="utf-8") as f:
                self._par = json.load(f)
        if self.iar_path.exists():
            with open(self.iar_path, "r", encoding="utf-8") as f:
                self._iar = json.load(f)

    def _save_par(self):
        with open(self.par_path, "w", encoding="utf-8") as f:
            json.dump(self._par, f, ensure_ascii=False, indent=2)

    def _save_iar(self):
        with open(self.iar_path, "w", encoding="utf-8") as f:
            json.dump(self._iar, f, ensure_ascii=False, indent=2)

    # ── PAR 操作 ──────────────────────────────────────────
    def add_to_par(self, factor: dict) -> str:
        """将验证通过的因子加入PAR，返回factor_id。"""
        factor.setdefault("created_at", datetime.now().isoformat())
        factor.setdefault("validated", False)
        factor.setdefault("in_iar", False)
        factor.setdefault("active", True)
        factor.setdefault("score", 0.0)
        factor.setdefault("ic_mean", 0.0)
        factor.setdefault("icir", 0.0)
        factor.setdefault("ic_win_rate", 0.0)
        factor.setdefault("annual_return", 0.0)
        factor.setdefault("sharpe", 0.0)
        factor.setdefault("max_drawdown", 0.0)

        # 自动生成 ID
        if "id" not in factor:
            import hashlib
            factor["id"] = "f_" + hashlib.md5(
                (factor.get("name","") + factor.get("code","")).encode()
            ).hexdigest()[:8]

        # 检查重复
        existing_ids = {f["id"] for f in self._par}
        if factor["id"] not in existing_ids:
            self._par.append(factor)
            self._save_par()

        return factor["id"]

    def update_par_metrics(self, factor_id: str, metrics: dict):
        """更新PAR中某因子的绩效指标。"""
        for f in self._par:
            if f["id"] == factor_id:
                f.update(metrics)
                break
        self._save_par()

    def deactivate_par(self, factor_id: str):
        """将PAR中绩效持续不佳的因子标记为失效。"""
        for f in self._par:
            if f["id"] == factor_id:
                f["active"] = False
                f["in_iar"] = False
                break
        self._save_par()
        # 从IAR中也移除
        self._iar = [f for f in self._iar if f["id"] != factor_id]
        self._save_iar()

    # ── IAR 操作（策略筛选层核心）────────────────────────
    def rotate_iar(self):
        """
        IAR轮动：按综合评分从PAR选Top PAR_TOP_RATIO进入IAR。
        约束：每个类别保持均衡，避免风格集中。
        """
        active_par = [f for f in self._par if f.get("active", True) and f.get("validated", False)]
        if not active_par:
            print("[FactorStore] PAR中暂无有效因子，IAR轮动跳过。")
            return

        # 按综合评分排序
        active_par.sort(key=lambda f: f.get("score", 0), reverse=True)
        top_n = max(1, int(len(active_par) * PAR_TOP_RATIO))
        top_n = min(top_n, IAR_MAX_FACTORS)

        # 类别均衡：每类最多放 ceil(top_n / n_categories) 个
        selected = []
        category_count: dict[str, int] = {}
        n_categories = len({f.get("category", "unknown") for f in active_par})
        per_cat_limit = max(2, top_n // max(n_categories, 1))

        for f in active_par:
            cat = f.get("category", "unknown")
            if category_count.get(cat, 0) < per_cat_limit and len(selected) < top_n:
                selected.append(f)
                category_count[cat] = category_count.get(cat, 0) + 1

        # 更新 in_iar 标记
        selected_ids = {f["id"] for f in selected}
        for f in self._par:
            f["in_iar"] = f["id"] in selected_ids

        self._iar = [f for f in active_par if f["id"] in selected_ids]
        self._save_par()
        self._save_iar()
        print(f"[FactorStore] IAR轮动完成：{len(self._iar)} 个因子入库。")

    # ── 查询 ──────────────────────────────────────────────
    def get_par(self, active_only: bool = True) -> list[dict]:
        if active_only:
            return [f for f in self._par if f.get("active", True)]
        return list(self._par)

    def get_iar(self) -> list[dict]:
        return list(self._iar)

    def par_stats(self) -> dict:
        active = self.get_par()
        categories = {}
        for f in active:
            cat = f.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total": len(active),
            "validated": sum(1 for f in active if f.get("validated")),
            "categories": categories,
        }

    def iar_stats(self) -> dict:
        return {
            "total": len(self._iar),
            "avg_ic": round(sum(f.get("ic_mean",0) for f in self._iar) / max(len(self._iar),1), 4),
            "avg_score": round(sum(f.get("score",0) for f in self._iar) / max(len(self._iar),1), 2),
        }
