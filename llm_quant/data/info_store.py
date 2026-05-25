"""
非结构化信息库（Info Store）
功能：
  - JSON持久化存储
  - TF-IDF语义检索（简化版RAG）
  - 时间衰减权重管理
"""

from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.config import INFO_DB_PATH, RAG_TOP_K


class InfoStore:
    """
    非结构化金融信息库。
    每条记录格式：
    {
        "id":        str,    # 内容hash
        "title":     str,
        "content":   str,
        "summary":   str,    # LLM生成摘要
        "tags":      list,   # LLM提取标签
        "source":    str,    # 来源
        "date":      str,    # ISO格式日期
        "logic":     str,    # 投资逻辑（LLM解析）
        "weight":    float,  # 时间衰减权重
    }
    """

    def __init__(self, db_path: Path = INFO_DB_PATH):
        self.db_path = db_path
        self._records: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._load()

    # ── 持久化 ────────────────────────────────────────────
    def _load(self):
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                self._records = json.load(f)
            self._build_index()

    def _save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    # ── 添加记录 ──────────────────────────────────────────
    def add(self, record: dict) -> str:
        """添加一条信息记录，返回记录ID。"""
        content = record.get("title", "") + record.get("content", "")
        rec_id  = hashlib.md5(content.encode()).hexdigest()[:12]

        # 去重
        existing_ids = {r["id"] for r in self._records}
        if rec_id in existing_ids:
            return rec_id

        record["id"]     = rec_id
        record["weight"] = 1.0
        record.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
        record.setdefault("summary", "")
        record.setdefault("tags", [])
        record.setdefault("logic", "")

        self._records.append(record)
        self._build_index()
        self._save()
        return rec_id

    def add_many(self, records: list[dict]) -> list[str]:
        return [self.add(r) for r in records]

    # ── TF-IDF 索引 ───────────────────────────────────────
    def _build_index(self):
        if not self._records:
            self._vectorizer = None
            self._tfidf_matrix = None
            return
        corpus = [
            r.get("title", "") + " " + r.get("summary", "") + " " + r.get("logic", "")
            for r in self._records
        ]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",     # 字符N-gram，适合中文
            ngram_range=(2, 4),
            max_features=5000,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

    # ── 语义检索 ──────────────────────────────────────────
    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """基于TF-IDF相似度检索最相关的信息记录。"""
        if self._vectorizer is None or not self._records:
            return []

        q_vec = self._vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        # 结合时间衰减权重
        weights = np.array([r.get("weight", 1.0) for r in self._records])
        scores  = sims * weights

        top_idx = scores.argsort()[::-1][:top_k]
        return [self._records[i] for i in top_idx if scores[i] > 0]

    # ── 时间衰减更新 ──────────────────────────────────────
    def decay_weights(self, half_life_days: int = 30):
        """对所有记录按发布时间施加指数衰减。"""
        today = datetime.now()
        for r in self._records:
            try:
                pub_date = datetime.fromisoformat(r.get("date", today.isoformat()))
                delta    = (today - pub_date).days
                r["weight"] = float(np.exp(-0.693 * delta / half_life_days))
            except Exception:
                r["weight"] = 0.5
        self._save()

    # ── 查询 ──────────────────────────────────────────────
    def __len__(self):
        return len(self._records)

    def get_all(self) -> list[dict]:
        return list(self._records)

    def get_recent(self, n: int = 10) -> list[dict]:
        sorted_records = sorted(
            self._records,
            key=lambda r: r.get("date", ""),
            reverse=True,
        )
        return sorted_records[:n]

    def stats(self) -> dict:
        return {
            "total_records": len(self._records),
            "sources": list({r.get("source","?") for r in self._records}),
        }
