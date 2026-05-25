"""
第1层：信息提取层（Info Extraction Layer）
功能：
  - 爬取新浪财经、东方财富等财经新闻
  - 爬取学术/研报摘要（简化版）
  - 输出标准化信息记录
"""

from __future__ import annotations
import re
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_quant.config import CRAWL_TIMEOUT, CRAWL_MAX_ITEMS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ── 通用文本清理 ──────────────────────────────────────────
def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # 去除 JS 残留
    text = re.sub(r"<[^>]+>", "", text)
    return text[:2000]   # 截断避免过长


# ── 新浪财经头条 ──────────────────────────────────────────
def crawl_sina_finance(max_items: int = CRAWL_MAX_ITEMS) -> list[dict]:
    """爬取新浪财经首页资讯列表。"""
    url = "https://finance.sina.com.cn/roll/index.d.html?cateId=92&type=1"
    records = []
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=CRAWL_TIMEOUT)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        items = soup.select("ul.list_009 li") or soup.select("ul li")
        for li in items[:max_items]:
            a = li.find("a")
            if not a:
                continue
            title = _clean_text(a.get_text())
            link  = a.get("href", "")
            if not title or len(title) < 5:
                continue

            content = _fetch_article(link) if link.startswith("http") else ""
            records.append({
                "title":   title,
                "content": content,
                "url":     link,
                "source":  "新浪财经",
                "date":    datetime.now().strftime("%Y-%m-%d"),
            })
            time.sleep(0.3)
    except Exception as e:
        print(f"[InfoExtractor] 新浪财经爬取失败: {e}")

    return records


# ── 东方财富财经新闻 ──────────────────────────────────────
def crawl_eastmoney(max_items: int = CRAWL_MAX_ITEMS) -> list[dict]:
    """通过东方财富 API 获取要闻列表。"""
    url = (
        "https://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_"
        "50_1_.html?callback=jQuery"
    )
    records = []
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=CRAWL_TIMEOUT)
        text = resp.text
        # 提取 JSON 主体
        m = re.search(r"jQuery\((.*)\)$", text, re.DOTALL)
        if m:
            import json
            data = json.loads(m.group(1))
            items = data.get("LivesList", [])
            for item in items[:max_items]:
                title   = _clean_text(item.get("title", ""))
                content = _clean_text(item.get("digest", "") or item.get("content", ""))
                date_str = item.get("showtime", datetime.now().strftime("%Y-%m-%d"))
                if title:
                    records.append({
                        "title":   title,
                        "content": content,
                        "source":  "东方财富",
                        "date":    date_str[:10],
                    })
    except Exception as e:
        print(f"[InfoExtractor] 东方财富爬取失败: {e}")

    return records


# ── 文章正文抓取 ──────────────────────────────────────────
def _fetch_article(url: str, max_chars: int = 1000) -> str:
    """尝试抓取文章正文（容错）。"""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=CRAWL_TIMEOUT)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        # 取主要内容块
        for tag in ["article", "div.article", "div#article", "div.content"]:
            el = soup.select_one(tag)
            if el:
                return _clean_text(el.get_text())[:max_chars]
        # fallback：body text
        body = soup.find("body")
        if body:
            return _clean_text(body.get_text())[:max_chars]
    except Exception:
        pass
    return ""


# ── 手动添加自定义文本 ────────────────────────────────────
def add_manual_text(title: str, content: str, source: str = "手动") -> dict:
    """将用户手动输入的文本包装成标准记录。"""
    return {
        "title":   title,
        "content": content,
        "source":  source,
        "date":    datetime.now().strftime("%Y-%m-%d"),
    }


# ── 统一入口 ──────────────────────────────────────────────
class InfoExtractor:
    """
    第1层：信息提取层。
    编排多个爬虫，输出标准化记录列表。
    """

    def extract(self, sources: list[str] | None = None) -> list[dict]:
        """
        从指定来源批量抓取信息。
        sources: ["sina", "eastmoney"]，默认全部。
        """
        if sources is None:
            sources = ["sina", "eastmoney"]

        all_records = []
        if "sina" in sources:
            print("[Layer1] 正在抓取新浪财经...")
            all_records.extend(crawl_sina_finance())
        if "eastmoney" in sources:
            print("[Layer1] 正在抓取东方财富...")
            all_records.extend(crawl_eastmoney())

        print(f"[Layer1] 共抓取 {len(all_records)} 条原始信息。")
        return all_records

    def add_manual(self, title: str, content: str, source: str = "手动") -> dict:
        return add_manual_text(title, content, source)
