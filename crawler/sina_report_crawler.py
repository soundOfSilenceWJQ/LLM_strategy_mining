from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE = "https://stock.finance.sina.com.cn"
SEARCH_URL = BASE + "/stock/go.php/vReport_List/kind/search/index.phtml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

EXCLUDED_TYPES = {"公司", "行业", "债券", "创业板", "中小板", "基金", "晨报"}


@dataclass
class ReportRecord:
    title: str
    url: str
    report_type: str
    publish_date: str
    org: str
    analyst: str
    content: str
    page: int


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def to_abs_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    return urljoin(BASE, url)


def request_html(session: requests.Session, url: str, timeout: float = 15.0) -> str:
    resp = session.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "gb2312"
    return resp.text


def parse_list_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.tb_01")
    if not table:
        return []

    rows = []
    for tr in table.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue

        # 数据行第一列通常是序号
        serial = clean_text(tds[0].get_text())
        if not serial.isdigit():
            continue

        a = tds[1].find("a")
        title_text = a.get("title") if a and a.get("title") else tds[1].get_text()
        link_href = a.get("href") if a and a.get("href") else ""
        title = clean_text(str(title_text))
        link = to_abs_url(str(link_href))

        report_type = clean_text(tds[2].get_text())
        publish_date = clean_text(tds[3].get_text())
        org = clean_text(tds[4].get_text())
        analyst = clean_text(tds[5].get_text())

        if not title or not link:
            continue

        rows.append(
            {
                "title": title,
                "url": link,
                "report_type": report_type,
                "publish_date": publish_date,
                "org": org,
                "analyst": analyst,
            }
        )

    return rows


def parse_report_detail(html: str) -> tuple[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.content")
    if not main:
        return "", {}

    h1 = main.select_one("h1")
    title = clean_text(h1.get_text(" ", strip=True) if h1 else "")

    # 类别、机构、研究员、日期在creab区域
    meta = {
        "category": "",
        "org": "",
        "analyst": "",
        "publish_date": "",
    }
    creab = main.select_one("div.creab")
    if creab:
        text = creab.get_text(" ", strip=True)
        m_cat = re.search(r"类别：\s*([^\s]+)", text)
        m_org = re.search(r"机构：\s*([^\s]+)", text)
        m_ana = re.search(r"研究员：\s*([^\s]+)", text)
        m_date = re.search(r"日期：\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
        if m_cat:
            meta["category"] = clean_text(m_cat.group(1))
        if m_org:
            meta["org"] = clean_text(m_org.group(1))
        if m_ana:
            meta["analyst"] = clean_text(m_ana.group(1))
        if m_date:
            meta["publish_date"] = clean_text(m_date.group(1))

    body = main.select_one("div.blk_container")
    if not body:
        return title, meta

    # 把<br>转为空格后抽文本
    for br in body.find_all("br"):
        br.replace_with(" ")
    content = clean_text(body.get_text(" ", strip=True))
    return title, {**meta, "content": content}


def crawl_reports_by_date(
    pubdate: str,
    max_pages: Optional[int] = None,
    sleep_seconds: float = 0.25,
    min_content_len: int = 80,
) -> list[ReportRecord]:
    session = requests.Session()
    seen_urls: set[str] = set()
    seen_page_fingerprint: set[tuple[str, ...]] = set()
    records: list[ReportRecord] = []

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break

        params = {
            "t1": "6",
            "symbol": "",
            "pubdate": pubdate,
            "p": str(page),
        }
        html = request_html(session, SEARCH_URL, timeout=20)
        # 页面是通过query string读取，直接拼URL更稳定
        if page != 1:
            url = f"{SEARCH_URL}?t1=6&symbol=&pubdate={pubdate}&p={page}"
            html = request_html(session, url, timeout=20)
        else:
            url = f"{SEARCH_URL}?t1=6&symbol=&pubdate={pubdate}"
            html = request_html(session, url, timeout=20)

        rows = parse_list_page(html)
        if not rows:
            break

        fingerprint = tuple(sorted(r["url"] for r in rows))
        if fingerprint in seen_page_fingerprint:
            break
        seen_page_fingerprint.add(fingerprint)

        kept_on_page = 0
        for row in rows:
            r_type = row["report_type"]
            if r_type in EXCLUDED_TYPES:
                continue
            if row["url"] in seen_urls:
                continue
            seen_urls.add(row["url"])

            try:
                detail_html = request_html(session, row["url"], timeout=20)
                d_title, d_meta = parse_report_detail(detail_html)
                content = clean_text(d_meta.get("content", ""))
                if len(content) < min_content_len:
                    continue

                record = ReportRecord(
                    title=d_title or row["title"],
                    url=row["url"],
                    report_type=d_meta.get("category") or r_type,
                    publish_date=d_meta.get("publish_date") or row["publish_date"],
                    org=d_meta.get("org") or row["org"],
                    analyst=d_meta.get("analyst") or row["analyst"],
                    content=content,
                    page=page,
                )

                if record.report_type in EXCLUDED_TYPES:
                    continue

                records.append(record)
                kept_on_page += 1
            except Exception as e:
                print(f"[WARN] 详情抓取失败: {row['url']} | {e}")
            finally:
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        print(
            f"[INFO] page={page} list_rows={len(rows)} kept={kept_on_page} total_kept={len(records)}"
        )
        page += 1

    return records


def dump_json(records: list[ReportRecord], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)


def dump_csv(records: list[ReportRecord], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "title",
        "url",
        "report_type",
        "publish_date",
        "org",
        "analyst",
        "content",
        "page",
    ]
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def iterate_dates(start_date: str, end_date: str) -> list[str]:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("start_date cannot be later than end_date")

    dates: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description="新浪财经股票研报爬虫（按日期搜索）")
    parser.add_argument("--date", default="2026-05-15", help="发布日期，格式YYYY-MM-DD")
    parser.add_argument("--start-date", help="起始日期，格式YYYY-MM-DD")
    parser.add_argument("--end-date", help="结束日期，格式YYYY-MM-DD")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="最多抓取页数。为None时可手动改代码取消限制。",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="每篇详情请求的间隔秒数")
    parser.add_argument("--min-content-len", type=int, default=80, help="正文最小长度阈值")
    parser.add_argument(
        "--out-dir",
        default="crawler/output",
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date 和 --end-date 必须同时提供")

        all_records: list[ReportRecord] = []
        seen_urls: set[str] = set()
        dates = iterate_dates(args.start_date, args.end_date)
        for day in dates:
            print(f"[INFO] 开始抓取日期: {day}")
            day_records = crawl_reports_by_date(
                pubdate=day,
                max_pages=args.max_pages,
                sleep_seconds=args.sleep,
                min_content_len=args.min_content_len,
            )

            added = 0
            for r in day_records:
                if r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                all_records.append(r)
                added += 1

            print(
                f"[INFO] 日期={day} 抓取={len(day_records)} 新增={added} 累计={len(all_records)}"
            )

        records = all_records
        out_json = out_dir / f"sina_reports_{args.start_date}_{args.end_date}.json"
        out_csv = out_dir / f"sina_reports_{args.start_date}_{args.end_date}.csv"
    else:
        records = crawl_reports_by_date(
            pubdate=args.date,
            max_pages=args.max_pages,
            sleep_seconds=args.sleep,
            min_content_len=args.min_content_len,
        )
        out_json = out_dir / f"sina_reports_{args.date}.json"
        out_csv = out_dir / f"sina_reports_{args.date}.csv"

    dump_json(records, out_json)
    dump_csv(records, out_csv)

    print(f"[DONE] records={len(records)}")
    print(f"[DONE] json={out_json}")
    print(f"[DONE] csv={out_csv}")


if __name__ == "__main__":
    main()
