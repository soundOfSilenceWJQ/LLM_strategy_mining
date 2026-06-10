from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://finance.sina.com.cn"
HEAD_URL = BASE_URL + "/head/finance{date}{period}.shtml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://finance.sina.com.cn/",
}

WEIBO_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Referer": "https://weibo.com/",
    "Accept": "application/json,text/plain,*/*",
}


@dataclass
class NewsRecord:
    section: str
    title: str
    url: str
    home_time: str
    publish_time: str
    content: str


SECTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "要闻": ("blk_yw_1", "blk_yw_2", "blk_yw_3", "blk_yw_4"),
    "证券": ("blk_yw_zq_01_1", "blk_yw_zq_01_2", "blk_yw_zq_01_3", "blk_yw_zq_01_4"),
    "产经": ("blk_industry_",),
    "国内财经": ("blk_newsinland_01", "blk_guoneiNews_"),
    "大盘评述": ("blk_dpzs_01",),
    "博客看市": ("blk_blogom_01",),
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def to_abs_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    return urljoin(BASE_URL, url)


def is_target_article_url(url: str) -> bool:
    if not url:
        return False
    if url.startswith("javascript:") or url.startswith("mailto:"):
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    if "weibo.com" in host:
        return "ttarticle" in url
    if "sina.com.cn" not in host:
        return False
    return (
        "/doc-" in path
        or "/detail-" in path
        or "/2026-" in path
        or "/2025-" in path
        or "/2024-" in path
        or "/2023-" in path
        or "/zt_d/" in path
        or "/article/" in path
    )


def request_html(session: requests.Session, url: str, timeout: float = 20.0) -> str:
    resp = session.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_time_string(value: str) -> str:
    value = clean_text(value)
    if not value:
        return ""

    candidates = [value]
    if value.endswith("Z"):
        candidates.append(value[:-1] + "+00:00")

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(value, pattern)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return value


def extract_meta_time(soup: BeautifulSoup) -> str:
    meta_keys = [
        ("property", "article:published_time"),
        ("property", "bytedance:published_time"),
        ("property", "bytedance:lrDate_time"),
        ("name", "weibo: article:create_at"),
        ("property", "article:modified_time"),
    ]
    for attr_name, attr_value in meta_keys:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            return parse_time_string(tag["content"])
    return ""


def extract_title_from_soup(soup: BeautifulSoup) -> str:
    title_tag = soup.find("meta", attrs={"property": "og:title"})
    if title_tag and title_tag.get("content"):
        title = clean_text(title_tag["content"])
        if title:
            return title

    for selector in ("h1", "h2", "title"):
        tag = soup.find(selector)
        if tag:
            title = clean_text(tag.get_text(" ", strip=True))
            if title:
                return title
    return ""


def extract_article_content(soup: BeautifulSoup) -> str:
    body = soup.select_one("#artibody, .article-content, .article")
    if not body:
        return ""

    paras: list[str] = []
    for paragraph in body.find_all("p"):
        text = clean_text(paragraph.get_text(" ", strip=True))
        if not text:
            continue
        if text.startswith("(sinaads"):
            continue
        if text.startswith("登录新浪财经APP"):
            continue
        if text.startswith("专题："):
            continue
        if text.startswith("海量资讯、精准解读"):
            continue
        if text.startswith("责任编辑："):
            continue
        if text.startswith("新浪财经声明"):
            continue
        if text.startswith("郑重声明："):
            continue
        paras.append(text)

    if paras:
        return clean_text(" ".join(paras))

    raw = clean_text(body.get_text(" ", strip=True))
    raw = re.sub(r"^登录新浪财经APP.*?专题：.*?\s+", "", raw)
    return raw


def parse_finance_detail(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title_from_soup(soup)
    publish_time = extract_meta_time(soup)
    content = extract_article_content(soup)
    return title, publish_time, content


def parse_weibo_detail(url: str, session: requests.Session) -> tuple[str, str, str]:
    match = re.search(r"id=(\d+)", url)
    if not match:
        return "", "", ""

    oid = match.group(1)
    detail_url = f"https://weibo.com/ttarticle/x/m/aj/detail?id={oid}"
    resp = session.get(detail_url, headers=WEIBO_HEADERS, timeout=25.0)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) != "100000":
        return "", "", ""
    data = payload.get("data") or {}

    title = clean_text(data.get("title", ""))
    publish_time = clean_text(data.get("complete_create_at") or data.get("create_at") or "")

    content_html = data.get("content") or ""
    content_soup = BeautifulSoup(content_html, "html.parser")
    content = clean_text(content_soup.get_text(" ", strip=True))
    content = re.sub(r"^原文\s*[:：]\s*", "", content)

    return title, publish_time, content


def parse_detail(url: str, session: requests.Session) -> tuple[str, str, str]:
    if "weibo.com/ttarticle" in url:
        return parse_weibo_detail(url, session)
    html = request_html(session, url, timeout=25.0)
    return parse_finance_detail(html)


def collect_section_records(soup: BeautifulSoup, section: str) -> list[dict]:
    prefixes = SECTION_PREFIXES[section]
    results: list[dict] = []
    seen: set[str] = set()

    nodes: list[Tag] = []
    for prefix in prefixes:
        if prefix.endswith("_"):
            nodes.extend(
                node
                for node in soup.find_all(attrs={"data-sudaclick": re.compile(rf"^{re.escape(prefix)}")})
            )
        else:
            nodes.extend(
                node
                for node in soup.find_all(attrs={"data-sudaclick": prefix})
            )

    for node in nodes:
        for anchor in node.find_all("a", href=True):
            href = to_abs_url(anchor.get("href", ""))
            if not is_target_article_url(href):
                continue
            title = clean_text(anchor.get("title") or anchor.get_text(" ", strip=True))
            if not title:
                continue
            if href in seen:
                continue

            home_time = ""
            li = anchor.find_parent("li")
            if li:
                time_node = li.select_one("span.time, span.bd_i_time, span.fright")
                if time_node:
                    home_time = clean_text(time_node.get_text(" ", strip=True))
                if not home_time:
                    m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*$", clean_text(li.get_text(" ", strip=True)))
                    if m:
                        home_time = m.group(1)

            results.append({
                "section": section,
                "title": title,
                "url": href,
                "home_time": home_time,
            })
            seen.add(href)

    return results


def collect_home_records(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for section in SECTION_PREFIXES:
        for row in collect_section_records(soup, section):
            key = (row["section"], row["url"])
            if key in seen:
                continue
            seen.add(key)
            records.append(row)

    return records


def crawl_news_page(date: str, period: str, max_items: Optional[int] = None, sleep_seconds: float = 0.15) -> list[NewsRecord]:
    session = requests.Session()
    url = HEAD_URL.format(date=date.replace("-", ""), period=period)
    html = request_html(session, url, timeout=25.0)
    home_rows = collect_home_records(html)

    if max_items is not None and max_items > 0:
        home_rows = home_rows[:max_items]

    records: list[NewsRecord] = []
    for index, row in enumerate(home_rows, start=1):
        title, publish_time, content = "", "", ""
        try:
            title, publish_time, content = parse_detail(row["url"], session)
        except Exception as exc:
            print(f"[WARN] detail failed: {row['url']} | {exc}")

        record = NewsRecord(
            section=row["section"],
            title=title or row["title"],
            url=row["url"],
            home_time=row["home_time"],
            publish_time=publish_time or row["home_time"],
            content=content,
        )
        records.append(record)

        print(
            f"[INFO] {index:03d} section={record.section} time={record.publish_time or record.home_time} title={record.title[:40]}"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return records


def dump_json(records: list[NewsRecord], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in records], f, ensure_ascii=False, indent=2)


def dump_csv(records: list[NewsRecord], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "title", "url", "home_time", "publish_time", "content"]
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def iterate_dates(start_date: str, end_date: str):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    current = start_dt
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def parse_periods(periods_arg: str) -> list[str]:
    periods = [clean_text(part).lower() for part in periods_arg.split(",")]
    periods = [period for period in periods if period]
    if not periods:
        raise ValueError("periods cannot be empty")
    invalid = [period for period in periods if period not in {"am", "pm"}]
    if invalid:
        raise ValueError(f"invalid periods: {invalid}")
    return periods


def append_fail_log(log_path: Path, run_date: str, period: str, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{run_date},{period},{clean_text(message)}\n")


def run_range_crawl(
    *,
    start_date: str,
    end_date: str,
    periods: list[str],
    out_dir: Path,
    sleep_seconds: float,
    max_items: Optional[int],
    resume: bool,
    fail_log_path: Path,
) -> None:
    total_jobs = 0
    done_jobs = 0
    skipped_jobs = 0
    failed_jobs = 0

    for run_date in iterate_dates(start_date, end_date):
        for period in periods:
            total_jobs += 1
            suffix = f"{run_date}_{period}"
            out_json = out_dir / f"sina_news_{suffix}.json"
            out_csv = out_dir / f"sina_news_{suffix}.csv"

            if resume and out_json.exists() and out_csv.exists():
                skipped_jobs += 1
                print(f"[SKIP] {suffix} already exists")
                continue

            try:
                print(f"[RUN] {suffix}")
                records = crawl_news_page(
                    date=run_date,
                    period=period,
                    max_items=max_items,
                    sleep_seconds=sleep_seconds,
                )
                dump_json(records, out_json)
                dump_csv(records, out_csv)
                done_jobs += 1
                print(f"[DONE] {suffix} records={len(records)}")
            except Exception as exc:
                failed_jobs += 1
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[FAIL] {suffix} {msg}")
                append_fail_log(fail_log_path, run_date, period, msg)

    print(
        "[SUMMARY] "
        f"total_jobs={total_jobs} done={done_jobs} skipped={skipped_jobs} failed={failed_jobs}"
    )
    if failed_jobs > 0:
        print(f"[SUMMARY] fail_log={fail_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="新浪财经首页新闻爬虫")
    parser.add_argument("--date", default="2026-04-30", help="日期，格式YYYY-MM-DD")
    parser.add_argument("--start-date", default="", help="开始日期，格式YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="结束日期，格式YYYY-MM-DD")
    parser.add_argument("--period", default="am", choices=["am", "pm"], help="上午或下午")
    parser.add_argument("--periods", default="", help="区间抓取时使用，多个时段逗号分隔，如 am,pm")
    parser.add_argument("--out-dir", default="crawler/output", help="输出目录")
    parser.add_argument("--sleep", type=float, default=0.15, help="详情页请求间隔秒数")
    parser.add_argument("--max-items", type=int, default=0, help="最多抓取多少条，0表示不限制")
    parser.add_argument("--resume", action="store_true", help="断点恢复：已存在输出则跳过")
    parser.add_argument("--fail-log", default="", help="失败任务日志路径")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    max_items = None if args.max_items <= 0 else args.max_items

    if args.start_date or args.end_date:
        start_date = args.start_date or args.date
        end_date = args.end_date or start_date
        periods = parse_periods(args.periods) if args.periods else ["am", "pm"]
        fail_log_path = Path(args.fail_log) if args.fail_log else out_dir / "sina_news_failures.log"

        run_range_crawl(
            start_date=start_date,
            end_date=end_date,
            periods=periods,
            out_dir=out_dir,
            sleep_seconds=args.sleep,
            max_items=max_items,
            resume=args.resume,
            fail_log_path=fail_log_path,
        )
        return

    records = crawl_news_page(
        date=args.date,
        period=args.period,
        max_items=max_items,
        sleep_seconds=args.sleep,
    )

    suffix = f"{args.date}_{args.period}"
    out_json = out_dir / f"sina_news_{suffix}.json"
    out_csv = out_dir / f"sina_news_{suffix}.csv"

    dump_json(records, out_json)
    dump_csv(records, out_csv)

    section_counter: dict[str, int] = {}
    for record in records:
        section_counter[record.section] = section_counter.get(record.section, 0) + 1

    print(f"[DONE] records={len(records)}")
    print(f"[DONE] section_counts={json.dumps(section_counter, ensure_ascii=False, sort_keys=True)}")
    print(f"[DONE] json={out_json}")
    print(f"[DONE] csv={out_csv}")


if __name__ == "__main__":
    main()