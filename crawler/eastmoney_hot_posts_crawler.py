from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_HOST = "https://guba.eastmoney.com"
BOARD_SPECS = [
    {"board": "上证指数吧", "board_code": "zssh000001"},
    {"board": "沪深300吧", "board_code": "zssh000300"},
    {"board": "财经评论吧", "board_code": "cjpl"},
]
CUTOFF_DATE = datetime(2023, 1, 1)
OUT_DIR = Path("crawler/output")
OUT_JSON = OUT_DIR / "eastmoney_three_boards_until_2023_01.json"
SLEEP_SECONDS = 0.15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": BASE_HOST + "/",
}


@dataclass
class HotPostRecord:
    board: str
    board_code: str
    list_url: str
    publish_time: str
    read_count: int
    comment_count: int
    title: str
    post_url: str
    author: str
    author_url: str
    update_time: str
    content: str


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def to_abs_url(url: str, base_url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base_url, url)


def to_int(value: str) -> int:
    text = clean_text(value)
    if not text:
        return 0

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([万亿]?)", text)
    if not match:
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else 0

    number = float(match.group(1))
    unit = match.group(2)
    if unit == "万":
        number *= 10000
    elif unit == "亿":
        number *= 100000000
    return int(number)


def parse_datetime(value: str) -> datetime | None:
    value = clean_text(value)
    if not value:
        return None

    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value[:19], pattern)
        except ValueError:
            continue
    return None


def request_html(session: requests.Session, url: str, timeout: float = 20.0) -> str:
    for candidate in (url, url.replace("https://", "http://", 1) if url.startswith("https://") else url):
        try:
            resp = session.get(candidate, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
                resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc  # type: ignore[name-defined]


def board_list_url(board_code: str, page_no: int) -> str:
    return f"{BASE_HOST}/list,{board_code}.html" if page_no == 1 else f"{BASE_HOST}/list,{board_code}_{page_no}.html"


def parse_list_page(html: str, board: str, board_code: str, list_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    table_rows = soup.select("table tr")
    if table_rows:
        for tr in table_rows:
            tds = tr.select("td")
            if len(tds) < 5:
                continue

            title_a = tds[2].select_one("a[href]")
            author_a = tds[3].select_one("a[href]")
            if not title_a or not author_a:
                continue

            rows.append(
                {
                    "board": board,
                    "board_code": board_code,
                    "list_url": list_url,
                    "read_count": to_int(tds[0].get_text(" ", strip=True)),
                    "comment_count": to_int(tds[1].get_text(" ", strip=True)),
                    "title": clean_text(str(title_a.get("title") or title_a.get_text(" ", strip=True))),
                    "post_url": to_abs_url(str(title_a.get("href") or ""), list_url),
                    "author": clean_text(author_a.get_text(" ", strip=True)),
                    "author_url": to_abs_url(str(author_a.get("href") or ""), list_url),
                    "update_time": clean_text(tds[4].get_text(" ", strip=True)),
                }
            )
        if rows:
            return rows

    for row in soup.select("div.articleh.normal_post"):
        title_a = row.select_one("span.l3.a3 a[href]") or row.select_one("span.l3 a[href]")
        author_a = row.select_one("span.l4.a4 a[href]") or row.select_one("span.l4 a[href]")
        if not title_a or not author_a:
            continue

        rows.append(
            {
                "board": board,
                "board_code": board_code,
                "list_url": list_url,
                "read_count": to_int((row.select_one("span.l1.a1") or row).get_text(" ", strip=True)),
                "comment_count": to_int((row.select_one("span.l2.a2") or row).get_text(" ", strip=True)),
                "title": clean_text(str(title_a.get("title") or title_a.get_text(" ", strip=True))),
                "post_url": to_abs_url(str(title_a.get("href") or ""), list_url),
                "author": clean_text(author_a.get_text(" ", strip=True)),
                "author_url": to_abs_url(str(author_a.get("href") or ""), list_url),
                "update_time": clean_text((row.select_one("span.l5.a5") or row.select_one("span.l5") or row).get_text(" ", strip=True)),
            }
        )

    return rows


def extract_detail_info(html: str) -> tuple[str, str]:
    publish_time = ""
    for pattern in (
        r'"post_publish_time":"([^"]+)"',
        r'"post_display_time":"([^"]+)"',
        r'"publish_time":"([^"]+)"',
    ):
        match = re.search(pattern, html)
        if match:
            publish_time = clean_text(match.group(1))
            break

    soup = BeautifulSoup(html, "html.parser")
    body = (
        soup.select_one(".article.page-article .article-body")
        or soup.select_one(".article-body")
        or soup.select_one(".xeditor_content.app_h5_modify")
        or soup.select_one("article")
    )
    if not body:
        return publish_time, ""

    paras: list[str] = []
    for paragraph in body.find_all("p"):
        text = clean_text(paragraph.get_text(" ", strip=True))
        if not text:
            continue
        if text.startswith(("以下内容仅代表作者", "股市有风险", "炒股第一步，先开个股票账户")):
            continue
        if "本文作者可以追加内容" in text:
            continue
        paras.append(text)

    if paras:
        return publish_time, "\n".join(paras)
    return publish_time, clean_text(body.get_text(" ", strip=True))


def fetch_detail_info(session: requests.Session, post_url: str) -> tuple[str, str]:
    html = request_html(session, post_url, timeout=25.0)
    return extract_detail_info(html)


def dedupe_by_url(records: Iterable[HotPostRecord]) -> list[HotPostRecord]:
    seen: set[str] = set()
    result: list[HotPostRecord] = []
    for record in records:
        if record.post_url in seen:
            continue
        seen.add(record.post_url)
        result.append(record)
    return result


def crawl_board(session: requests.Session, board: str, board_code: str) -> list[HotPostRecord]:
    records: list[HotPostRecord] = []
    page_no = 1

    while True:
        list_url = board_list_url(board_code, page_no)
        try:
            list_html = request_html(session, list_url, timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] list failed board={board} page={page_no} url={list_url} err={exc}")
            break

        page_rows = parse_list_page(list_html, board=board, board_code=board_code, list_url=list_url)
        if not page_rows:
            print(f"[INFO] board={board} page={page_no} no rows, stop")
            break

        reached_cutoff = False
        kept_on_page = 0
        for idx, row in enumerate(page_rows, start=1):
            try:
                publish_time, content = fetch_detail_info(session, row["post_url"])
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] detail failed board={board} page={page_no} idx={idx} url={row['post_url']} err={exc}")
                continue

            publish_dt = parse_datetime(publish_time)
            if publish_dt and publish_dt < CUTOFF_DATE:
                reached_cutoff = True
                continue

            records.append(
                HotPostRecord(
                    board=row["board"],
                    board_code=row["board_code"],
                    list_url=row["list_url"],
                    publish_time=publish_time,
                    read_count=row["read_count"],
                    comment_count=row["comment_count"],
                    title=row["title"],
                    post_url=row["post_url"],
                    author=row["author"],
                    author_url=row["author_url"],
                    update_time=row["update_time"],
                    content=content,
                )
            )
            kept_on_page += 1
            print(
                f"[INFO] board={board} page={page_no} idx={idx} publish_time={publish_time or 'unknown'} "
                f"content_len={len(content)}"
            )

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

        print(f"[INFO] board={board} page={page_no} list_rows={len(page_rows)} kept={kept_on_page} total={len(records)}")
        if reached_cutoff:
            print(f"[INFO] board={board} reached cutoff {CUTOFF_DATE:%Y-%m}, stop")
            break

        page_no += 1

    return dedupe_by_url(records)


def crawl_all() -> list[HotPostRecord]:
    session = requests.Session()
    all_records: list[HotPostRecord] = []
    for spec in BOARD_SPECS:
        board_records = crawl_board(session, board=spec["board"], board_code=spec["board_code"])
        all_records.extend(board_records)
    return dedupe_by_url(all_records)


def dump_json(records: list[HotPostRecord], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    records = crawl_all()
    dump_json(records, OUT_JSON)
    print(f"[DONE] total={len(records)}")
    print(f"[DONE] json={OUT_JSON}")


if __name__ == "__main__":
    main()
