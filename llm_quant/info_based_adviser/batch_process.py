from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_quant.info_based_adviser.agent import InfoBasedAdviserAgent, save_json


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_weak_sample(
    record: dict[str, Any],
    min_content_chars: int = 80,
    allow_title_only: bool = False,
) -> tuple[bool, str]:
    title = normalize_text(str(record.get("title", "")))
    content = normalize_text(str(record.get("content", "")))
    url = str(record.get("url", "")).strip().lower()
    publish_time = str(record.get("publish_time", "")).strip()

    content_len = len(content)
    title_len = len(title)
    topic_like_url = any(token in url for token in ["/zt_d/", "subject-", "index.shtml"]) or "client.sina.com.cn/zt_d/" in url

    if not title:
        return True, "missing_title"

    if content_len == 0 and not allow_title_only:
        return True, "empty_content"

    if content_len < min_content_chars:
        if topic_like_url:
            return True, "topic_page_with_short_content"
        if not publish_time:
            return True, "short_content_without_publish_time"
        if title_len > 0 and content_len <= max(20, title_len // 2):
            return True, "content_too_short"

    return False, ""


def filter_weak_samples(
    records: list[dict[str, Any]],
    min_content_chars: int,
    allow_title_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    for record in records:
        weak, reason = is_weak_sample(
            record,
            min_content_chars=min_content_chars,
            allow_title_only=allow_title_only,
        )
        if weak:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept.append(record)

    return kept, reasons


def flatten_for_csv(item: dict[str, Any]) -> dict[str, Any]:
    src = item.get("source_record", {})
    ana = item.get("analysis", {})
    impact = ana.get("market_impact", {})
    sug = ana.get("investment_suggestion", {})

    return {
        "source_type": src.get("source_type", ""),
        "section": src.get("section", ""),
        "report_type": src.get("report_type", ""),
        "title": src.get("title", ""),
        "url": src.get("url", ""),
        "publish_time": src.get("publish_time", ""),
        "publish_date": src.get("publish_date", ""),
        "org": src.get("org", ""),
        "analyst": src.get("analyst", ""),
        "main_content": ana.get("main_content", ""),
        "pricing_logic": ana.get("pricing_logic", ""),
        "direction": impact.get("direction", ""),
        "target_assets": " | ".join(impact.get("target_assets", []) or []),
        "horizon_days": impact.get("horizon_days", ""),
        "confidence": impact.get("confidence", ""),
        "impact_reasoning": impact.get("reasoning", ""),
        "action": sug.get("action", ""),
        "for_investor": sug.get("for_investor", ""),
        "holding_period_days": sug.get("holding_period_days", ""),
        "position_guidance": sug.get("position_guidance", ""),
        "risk_controls": " | ".join(sug.get("risk_controls", []) or []),
        "risk_warnings": " | ".join(ana.get("risk_warnings", []) or []),
        "evidence_quotes": " | ".join(ana.get("evidence_quotes", []) or []),
        "processed_at": item.get("processed_at", ""),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [flatten_for_csv(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    if not flat_rows:
        return

    fieldnames = list(flat_rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


async def run_batch(
    records: list[dict[str, Any]],
    concurrency: int,
    request_timeout: int,
) -> list[dict[str, Any]]:
    agent = InfoBasedAdviserAgent()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    out: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    finished = 0
    finished_lock = asyncio.Lock()

    for rec in records:
        await q.put(rec)

    async def worker(worker_id: int) -> None:
        nonlocal finished
        while True:
            rec = await q.get()
            try:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(agent.analyze_record, rec),
                        timeout=max(1, request_timeout),
                    )
                except asyncio.TimeoutError:
                    result = agent._with_metadata(
                        rec,
                        agent._fallback(rec, f"request_timeout_{request_timeout}s"),
                    )
                async with lock:
                    out.append(result)
                async with finished_lock:
                    finished += 1
                    if finished % 10 == 0 or finished == len(records):
                        print(f"[PROGRESS] finished={finished}/{len(records)}")
            finally:
                q.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    await q.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    return out


def dedup_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        src = r.get("source_record", {})
        key = f"{src.get('url','')}|{src.get('title','')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent processing for single-item information advice")
    parser.add_argument(
        "--input",
        default="crawler/output/sina_news/sections/sina_news_产经.json",
        help="Input news/report json list",
    )
    parser.add_argument(
        "--output-json",
        default="llm_quant/store/info_advice_trial5.json",
        help="Output json path",
    )
    parser.add_argument(
        "--output-csv",
        default="llm_quant/store/info_advice_trial5.csv",
        help="Output csv path",
    )
    parser.add_argument("--limit", type=int, default=200, help="Only process first N records")
    parser.add_argument("--concurrency", type=int, default=100, help="Concurrent workers")
    parser.add_argument(
        "--disable-weak-filter",
        action="store_true",
        help="Disable weak-sample preprocessing",
    )
    parser.add_argument(
        "--min-content-chars",
        type=int,
        default=80,
        help="Minimum content length to keep a sample",
    )
    parser.add_argument(
        "--allow-title-only",
        action="store_true",
        help="Allow title-only records to pass preprocessing",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()

    raw_records = load_records(Path(args.input))
    filtered_reasons: dict[str, int] = {}

    if args.disable_weak_filter:
        records = raw_records
    else:
        records, filtered_reasons = filter_weak_samples(
            raw_records,
            min_content_chars=args.min_content_chars,
            allow_title_only=args.allow_title_only,
        )

    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        raise RuntimeError("No valid input records to process after preprocessing.")

    results = asyncio.run(run_batch(records=records, concurrency=args.concurrency, request_timeout=args.request_timeout))
    results = dedup_results(results)

    save_json(Path(args.output_json), results)
    save_csv(Path(args.output_csv), results)

    print(
        json.dumps(
            {
                "input": args.input,
                "raw_records": len(raw_records),
                "kept_after_filter": len(raw_records) if args.disable_weak_filter else len(raw_records) - sum(filtered_reasons.values()),
                "filtered_out": 0 if args.disable_weak_filter else sum(filtered_reasons.values()),
                "filtered_reasons": filtered_reasons,
                "processed": len(records),
                "saved": len(results),
                "output_json": args.output_json,
                "output_csv": args.output_csv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
