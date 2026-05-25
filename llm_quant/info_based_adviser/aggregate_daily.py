from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def normalize_news_record(record: dict[str, Any], source_file: str) -> dict[str, Any] | None:
    publish_time = str(record.get("publish_time", "")).strip()
    if not publish_time:
        return None
    publish_date = publish_time[:10]
    return {
        "source_type": "news",
        "section": str(record.get("section", "")).strip(),
        "title": str(record.get("title", "")).strip(),
        "url": str(record.get("url", "")).strip(),
        "publish_time": publish_time,
        "publish_date": publish_date,
        "content": str(record.get("content", "")).strip(),
        "source_file": source_file,
    }


def normalize_report_record(record: dict[str, Any], source_file: str) -> dict[str, Any] | None:
    publish_date = str(record.get("publish_date", "")).strip()
    if not publish_date:
        return None
    return {
        "source_type": "report",
        "section": "研报",
        "report_type": str(record.get("report_type", "")).strip(),
        "title": str(record.get("title", "")).strip(),
        "url": str(record.get("url", "")).strip(),
        "publish_time": publish_date,
        "publish_date": publish_date,
        "org": str(record.get("org", "")).strip(),
        "analyst": str(record.get("analyst", "")).strip(),
        "content": str(record.get("content", "")).strip(),
        "source_file": source_file,
    }


def aggregate_news(news_sections_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(news_sections_dir.glob("*.json")):
        for record in read_json_list(path):
            normalized = normalize_news_record(record, path.name)
            if normalized is not None:
                rows.append(normalized)
    return rows


def aggregate_reports(report_json_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_json_list(report_json_path):
        normalized = normalize_report_record(record, report_json_path.name)
        if normalized is not None:
            rows.append(normalized)
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate daily news/report files for info-based adviser pipeline")
    parser.add_argument(
        "--news-dir",
        default="crawler/output/sina_news/sections",
        help="Directory containing section-level news JSON files",
    )
    parser.add_argument(
        "--reports-json",
        default="crawler/output/reports/sina_reports_2023-09-01_2026-05-15.json",
        help="Path to merged reports JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_quant/store/daily_aggregated",
        help="Directory for per-day aggregated files",
    )
    parser.add_argument(
        "--start-date",
        default="",
        help="Optional inclusive start date, e.g. 2023-09-01",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="Optional inclusive end date, e.g. 2023-09-10",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    news_rows = aggregate_news(Path(args.news_dir))
    report_rows = aggregate_reports(Path(args.reports_json))
    all_rows = news_rows + report_rows

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        day = row.get("publish_date", "")
        if not day:
            continue
        if args.start_date and day < args.start_date:
            continue
        if args.end_date and day > args.end_date:
            continue
        grouped[day].append(row)

    summary: list[dict[str, Any]] = []
    daily_dir = output_dir / "days"
    for day in sorted(grouped):
        items = sorted(
            grouped[day],
            key=lambda x: (x.get("publish_time", ""), x.get("source_type", ""), x.get("title", "")),
        )
        out_path = daily_dir / f"info_bundle_{day}.json"
        save_json(out_path, items)
        news_count = sum(1 for x in items if x.get("source_type") == "news")
        report_count = sum(1 for x in items if x.get("source_type") == "report")
        summary.append({
            "date": day,
            "items": len(items),
            "news": news_count,
            "reports": report_count,
            "file": str(out_path).replace('\\', '/'),
        })

    save_json(output_dir / "daily_index.json", summary)
    print(json.dumps({
        "days": len(summary),
        "total_items": sum(x["items"] for x in summary),
        "output_dir": str(output_dir),
        "sample": summary[:3],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
