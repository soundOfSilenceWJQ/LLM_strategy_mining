from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_quant.info_based_summarize.agent import InfoBasedSummarizerAgent


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_one_day(input_path: Path, output_dir: Path) -> dict[str, Any]:
    analyzed_rows = read_json_list(input_path)
    if not analyzed_rows:
        raise RuntimeError(f"No analyzed rows found: {input_path}")

    date = input_path.stem.replace("advice_", "")
    agent = InfoBasedSummarizerAgent()
    result = agent.summarize_day(date=date, analyzed_rows=analyzed_rows)
    save_json(output_dir / f"summary_{date}.json", result)
    return {
        "date": date,
        "items": result["counts"]["items"],
        "news": result["counts"]["news"],
        "reports": result["counts"]["reports"],
        "stance": result["summary"]["investor_advice"].get("stance", ""),
        "summary_file": str((output_dir / f"summary_{date}.json")).replace('\\', '/'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily summary from analyzed mixed information files")
    parser.add_argument(
        "--input-dir",
        default="llm_quant/store/daily_advice",
        help="Directory containing per-day analyzed advice files",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_quant/store/daily_summary",
        help="Directory for per-day summary outputs",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Optional single day to summarize, e.g. 2023-09-01",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.date:
        files = [input_dir / f"advice_{args.date}.json"]
    else:
        files = sorted(input_dir.glob("advice_*.json"))

    results: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        results.append(summarize_one_day(path, output_dir))

    save_csv(output_dir / "daily_summary_index.csv", results)
    save_json(output_dir / "daily_summary_index.json", results)
    print(json.dumps({
        "days": len(results),
        "output_dir": str(output_dir),
        "sample": results[:3],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
