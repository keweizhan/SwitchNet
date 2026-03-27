"""
aggregate_results.py -- Collect results/*_summary.json into one flat CSV.

Scans the results/ directory for files matching *_summary.json, extracts
key metrics from each, and writes results/aggregated_results.csv.

Fields in the CSV:
  tag              - experiment name derived from filename (strip _summary.json)
  n_utterances     - overall.n_utterances
  overall_wer      - overall.wer
  overall_mer      - overall.mer
  sp_wer           - switch_point.wer  (null -> "")
  sp_mer           - switch_point.mer  (null -> "")
  sp_n             - switch_point.n    (null -> "")

Usage:
  # All experiments
  python scripts/aggregate_results.py

  # Only A2 experiments
  python scripts/aggregate_results.py --pattern a2_

  # Custom output path
  python scripts/aggregate_results.py --output results/a2_summary_table.csv --pattern a2_

  # Sort by overall_wer ascending
  python scripts/aggregate_results.py --sort overall_wer
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = RESULTS_DIR / "aggregated_results.csv"

FIELDNAMES = [
    "tag",
    "n_utterances",
    "overall_wer",
    "overall_mer",
    "sp_wer",
    "sp_mer",
    "sp_n",
]

SORT_KEYS = set(FIELDNAMES) - {"tag"}


def _safe(d, *keys, default=""):
    """Drill into nested dict d via keys; return default if any key is missing or None."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d if d is not None else default


def load_summary(path: Path) -> dict:
    """Parse one *_summary.json and return a flat row dict."""
    tag = path.name.replace("_summary.json", "")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARNING: could not parse {path.name}: {e}", file=sys.stderr)
        return {"tag": tag, "n_utterances": "", "overall_wer": "", "overall_mer": "",
                "sp_wer": "", "sp_mer": "", "sp_n": ""}

    sp = data.get("switch_point") or {}

    return {
        "tag":          tag,
        "n_utterances": _safe(data, "overall", "n_utterances"),
        "overall_wer":  _safe(data, "overall", "wer"),
        "overall_mer":  _safe(data, "overall", "mer"),
        "sp_wer":       _safe(sp, "wer"),
        "sp_mer":       _safe(sp, "mer"),
        "sp_n":         _safe(sp, "n"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate SwitchNet experiment summaries into CSV")
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Directory to scan for *_summary.json files (default: results/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <results-dir>/aggregated_results.csv)",
    )
    parser.add_argument(
        "--pattern", default=None,
        help="Only include files whose tag contains this substring",
    )
    parser.add_argument(
        "--sort", default="tag", choices=FIELDNAMES,
        help="Sort output rows by this column (default: tag)",
    )
    parser.add_argument(
        "--descending", action="store_true",
        help="Sort descending instead of ascending",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "aggregated_results.csv"

    # Collect summary files
    summary_files = sorted(results_dir.glob("*_summary.json"))
    if not summary_files:
        print(f"No *_summary.json files found in {results_dir}")
        return

    # Filter by pattern
    if args.pattern:
        summary_files = [f for f in summary_files if args.pattern in f.name]
        if not summary_files:
            print(f"No files matching pattern {args.pattern!r} in {results_dir}")
            return

    rows = [load_summary(f) for f in summary_files]

    # Sort
    def sort_key(row):
        v = row[args.sort]
        # Put empty values last regardless of direction
        if v == "":
            return (1, 0)
        try:
            return (0, float(v))
        except (ValueError, TypeError):
            return (0, str(v))

    rows.sort(key=sort_key, reverse=args.descending)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {output_path}")

    # Print a quick preview table
    col_w = {
        "tag": max(len(r["tag"]) for r in rows),
        "n_utterances": 4,
        "overall_wer": 11,
        "overall_mer": 11,
        "sp_wer": 8,
        "sp_mer": 8,
        "sp_n": 6,
    }
    header = (
        f"{'tag':<{col_w['tag']}}  "
        f"{'n':>{col_w['n_utterances']}}  "
        f"{'overall_wer':>{col_w['overall_wer']}}  "
        f"{'overall_mer':>{col_w['overall_mer']}}  "
        f"{'sp_wer':>{col_w['sp_wer']}}  "
        f"{'sp_mer':>{col_w['sp_mer']}}  "
        f"{'sp_n':>{col_w['sp_n']}}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(v, width, is_float=False):
            if v == "":
                return " " * width
            if is_float:
                try:
                    return f"{float(v):.4f}".rjust(width)
                except (ValueError, TypeError):
                    pass
            return str(v).rjust(width)

        print(
            f"{r['tag']:<{col_w['tag']}}  "
            f"{fmt(r['n_utterances'], col_w['n_utterances'])}  "
            f"{fmt(r['overall_wer'],  col_w['overall_wer'],  True)}  "
            f"{fmt(r['overall_mer'],  col_w['overall_mer'],  True)}  "
            f"{fmt(r['sp_wer'],       col_w['sp_wer'],       True)}  "
            f"{fmt(r['sp_mer'],       col_w['sp_mer'],       True)}  "
            f"{fmt(r['sp_n'],         col_w['sp_n'])}"
        )


if __name__ == "__main__":
    main()
