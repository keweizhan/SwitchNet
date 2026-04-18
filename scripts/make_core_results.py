"""
make_core_results.py -- Produce a curated core-results table from experiment summaries.

Reads results/*_summary.json for a fixed list of main project experiments and
writes two output files:

  results/core_results.csv   -- flat CSV, suitable for pandas / spreadsheet import
  results/core_results.md    -- GitHub-flavoured Markdown table, paste into README

Experiment groups included:
  A1  Order effect         (oracle-segment bilingual, 100 utterances, GPU)
  A2  Pause vs no-pause    (full-concat bilingual, 100 utterances, CPU)
  A3  Arrangement effect   (A/B/A pattern bilingual, 100 utterances, CPU)
  PREPROC  Preprocessing ablation  (oracle-segment bilingual, 20/10 utterances, CPU)

Usage:
  python scripts/make_core_results.py
  python scripts/make_core_results.py --results-dir results --output-dir results
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Core experiment registry
# Each entry: (group, tag)
# Order is the display order in the output tables.
# ---------------------------------------------------------------------------
CORE_EXPERIMENTS = [
    # A1: order effect -- oracle segments, 100 utterances
    # NOTE: "en-es" and "es-en" refer to the language order of the bilingual
    # clips (which language appears first), not a directional claim about
    # transcription performance.
    ("A1", "bilingual_en-es_100_seg"),
    ("A1", "bilingual_es-en_100_seg"),

    # A2: pause vs no-pause -- full_concat mode, 100 utterances, CPU run
    ("A2", "a2_es-en_100_nopause_cpu"),
    ("A2", "a2_es-en_100_pause05_cpu"),

    # A3: arrangement effect -- A/B/A patterns, 100 utterances, CPU run
    ("A3", "a3_en-es-en_100_pause05_cpu"),
    ("A3", "a3_es-en-es_100_pause05_cpu"),

    # PREPROC: preprocessing ablation -- oracle segments, CPU run
    # raw_20 and normalize_20 share the same 20-utterance set.
    # preemphasis_10 is on a smaller 10-utterance set (included for contrast).
    ("PREPROC", "preproc_raw_20"),
    ("PREPROC", "preproc_normalize_20"),
    ("PREPROC", "preproc_preemphasis_10"),
]

FIELDNAMES = ["group", "tag", "n_utterances", "overall_wer", "overall_mer", "sp_wer", "sp_mer"]


def _safe(d, *keys, default=""):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d if d is not None else default


def load_row(results_dir: Path, group: str, tag: str) -> dict:
    path = results_dir / f"{tag}_summary.json"
    base = {"group": group, "tag": tag,
            "n_utterances": "", "overall_wer": "", "overall_mer": "",
            "sp_wer": "", "sp_mer": ""}
    if not path.exists():
        print(f"  WARNING: {path.name} not found -- row will be empty", file=sys.stderr)
        return base
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARNING: could not parse {path.name}: {e}", file=sys.stderr)
        return base

    sp = data.get("switch_point") or {}
    return {
        "group":        group,
        "tag":          tag,
        "n_utterances": _safe(data, "overall", "n_utterances"),
        "overall_wer":  _safe(data, "overall", "wer"),
        "overall_mer":  _safe(data, "overall", "mer"),
        "sp_wer":       _safe(sp, "wer"),
        "sp_mer":       _safe(sp, "mer"),
    }


def write_csv(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV  -> {path}")


def fmt(v, decimals=4):
    """Format a numeric cell; return empty string if blank."""
    if v == "":
        return ""
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def write_md(rows, path: Path):
    headers = ["group", "tag", "n", "overall_wer", "overall_mer", "sp_wer", "sp_mer"]
    # Column widths
    col_data = {h: [h] for h in headers}
    for r in rows:
        col_data["group"].append(r["group"])
        col_data["tag"].append(r["tag"])
        col_data["n"].append(str(r["n_utterances"]))
        col_data["overall_wer"].append(fmt(r["overall_wer"]))
        col_data["overall_mer"].append(fmt(r["overall_mer"]))
        col_data["sp_wer"].append(fmt(r["sp_wer"]))
        col_data["sp_mer"].append(fmt(r["sp_mer"]))

    widths = {h: max(len(s) for s in col_data[h]) for h in headers}

    def row_line(cells):
        return "| " + " | ".join(c.ljust(widths[h]) for h, c in zip(headers, cells)) + " |"

    sep_line = "| " + " | ".join("-" * widths[h] for h in headers) + " |"

    lines = [row_line(headers), sep_line]
    for r in rows:
        cells = [
            r["group"],
            r["tag"],
            str(r["n_utterances"]),
            fmt(r["overall_wer"]),
            fmt(r["overall_mer"]),
            fmt(r["sp_wer"]),
            fmt(r["sp_mer"]),
        ]
        lines.append(row_line(cells))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote MD   -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Produce curated core-results table")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing *_summary.json files (default: results/)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write core_results.csv and core_results.md (default: results/)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_row(results_dir, group, tag) for group, tag in CORE_EXPERIMENTS]

    write_csv(rows, output_dir / "core_results.csv")
    write_md(rows,  output_dir / "core_results.md")

    # Terminal preview
    print()
    for line in open(output_dir / "core_results.md", encoding="utf-8"):
        print(line, end="")


if __name__ == "__main__":
    main()
