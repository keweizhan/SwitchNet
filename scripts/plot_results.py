"""
plot_results.py -- Generate comparison plots from aggregated_results.csv.

Reads results/aggregated_results.csv (or a custom path) and writes three
PNG plots to results/plots/:

  wer_by_group.png    Bar chart of overall_wer and sp_wer, grouped by
                      experiment series (A1 / A2 / A3 / preproc / baseline).

  wer_vs_sp_wer.png   Scatter of overall_wer vs switch-point WER for all
                      bilingual entries that have both metrics.

  backend_compare.png Side-by-side overall_wer for whisper vs whisperx on
                      matched tags (tags where both wx_<tag> and <tag> exist).
                      Skipped if no whisperx results are present yet.

Requires matplotlib (pip install matplotlib).

Usage:
  python scripts/plot_results.py
  python scripts/plot_results.py --csv results/aggregated_results.csv
  python scripts/plot_results.py --output-dir results/plots --show
"""

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> list[dict]:
    """Load aggregated_results.csv into a list of row dicts."""
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run: python scripts/aggregate_results.py")
        sys.exit(1)
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str):
    """Parse a float from a row dict; return None if missing or blank."""
    v = row.get(key, "")
    if v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Experiment grouping
# ---------------------------------------------------------------------------

_GROUP_RULES = [
    ("a1",      lambda t: t.startswith(("bilingual_en-es_", "bilingual_es-en_"))),
    ("a2",      lambda t: t.startswith("a2_")),
    ("a3",      lambda t: t.startswith("a3_")),
    ("preproc", lambda t: t.startswith("preproc_")),
    ("baseline",lambda t: t.startswith(("en_", "es_"))),
    ("bench",   lambda t: t.startswith("bench_")),
]

def _group(tag: str) -> str:
    for name, pred in _GROUP_RULES:
        if pred(tag):
            return name
    return "other"


# ---------------------------------------------------------------------------
# Plot 1: WER by group
# ---------------------------------------------------------------------------

def plot_wer_by_group(rows: list[dict], output_dir: Path, show: bool):
    import matplotlib.pyplot as plt

    # Filter to rows with at least overall_wer
    plot_rows = [r for r in rows if _float(r, "overall_wer") is not None]
    if not plot_rows:
        print("  wer_by_group.png: no rows with overall_wer — skipped.")
        return

    # Sort by group then tag for a consistent visual order
    group_order = {g: i for i, (g, _) in enumerate(_GROUP_RULES)}
    plot_rows.sort(key=lambda r: (group_order.get(_group(r["tag"]), 99), r["tag"]))

    tags    = [r["tag"] for r in plot_rows]
    overall = [_float(r, "overall_wer") for r in plot_rows]
    sp_wer  = [_float(r, "sp_wer") for r in plot_rows]

    x = range(len(tags))
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 0.55), 5))
    ax.bar([i - bar_w / 2 for i in x], overall, bar_w,
           label="overall WER", color="#4c72b0")
    # Draw switch-point WER only for rows that have it
    for i, v in enumerate(sp_wer):
        if v is not None:
            ax.bar(i + bar_w / 2, v, bar_w, color="#dd8452",
                   label="switch-pt WER" if i == next(
                       (j for j, w in enumerate(sp_wer) if w is not None), -1
                   ) else "_nolegend_")

    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("WER")
    ax.set_title("Overall WER and Switch-Point WER by Experiment")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "wer_by_group.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Overall WER vs Switch-Point WER scatter
# ---------------------------------------------------------------------------

def plot_wer_vs_sp_wer(rows: list[dict], output_dir: Path, show: bool):
    import matplotlib.pyplot as plt

    scatter_rows = [
        r for r in rows
        if _float(r, "overall_wer") is not None and _float(r, "sp_wer") is not None
    ]
    if not scatter_rows:
        print("  wer_vs_sp_wer.png: no rows with both overall_wer and sp_wer — skipped.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in scatter_rows:
        x = _float(r, "overall_wer")
        y = _float(r, "sp_wer")
        ax.scatter(x, y, s=60, zorder=3)
        ax.annotate(r["tag"], (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, alpha=0.8)

    # Diagonal reference line (sp_wer == overall_wer)
    lim = max(
        max(_float(r, "overall_wer") for r in scatter_rows),
        max(_float(r, "sp_wer") for r in scatter_rows),
    ) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1, label="sp_wer = overall_wer")

    ax.set_xlabel("Overall WER")
    ax.set_ylabel("Switch-Point WER")
    ax.set_title("Overall WER vs Switch-Point WER")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = output_dir / "wer_vs_sp_wer.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Backend comparison (whisper vs whisperx on matched tags)
# ---------------------------------------------------------------------------

def plot_backend_compare(rows: list[dict], output_dir: Path, show: bool):
    import matplotlib.pyplot as plt

    wx_tags   = {r["tag"][3:]: r for r in rows if r["tag"].startswith("wx_")}
    base_tags = {r["tag"]: r    for r in rows if not r["tag"].startswith("wx_")}

    matched = sorted(wx_tags.keys() & base_tags.keys())
    if not matched:
        print("  backend_compare.png: no matched whisper/whisperx tag pairs — skipped.")
        return

    whisper_wer  = [_float(base_tags[t], "overall_wer") for t in matched]
    whisperx_wer = [_float(wx_tags[t],   "overall_wer") for t in matched]

    x = range(len(matched))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(matched) * 1.2), 5))
    ax.bar([i - bar_w / 2 for i in x], whisper_wer,  bar_w,
           label="openai-whisper", color="#4c72b0")
    ax.bar([i + bar_w / 2 for i in x], whisperx_wer, bar_w,
           label="whisperx",       color="#55a868")

    ax.set_xticks(list(x))
    ax.set_xticklabels(matched, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Overall WER")
    ax.set_title("openai-whisper vs WhisperX: Overall WER")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "backend_compare.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate WER comparison plots from aggregated_results.csv"
    )
    parser.add_argument(
        "--csv", default="results/aggregated_results.csv",
        help="Path to aggregated_results.csv (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output-dir", default="results/plots",
        help="Directory to write PNG files (default: results/plots)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display plots interactively (requires a display)",
    )
    parser.add_argument(
        "--pattern", default=None,
        help="Only plot rows whose tag contains this substring",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    csv_path   = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    if args.pattern:
        rows = [r for r in rows if args.pattern in r["tag"]]
    if not rows:
        print("No rows to plot.")
        return

    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"Writing plots to {output_dir}/\n")

    plot_wer_by_group(rows, output_dir, args.show)
    plot_wer_vs_sp_wer(rows, output_dir, args.show)
    plot_backend_compare(rows, output_dir, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
