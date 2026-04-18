"""
compare_baselines.py -- Compare Whisper vs WhisperX across EN / ES / Bilingual conditions.

Reads pre-existing *_summary.json files from results/ and prints a markdown
comparison table. Missing files are skipped with a note so you can run
whichever evaluations are still pending.

Expected summary files (all in results/):
  en_librispeech_50_large_cpu_summary.json    <- Whisper EN
  wx_en_librispeech_50_large_cpu_summary.json <- WhisperX EN  (run if missing)
  es_mls_50_large_cpu_summary.json            <- Whisper ES
  wx_es_mls_50_large_cpu_summary.json         <- WhisperX ES  (run if missing)
  bilingual_es-en_50_large_cpu_summary.json   <- Whisper Bilingual
  wx_bilingual_es-en_50_large_cpu_summary.json<- WhisperX Bilingual

Generate missing WhisperX results with:
  python scripts/run_eval_whisperx.py \\
      --manifest data/manifests/en_librispeech_test.jsonl \\
      --model large-v3 --device cpu --compute-type int8 --max 50 \\
      --tag wx_en_librispeech_50_large_cpu

  python scripts/run_eval_whisperx.py \\
      --manifest data/manifests/es_mls_test.jsonl \\
      --model large-v3 --device cpu --compute-type int8 --max 50 \\
      --tag wx_es_mls_50_large_cpu

Usage:
  python scripts/compare_baselines.py
  python scripts/compare_baselines.py --output results/baseline_comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results")

EXPERIMENTS = [
    # (condition, backend, tag)
    ("EN",        "whisper",  "en_librispeech_50_large_cpu"),
    ("EN",        "whisperx", "wx_en_librispeech_50_large_cpu"),
    ("ES",        "whisper",  "es_mls_50_large_cpu"),
    ("ES",        "whisperx", "wx_es_mls_50_large_cpu"),
    ("Bilingual", "whisper",  "bilingual_es-en_50_large_cpu"),
    ("Bilingual", "whisperx", "wx_bilingual_es-en_50_large_cpu"),
]


def load_summary(tag: str) -> dict | None:
    path = RESULTS_DIR / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: could not parse {path.name}: {e}", file=sys.stderr)
        return None


def fmt_pct(v) -> str:
    if v is None or v == "":
        return "—"
    return f"{float(v):.2%}"


def fmt_delta(v_wx, v_w) -> str:
    if v_wx is None or v_w is None:
        return "—"
    delta = (float(v_wx) - float(v_w)) / float(v_w) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Whisper vs WhisperX baseline comparison")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    rows = []
    for condition, backend, tag in EXPERIMENTS:
        data = load_summary(tag)
        if data is None:
            print(f"  MISSING: results/{tag}_summary.json — skipping", file=sys.stderr)
            rows.append({
                "condition": condition, "backend": backend,
                "wer": None, "mer": None, "sp_wer": None, "sp_mer": None,
                "n": None, "tag": tag,
            })
        else:
            sp = data.get("switch_point") or {}
            rows.append({
                "condition": condition,
                "backend":   backend,
                "wer":       data["overall"].get("wer"),
                "mer":       data["overall"].get("mer"),
                "sp_wer":    sp.get("wer"),
                "sp_mer":    sp.get("mer"),
                "n":         data["overall"].get("n_utterances"),
                "tag":       tag,
            })

    # Build paired view
    pairs = {}  # condition -> {whisper: row, whisperx: row}
    for r in rows:
        pairs.setdefault(r["condition"], {})[r["backend"]] = r

    # Print markdown table
    print("\n## Whisper vs WhisperX — Single-Language and Bilingual Baselines")
    print(f"{'Condition':<12} {'Backend':<10} {'n':>4}  {'WER':>8}  {'MER':>8}  {'SP-WER':>8}")
    print("-" * 62)
    for condition in ["EN", "ES", "Bilingual"]:
        for backend in ["whisper", "whisperx"]:
            r = pairs.get(condition, {}).get(backend)
            if r is None:
                continue
            n_str  = str(r["n"]) if r["n"] is not None else "—"
            wer_s  = fmt_pct(r["wer"])
            mer_s  = fmt_pct(r["mer"])
            sp_s   = fmt_pct(r["sp_wer"])
            print(f"{condition:<12} {backend:<10} {n_str:>4}  {wer_s:>8}  {mer_s:>8}  {sp_s:>8}")
        # Print relative improvement row if both backends present
        w  = pairs.get(condition, {}).get("whisper")
        wx = pairs.get(condition, {}).get("whisperx")
        if w and wx and w["wer"] is not None and wx["wer"] is not None:
            d_wer = fmt_delta(wx["wer"], w["wer"])
            d_mer = fmt_delta(wx["mer"], w["mer"])
            d_sp  = fmt_delta(wx["sp_wer"], w["sp_wer"])
            print(f"{'':12} {'Δ (rel)':10} {'':>4}  {d_wer:>8}  {d_mer:>8}  {d_sp:>8}")
        print()

    # Save CSV
    if args.output:
        csv_path = Path(args.output)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "condition", "backend", "n", "wer", "mer", "sp_wer", "sp_mer", "tag"
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
