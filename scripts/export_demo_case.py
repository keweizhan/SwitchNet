"""
export_demo_case.py — Generate a demo comparison folder for one bilingual entry.

Always produces (no model required):
  results/subtitles/demo_cases/<entry_id>/
    reference.srt    -- ground-truth SRT with explicit [ES→EN] boundary cue
    reference.txt    -- plain-text reference showing [ES] / SWITCH / [EN]
    comparison.md    -- side-by-side markdown summary

Optional (supply pre-existing result files to add model SRTs):
    whisper.srt      -- from --whisper-jsonl (eval JSONL, timing inferred from audio)
                        or --whisper-sidecar  (export_subtitles.py sidecar JSON)
    whisperx.srt     -- from --whisperx-jsonl (WhisperX eval JSONL)

Both Whisper and WhisperX eval JSONLs use the same segment_outputs schema
{position, language, reference, hypothesis}, so a single code path handles both.
Timing is inferred from the manifest segment audio durations (no model re-run).

Usage
-----
  # Reference only (instant, no model):
  python scripts/export_demo_case.py \\
      --manifest data/manifests/bilingual_es-en_50.jsonl \\
      --limit 1

  # Full comparison — Whisper + WhisperX from existing eval JSONLs:
  python scripts/export_demo_case.py \\
      --manifest        data/manifests/bilingual_es-en_50.jsonl \\
      --entry-id        bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001 \\
      --whisper-jsonl   results/bilingual_es-en_50_large_cpu.jsonl \\
      --whisperx-jsonl  results/wx_bilingual_es-en_50_large_cpu.jsonl

  # Whisper from export_subtitles.py sidecar (pre-processed, has exact timing):
  python scripts/export_demo_case.py \\
      --manifest        data/manifests/bilingual_es-en_50.jsonl \\
      --entry-id        bilingual_es-en_0022_... \\
      --whisper-sidecar results/subtitles/some_run/bilingual_es-en_0022_....json \\
      --whisperx-jsonl  results/wx_bilingual_es-en_50_large_cpu.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import load_manifest
from src.asr.subtitles import (
    SubtitleCue,
    build_segment_level_cues,
    write_srt,
    write_reference_srt,
    write_reference_txt,
    write_comparison_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonl_to_srt(jsonl_path: Path, entry, output_path: Path) -> int:
    """
    Build a model .srt from a transcription eval JSONL (Whisper or WhisperX).

    Both backends write the same segment_outputs schema:
        [{"position": int, "language": str, "reference": str, "hypothesis": str}, ...]

    Timing is inferred from the manifest entry's segment audio durations via
    build_segment_level_cues (same path used by export_subtitles.py).

    Returns the number of cues written.
    Raises ValueError if the entry ID is not found in the JSONL.
    """
    rec = None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id") == entry.id:
                rec = r
                break

    if rec is None:
        raise ValueError(f"Entry {entry.id!r} not found in {jsonl_path}")

    seg_outputs = rec.get("segment_outputs")
    if not seg_outputs:
        # Flat record (no per-segment split): wrap as a single segment output.
        seg_outputs = [{
            "position":   0,
            "language":   entry.language,
            "reference":  entry.transcript,
            "hypothesis": rec.get("hypothesis", ""),
        }]

    cues = build_segment_level_cues(
        seg_outputs,
        entry.segments,
        infer_timing_from_audio=True,
    )
    write_srt(cues, output_path, add_period=True, max_chars_per_line=42,
              subtitle_mode="english")
    return len(cues)


def _sidecar_to_srt(sidecar_path: Path, output_path: Path) -> int:
    """
    Rebuild a Whisper .srt from an export_subtitles.py sidecar JSON.

    The sidecar already has pre-computed cue timestamps and cleaned text,
    so this skips audio duration inference entirely.
    Returns the number of cues written.
    """
    with open(sidecar_path, encoding="utf-8") as f:
        data = json.load(f)

    cues = []
    for c in data.get("cues", []):
        text = c.get("text", "").strip()
        if not text:
            continue
        cues.append(SubtitleCue(
            start=float(c["start"]),
            end=float(c["end"]),
            text=text,
            source_language=c.get("source_language"),
            timing_source=c.get("timing_source", "sidecar"),
            source_text=c.get("source_text"),
        ))

    subtitle_mode = data.get("subtitle_mode", "english")
    write_srt(cues, output_path, add_period=False,
              max_chars_per_line=42, subtitle_mode=subtitle_mode)
    return len(cues)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a demo comparison folder for one bilingual manifest entry. "
            "reference.srt / reference.txt / comparison.md are always produced. "
            "whisper.srt and whisperx.srt are produced when the corresponding "
            "result files are supplied."
        )
    )
    parser.add_argument(
        "--manifest", required=True,
        help="JSONL manifest file containing the entry",
    )
    parser.add_argument(
        "--entry-id", default=None,
        help="Process exactly this entry ID (overrides --limit)",
    )
    parser.add_argument(
        "--limit", type=int, default=1,
        help="Process the first N bilingual entries when --entry-id is not set (default 1)",
    )
    parser.add_argument(
        "--output-root", default="results/subtitles/demo_cases",
        help="Root output dir; each entry writes to <output-root>/<entry_id>/",
    )
    # --- Whisper sources (pick one) ---
    whisper_group = parser.add_mutually_exclusive_group()
    whisper_group.add_argument(
        "--whisper-jsonl", default=None, metavar="PATH",
        help=(
            "Whisper eval JSONL (results/<tag>.jsonl from run_eval.py). "
            "Timing is inferred from manifest audio durations."
        ),
    )
    whisper_group.add_argument(
        "--whisper-sidecar", default=None, metavar="PATH",
        help=(
            "export_subtitles.py sidecar JSON for this entry. "
            "Has pre-computed timestamps; preferred when available."
        ),
    )
    # --- WhisperX source ---
    parser.add_argument(
        "--whisperx-jsonl", default=None, metavar="PATH",
        help=(
            "WhisperX eval JSONL (results/wx_<tag>.jsonl from run_eval_whisperx.py). "
            "Same segment_outputs schema as Whisper JSONL."
        ),
    )
    args = parser.parse_args()

    # --- load and select entries -------------------------------------------
    all_entries = load_manifest(args.manifest)
    entries = [e for e in all_entries if e.language == "bilingual"]

    if args.entry_id:
        entries = [e for e in entries if e.id == args.entry_id]
        if not entries:
            print(f"No bilingual entry with id={args.entry_id!r} in {args.manifest}.")
            sys.exit(1)
    else:
        entries = entries[: args.limit]

    if not entries:
        print("No bilingual entries found. Nothing to do.")
        sys.exit(1)

    output_root       = Path(args.output_root)
    whisper_jsonl     = Path(args.whisper_jsonl)    if args.whisper_jsonl    else None
    whisper_sidecar   = Path(args.whisper_sidecar)  if args.whisper_sidecar  else None
    whisperx_jsonl    = Path(args.whisperx_jsonl)   if args.whisperx_jsonl   else None

    # --- process each entry ------------------------------------------------
    for entry in entries:
        out_dir = output_root / entry.id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"Entry : {entry.id}")
        print(f"Output: {out_dir}")
        print(f"{'='*72}")

        # ── Reference (always, no model) ────────────────────────────────────
        write_reference_srt(entry.segments, out_dir / "reference.srt")
        print("  [OK] reference.srt")

        write_reference_txt(entry.segments, out_dir / "reference.txt")
        print("  [OK] reference.txt")

        # ── Whisper SRT ─────────────────────────────────────────────────────
        whisper_srt_path: Path | None = None
        if whisper_jsonl:
            if whisper_jsonl.exists():
                try:
                    p = out_dir / "whisper.srt"
                    n = _jsonl_to_srt(whisper_jsonl, entry, p)
                    whisper_srt_path = p
                    print(f"  [OK] whisper.srt  ({n} cues, from JSONL)")
                except ValueError as e:
                    print(f"  [SKIP] whisper.srt — {e}")
            else:
                print(f"  [SKIP] --whisper-jsonl not found: {whisper_jsonl}")
        elif whisper_sidecar:
            if whisper_sidecar.exists():
                p = out_dir / "whisper.srt"
                n = _sidecar_to_srt(whisper_sidecar, p)
                whisper_srt_path = p
                print(f"  [OK] whisper.srt  ({n} cues, from sidecar)")
            else:
                print(f"  [SKIP] --whisper-sidecar not found: {whisper_sidecar}")

        # ── WhisperX SRT ────────────────────────────────────────────────────
        whisperx_srt_path: Path | None = None
        if whisperx_jsonl:
            if whisperx_jsonl.exists():
                try:
                    p = out_dir / "whisperx.srt"
                    n = _jsonl_to_srt(whisperx_jsonl, entry, p)
                    whisperx_srt_path = p
                    print(f"  [OK] whisperx.srt ({n} cues, from JSONL)")
                except ValueError as e:
                    print(f"  [SKIP] whisperx.srt — {e}")
            else:
                print(f"  [SKIP] --whisperx-jsonl not found: {whisperx_jsonl}")

        # ── comparison.md ───────────────────────────────────────────────────
        srt_map = {"Whisper": whisper_srt_path, "WhisperX": whisperx_srt_path}
        write_comparison_md(
            entry.id,
            out_dir / "comparison.md",
            entry.segments,
            srt_map,
            manifest_path=args.manifest,
        )
        print("  [OK] comparison.md")

        # ── directory listing ───────────────────────────────────────────────
        print(f"\n  Files written to: {out_dir}")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name}")

    print(f"\nDone. Output root: {output_root}")


if __name__ == "__main__":
    main()
