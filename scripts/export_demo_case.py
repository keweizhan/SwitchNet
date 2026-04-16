"""
export_demo_case.py — Generate a demo comparison folder for one bilingual entry.

Produces (no model required):
  results/subtitles/demo_cases/<entry_id>/
    reference.srt    -- ground-truth SRT with explicit [ES→EN] boundary cue
    reference.txt    -- plain-text reference showing [ES] / SWITCH / [EN]
    comparison.md    -- side-by-side markdown summary

Optionally (if --whisper-sidecar is supplied):
    whisper.srt      -- rebuilt from an existing export_subtitles.py sidecar JSON
                        without re-running the model

whisperx.srt is noted as pending in comparison.md (WhisperX subtitle export
not yet wired into the pipeline).

Usage
-----
  # First bilingual entry in the 50-pair es-en manifest (reference only):
  python scripts/export_demo_case.py \\
      --manifest data/manifests/bilingual_es-en_50.jsonl \\
      --limit 1

  # Specific entry by ID:
  python scripts/export_demo_case.py \\
      --manifest data/manifests/bilingual_es-en_50.jsonl \\
      --entry-id bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001

  # With an existing Whisper sidecar to also generate whisper.srt:
  python scripts/export_demo_case.py \\
      --manifest data/manifests/bilingual_es-en_50.jsonl \\
      --entry-id bilingual_es-en_0022_mls_es_10667_9310_000089_ls_5105-28241-0001 \\
      --whisper-sidecar results/subtitles/some_run/<entry_id>.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import load_manifest
from src.asr.subtitles import (
    SubtitleCue,
    write_srt,
    write_reference_srt,
    write_reference_txt,
    write_comparison_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sidecar_to_srt(sidecar_path: Path, output_path: Path) -> int:
    """
    Rebuild a model .srt from an existing export_subtitles.py sidecar JSON.

    The sidecar already contains cleaned cue text and timestamps, so this
    avoids re-running Whisper.  Returns the number of cues written.
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
            "Generate a demo comparison folder (reference.srt, reference.txt, "
            "comparison.md) for one bilingual manifest entry.  No model required."
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
        help=(
            "Root output directory.  Each entry writes to "
            "<output-root>/<entry_id>/  (default: results/subtitles/demo_cases)"
        ),
    )
    parser.add_argument(
        "--whisper-sidecar", default=None,
        help=(
            "Path to an existing export_subtitles.py sidecar JSON for this entry. "
            "When supplied, whisper.srt is generated from the cached cue data "
            "without re-running the model."
        ),
    )
    args = parser.parse_args()

    # --- load and select entries -------------------------------------------
    all_entries = load_manifest(args.manifest)
    entries = [e for e in all_entries if e.language == "bilingual"]

    if args.entry_id:
        entries = [e for e in entries if e.id == args.entry_id]
        if not entries:
            print(f"No bilingual entry with id={args.entry_id!r} found in {args.manifest}.")
            sys.exit(1)
    else:
        entries = entries[: args.limit]

    if not entries:
        print("No bilingual entries found. Nothing to do.")
        sys.exit(1)

    output_root = Path(args.output_root)
    whisper_sidecar = Path(args.whisper_sidecar) if args.whisper_sidecar else None

    # --- process each entry ------------------------------------------------
    for entry in entries:
        out_dir = output_root / entry.id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"Entry : {entry.id}")
        print(f"Output: {out_dir}")
        print(f"{'='*72}")

        # Always: reference files (no model)
        ref_srt = out_dir / "reference.srt"
        ref_txt = out_dir / "reference.txt"

        write_reference_srt(entry.segments, ref_srt)
        print(f"  [OK] reference.srt")

        write_reference_txt(entry.segments, ref_txt)
        print(f"  [OK] reference.txt")

        # Optional: whisper.srt from existing sidecar
        whisper_srt_path: Path | None = None
        if whisper_sidecar:
            if whisper_sidecar.exists():
                whisper_srt = out_dir / "whisper.srt"
                n = _sidecar_to_srt(whisper_sidecar, whisper_srt)
                whisper_srt_path = whisper_srt
                print(f"  [OK] whisper.srt  ({n} cues, from sidecar)")
            else:
                print(f"  [SKIP] --whisper-sidecar not found: {whisper_sidecar}")

        # Always: comparison.md
        srt_map = {
            "Whisper":  whisper_srt_path,
            "WhisperX": None,   # pending WhisperX subtitle export
        }
        comp_md = out_dir / "comparison.md"
        write_comparison_md(
            entry.id, comp_md, entry.segments, srt_map,
            manifest_path=args.manifest,
        )
        print(f"  [OK] comparison.md")

        # Summary
        print(f"\n  Files written to: {out_dir}")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name}")

    print(f"\nDone. Output root: {output_root}")


if __name__ == "__main__":
    main()
