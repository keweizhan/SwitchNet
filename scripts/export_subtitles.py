"""
export_subtitles.py -- English subtitle export for bilingual manifest entries.

Runs oracle-segment transcription on each bilingual entry, then writes:
  - One .srt per entry (standard subtitle file, timestamps in HH:MM:SS,mmm)
  - One sidecar .json per entry (per-segment details for debugging)

By default every segment is transcribed in its source language.
Pass --translate-es to run Whisper's translate task on Spanish segments,
producing English output for all cues.

Pass --use-whisper-segments to use Whisper's internal segment boundaries for
fine-grained subtitle cues instead of one cue per manifest segment.

Pass --max-chars-per-cue or --max-words-per-cue (V1.2) to apply rule-based
cue splitting.  Long cues are broken at sentence/clause boundaries and
durations distributed proportionally by word count.

Pass --subtitle-mode bilingual (V1.2) to show the original Spanish text on
line 1 and the English translation on line 2 for Spanish segments.
English segments are always shown as a single English line.

Examples:
  # V1.0 mode (one cue per manifest segment, translate ES->EN)
  python scripts/export_subtitles.py \\
      --manifest    data/manifests/bilingual_es-en_100.jsonl \\
      --output-dir  results/subtitles/es-en_100 \\
      --model       large-v3 \\
      --translate-es

  # V1.1 mode (Whisper-segment splitting, translate ES->EN, wrap at 42 chars)
  python scripts/export_subtitles.py \\
      --manifest           data/manifests/bilingual_es-en_100.jsonl \\
      --output-dir         results/subtitles/es-en_100_v11 \\
      --model              large-v3 \\
      --translate-es \\
      --use-whisper-segments

  # V1.2 mode (rule-based cue splitting by char count + bilingual display)
  python scripts/export_subtitles.py \\
      --manifest           data/manifests/bilingual_es-en_100.jsonl \\
      --output-dir         results/subtitles/es-en_100_v12 \\
      --model              large-v3 \\
      --translate-es \\
      --max-chars-per-cue  120 \\
      --subtitle-mode      bilingual

  # V1.2 mode with word count limit instead of char count
  python scripts/export_subtitles.py \\
      --manifest           data/manifests/bilingual_es-en_100.jsonl \\
      --output-dir         results/subtitles/es-en_100_v12_words \\
      --model              large-v3 \\
      --translate-es \\
      --max-words-per-cue  20 \\
      --subtitle-mode      bilingual

  # Single sample smoke test (V1.2, bilingual mode)
  python scripts/export_subtitles.py \\
      --manifest           data/manifests/bilingual_smoke.jsonl \\
      --output-dir         results/subtitles/smoke_v12 \\
      --model              large-v3 \\
      --translate-es \\
      --max-chars-per-cue  120 \\
      --subtitle-mode      bilingual \\
      --limit              1
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import load_manifest
from src.asr.transcribe import Transcriber
from src.asr.subtitles import (
    build_segment_level_cues,
    write_srt,
    apply_rule_based_splitting,
)


# ---------------------------------------------------------------------------
# Sidecar helper
# ---------------------------------------------------------------------------

def _build_sidecar(
    entry,
    segment_outputs: list[dict],
    cues,
    task_map: dict,
    use_whisper_segments: bool,
    subtitle_mode: str,
    max_chars_per_cue: int | None,
    max_words_per_cue: int | None,
) -> dict:
    """Build the per-entry sidecar dict written alongside the .srt."""
    return {
        "id": entry.id,
        "bilingual_mode": "oracle_segments",
        "task_map": task_map,
        "subtitle_mode": subtitle_mode,
        "use_whisper_segments": use_whisper_segments,
        "max_chars_per_cue": max_chars_per_cue,
        "max_words_per_cue": max_words_per_cue,
        "cue_count": sum(1 for c in cues if c.text.strip()),
        "segments": [
            {
                "position": out["position"],
                "language": out["language"],
                "reference": out["reference"],
                "hypothesis": out["hypothesis"],
                "whisper_segment_count": (
                    len(out["whisper_segments"])
                    if "whisper_segments" in out else None
                ),
            }
            for out in segment_outputs
        ],
        "cues": [
            {
                "index": i + 1,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "source_language": cue.source_language,
                "source_text": cue.source_text,
                "timing_source": cue.timing_source,
            }
            for i, cue in enumerate(cues)
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwitchNet: export subtitles for bilingual manifest entries"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="JSONL manifest file (bilingual entries will be selected automatically)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write .srt and .json sidecar files",
    )
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda | cpu (auto-detect if omitted)",
    )
    parser.add_argument(
        "--sample-id", default=None,
        help="Process only the entry with this ID (overrides --limit)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N bilingual entries (debug / smoke test)",
    )
    parser.add_argument(
        "--translate-es", action="store_true",
        help=(
            "Use Whisper translate task for Spanish segments so all cues are "
            "in English. Without this flag every segment is transcribed in its "
            "source language."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .srt files (default: skip already-exported entries)",
    )
    # V1.1 flags
    parser.add_argument(
        "--use-whisper-segments", action="store_true",
        help=(
            "Use Whisper's internal segment boundaries to create multiple short "
            "subtitle cues per manifest segment instead of one cue per segment. "
            "Falls back to one-cue-per-segment if Whisper returns no segments."
        ),
    )
    parser.add_argument(
        "--max-cue-seconds", type=float, default=None,
        help=(
            "Cap each subtitle cue's duration to this many seconds. "
            "Only affects --use-whisper-segments mode. "
            "The cue text is not split -- only the end timestamp is clamped."
        ),
    )
    parser.add_argument(
        "--max-chars-per-line", type=int, default=42,
        help=(
            "Wrap subtitle lines longer than this many characters (default 42). "
            "Set to a large value to disable wrapping."
        ),
    )
    # V1.2 flags
    parser.add_argument(
        "--max-chars-per-cue", type=int, default=None,
        help=(
            "V1.2: Split cue text if it exceeds this many characters. "
            "Uses rule-based splitting (sentence -> clause -> midpoint). "
            "Disabled by default. Suggested starting value: 120."
        ),
    )
    parser.add_argument(
        "--max-words-per-cue", type=int, default=None,
        help=(
            "V1.2: Split cue text if it exceeds this many words. "
            "Uses rule-based splitting (sentence -> clause -> midpoint). "
            "Disabled by default. Suggested starting value: 20."
        ),
    )
    parser.add_argument(
        "--subtitle-mode",
        default="english",
        choices=["english", "bilingual"],
        help=(
            "V1.2: Subtitle display mode. "
            "'english' (default): English text only for all cues. "
            "'bilingual': Spanish cues show source text on line 1 and English "
            "translation on line 2. English cues are shown as a single line. "
            "Requires --translate-es to be meaningful for Spanish segments."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task_map
    task_map = {
        "en": "transcribe",
        "es": "translate" if args.translate_es else "transcribe",
    }

    # Load and filter manifest
    all_entries = load_manifest(args.manifest)
    bilingual = [e for e in all_entries if e.language == "bilingual"]

    if not bilingual:
        print(f"No bilingual entries found in {args.manifest}. Nothing to do.")
        return

    if args.sample_id:
        bilingual = [e for e in bilingual if e.id == args.sample_id]
        if not bilingual:
            print(f"No entry with id={args.sample_id!r} found.")
            return

    if args.limit is not None:
        bilingual = bilingual[: args.limit]

    print(f"Manifest         : {args.manifest}")
    print(f"Entries selected : {len(bilingual)}")
    print(f"Task map         : {task_map}")
    print(f"Subtitle mode    : {args.subtitle_mode}")
    print(f"Whisper segments : {args.use_whisper_segments}")
    print(f"Max cue seconds  : {args.max_cue_seconds}")
    print(f"Max chars/cue    : {args.max_chars_per_cue}")
    print(f"Max words/cue    : {args.max_words_per_cue}")
    print(f"Max chars/line   : {args.max_chars_per_line}")
    print(f"Output dir       : {output_dir}")
    print()

    # Load model
    transcriber = Transcriber(model_size=args.model, device=args.device)

    # Process each entry
    skipped = 0
    written = 0
    errors = 0

    for i, entry in enumerate(bilingual):
        srt_path  = output_dir / f"{entry.id}.srt"
        json_path = output_dir / f"{entry.id}.json"

        if srt_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(bilingual)}] SKIP (exists): {entry.id}")
            skipped += 1
            continue

        print(f"[{i+1}/{len(bilingual)}] {entry.id}", end=" ... ", flush=True)

        try:
            # Transcribe
            _, segment_outputs = transcriber._transcribe_bilingual_oracle(
                entry,
                task_map=task_map,
                return_whisper_segments=args.use_whisper_segments,
            )

            # Build cues (one per manifest segment, with optional Whisper sub-segs)
            cues = build_segment_level_cues(
                segment_outputs,
                entry.segments,
                infer_timing_from_audio=True,
                use_whisper_segments=args.use_whisper_segments,
                max_cue_seconds=args.max_cue_seconds,
            )

            # V1.2: apply rule-based splitting if requested
            cues = apply_rule_based_splitting(
                cues,
                max_chars_per_cue=args.max_chars_per_cue,
                max_words_per_cue=args.max_words_per_cue,
            )

            # Write .srt
            write_srt(
                cues,
                srt_path,
                add_period=True,
                max_chars_per_line=args.max_chars_per_line,
                subtitle_mode=args.subtitle_mode,
            )

            # Write sidecar .json
            sidecar = _build_sidecar(
                entry, segment_outputs, cues, task_map,
                use_whisper_segments=args.use_whisper_segments,
                subtitle_mode=args.subtitle_mode,
                max_chars_per_cue=args.max_chars_per_cue,
                max_words_per_cue=args.max_words_per_cue,
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False)

            n_cues = sidecar["cue_count"]
            print(f"done ({n_cues} cues)")
            written += 1

        except Exception as exc:
            print(f"ERROR: {exc}")
            errors += 1

    print()
    print(f"Done. Written={written}  Skipped={skipped}  Errors={errors}")
    print(f"SRT files in: {output_dir}")


if __name__ == "__main__":
    main()
