"""
export_subtitles.py — English subtitle export for bilingual manifest entries.

Runs oracle-segment transcription on each bilingual entry, then writes:
  - One .srt per entry (standard subtitle file, timestamps in HH:MM:SS,mmm)
  - One sidecar .json per entry (per-segment details for debugging / downstream use)

By default every segment is transcribed in its source language.
Pass --translate-es to run Whisper's translate task on Spanish segments,
producing English output for all cues.

Examples:
  # Transcribe (not translate) — useful for checking pipeline without GPU
  python scripts/export_subtitles.py \\
      --manifest data/manifests/bilingual_smoke.jsonl \\
      --output-dir results/subtitles/smoke \\
      --model large-v3

  # Translate ES → EN (subtitle export mode)
  python scripts/export_subtitles.py \\
      --manifest data/manifests/bilingual_es-en_100.jsonl \\
      --output-dir results/subtitles/es-en_100 \\
      --model large-v3 \\
      --translate-es

  # Single sample smoke test
  python scripts/export_subtitles.py \\
      --manifest data/manifests/bilingual_smoke.jsonl \\
      --output-dir results/subtitles/smoke \\
      --model large-v3 \\
      --limit 1
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import load_manifest
from src.asr.transcribe import Transcriber
from src.asr.subtitles import build_segment_level_cues, write_srt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sidecar(
    entry,
    segment_outputs: list[dict],
    cues,
    task_map: dict,
) -> dict:
    """Build the per-entry sidecar dict written alongside the .srt."""
    return {
        "id": entry.id,
        "bilingual_mode": "oracle_segments",
        "task_map": task_map,
        "segments": [
            {
                "position": out["position"],
                "language": out["language"],
                "reference": out["reference"],
                "hypothesis": out["hypothesis"],
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
            }
            for i, cue in enumerate(cues)
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwitchNet: export English subtitles for bilingual manifest entries"
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
            "in English.  Without this flag every segment is transcribed in its "
            "source language."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .srt files (default: skip already-exported entries)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build task_map ---
    task_map = {
        "en": "transcribe",
        "es": "translate" if args.translate_es else "transcribe",
    }

    # --- Load and filter manifest ---
    all_entries = load_manifest(args.manifest)
    bilingual = [e for e in all_entries if e.language == "bilingual"]

    if not bilingual:
        print(f"No bilingual entries found in {args.manifest}. Nothing to do.")
        return

    # Filter by --sample-id
    if args.sample_id:
        bilingual = [e for e in bilingual if e.id == args.sample_id]
        if not bilingual:
            print(f"No entry with id={args.sample_id!r} found.")
            return

    # Apply --limit
    if args.limit is not None:
        bilingual = bilingual[: args.limit]

    print(f"Manifest: {args.manifest}")
    print(f"Bilingual entries to process: {len(bilingual)}")
    print(f"Task map: {task_map}")
    print(f"Output dir: {output_dir}")
    print()

    # --- Load model ---
    transcriber = Transcriber(model_size=args.model, device=args.device)

    # --- Process each entry ---
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
            # --- Transcribe ---
            _, segment_outputs = transcriber._transcribe_bilingual_oracle(
                entry, task_map=task_map
            )

            # --- Build cues (timing inferred from audio if needed) ---
            cues = build_segment_level_cues(
                segment_outputs,
                entry.segments,
                infer_timing_from_audio=True,
            )

            # --- Write .srt ---
            write_srt(cues, srt_path)

            # --- Write sidecar .json ---
            sidecar = _build_sidecar(entry, segment_outputs, cues, task_map)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False)

            n_cues = sum(1 for c in cues if c.text.strip())
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
