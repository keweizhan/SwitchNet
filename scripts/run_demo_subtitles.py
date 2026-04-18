"""
run_demo_subtitles.py -- Run three predefined subtitle demo cases.

Wraps export_subtitles.py with fixed parameters for each case and writes
all output to results/demo_cases/<case_name>/.  Intended as a one-command
demo that shows the three main subtitle modes side-by-side.

Demo cases:
  demo_en_only      -- English-only subtitles (one cue per manifest segment)
  demo_bilingual    -- Bilingual display (Spanish source + English translation)
  demo_bilingual_split -- Bilingual + rule-based cue splitting (max 15 words)

Usage:
  # Preview commands without running them:
  python scripts/run_demo_subtitles.py --dry-run

  # Run all three demo cases (requires large-v3 model):
  python scripts/run_demo_subtitles.py

  # Run on a specific manifest instead of the smoke default:
  python scripts/run_demo_subtitles.py --manifest data/manifests/bilingual_es-en_100.jsonl --limit 3

  # Run with a smaller/faster model:
  python scripts/run_demo_subtitles.py --model base
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Demo case definitions
# ---------------------------------------------------------------------------

DEMO_CASES = [
    {
        "name": "demo_en_only",
        "description": "English-only subtitles, one cue per manifest segment.",
        "extra_args": [
            "--translate-es",
            "--subtitle-mode", "english",
        ],
    },
    {
        "name": "demo_bilingual",
        "description": "Bilingual display: Spanish source on line 1, English translation on line 2.",
        "extra_args": [
            "--translate-es",
            "--subtitle-mode", "bilingual",
        ],
    },
    {
        "name": "demo_bilingual_split",
        "description": "Bilingual display with rule-based cue splitting (max 15 words per cue).",
        "extra_args": [
            "--translate-es",
            "--subtitle-mode", "bilingual",
            "--max-words-per-cue", "15",
        ],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_command(
    case: dict,
    manifest: str,
    model: str,
    device: str | None,
    limit: int | None,
    output_root: Path,
) -> list[str]:
    output_dir = str(output_root / case["name"])
    cmd = [
        sys.executable, "scripts/export_subtitles.py",
        "--manifest",   manifest,
        "--output-dir", output_dir,
        "--model",      model,
    ]
    if device:
        cmd += ["--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += case["extra_args"]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run three predefined subtitle demo cases via export_subtitles.py"
    )
    parser.add_argument(
        "--manifest", default="data/manifests/bilingual_smoke.jsonl",
        help="Input manifest (default: bilingual_smoke.jsonl — 1 entry smoke test)",
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
        "--limit", type=int, default=None,
        help="Process at most N entries per case (default: all entries in manifest)",
    )
    parser.add_argument(
        "--output-root", default="results/demo_cases",
        help="Root output directory; each case writes to a subdirectory (default: results/demo_cases)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run without executing them",
    )
    parser.add_argument(
        "--cases", nargs="+",
        choices=[c["name"] for c in DEMO_CASES],
        default=None,
        help="Run only the named demo case(s). Default: run all three.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    selected = (
        [c for c in DEMO_CASES if c["name"] in args.cases]
        if args.cases else DEMO_CASES
    )

    if args.dry_run:
        print("=== Dry run — commands that would be executed ===\n")
        for case in selected:
            cmd = build_command(
                case, args.manifest, args.model, args.device, args.limit, output_root
            )
            print(f"# {case['name']}: {case['description']}")
            print(" ".join(cmd))
            print()
        return

    # Run each case
    failed = []
    for i, case in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected)}] {case['name']}")
        print(f"  {case['description']}")
        print(f"{'='*60}\n")

        cmd = build_command(
            case, args.manifest, args.model, args.device, args.limit, output_root
        )
        print("Command:", " ".join(cmd), "\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: case '{case['name']}' exited with code {result.returncode}")
            failed.append(case["name"])

    print(f"\n{'='*60}")
    print(f"Demo complete. Output root: {output_root}")
    if failed:
        print(f"FAILED cases: {failed}")
        sys.exit(1)
    else:
        total = sum(1 for _ in (output_root).rglob("*.srt"))
        print(f"Total .srt files written: {total}")
        print("Run --dry-run to preview commands for any case.")


if __name__ == "__main__":
    main()
