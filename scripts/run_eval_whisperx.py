"""
run_eval_whisperx.py -- WhisperX evaluation driver for SwitchNet.

Drop-in replacement for run_eval.py that uses WhisperX as the ASR backend.
Evaluation (WER / MER / switch-point WER) is performed by the same
evaluate_results() call used in run_eval.py, so results are directly
comparable and aggregate_results.py picks them up without changes.

Install WhisperX before running:
    pip install whisperx

Examples:
  # Monolingual English baseline
  python scripts/run_eval_whisperx.py \\
      --manifest data/manifests/en_librispeech_test.jsonl \\
      --model    large-v3 \\
      --tag      wx_en_librispeech_baseline

  # Bilingual oracle-segment (same decoding strategy as run_eval.py default)
  python scripts/run_eval_whisperx.py \\
      --manifest data/manifests/bilingual_es-en_100.jsonl \\
      --model    large-v3 \\
      --tag      wx_bilingual_es-en_100_oracle

  # Bilingual oracle, CPU only (int8)
  python scripts/run_eval_whisperx.py \\
      --manifest      data/manifests/bilingual_smoke.jsonl \\
      --model         large-v3 \\
      --device        cpu \\
      --compute-type  int8 \\
      --tag           wx_smoke_cpu \\
      --max           1

  # Full-concat mode (A2 equivalent)
  python scripts/run_eval_whisperx.py \\
      --manifest       data/manifests/bilingual_es-en_100_pause05.jsonl \\
      --model          large-v3 \\
      --tag            wx_a2_es-en_100_pause05 \\
      --bilingual-mode full_concat

  # Re-run eval only (skip transcription if JSONL already exists)
  python scripts/run_eval_whisperx.py \\
      --manifest        data/manifests/bilingual_es-en_100.jsonl \\
      --tag             wx_bilingual_es-en_100_oracle \\
      --skip-transcribe
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.transcribe_whisperx import WhisperXTranscriber
from src.asr.evaluate import evaluate_results


def main():
    parser = argparse.ArgumentParser(
        description="SwitchNet full eval pipeline using WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest",  required=True, help="Input JSONL manifest")
    parser.add_argument("--model",     default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--device",    default=None,
                        help="cuda | cpu  (auto-detect if omitted)")
    parser.add_argument("--compute-type", default=None, dest="compute_type",
                        help="float16 (GPU default) | int8 (CPU default)")
    parser.add_argument("--tag",       required=True,
                        help="Experiment tag — used as results/<tag>.jsonl and results/<tag>_summary.json")
    parser.add_argument("--max",       type=int, default=None,
                        help="Limit number of manifest entries (debug / smoke test)")
    parser.add_argument("--window",    type=int, default=5,
                        help="Switch-point evaluation window in words (default: 5)")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription step if results/<tag>.jsonl already exists")
    parser.add_argument(
        "--bilingual-mode",
        default="oracle_segments",
        choices=["oracle_segments", "full_concat"],
        help=(
            "Bilingual decoding strategy (only affects language='bilingual' entries).\n"
            "  oracle_segments (default): decode each segment with its known language forced.\n"
            "  full_concat: concatenate all segment audio with pause_s silence gaps\n"
            "               and decode in one WhisperX pass (A2 equivalent)."
        ),
    )
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / f"{args.tag}.jsonl"
    summary_path = results_dir / f"{args.tag}_summary.json"

    # ------------------------------------------------------------------ #
    # Step 1: Transcribe                                                   #
    # ------------------------------------------------------------------ #
    if args.skip_transcribe and results_path.exists():
        print(f"Skipping transcription (--skip-transcribe), using {results_path}")
    else:
        if results_path.exists():
            print(f"Results file exists at {results_path}. Overwriting...")
        t = WhisperXTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        t.transcribe_manifest(
            args.manifest,
            output_path=results_path,
            max_entries=args.max,
            bilingual_mode=args.bilingual_mode,
        )

    # ------------------------------------------------------------------ #
    # Step 2: Evaluate — identical call to run_eval.py                    #
    # ------------------------------------------------------------------ #
    evaluate_results(
        results_path=results_path,
        manifest_path=args.manifest,
        output_path=summary_path,
        window_words=args.window,
    )


if __name__ == "__main__":
    main()
