"""
run_eval.py -- One-shot evaluation driver: manifest -> transcription -> metrics.

This is the command-line entry point that ties together transcribe + evaluate.
Useful for running experiments without opening a notebook.

Examples:
  # Spanish baseline, 50 utterances
  python scripts/run_eval.py \
      --manifest data/manifests/es_mls_test.jsonl \
      --model    large-v3 \
      --tag      es_mls_raw \
      --max      50

  # Bilingual oracle-segment mode (default)
  python scripts/run_eval.py \
      --manifest data/manifests/bilingual_es-en_100.jsonl \
      --model    large-v3 \
      --tag      bilingual_es-en_100_oracle

  # Bilingual full-concat / A2 mode
  python scripts/run_eval.py \
      --manifest       data/manifests/bilingual_es-en_100.jsonl \
      --model          large-v3 \
      --tag            a2_es-en_100_nopause_fc \
      --bilingual-mode full_concat

  # Preprocessing ablation: preemphasis on bilingual oracle
  python scripts/run_eval.py \
      --manifest   data/manifests/bilingual_es-en_100.jsonl \
      --model      large-v3 \
      --tag        bilingual_es-en_100_preemphasis \
      --preprocess preemphasis
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root: python scripts/run_eval.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.transcribe import Transcriber
from src.asr.evaluate import evaluate_results
from src.audio.preprocess import get_preprocessor, MODES as PREPROCESS_MODES


def main():
    parser = argparse.ArgumentParser(description="SwitchNet full eval pipeline")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--model",    default="large-v3", help="Whisper model size")
    parser.add_argument("--device",   default=None,       help="cuda | cpu (auto-detect)")
    parser.add_argument("--tag",      required=True,      help="Experiment tag for output file naming")
    parser.add_argument("--max",      type=int, default=None, help="Limit entries (debug)")
    parser.add_argument("--window",   type=int, default=5,    help="Switch-point window (words)")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription if results file already exists")
    parser.add_argument(
        "--bilingual-mode",
        default="oracle_segments",
        choices=["oracle_segments", "full_concat"],
        help=(
            "Bilingual decoding strategy (only affects language='bilingual' entries). "
            "'oracle_segments' (default): decode each segment with forced language. "
            "'full_concat': concatenate all segment audio with pause_s silence gaps "
            "and decode in one Whisper pass (A2 pause-vs-no-pause experiment)."
        ),
    )
    parser.add_argument(
        "--preprocess",
        default="raw",
        choices=PREPROCESS_MODES,
        help=(
            "Audio preprocessing applied before Whisper inference. "
            "'raw' (default): no preprocessing, baseline behaviour unchanged. "
            "'normalize': peak amplitude normalization to [-1, 1]. "
            "'preemphasis': first-order high-pass pre-emphasis filter (coef=0.97). "
            "Include the condition name in --tag to keep results distinguishable, "
            "e.g. --tag bilingual_es-en_20_preemphasis."
        ),
    )
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / f"{args.tag}.jsonl"
    summary_path = results_dir / f"{args.tag}_summary.json"

    # Step 1: Transcribe
    if args.skip_transcribe and results_path.exists():
        print(f"Skipping transcription (--skip-transcribe), using {results_path}")
    else:
        if results_path.exists():
            print(f"Results file exists at {results_path}. Overwriting...")
        preprocessor = get_preprocessor(args.preprocess)
        if preprocessor is not None:
            print(f"Preprocessing: {args.preprocess}")
        t = Transcriber(
            model_size=args.model,
            device=args.device,
            preprocessor=preprocessor,
        )
        t.transcribe_manifest(
            args.manifest,
            output_path=results_path,
            max_entries=args.max,
            bilingual_mode=args.bilingual_mode,
        )

    # Step 2: Evaluate
    evaluate_results(
        results_path=results_path,
        manifest_path=args.manifest,
        output_path=summary_path,
        window_words=args.window,
    )


if __name__ == "__main__":
    main()