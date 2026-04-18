# archive/

This directory holds files from the project's earlier prototype phase (pre-SwitchNet pipeline).
They are preserved for reference but are **not part of the active codebase**.

## Contents

| Path | What it is |
|---|---|
| `transcribe.py` | Standalone `faster-whisper` transcription tool (SRT/VTT output). Predates the current pipeline. Requires `pip install faster-whisper`, which is **not** in `requirements.txt`. |
| `example_usage.py` | Usage examples for `transcribe.py`. Depends on the above. |
| `eval_librispeech_subset.py` | Manual LibriSpeech eval script that calls `transcribe.py` via subprocess and computes WER by hand. Superseded by `scripts/run_eval.py` + `src/asr/evaluate.py`. |
| `eval_outputs/` | Output artifacts from `eval_librispeech_subset.py` runs (SRT files, CSV reports, summaries). |
| `file/` | Original project directory for the standalone `faster-whisper` tool, including its own README. |
| `outputs/` | Prototype demo audio (`live_demo.wav`) and preview artifacts. |

## Active pipeline

All current work lives in:

```
scripts/   ← CLI entry points (build_manifests, run_eval, export_subtitles, aggregate_results)
src/       ← library layer (asr, audio, data, utils)
data/      ← manifests + raw audio
results/   ← experiment outputs
```

See the top-level README for usage.
