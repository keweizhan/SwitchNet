# SwitchNet

A bilingual / code-switching ASR evaluation and demo framework built around Whisper and WhisperX.
Developed for EE519 at USC.

SwitchNet is **not** a trained ASR model. It is an evaluation pipeline, analysis tool, and interactive demo application that compares how different Whisper-family backends handle English–Spanish code-switching speech, switch-point behavior, and subtitle generation.

Everything is driven by JSONL manifest files, which keep data preparation, inference, and evaluation cleanly separated and repeatable.

---

## Key Capabilities

- **Manifest-driven bilingual evaluation** — oracle-segment and full-concat decoding modes
- **Switch-point WER / MER** — metrics computed over a word window around each language boundary
- **Whisper vs WhisperX backend comparison** — system-level side-by-side on the same manifest
- **Subtitle export and demo-case packaging** — per-entry `.srt` files with English or bilingual display; `export_demo_case.py` bundles reference + Whisper + WhisperX outputs
- **Streamlit app** — four-tab interactive demo: Reference audio/transcript, side-by-side Comparison, WER Summary, and Live Recording
- **Live Recording tab** — record in-browser, run Whisper base on CPU, compute WER against a typed reference
- **Multilingual live-recording** — Auto (Whisper language detection), English, Spanish, or any BCP-47 code (fr, zh, de, ja, …)

---

## Core Findings

Evaluated on 50 bilingual ES→EN utterances using `large-v3` on CPU.

| Metric | Whisper (oracle) | WhisperX (oracle) | Relative improvement |
|---|---|---|---|
| Overall WER | 0.0540 | 0.0356 | −34.1 % |
| Overall MER | 0.0523 | 0.0346 | −33.8 % |
| Switch-point WER | 0.1120 | 0.0940 | −16.1 % |
| Switch-point MER | 0.1102 | 0.0929 | −15.7 % |

WhisperX improves across all metrics. Gains at switch points are smaller than overall gains, suggesting improvements come partly from better general ASR quality and partly from reduced hallucination / boundary failures near language switches. This is a **system-level comparison** — the two backends differ in alignment, VAD, and decoding strategy, not just the underlying model weights.

---

## Repository Structure

```
SwitchNet/
├── app/
│   └── streamlit_app.py          # Four-tab Streamlit demo app
│
├── scripts/
│   ├── run_eval.py               # Main eval driver: transcribe + evaluate (Whisper)
│   ├── run_eval_whisperx.py      # Same workflow using WhisperX backend
│   ├── export_subtitles.py       # Generate .srt files from bilingual manifests
│   ├── export_demo_case.py       # Package one entry: reference + Whisper + WhisperX outputs
│   ├── run_demo_subtitles.py     # Run three preset demo subtitle cases
│   ├── aggregate_results.py      # Collect *_summary.json files into one CSV
│   ├── plot_results.py           # WER comparison plots from aggregated_results.csv
│   ├── build_manifests.py        # Build JSONL manifests from raw datasets
│   └── make_core_results.py      # Curated core-results table (hardcoded experiment list)
│
├── src/
│   ├── asr/
│   │   ├── transcribe.py         # Whisper inference (mono, oracle-segment, full-concat)
│   │   ├── transcribe_whisperx.py # WhisperX inference (mirrors Transcriber interface)
│   │   ├── evaluate.py           # WER / MER / switch-point metrics
│   │   └── subtitles.py          # SRT cue building, timing, text cleanup
│   ├── audio/
│   │   └── preprocess.py         # Optional audio preprocessing (denoising, normalization)
│   ├── data/
│   │   └── manifest.py           # ManifestEntry / Segment dataclasses, load/save helpers
│   └── utils/
│       └── normalize.py          # Language-aware text normalization for WER
│
├── data/
│   ├── manifests/                # JSONL manifest files
│   ├── LibriSpeech/              # English audio (test-clean, .flac)
│   └── mls_spanish/              # Spanish audio (test, .opus)
│
├── results/
│   ├── subtitles/
│   │   └── demo_cases/           # Per-entry bundled demo outputs (<entry_id>/)
│   ├── plots/                    # Generated comparison charts
│   ├── aggregated_results.csv
│   └── *.jsonl / *_summary.json  # Per-run transcription outputs and eval summaries
│
├── notebooks/
│   ├── whisper_vs_whisperx_demo.ipynb  # Notebook walkthrough of the comparison
│   └── demo.ipynb
│
└── archive/                      # Legacy scripts and exploratory artifacts
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core packages:
- `openai-whisper` — Whisper inference
- `torch` — GPU/CPU detection and tensor ops
- `numpy` — audio array handling
- `librosa` — audio loading and duration inference
- `jiwer` — WER / MER computation
- `streamlit` — interactive demo app
- `whisperx` — optional; required only for `run_eval_whisperx.py`

FFmpeg must be on the system path for Whisper's audio decoding:

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Linux
choco install ffmpeg       # Windows
```

Python 3.10+ recommended.

---

## Quick Start

### Run standard Whisper evaluation
```bash
python scripts/run_eval.py \
    --manifest data/manifests/bilingual_es-en_50.jsonl \
    --model    large-v3 \
    --tag      bilingual_es-en_50_large_cpu
```

### Run WhisperX evaluation
```bash
python scripts/run_eval_whisperx.py \
    --manifest data/manifests/bilingual_es-en_50.jsonl \
    --model    large-v3 \
    --tag      wx_bilingual_es-en_50_large_cpu
```

### Export a demo case (bundles reference + Whisper + WhisperX)
```bash
python scripts/export_demo_case.py \
    --manifest       data/manifests/bilingual_es-en_50.jsonl \
    --whisper-jsonl  results/bilingual_es-en_50_large_cpu.jsonl \
    --whisperx-jsonl results/wx_bilingual_es-en_50_large_cpu.jsonl \
    --limit 5
```

### Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## Streamlit App

The app has four tabs. It uses precomputed JSONL files for the Whisper / WhisperX comparison tabs, ensuring stable and fast demo playback without re-running inference.

### Reference tab
Displays the reference audio and ground-truth transcript for a selected bilingual entry. Shows the ES and EN segments separately with a language-switch banner.

### Comparison tab
Side-by-side timed subtitle comparison: Reference cues | Whisper | WhisperX. Useful for spotting where each system diverges from the ground truth.

### WER Summary tab
Per-entry and corpus-level WER / MER table, plus a bar chart comparing Whisper vs WhisperX across all entries in the loaded JSONL.

### Live Recording tab
Record a clip in the browser, transcribe with a configurable Whisper model and device, and evaluate against a typed reference transcript.

#### Model / Device selectors
- **Whisper model**: `base` (default) / `small` / `medium` / `large-v3`
- **Device**: `auto` (CUDA if available, else CPU) / `cpu` / `cuda`
- A warning is shown when `large-v3` or `medium` is selected on CPU — these models take several minutes per clip

Two ASR demo modes:

#### Monolingual Mode
- **Language selector**: Auto / English / Spanish / Other (any BCP-47 code)
- Single Whisper run, shows detected language, WER/MER, sub/del/ins counts
- If the reference contains `[lang]` tags (e.g. `[en] hello [es] mundo`), a warning is shown and tags are **stripped before WER computation** so they are never counted as reference words

#### Code-Switch Challenge Mode
- Preset mixed-language examples or free-form `[lang]`-tagged reference:
  - EN/ES — homework deadline
  - ES/EN — start the experiment
  - EN/ES — explain WER vs MER
  - EN/ES/ZH — mixed three languages
  - ZH/EN — finish the experiment (`今天下午 … 然后整理结果`)
  - ZH/EN — code-switching is hard (`这个例子展示了 … 因为模型需要识别语言切换`)
- Tags supported: `[en]`, `[es]`, `[zh]`, and any 2–3 letter BCP-47 code
- Parsed-segments table shows each language span
- **Four decoding strategies** compared side by side (all use the same selected model/device — no model reload between strategies):
  | Strategy | Whisper language parameter |
  |---|---|
  | Auto-global | `None` (Whisper detects) |
  | Forced-English | `"en"` |
  | Forced-Spanish | `"es"` |
  | Dominant-language | `"en"` or `"es"` inferred from tag word counts |
- Results table: strategy · detected · WER · MER · CER (if CJK) · sub · del · ins
- CER (Character Error Rate) shown for CJK text (word segmentation not required)
- Note: auto-global detection is **not** segment-level language ID — mixed-language speech may need segment-level routing for accurate transcription

---

## Experimental Findings (A1 / A2 / A3)

All A-series experiments use `large-v3` on 100-pair bilingual manifests from LibriSpeech (EN) and MLS Spanish (ES).

### A1 — Segment order effect under oracle routing
Oracle-segment mode decodes each segment independently with its known language forced.

| Condition | WER | MER |
|---|---|---|
| EN→ES | 0.0385 | 0.0381 |
| ES→EN | 0.0348 | 0.0345 |

Small consistent difference; ES-first is slightly lower. Likely reflects utterance-level sampling variation rather than a strong position effect.

### A2 — Pause vs no-pause under full-concat decoding
Full-concat decodes all segments as one waveform with `language=None`.

| Condition | Overall WER | Switch-pt WER |
|---|---|---|
| No pause  | 0.3170 | 0.4257 |
| 0.5s pause | 0.3008 | 0.4076 |

Adding a real 0.5s silence gap gives a small but consistent improvement. Full-concat WER is much higher than oracle-segment WER — expected, since Whisper receives no language hint.

### A3 — 3-segment arrangement effect
All conditions use 0.5s pauses and full-concat decoding.

| Condition | Overall WER | Switch-pt WER |
|---|---|---|
| ES→EN (2-seg baseline) | 0.3008 | 0.4076 |
| ES→EN→ES (3-seg)       | 0.2425 | 0.5880 |
| EN→ES→EN (3-seg)       | 0.2210 | 0.3385 |

Overall WER improves in both 3-segment conditions, but switch-point WER diverges: EN→ES→EN improves at switch boundaries while ES→EN→ES gets noticeably worse. Language arrangement matters beyond simply adding more switches.

---

## Clean Local Demo Data

The full 50-entry bilingual manifest (`data/manifests/bilingual_es-en_50.jsonl`) was built on a machine that had the complete MLS Spanish test set. The local audio subset in this repo does not include every specific utterance referenced by that manifest, so the Streamlit demo will show "audio not available" for most entries.

To build a clean demo from the audio files that **are** present locally:

```bash
# Step 1 — build a small bilingual manifest from locally available audio
python scripts/build_manifests.py bilingual \
    --en-manifest data/manifests/en_librispeech_test.jsonl \
    --es-manifest data/manifests/es_mls_local.jsonl \
    --output      data/manifests/bilingual_local_demo.jsonl \
    --pairs 10 --pattern es-en

# Step 2 — run Whisper inference
python scripts/run_eval.py \
    --manifest data/manifests/bilingual_local_demo.jsonl \
    --model large-v3 --tag local_demo_large_cpu

# Step 3 — run WhisperX inference
python scripts/run_eval_whisperx.py \
    --manifest data/manifests/bilingual_local_demo.jsonl \
    --model large-v3 --tag wx_local_demo_large_cpu

# Step 4 — filter to fully-present entries and write clean demo files
python scripts/prepare_clean_demo_data.py \
    --manifest            data/manifests/bilingual_local_demo.jsonl \
    --whisper-jsonl       results/local_demo_large_cpu.jsonl \
    --whisperx-jsonl      results/wx_local_demo_large_cpu.jsonl \
    --out-manifest        data/demo/bilingual_es-en_clean_demo.jsonl \
    --out-whisper-jsonl   results/demo/bilingual_es-en_clean_whisper.jsonl \
    --out-whisperx-jsonl  results/demo/bilingual_es-en_clean_whisperx.jsonl \
    --max-entries 30
```

Step 4 produces **five** output files:

| File | Purpose |
|---|---|
| `data/demo/bilingual_es-en_clean_demo.jsonl` | Filtered manifest (audio-present entries only) |
| `results/demo/bilingual_es-en_clean_whisper.jsonl` | Whisper result slice |
| `results/demo/bilingual_es-en_clean_whisperx.jsonl` | WhisperX result slice |
| `results/demo/bilingual_es-en_clean_whisper_summary.json` | WER/MER summary (generated by the script) |
| `results/demo/bilingual_es-en_clean_whisperx_summary.json` | WER/MER summary (generated by the script) |

The summary JSON files are computed by the script using the same `evaluate_results` function as
the main eval pipeline, so WER/MER numbers are consistent with full-experiment results.

Once `data/demo/bilingual_es-en_clean_demo.jsonl` exists, the Streamlit app automatically
prefers all five files (shown with a ✅ badge in the sidebar). Broken entries are
**dropped, not faked** — the filter only keeps entries where every audio file is confirmed present.

If a `*_summary.json` file is missing (e.g. after manually copying a JSONL slice), the
Streamlit WER Summary tab falls back to computing WER/MER directly from the JSONL and shows a
small note: *"Summary computed from JSONL because no \*\_summary.json file was found."*

---

## Notes and Caveats

- **Bilingual samples are synthetic.** Each entry is built by concatenating real monolingual utterances from separate corpora. This is a controlled approximation, not natural code-switched speech.
- **Whisper vs WhisperX is a system-level comparison.** The two backends differ in VAD, alignment, and decoding strategy, not just model weights. Observed WER differences reflect the full pipeline, not an isolated ablation of a single component.
- **Switch-point metrics are a key focus.** Overall WER can look reasonable while near-switch behavior is poor. Always check both.
- **Manifest paths are absolute and machine-specific.** If you move the data or change machines, re-run `build_manifests.py` to regenerate.
- **Live recording WER is informative only when the reference matches the spoken language.** The normalization pipeline does its best for unsupported language codes, but accuracy degrades.
- **Audio playback in the Streamlit app depends on local codec availability.** `.opus` and `.flac` files may not play in all browsers without the appropriate system codecs.
- **CPU vs GPU for full-concat.** CPU inference (fp32) produced lower WER than GPU (fp16) for full-concat runs on this machine. Oracle-segment runs were not systematically compared across devices.

---

## Demo Assets

| Asset | Description |
|---|---|
| `app/streamlit_app.py` | Main interactive demo app |
| `notebooks/whisper_vs_whisperx_demo.ipynb` | Notebook walkthrough of the Whisper vs WhisperX comparison |
| `results/subtitles/demo_cases/<entry_id>/` | Per-entry bundled demo outputs: reference, Whisper, WhisperX SRTs and JSON sidecars |

---

## Manifest Format (Reference)

**Monolingual entry:**
```json
{
  "id": "ls_1089-134686-0000",
  "audio_path": "/abs/path/to/1089-134686-0000.flac",
  "language": "en",
  "transcript": "HE HOPED THERE WOULD BE STEW FOR DINNER"
}
```

**Bilingual entry:**
```json
{
  "id": "bilingual_0000_ls_4970-29093-0015_mls_es_8585_9503_000061",
  "audio_path": "__multi__",
  "language": "bilingual",
  "transcript": "YOU CAN BEGIN BY CARRYING A ROD después de haber bebido masqué un poco de tabaco",
  "segments": [
    { "start": 0.0, "end": 3.3, "language": "en", "transcript": "YOU CAN BEGIN BY CARRYING A ROD",
      "audio_path": "/abs/path/to/4970-29093-0015.flac", "pause_s": 0.0 },
    { "start": 3.3, "end": 13.6, "language": "es", "transcript": "después de haber bebido masqué un poco de tabaco",
      "audio_path": "/abs/path/to/8585_9503_000061.opus", "pause_s": 0.0 }
  ]
}
```

Key fields: `audio_path` is `"__multi__"` for bilingual entries (use `seg.audio_path` per segment). `language` is `"en"`, `"es"`, or `"bilingual"`. Segment `start`/`end` are populated only if the source entry had `duration_s`; subtitle export infers timing from audio otherwise.
