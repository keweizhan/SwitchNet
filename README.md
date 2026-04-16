# SwitchNet

A manifest-driven pipeline for evaluating Whisper on bilingual / code-switching speech. Built for EE519.

The core idea is to study how Whisper handles English-Spanish code-switching by constructing controlled synthetic bilingual samples from real monolingual corpora (LibriSpeech and MLS Spanish), running inference under different conditions, and measuring WER at the utterance level and near language switch points.

Everything is driven by JSONL manifest files, which keeps data preparation, inference, and evaluation cleanly separated and repeatable.

---

## Current Capabilities

- Monolingual English evaluation (LibriSpeech test-clean)
- Monolingual Spanish evaluation (MLS Spanish test)
- Bilingual oracle-segment routing: each segment decoded separately with its known language forced
- Bilingual full-concat decoding: all segments concatenated with optional silence gaps, decoded in one Whisper pass
- Switch-point WER: WER computed over a word window around each language boundary
- Controlled experiments: 2-segment order effect (A1), pause vs no-pause (A2), 3-segment arrangement (A3)
- Subtitle export for bilingual samples: English-only (default) or optional bilingual display mode; `.srt` output; rule-based cue splitting for long cues (V1.2)

---

## Repository Structure

```
SwitchNet/
+-- scripts/
|   +-- build_manifests.py    # Build JSONL manifests from raw datasets
|   +-- run_eval.py           # Main eval driver: transcribe + evaluate in one command
|   +-- export_subtitles.py   # Export English or bilingual .srt files for bilingual entries
|   +-- aggregate_results.py  # Collect all *_summary.json files into one CSV table
|   +-- make_core_results.py  # Produce curated core-results table for paper (hardcodes experiment list)
|   +-- run_eval_whisperx.py  # Same as run_eval.py but uses WhisperX backend (optional, pip install whisperx)
|   +-- run_demo_subtitles.py # Run three predefined subtitle demo cases; output to results/demo_cases/
|   +-- plot_results.py      # Generate WER comparison plots from aggregated_results.csv (requires matplotlib)
|
+-- src/
|   +-- asr/
|   |   +-- transcribe.py     # Whisper inference; handles mono, oracle-segment, full-concat
|   |   +-- evaluate.py       # WER / MER / switch-point WER metrics
|   |   +-- subtitles.py      # SRT building: cue timing, text cleanup, file writing
|   +-- data/
|   |   +-- manifest.py       # ManifestEntry / Segment dataclasses, load/save helpers
|   +-- utils/
|       +-- normalize.py      # Language-aware text normalization for WER
|
+-- data/
|   +-- manifests/            # JSONL manifest files (one per dataset / experiment)
|   +-- LibriSpeech/          # English audio (test-clean, .flac)
|   +-- mls_spanish/          # Spanish audio (test, .opus)
|
+-- results/                  # Transcription outputs (.jsonl) and eval summaries (.json)
```

**scripts/** are entry points meant to be run from the repo root.
**src/** is the library layer -- scripts import from here, nothing in src imports from scripts.

---

## Dependencies

The code uses **openai-whisper** (the `whisper` package), not faster-whisper.

```bash
pip install -r requirements.txt
```

Which installs:

- `openai-whisper` -- inference
- `torch` -- CUDA detection (`torch.cuda.is_available()`) and tensor ops
- `numpy` -- audio array handling in the bilingual concat path
- `librosa` -- audio loading and duration inference (used in bilingual paths and subtitle timing)
- `jiwer` -- WER / MER computation

FFmpeg must be available on the system path for Whisper's audio decoding:

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

Python 3.10+ recommended (the code uses `str | Path` union syntax).

---

## Manifest Format

The entire pipeline is manifest-driven. A manifest is a JSONL file where each line is one sample.

**Monolingual entry:**
```json
{
  "id": "ls_1089-134686-0000",
  "audio_path": "/abs/path/to/1089-134686-0000.flac",
  "language": "en",
  "transcript": "HE HOPED THERE WOULD BE STEW FOR DINNER"
}
```

**Bilingual entry** (synthetic concat of two separate audio files):
```json
{
  "id": "bilingual_0000_ls_4970-29093-0015_mls_es_8585_9503_000061",
  "audio_path": "__multi__",
  "language": "bilingual",
  "transcript": "YOU CAN BEGIN BY CARRYING A ROD después de haber bebido masqué un poco de tabaco",
  "segments": [
    {
      "start": 0.0, "end": 3.3,
      "language": "en",
      "transcript": "YOU CAN BEGIN BY CARRYING A ROD",
      "audio_path": "/abs/path/to/4970-29093-0015.flac",
      "pause_s": 0.0
    },
    {
      "start": 3.3, "end": 13.6,
      "language": "es",
      "transcript": "después de haber bebido masqué un poco de tabaco",
      "audio_path": "/abs/path/to/8585_9503_000061.opus",
      "pause_s": 0.0
    }
  ]
}
```

Key fields:
- `audio_path`: real path for monolingual entries; `"__multi__"` sentinel for bilingual entries (use `seg.audio_path` instead)
- `language`: `"en"`, `"es"`, or `"bilingual"`
- `segments`: list of `Segment` objects; required for bilingual entries; each has its own `audio_path`, `language`, `transcript`, and `pause_s`
- `duration_s`: optional; used for RTF reporting; populated by the `durations` sub-command
- `start` / `end`: timing in seconds; often `0.0` in current manifests because source entries lack `duration_s` -- timing is inferred from audio at runtime where needed

Audio paths in the current manifests are absolute Windows paths. If you move the repo or run on a different machine, re-run `build_manifests.py` to regenerate them.

---

## Main Workflows

### 1. Build manifests

```bash
# LibriSpeech (English)
python scripts/build_manifests.py librispeech \
    --ls-root data/LibriSpeech \
    --split   test-clean \
    --output  data/manifests/en_librispeech_test.jsonl

# MLS Spanish
python scripts/build_manifests.py mls \
    --mls-root data/mls_spanish \
    --split    test \
    --output   data/manifests/es_mls_test.jsonl \
    --max      200

# Bilingual 2-segment, ES-first, 100 pairs
python scripts/build_manifests.py bilingual \
    --en-manifest data/manifests/en_librispeech_test.jsonl \
    --es-manifest data/manifests/es_mls_test.jsonl \
    --output      data/manifests/bilingual_es-en_100.jsonl \
    --pairs       100 \
    --pattern     es-en

# Bilingual 2-segment with 0.5s pause gaps
python scripts/build_manifests.py bilingual \
    --en-manifest data/manifests/en_librispeech_test.jsonl \
    --es-manifest data/manifests/es_mls_test.jsonl \
    --output      data/manifests/bilingual_es-en_100_pause05.jsonl \
    --pairs       100 \
    --pattern     es-en \
    --pause-s     0.5

# 3-segment EN->ES->EN
python scripts/build_manifests.py bilingual \
    --en-manifest data/manifests/en_librispeech_test.jsonl \
    --es-manifest data/manifests/es_mls_test.jsonl \
    --output      data/manifests/bilingual_en-es-en_100_pause05.jsonl \
    --pairs       100 \
    --pattern     en-es-en \
    --pause-s     0.5

# (Optional) populate duration_s fields after building
python scripts/build_manifests.py durations \
    --manifest data/manifests/en_librispeech_test.jsonl
```

Supported patterns for bilingual: `en-es`, `es-en`, `mixed`, `en-es-en`, `es-en-es`.

### 2. Run evaluation

```bash
# Monolingual Spanish baseline
python scripts/run_eval.py \
    --manifest data/manifests/es_mls_test.jsonl \
    --model    large-v3 \
    --tag      es_mls_baseline

# Bilingual oracle-segment (default mode)
python scripts/run_eval.py \
    --manifest data/manifests/bilingual_es-en_100.jsonl \
    --model    large-v3 \
    --tag      bilingual_es-en_100_oracle

# Bilingual full-concat, no pause
python scripts/run_eval.py \
    --manifest         data/manifests/bilingual_es-en_100.jsonl \
    --model            large-v3 \
    --tag              a2_es-en_100_nopause \
    --bilingual-mode   full_concat

# Bilingual full-concat, 0.5s pause
python scripts/run_eval.py \
    --manifest         data/manifests/bilingual_es-en_100_pause05.jsonl \
    --model            large-v3 \
    --tag              a2_es-en_100_pause05 \
    --bilingual-mode   full_concat

# Run on CPU explicitly
python scripts/run_eval.py \
    --manifest data/manifests/bilingual_es-en_100.jsonl \
    --model    large-v3 \
    --device   cpu \
    --tag      a2_es-en_100_nopause_cpu \
    --bilingual-mode full_concat

# Skip transcription if results file already exists (re-run eval only)
python scripts/run_eval.py \
    --manifest        data/manifests/bilingual_es-en_100.jsonl \
    --tag             a2_es-en_100_nopause_cpu \
    --skip-transcribe
```

Results are written to `results/<tag>.jsonl` (per-utterance hypotheses) and `results/<tag>_summary.json` (aggregate metrics).

### 3. Export subtitles

```bash
# English subtitles (default): translate ES->EN, one cue per segment
python scripts/export_subtitles.py \
    --manifest    data/manifests/bilingual_smoke.jsonl \
    --output-dir  results/subtitles/demo_en \
    --model       large-v3 \
    --translate-es \
    --limit       1

# Bilingual mode: Spanish cues show source text on line 1, English translation on line 2
python scripts/export_subtitles.py \
    --manifest       data/manifests/bilingual_smoke.jsonl \
    --output-dir     results/subtitles/demo_bilingual \
    --model          large-v3 \
    --translate-es \
    --subtitle-mode  bilingual \
    --limit          1

# Bilingual mode with rule-based cue splitting (split cues that exceed 15 words)
python scripts/export_subtitles.py \
    --manifest          data/manifests/bilingual_smoke.jsonl \
    --output-dir        results/subtitles/demo_split \
    --model             large-v3 \
    --translate-es \
    --subtitle-mode     bilingual \
    --max-words-per-cue 15 \
    --limit             1

# 5-entry batch validation
python scripts/export_subtitles.py \
    --manifest    data/manifests/bilingual_es-en_100.jsonl \
    --output-dir  results/subtitles/es-en_batch5 \
    --model       large-v3 \
    --translate-es \
    --limit       5

# Full run
python scripts/export_subtitles.py \
    --manifest    data/manifests/bilingual_es-en_100.jsonl \
    --output-dir  results/subtitles/es-en_100 \
    --model       large-v3 \
    --translate-es
```

Without `--translate-es`, every segment is transcribed in its source language (useful for debugging the pipeline without caring about English output).

**Quick demo (three modes, one command):**

```bash
# Preview what will run:
python scripts/run_demo_subtitles.py --dry-run

# Run all three demo cases on the smoke manifest (1 bilingual entry):
python scripts/run_demo_subtitles.py

# Output goes to results/demo_cases/demo_en_only/, demo_bilingual/, demo_bilingual_split/
```

Output per entry: one `<id>.srt` + one `<id>.json` sidecar with per-segment hypotheses and cue timing.

---

## Experiment Summary

All experiments use `large-v3` on 100-pair bilingual manifests drawn from LibriSpeech (EN) and MLS Spanish (ES).

### A1 - Segment order effect under oracle routing

Oracle-segment mode decodes each segment independently with its known language. A1 asks whether the order of EN/ES segments affects whole-utterance WER.

| Condition | WER | MER |
|---|---|---|
| EN->ES | 0.0385 | 0.0381 |
| ES->EN | 0.0348 | 0.0345 |

There is a small consistent difference, with ES-first slightly lower. The gap is not large enough to claim a stable mechanism -- it likely reflects variation in the specific utterance samples rather than a strong position effect.

### A2 - Pause vs no-pause under full-concat decoding

Full-concat mode concatenates all segment audio into one waveform and runs a single Whisper pass with `language=None`. A2 tests whether inserting a 0.5s real silence gap between segments helps.

| Condition | Overall WER | Overall MER | Switch-pt WER | Switch-pt MER |
|---|---|---|---|---|
| No pause  | 0.3170 | 0.3076 | 0.4257 | 0.4177 |
| 0.5s pause | 0.3008 | 0.2943 | 0.4076 | 0.3992 |

Adding a real 0.5s silence gap gives a small but consistent improvement across the board, including near switch points. Full-concat WER is much higher than oracle-segment WER -- the gap is expected since full-concat gives Whisper no language hint.

### A3 - 3-segment arrangement effect

A3 extends A2 to 3-segment patterns. All conditions use 0.5s pauses and full-concat decoding.

| Condition | Overall WER | Overall MER | Switch-pt WER | Switch-pt MER |
|---|---|---|---|---|
| ES->EN (2-seg baseline) | 0.3008 | 0.2943 | 0.4076 | 0.3992 |
| ES->EN->ES (3-seg)      | 0.2425 | 0.2294 | 0.5880 | 0.5748 |
| EN->ES->EN (3-seg)      | 0.2210 | 0.2123 | 0.3385 | 0.3322 |

Overall WER improves in both 3-segment conditions compared to the 2-segment baseline, but switch-point WER behaves differently: EN->ES->EN improves at switch points while ES->EN->ES gets noticeably worse. Language arrangement matters more than simply adding more switches.

---

## Subtitle Export (V1.2)

`scripts/export_subtitles.py` produces `.srt` subtitle files for bilingual manifest entries.

**How it works:**
- Loads a bilingual manifest and filters for `language="bilingual"` entries
- Runs oracle-segment transcription: each segment decoded separately with its known language
- English segments: `task="transcribe"` — output is English as-is
- Spanish segments: `task="translate"` when `--translate-es` is set — Whisper translates to English
- Timing is taken from `segment.start`/`segment.end` when populated; otherwise inferred from audio file duration via librosa
- Cue timestamps are built cumulatively, respecting `pause_s` gaps, so timestamps remain monotonic
- Light text cleanup before writing: whitespace normalization, first-character capitalization, terminal period added if absent

**Display modes:**

| Mode | What you get |
|---|---|
| `--subtitle-mode english` (default) | One English line per cue for all segments |
| `--subtitle-mode bilingual` | Spanish cues: source text on line 1, English translation on line 2 |

**Rule-based cue splitting (V1.2):**

Pass `--max-words-per-cue N` (or `--max-chars-per-cue N`) to split long cues into shorter sub-cues. The splitter tries boundaries in order: sentence-ending punctuation → clause marks → word-count midpoint. The midpoint fallback avoids ending a chunk on short function words (the, a, to, of, etc.) where possible.

Sub-cue durations are distributed proportionally by word count. Only the final sub-cue of a split may receive an appended period; intermediate sub-cues are left without added terminal punctuation.

If a cue is split, source text is not propagated to sub-cues (it cannot be reliably partitioned), so split bilingual cues fall back to English-only display.

**Example output (English mode, one entry):**

```
1
00:00:00,000 --> 00:00:03,325
You can begin by carrying a rod.

2
00:00:03,325 --> 00:00:08,657
After having drunk I masked a little tobacco.
```

**Limitations:**

- This is segment-level, manifest-aware subtitle export. It is not free-form code-switch subtitle generation — it relies on the manifest knowing where each language segment begins and ends.
- Bilingual display is most reliable on unsplit cues. Split cues always show English only.
- Split point quality improves on longer cues with natural punctuation. Short cues with no punctuation and many function words may still produce awkward splits.
- Translation quality is bounded by Whisper's `task="translate"` accuracy, which can produce wrong output on short or acoustically ambiguous segments.

---

## Practical Notes and Limitations

- **Bilingual samples are synthetic.** Each bilingual entry is constructed by concatenating two (or three) real monolingual utterances. This is not natural code-switched speech -- it is a controlled approximation.

- **Manifest paths are absolute and machine-specific.** If you move the data or change machines, re-run `build_manifests.py` to regenerate the manifests. There is no path remapping utility currently.

- **Segment timing is often 0.0 in current manifests.** The `start`/`end` fields on bilingual segments are only populated if the source entry had `duration_s`. Run `build_manifests.py durations` on your source manifests before building bilingual ones if you need accurate timing. Subtitle export handles this gracefully by inferring duration from audio, but it is slower.

- **CPU vs GPU for large-v3 full-concat.** On the current test machine, CPU inference produced lower WER than GPU for full-concat experiments. This was consistent enough to prefer CPU for those runs. The cause is likely numerical precision differences (GPU uses fp16, CPU uses fp32). Oracle-segment runs were not systematically compared.

- **Translation quality.** Whisper's `task="translate"` works reasonably for full sentences but can produce noticeably wrong output on short segments or segments with unusual vocabulary. The subtitle output for translated Spanish segments should be treated as a rough English rendering, not a reliable translation.

- **Subtitle export is manifest-aware, not automatic.** The pipeline knows segment boundaries from the manifest. It does not detect code-switching automatically from raw audio. Bilingual display mode is strongest on unsplit cues; rule-based splitting improves readability but does not guarantee clean phrase boundaries.

- **This is research/development tooling.** There is no error recovery, no resume-from-checkpoint, and no production-facing interface. If a single entry errors during a long run, the script logs the error and continues -- the results file will have a blank hypothesis for that entry.

---

## Future Work

- **Preprocessing ablations.** Test denoising or normalization before feeding Whisper, especially for the full-concat path where there is no language forcing.
- **Timestamp refinement.** Current cue timing comes from audio file durations or inferred segment lengths. Per-word timestamps from Whisper's output (`result["segments"]`) would give tighter alignment, especially after rule-based splitting.
- **Natural code-switching data.** The synthetic concatenation approach is a useful proxy but not the same as real code-switched speech. Evaluation on an actual CS corpus (e.g., Miami Bangor, SEAME) would be a meaningful next step.
