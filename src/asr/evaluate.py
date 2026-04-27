"""
evaluate.py — ASR evaluation for SwitchNet.

Metrics:
  - WER  (Word Error Rate)
  - MER  (Match Error Rate)
  - Switch-point WER: WER computed only on a ±N-word window around each
    language-switch boundary. This is the key metric for EE519.

Usage (CLI):
    python -m src.asr.evaluate \
        --results  results/es_whisper_large-v3.jsonl \
        --manifest data/manifests/es_common_voice.jsonl \
        --output   results/es_eval_summary.json

Usage (API):
    from src.asr.evaluate import evaluate_results, switchpoint_wer
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import jiwer

from src.data.manifest import ManifestEntry, load_manifest
from src.utils.normalize import normalize_text


# ---------------------------------------------------------------------------
# jiwer compatibility shim
# ---------------------------------------------------------------------------

def compute_word_measures(reference: str, hypothesis: str) -> dict:
    """Compute word-level ASR metrics compatible with jiwer ≥2 and ≥3.

    jiwer ≥3 removed ``compute_measures`` in favour of ``process_words``.
    This helper tries ``process_words`` first, falls back to
    ``compute_measures``, so the codebase works with either version.

    Args:
        reference:  Ground-truth transcript (already normalised).
        hypothesis: ASR output transcript (already normalised).

    Returns:
        Dict with keys: wer, mer, hits, substitutions, deletions, insertions.
    """
    if hasattr(jiwer, "process_words"):
        out = jiwer.process_words(reference, hypothesis)
        return {
            "wer":           float(out.wer),
            "mer":           float(out.mer),
            "hits":          int(out.hits),
            "substitutions": int(out.substitutions),
            "deletions":     int(out.deletions),
            "insertions":    int(out.insertions),
        }
    if hasattr(jiwer, "compute_measures"):
        m = jiwer.compute_measures(reference, hypothesis)
        return {
            "wer":           float(m.get("wer",           0.0)),
            "mer":           float(m.get("mer",           0.0)),
            "hits":          int(  m.get("hits",          0)),
            "substitutions": int(  m.get("substitutions", 0)),
            "deletions":     int(  m.get("deletions",     0)),
            "insertions":    int(  m.get("insertions",    0)),
        }
    raise RuntimeError(
        "Unsupported jiwer version: neither process_words nor compute_measures "
        "is available.  Install jiwer>=2.3: pip install 'jiwer>=2.3'"
    )


def _compute_measures_compat(refs, hyps) -> dict:
    """Internal helper: accepts string or list[str] for both ref and hyp."""
    if isinstance(refs, list):
        refs = " ".join(refs)
    if isinstance(hyps, list):
        hyps = " ".join(hyps)
    return compute_word_measures(refs, hyps)


# ---------------------------------------------------------------------------
# Text normalization (thin wrapper — reuse your existing normalize logic)
# ---------------------------------------------------------------------------

def _norm(text: str, language: str = "en") -> str:
    """Normalize text for WER computation."""
    return normalize_text(text, language=language)


# ---------------------------------------------------------------------------
# Per-entry metrics
# ---------------------------------------------------------------------------

def compute_entry_metrics(ref: str, hyp: str, language: str = "en") -> dict:
    ref_n = _norm(ref, language)
    hyp_n = _norm(hyp, language)

    if not ref_n.strip():
        return {"wer": None, "mer": None, "ref_words": 0, "hyp_words": len(hyp_n.split())}

    measures = compute_word_measures(ref_n, hyp_n)
    return {
        "wer": round(measures["wer"], 4),
        "mer": round(measures["mer"], 4),
        "substitutions": measures["substitutions"],
        "deletions":     measures["deletions"],
        "insertions":    measures["insertions"],
        "ref_words":     len(ref_n.split()),
        "hyp_words":     len(hyp_n.split()),
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation over a results file
# ---------------------------------------------------------------------------

def evaluate_results(
    results_path: str | Path,
    manifest_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    window_words: int = 5,
) -> dict:
    """
    Evaluate transcription results from transcribe.py output.

    Args:
        results_path:   JSONL produced by transcribe_manifest()
        manifest_path:  Optional: original manifest (needed for switch-point WER)
        output_path:    Optional: where to write the summary JSON
        window_words:   Context window (words) around switch points for local WER

    Returns:
        dict with aggregate metrics
    """
    results_path = Path(results_path)

    # Load transcription results
    results = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Load manifest entries (for segment/switch-point info)
    manifest_index: dict[str, ManifestEntry] = {}
    if manifest_path:
        for entry in load_manifest(manifest_path):
            manifest_index[entry.id] = entry

    per_entry = []
    all_refs, all_hyps = [], []
    lang_refs: dict[str, list] = {}
    lang_hyps: dict[str, list] = {}

    sp_windows_ref: list[str] = []  # switch-point window tokens
    sp_windows_hyp: list[str] = []

    # Segment-level accumulators (bilingual only)
    seg_lang_refs:   dict[str, list] = {}   # keyed by language ("en"/"es")
    seg_lang_hyps:   dict[str, list] = {}
    seg_pos_refs:    dict[str, list] = {}   # keyed by "position_0", "position_1", …
    seg_pos_hyps:    dict[str, list] = {}
    seg_lp_refs:     dict[str, list] = {}   # keyed by "en_pos0", "es_pos1", …
    seg_lp_hyps:     dict[str, list] = {}

    for r in results:
        if r.get("error"):
            continue
        ref = r["reference"]
        hyp = r["hypothesis"]
        lang = r.get("language", "en")

        m = compute_entry_metrics(ref, hyp, lang)
        m["id"] = r["id"]
        m["language"] = lang
        per_entry.append(m)

        ref_n = _norm(ref, lang)
        hyp_n = _norm(hyp, lang)
        all_refs.append(ref_n)
        all_hyps.append(hyp_n)
        lang_refs.setdefault(lang, []).append(ref_n)
        lang_hyps.setdefault(lang, []).append(hyp_n)

        # Switch-point windowed WER
        entry = manifest_index.get(r["id"])
        if entry and entry.has_switch_points():
            ref_toks = ref_n.split()
            hyp_toks = hyp_n.split()
            sp_segs = _extract_switchpoint_windows(entry, ref_toks, hyp_toks, window_words)
            sp_windows_ref.extend(sp_segs["ref"])
            sp_windows_hyp.extend(sp_segs["hyp"])

        # Segment-level accumulation (bilingual only)
        for seg in r.get("segment_outputs", []):
            slang = seg["language"]
            pos   = seg["position"]
            s_ref = _norm(seg["reference"], slang)
            s_hyp = _norm(seg["hypothesis"], slang)

            seg_lang_refs.setdefault(slang, []).append(s_ref)
            seg_lang_hyps.setdefault(slang, []).append(s_hyp)

            pos_key = f"position_{pos}"
            seg_pos_refs.setdefault(pos_key, []).append(s_ref)
            seg_pos_hyps.setdefault(pos_key, []).append(s_hyp)

            lp_key = f"{slang}_pos{pos}"
            seg_lp_refs.setdefault(lp_key, []).append(s_ref)
            seg_lp_hyps.setdefault(lp_key, []).append(s_hyp)

    # Aggregate WER (corpus-level, not mean of per-utterance WER)
    def corpus_wer(refs, hyps):
        m = _compute_measures_compat(refs, hyps)
        return round(m["wer"], 4), round(m["mer"], 4)

    overall_wer, overall_mer = corpus_wer(all_refs, all_hyps)

    per_lang_metrics = {}
    for lang in lang_refs:
        w, m_val = corpus_wer(lang_refs[lang], lang_hyps[lang])
        per_lang_metrics[lang] = {"wer": w, "mer": m_val, "n": len(lang_refs[lang])}

    # Segment-level summaries (present only when segment_outputs exist)
    per_segment_language: dict = {}
    per_position:         dict = {}
    per_language_position: dict = {}
    if seg_lang_refs:
        for k in seg_lang_refs:
            w, m_val = corpus_wer(seg_lang_refs[k], seg_lang_hyps[k])
            per_segment_language[k] = {"wer": w, "mer": m_val, "n": len(seg_lang_refs[k])}
        for k in seg_pos_refs:
            w, m_val = corpus_wer(seg_pos_refs[k], seg_pos_hyps[k])
            per_position[k] = {"wer": w, "mer": m_val, "n": len(seg_pos_refs[k])}
        for k in seg_lp_refs:
            w, m_val = corpus_wer(seg_lp_refs[k], seg_lp_hyps[k])
            per_language_position[k] = {"wer": w, "mer": m_val, "n": len(seg_lp_refs[k])}

    summary = {
        "overall": {
            "wer": overall_wer,
            "mer": overall_mer,
            "n_utterances": len(per_entry),
        },
        "per_language": per_lang_metrics,
        "per_entry": per_entry,
        "per_segment_language":  per_segment_language  or None,
        "per_position":          per_position          or None,
        "per_language_position": per_language_position or None,
    }

    # Switch-point WER (only if we have switch-point entries)
    if sp_windows_ref:
        sp_wer, sp_mer = corpus_wer(sp_windows_ref, sp_windows_hyp)
        summary["switch_point"] = {
            "wer": sp_wer,
            "mer": sp_mer,
            "window_words": window_words,
            "n_windows": len(sp_windows_ref),
        }
    else:
        summary["switch_point"] = None

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Evaluation summary written to {output_path}")

    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Switch-point windowed evaluation helpers
# ---------------------------------------------------------------------------

def _extract_switchpoint_windows(
    entry: ManifestEntry,
    ref_toks: List[str],
    hyp_toks: List[str],
    window_words: int,
) -> dict:
    """
    Estimate word positions of switch points by aligning segment boundaries
    to word counts, then extract ±window_words context.

    This is an approximation: we don't have word-level timestamps in the
    reference, so we distribute words proportionally across segments by duration.

    Returns {"ref": [...windows...], "hyp": [...windows...]}
    """
    if not entry.segments or not ref_toks:
        return {"ref": [], "hyp": []}

    total_dur = sum(s.end - s.start for s in entry.segments)
    if total_dur <= 0:
        # Fallback: no timing info — estimate word positions from per-segment
        # transcript token counts (normalized per segment language).
        seg_word_counts = [
            len(_norm(s.transcript, s.language).split())
            for s in entry.segments
        ]
        if sum(seg_word_counts) == 0:
            return {"ref": [], "hyp": []}
        sp_word_positions = []
        cumulative = 0
        for i, count in enumerate(seg_word_counts[:-1]):
            cumulative += count
            if entry.segments[i].language != entry.segments[i + 1].language:
                sp_word_positions.append(cumulative)
        ref_windows, hyp_windows = [], []
        for pos in sp_word_positions:
            lo = max(0, pos - window_words)
            hi = min(len(ref_toks), pos + window_words)
            ref_windows.append(" ".join(ref_toks[lo:hi]))
            h_lo = max(0, pos - window_words)
            h_hi = min(len(hyp_toks), pos + window_words)
            hyp_windows.append(" ".join(hyp_toks[h_lo:h_hi]))
        return {"ref": ref_windows, "hyp": hyp_windows}

    # Assign word counts proportionally to segment duration
    seg_word_counts = []
    running = 0
    for seg in entry.segments:
        frac = (seg.end - seg.start) / total_dur
        count = max(1, round(frac * len(ref_toks)))
        seg_word_counts.append(count)
        running += count

    # Compute switch-point word positions (boundary between adjacent segments)
    sp_word_positions = []
    cumulative = 0
    for i, count in enumerate(seg_word_counts[:-1]):
        cumulative += count
        if entry.segments[i].language != entry.segments[i + 1].language:
            sp_word_positions.append(cumulative)

    ref_windows = []
    hyp_windows = []
    for pos in sp_word_positions:
        lo = max(0, pos - window_words)
        hi = min(len(ref_toks), pos + window_words)
        ref_windows.append(" ".join(ref_toks[lo:hi]))

        # hyp may be shorter/longer; clip gracefully
        h_lo = max(0, pos - window_words)
        h_hi = min(len(hyp_toks), pos + window_words)
        hyp_windows.append(" ".join(hyp_toks[h_lo:h_hi]))

    return {"ref": ref_windows, "hyp": hyp_windows}


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 52)
    print("  SwitchNet Evaluation Summary")
    print("=" * 52)
    ov = summary["overall"]
    print(f"  Overall WER : {ov['wer']:.4f}  ({ov['n_utterances']} utterances)")
    print(f"  Overall MER : {ov['mer']:.4f}")
    print()
    for lang, m in summary.get("per_language", {}).items():
        print(f"  [{lang.upper()}]  WER={m['wer']:.4f}  MER={m['mer']:.4f}  n={m['n']}")
    sp = summary.get("switch_point")
    if sp:
        print()
        print(f"  Switch-point WER : {sp['wer']:.4f}  "
              f"(window=±{sp['window_words']} words, n={sp['n_windows']})")
        print(f"  Switch-point MER : {sp['mer']:.4f}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SwitchNet ASR evaluation")
    parser.add_argument("--results",   required=True,       help="JSONL from transcribe.py")
    parser.add_argument("--manifest",  default=None,        help="Original JSONL manifest (for switch-point WER)")
    parser.add_argument("--output",    default=None,        help="Output summary JSON path")
    parser.add_argument("--window",    type=int, default=5, help="Switch-point window in words")
    args = parser.parse_args()

    evaluate_results(
        results_path=args.results,
        manifest_path=args.manifest,
        output_path=args.output,
        window_words=args.window,
    )


if __name__ == "__main__":
    main()