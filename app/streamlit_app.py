"""
app/streamlit_app.py — SwitchNet bilingual ASR demo (Streamlit).

Runs locally with:
    streamlit run app/streamlit_app.py

Sidebar controls:
  Manifest  → one of the bilingual_es-en_*.jsonl manifests
  Entry     → any bilingual entry in the selected manifest
  Whisper   → eval JSONL (bilingual_es-en_50_large_cpu.jsonl by default)
  WhisperX  → eval JSONL (wx_bilingual_es-en_50_large_cpu.jsonl by default)

Tabs:
  Reference   — ES audio + transcript | switch banner | EN audio + transcript
  Comparison  — Reference / Whisper / WhisperX cues side-by-side (timed)
  WER Summary — per-entry and overall WER/MER table + bar chart
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.manifest import ManifestEntry, load_manifest
from src.asr.subtitles import SubtitleCue, build_segment_level_cues

# ── constants ─────────────────────────────────────────────────────────────────
MANIFESTS_DIR = ROOT / "data" / "manifests"
RESULTS_DIR   = ROOT / "results"

DEFAULT_MANIFEST   = "bilingual_es-en_50.jsonl"
DEFAULT_WHISPER    = "bilingual_es-en_50_large_cpu.jsonl"
DEFAULT_WHISPERX   = "wx_bilingual_es-en_50_large_cpu.jsonl"
DEFAULT_ENTRY_HINT = "bilingual_es-en_0022"      # partial match

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwitchNet — Bilingual ASR Demo",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .transcript-card {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f5f7fa;
        font-size: 1.08rem;
        line-height: 1.65;
        margin-top: 6px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }
    .transcript-card.es {
        background: linear-gradient(180deg, rgba(90, 24, 30, 0.92), rgba(52, 16, 20, 0.94));
        border-color: rgba(231, 76, 60, 0.45);
        box-shadow: inset 4px 0 0 #e74c3c, 0 10px 24px rgba(0, 0, 0, 0.22);
    }
    .transcript-card.en {
        background: linear-gradient(180deg, rgba(18, 66, 42, 0.92), rgba(11, 42, 28, 0.94));
        border-color: rgba(39, 174, 96, 0.45);
        box-shadow: inset 4px 0 0 #27ae60, 0 10px 24px rgba(0, 0, 0, 0.22);
    }
    .switch-banner {
        margin: 18px auto 14px;
        max-width: 640px;
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid rgba(255, 196, 87, 0.42);
        background: linear-gradient(180deg, rgba(120, 80, 18, 0.88), rgba(82, 56, 14, 0.94));
        color: #fff4cf;
        text-align: center;
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: 0.08em;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.2);
    }
    .cue-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
        font-size: 0.95rem;
        color: #f5f7fa;
    }
    .cue-table td {
        padding: 10px 12px;
        vertical-align: top;
    }
    .cue-table .time-cell {
        width: 86px;
        white-space: nowrap;
        color: #cbd5e1;
        font-size: 0.82rem;
        font-variant-numeric: tabular-nums;
    }
    .cue-table .badge-cell {
        width: 40px;
        padding-right: 6px;
    }
    .cue-table .text-cell {
        line-height: 1.55;
    }
    .cue-row.es td {
        background: rgba(103, 30, 39, 0.86);
        border-top: 1px solid rgba(231, 76, 60, 0.32);
        border-bottom: 1px solid rgba(231, 76, 60, 0.32);
    }
    .cue-row.en td {
        background: rgba(21, 77, 49, 0.86);
        border-top: 1px solid rgba(39, 174, 96, 0.32);
        border-bottom: 1px solid rgba(39, 174, 96, 0.32);
    }
    .cue-row.neutral td {
        background: rgba(44, 52, 64, 0.88);
        border-top: 1px solid rgba(148, 163, 184, 0.18);
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }
    .cue-row td:first-child {
        border-top-left-radius: 12px;
        border-bottom-left-radius: 12px;
    }
    .cue-row td:last-child {
        border-top-right-radius: 12px;
        border-bottom-right-radius: 12px;
    }
    .cue-switch td {
        padding: 12px 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 196, 87, 0.42);
        background: linear-gradient(180deg, rgba(120, 80, 18, 0.88), rgba(82, 56, 14, 0.94));
        color: #fff4cf;
        text-align: center;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .comparison-note {
        margin: 0 0 10px;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid rgba(245, 158, 11, 0.35);
        background: rgba(120, 80, 18, 0.22);
        color: #fde68a;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_manifest_cached(path: str) -> List[ManifestEntry]:
    return load_manifest(path)


# Repo root — used for cross-platform audio path resolution.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_audio_path(path_str: str) -> Path:
    """Return a Path that exists on the current machine.

    Manifests store absolute Windows paths (e.g. F:\\EE519\\...\\data\\...).
    On another OS those paths don't exist, but the relative part after the
    first 'data/' segment is stable.  We try:
      1. The stored path as-is.
      2. _REPO_ROOT / <relative-part-starting-at-'data/'>.
    Returns the first existing Path, or the original Path if neither exists
    (caller should check .exists() and degrade gracefully).
    """
    p = Path(path_str)
    if p.exists():
        return p
    # Normalise separators and find the 'data/' anchor
    norm = path_str.replace("\\", "/")
    marker = "data/"
    idx = norm.find(marker)
    if idx != -1:
        rel = norm[idx:]          # e.g. "data/mls_spanish/test/audio/..."
        candidate = _REPO_ROOT / rel
        if candidate.exists():
            return candidate
    return p  # not found; caller handles gracefully


@st.cache_data(show_spinner=False)
def _audio_wav_bytes(audio_path: str) -> Optional[bytes]:
    """Decode any audio (opus / flac / wav) → WAV bytes at native sample rate."""
    try:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_jsonl(path: str) -> Dict[str, dict]:
    """Load eval JSONL → {entry_id: record}."""
    records: Dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                records[r["id"]] = r
    except Exception:
        pass
    return records


@st.cache_data(show_spinner=False)
def _load_summary(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _audio_duration(audio_path: str) -> float:
    """Return duration in seconds; falls back to 5.0 on error."""
    try:
        import librosa
        resolved = str(resolve_audio_path(audio_path))
        return float(librosa.get_duration(path=resolved))
    except Exception:
        return 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Derived data (also cached so they survive widget interactions)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _switch_time_cached(entry_id: str, manifest_path: str) -> Optional[float]:
    """Cumulative offset (s) where the first ES→EN transition begins."""
    entries = _load_manifest_cached(manifest_path)
    entry = next((e for e in entries if e.id == entry_id), None)
    if entry is None:
        return None
    offset = 0.0
    prev_lang = None
    for seg in entry.segments:
        dur = _audio_duration(str(seg.audio_path))
        if prev_lang == "es" and seg.language == "en":
            return offset
        offset += dur
        prev_lang = seg.language
    return None


def _audio_duration_from_bytes(audio_bytes: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if frame_rate <= 0:
            return None
        return frame_count / frame_rate
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _live_transcriber_cached(model_size: str, device: str):
    from src.asr.transcribe import Transcriber

    return Transcriber(model_size=model_size, device=device)


def _transcribe_live_audio(
    audio_bytes: bytes, language: Optional[str], model_size: str = "base"
) -> tuple:
    """Transcribe recorded audio bytes via Whisper.

    Returns:
        (hypothesis: str, detected_language: str)
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        transcriber = _live_transcriber_cached(model_size=model_size, device="cpu")
        return transcriber._transcribe_file_ex(temp_path, language=language)
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _live_entry_metrics(ref: str, hyp: str, language: str) -> dict:
    from src.asr.evaluate import compute_entry_metrics

    # normalize.py already handles unknown language codes with basic lowercasing
    return compute_entry_metrics(ref, hyp, language=language)


@st.cache_data(show_spinner=False)
def _reference_cues_cached(entry_id: str, manifest_path: str) -> List[SubtitleCue]:
    """Build SubtitleCues from ground-truth segment transcripts."""
    entries = _load_manifest_cached(manifest_path)
    entry = next((e for e in entries if e.id == entry_id), None)
    if entry is None:
        return []
    cues: List[SubtitleCue] = []
    offset = 0.0
    for seg in entry.segments:
        dur = _audio_duration(str(seg.audio_path))
        cues.append(SubtitleCue(
            start=offset,
            end=offset + dur,
            text=seg.transcript or "",
            source_language=seg.language,
            timing_source="manifest",
        ))
        offset += dur
    return cues


@st.cache_data(show_spinner=False)
def _model_cues_cached(entry_id: str, manifest_path: str, jsonl_path: str) -> List[SubtitleCue]:
    """Build SubtitleCues from a Whisper or WhisperX eval JSONL record."""
    entries = _load_manifest_cached(manifest_path)
    entry = next((e for e in entries if e.id == entry_id), None)
    if entry is None:
        return []
    records = _load_jsonl(jsonl_path)
    rec = records.get(entry_id)
    if rec is None:
        return []
    seg_outputs = rec.get("segment_outputs")
    if not seg_outputs:
        seg_outputs = [{
            "position":   0,
            "language":   entry.language,
            "reference":  entry.transcript,
            "hypothesis": rec.get("hypothesis", ""),
        }]
    return build_segment_level_cues(
        seg_outputs,
        entry.segments,
        infer_timing_from_audio=True,
    )


@st.cache_data(show_spinner=False)
def _wer_for_entry(entry_id: str, jsonl_name: str) -> Optional[float]:
    """Look up this entry's WER from the corresponding *_summary.json."""
    stem = Path(jsonl_name).stem
    summary = _load_summary(str(RESULTS_DIR / f"{stem}_summary.json"))
    if not summary:
        return None
    for pe in summary.get("per_entry", []):
        if pe.get("id") == entry_id:
            return pe.get("wer")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _lang_badge(lang: Optional[str]) -> str:
    if lang == "es":
        return "🇪🇸"
    if lang == "en":
        return "🇺🇸"
    return "🌐"


def _transcript_card_html(text: str, lang: Optional[str]) -> str:
    lang_class = "es" if lang == "es" else "en"
    return f'<div class="transcript-card {lang_class}">{text or "(no transcript)"}</div>'


def _comparison_note_html(text: str) -> str:
    return f'<div class="comparison-note">{text}</div>'


def _overgeneration_note(seg_outputs: Optional[List[dict]], backend_label: str) -> Optional[str]:
    if not seg_outputs:
        return None

    for out in seg_outputs:
        ref_words = len((out.get("reference") or "").split())
        hyp_words = len((out.get("hypothesis") or "").split())
        if ref_words < 8:
            continue
        if hyp_words >= ref_words + 18 and hyp_words >= int(ref_words * 1.45):
            return (
                f"Possible boundary over-generation. Stored {backend_label} output "
                "continues beyond reference; this is a model behavior, not a UI artifact."
            )
    return None


def _cue_rows_html(cues: List[SubtitleCue], switch_time: Optional[float]) -> str:
    """Render cues as an HTML table; inserts a highlighted switch separator row."""
    rows: List[str] = []
    switch_inserted = False
    for c in cues:
        if switch_time is not None and not switch_inserted and c.start >= switch_time:
            rows.append(
                '<tr class="cue-switch">'
                '<td colspan="3">ES -> EN SWITCH</td></tr>'
            )
            switch_inserted = True
        badge = _lang_badge(c.source_language)
        text  = c.text.replace("\n", "<br>")
        lang_class = "es" if c.source_language == "es" else "en" if c.source_language == "en" else "neutral"
        rows.append(
            f'<tr class="cue-row {lang_class}">'
            f'<td class="time-cell">{_fmt_time(c.start)}</td>'
            f'<td class="badge-cell">{badge}</td>'
            f'<td class="text-cell">{text}</td></tr>'
        )
    return (
        '<table class="cue-table">'
        + "".join(rows)
        + "</table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _available_manifests() -> List[str]:
    return sorted(p.name for p in MANIFESTS_DIR.glob("bilingual_es-en_*.jsonl"))


def _available_jsonls(pattern: str) -> List[str]:
    return sorted(p.name for p in RESULTS_DIR.glob(pattern))


with st.sidebar:
    st.title("🎙️ SwitchNet Demo")
    st.caption("Bilingual ES→EN ASR comparison")

    # manifest
    manifest_choices = _available_manifests()
    manifest_idx = (
        manifest_choices.index(DEFAULT_MANIFEST)
        if DEFAULT_MANIFEST in manifest_choices else 0
    )
    manifest_name = st.selectbox("Manifest", manifest_choices, index=manifest_idx)
    manifest_path = str(MANIFESTS_DIR / manifest_name)

    with st.spinner("Loading manifest…"):
        all_entries = _load_manifest_cached(manifest_path)
    bilingual_entries = [e for e in all_entries if e.language == "bilingual"]

    # entry picker
    entry_ids = [e.id for e in bilingual_entries]
    default_entry_idx = next(
        (i for i, eid in enumerate(entry_ids) if DEFAULT_ENTRY_HINT in eid), 0
    )
    selected_id = st.selectbox(
        "Entry", entry_ids, index=default_entry_idx,
        format_func=lambda x: x.split("_", 3)[-1][:52],
    )
    entry: ManifestEntry = next(e for e in bilingual_entries if e.id == selected_id)

    # Whisper JSONL — exclude wx_ prefixed files
    whisper_choices = ["(none)"] + [
        n for n in _available_jsonls("bilingual_es-en_*.jsonl")
        if not n.startswith("wx_")
    ]
    w_default = (
        whisper_choices.index(DEFAULT_WHISPER)
        if DEFAULT_WHISPER in whisper_choices else 0
    )
    whisper_name = st.selectbox("Whisper JSONL", whisper_choices, index=w_default)
    whisper_path = str(RESULTS_DIR / whisper_name) if whisper_name != "(none)" else None

    # WhisperX JSONL
    whisperx_choices = ["(none)"] + _available_jsonls("wx_bilingual_es-en_*.jsonl")
    wx_default = (
        whisperx_choices.index(DEFAULT_WHISPERX)
        if DEFAULT_WHISPERX in whisperx_choices else 0
    )
    whisperx_name = st.selectbox("WhisperX JSONL", whisperx_choices, index=wx_default)
    whisperx_path = str(RESULTS_DIR / whisperx_name) if whisperx_name != "(none)" else None

    st.divider()
    st.caption(f"Streamlit {st.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute shared data
# ─────────────────────────────────────────────────────────────────────────────

sw_time = _switch_time_cached(entry.id, manifest_path)

# WER from summary JSONs (not from JSONL records — those lack a wer field)
wer_whisper  = _wer_for_entry(entry.id, whisper_name)  if whisper_name  != "(none)" else None
wer_whisperx = _wer_for_entry(entry.id, whisperx_name) if whisperx_name != "(none)" else None


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("SwitchNet — Bilingual ASR Demo")
st.markdown(
    f"**Entry:** `{entry.id}`  \n"
    f"**Structure:** {len(entry.segments)} segments "
    f"({' → '.join(s.language.upper() for s in entry.segments)})"
)

# Switch-time info bar
if sw_time is not None:
    st.info(f"ES→EN switch at **{_fmt_time(sw_time)}**", icon="🔀")

# WER scorecard strip — only when both backends loaded
if wer_whisper is not None or wer_whisperx is not None:
    cols = st.columns(4)
    if wer_whisper is not None:
        cols[0].metric("Whisper WER", f"{wer_whisper:.1%}")
    if wer_whisperx is not None:
        cols[1].metric("WhisperX WER", f"{wer_whisperx:.1%}")
    if wer_whisper is not None and wer_whisperx is not None:
        delta = wer_whisperx - wer_whisper
        cols[2].metric(
            "Δ WER (X − W)",
            f"{delta:+.1%}",
            delta_color="inverse",   # green when negative (WhisperX better)
        )
        if delta < -0.05:
            cols[3].markdown("✅ WhisperX **significantly** better")
        elif delta > 0.05:
            cols[3].markdown("⚠️ WhisperX **worse** on this entry")
        else:
            cols[3].markdown("≈ roughly equivalent")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_ref, tab_cmp, tab_wer, tab_live = st.tabs(
    ["📄 Reference", "⚖️ Comparison", "📊 WER Summary", "🎤 Live Recording"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Reference
# ══════════════════════════════════════════════════════════════════════════════

with tab_ref:
    st.subheader("Ground-truth reference (per segment)")
    st.caption("Audio + transcript for each language segment; ES→EN boundary shown below.")

    for i, seg in enumerate(entry.segments):
        flag  = "🇪🇸" if seg.language == "es" else "🇺🇸"
        label = "Spanish" if seg.language == "es" else "English"

        with st.container():
            st.markdown(f"**{flag} {label} — segment {i}**")

            _resolved_path = resolve_audio_path(str(seg.audio_path))
            wav = _audio_wav_bytes(str(_resolved_path))
            if wav:
                st.audio(wav, format="audio/wav")
            else:
                st.caption(f"_(audio unavailable — tried: `{_resolved_path}`)_")

            st.markdown(
                _transcript_card_html(seg.transcript or "(no transcript)", seg.language),
                unsafe_allow_html=True,
            )

        if i < len(entry.segments) - 1:
            st.markdown(
                '<div class="switch-banner">'
                "ES -> EN SWITCH"
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_cmp:
    st.subheader("Cue-by-cue comparison")
    st.caption(
        "Each column shows timestamped cues. "
        "The highlighted row marks the ES→EN switch boundary."
    )

    ref_cues = _reference_cues_cached(entry.id, manifest_path)
    w_records  = _load_jsonl(whisper_path)  if whisper_path  else {}
    wx_records = _load_jsonl(whisperx_path) if whisperx_path else {}
    w_rec  = w_records.get(entry.id)
    wx_rec = wx_records.get(entry.id)

    col_ref, col_w, col_wx = st.columns(3)

    with col_ref:
        st.markdown("#### 📄 Reference")
        st.markdown(_cue_rows_html(ref_cues, sw_time), unsafe_allow_html=True)

    with col_w:
        st.markdown("#### 🔵 Whisper")
        if whisper_path:
            w_cues = _model_cues_cached(entry.id, manifest_path, whisper_path)
            if w_cues:
                note = _overgeneration_note(w_rec.get("segment_outputs") if w_rec else None, "Whisper")
                if note:
                    st.markdown(_comparison_note_html(note), unsafe_allow_html=True)
                st.markdown(_cue_rows_html(w_cues, sw_time), unsafe_allow_html=True)
                if wer_whisper is not None:
                    st.caption(f"WER = **{wer_whisper:.1%}**")
            else:
                st.caption("_(entry not found in selected JSONL)_")
        else:
            st.caption("_(no Whisper JSONL selected)_")

    with col_wx:
        st.markdown("#### 🟢 WhisperX")
        if whisperx_path:
            wx_cues = _model_cues_cached(entry.id, manifest_path, whisperx_path)
            if wx_cues:
                note = _overgeneration_note(wx_rec.get("segment_outputs") if wx_rec else None, "WhisperX")
                if note:
                    st.markdown(_comparison_note_html(note), unsafe_allow_html=True)
                st.markdown(_cue_rows_html(wx_cues, sw_time), unsafe_allow_html=True)
                if wer_whisperx is not None:
                    st.caption(f"WER = **{wer_whisperx:.1%}**")
            else:
                st.caption("_(entry not found in selected JSONL)_")
        else:
            st.caption("_(no WhisperX JSONL selected)_")

    # ── per-segment text expanders ────────────────────────────────────────────
    if w_rec or wx_rec:
        st.divider()
        st.subheader("Per-segment hypothesis text")

        w_segs  = w_rec.get("segment_outputs",  []) if w_rec  else []
        wx_segs = wx_rec.get("segment_outputs", []) if wx_rec else []
        n = max(len(entry.segments), len(w_segs), len(wx_segs))

        for i in range(n):
            ref_seg = entry.segments[i] if i < len(entry.segments) else None
            w_seg   = w_segs[i]          if i < len(w_segs)         else None
            wx_seg  = wx_segs[i]         if i < len(wx_segs)        else None
            lang    = ref_seg.language.upper() if ref_seg else "?"
            badge   = "🇪🇸" if lang == "ES" else "🇺🇸"

            with st.expander(f"{badge} Segment {i} [{lang}]", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Reference**")
                    st.write(ref_seg.transcript if ref_seg else "—")
                with c2:
                    st.markdown("**Whisper**")
                    hyp = w_seg["hypothesis"] if w_seg else "—"
                    st.write(hyp)
                with c3:
                    st.markdown("**WhisperX**")
                    hyp = wx_seg["hypothesis"] if wx_seg else "—"
                    st.write(hyp)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WER Summary
# ══════════════════════════════════════════════════════════════════════════════

with tab_wer:
    import pandas as pd

    st.subheader("WER / MER summary — all 50 entries")

    def _build_summary_df(jsonl_name: str, label: str) -> Optional[pd.DataFrame]:
        if not jsonl_name or jsonl_name == "(none)":
            return None
        stem = Path(jsonl_name).stem
        summary = _load_summary(str(RESULTS_DIR / f"{stem}_summary.json"))
        if not summary:
            st.warning(f"No summary JSON found for {jsonl_name}")
            return None
        rows = []
        for pe in summary.get("per_entry", []):
            rows.append({
                "backend": label,
                "id":      pe["id"].split("_", 3)[-1][:42],
                "WER":     round(pe.get("wer",  float("nan")), 4),
                "MER":     round(pe.get("mer",  float("nan")), 4),
                "ins":     pe.get("insertions",    0),
                "del":     pe.get("deletions",     0),
                "sub":     pe.get("substitutions", 0),
            })
        return pd.DataFrame(rows)

    df_w  = _build_summary_df(whisper_name,  "Whisper")
    df_wx = _build_summary_df(whisperx_name, "WhisperX")

    frames = [df for df in [df_w, df_wx] if df is not None]
    if not frames:
        st.info("Select at least one JSONL in the sidebar to see WER statistics.")
    else:
        combined = pd.concat(frames, ignore_index=True)

        # ── overall scorecard ─────────────────────────────────────────────────
        st.markdown("#### Overall (mean across all entries)")
        overall_cols = st.columns(len(frames) * 2)
        for k, (lbl, grp) in enumerate(combined.groupby("backend", sort=False)):
            overall_cols[k * 2    ].metric(f"{lbl} — WER", f"{grp['WER'].mean():.1%}")
            overall_cols[k * 2 + 1].metric(f"{lbl} — MER", f"{grp['MER'].mean():.1%}")

        st.divider()

        # ── per-entry pivot table ─────────────────────────────────────────────
        st.markdown("#### Per-entry WER")
        if len(frames) == 2:
            pivot = combined.pivot_table(
                index="id", columns="backend", values="WER"
            ).reset_index()
            pivot.columns.name = None
            if "Whisper" in pivot.columns and "WhisperX" in pivot.columns:
                pivot["Δ WER"] = pivot["WhisperX"] - pivot["Whisper"]
                pivot = pivot.sort_values("Δ WER")
                fmt = {"WER": "{:.1%}", "MER": "{:.1%}"}
                if "Whisper" in pivot.columns:
                    fmt["Whisper"]  = "{:.1%}"
                if "WhisperX" in pivot.columns:
                    fmt["WhisperX"] = "{:.1%}"
                fmt["Δ WER"] = "{:+.1%}"

                def _color_delta(v):
                    if pd.isna(v):
                        return ""
                    return "color: green" if v < 0 else ("color: #c0392b" if v > 0 else "")

                styler = pivot.style.format(fmt)
                _map_fn = getattr(styler, "map", None) or styler.applymap
                st.dataframe(
                    _map_fn(_color_delta, subset=["Δ WER"]),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.dataframe(pivot, use_container_width=True)
        else:
            tbl = combined[["backend", "id", "WER", "MER", "ins", "del", "sub"]]
            st.dataframe(
                tbl.style.format({"WER": "{:.1%}", "MER": "{:.1%}"}),
                use_container_width=True,
                height=400,
            )

        st.divider()

        # ── WER bar chart ─────────────────────────────────────────────────────
        st.markdown("#### Per-entry WER — Whisper vs WhisperX")
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            backends = list(combined["backend"].unique())
            all_ids  = list(combined["id"].unique())
            n_entries = len(all_ids)
            x     = np.arange(n_entries)
            width = 0.35
            colors = ["#4a90d9", "#27ae60"]

            fig, ax = plt.subplots(figsize=(max(12, n_entries * 0.35), 4))

            for k, (backend, grp) in enumerate(combined.groupby("backend", sort=False)):
                id_to_wer = dict(zip(grp["id"], grp["WER"]))
                wers = [id_to_wer.get(eid, float("nan")) for eid in all_ids]
                offset = (k - (len(backends) - 1) / 2) * width
                ax.bar(x + offset, wers, width,
                       label=backend, color=colors[k % len(colors)], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_ids, rotation=75, ha="right", fontsize=6)
            ax.set_ylabel("WER")
            ax.set_title("Per-entry WER — Whisper vs WhisperX (large-v3, CPU, oracle segments)")
            ax.legend()
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0%}")
            )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not render bar chart: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Recording
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Live recording ASR demo")
    st.caption(
        "Record a short clip in the browser, run Whisper on CPU, and compare "
        "against your own reference transcript."
    )

    # --- Language selector --------------------------------------------------
    _LANG_OPTS = ["Auto (detect)", "English", "Spanish", "Other"]
    live_language_label = st.selectbox(
        "Spoken language",
        _LANG_OPTS,
        index=0,
        key="live_language",
        help="Auto lets Whisper detect the language. Choose Other to enter any BCP-47 code.",
    )

    live_language_code: Optional[str] = None  # None → Whisper auto-detect
    if live_language_label == "English":
        live_language_code = "en"
    elif live_language_label == "Spanish":
        live_language_code = "es"
    elif live_language_label == "Other":
        _custom = st.text_input(
            "Language code (e.g. fr, zh, de, ja, ko, ar …)",
            max_chars=10,
            placeholder="fr",
            key="live_language_custom",
        ).strip().lower()
        live_language_code = _custom if _custom else None

    # --- Audio input --------------------------------------------------------
    live_audio = None
    live_audio_bytes = b""
    live_duration = None
    if hasattr(st, "audio_input"):
        live_audio = st.audio_input(
            "Microphone recording",
            sample_rate=16000,
            key="live_audio_input",
        )
        live_audio_bytes = live_audio.getvalue() if live_audio is not None else b""
        live_duration = _audio_duration_from_bytes(live_audio_bytes) if live_audio_bytes else None

        if live_audio_bytes:
            if live_duration is not None:
                st.caption(f"Recording duration: **{live_duration:.1f}s**")
        else:
            st.info(
                "Record audio, then stop. If no clip appears, check your browser's "
                "microphone permissions."
            )
    else:
        st.error("This Streamlit version does not support `st.audio_input`.")

    live_reference = st.text_area(
        "Reference transcript",
        height=140,
        placeholder="Enter the expected transcript here to compute WER.",
        key="live_reference",
    )

    if st.button("Run ASR", type="primary", key="live_run_asr"):
        if not live_audio_bytes:
            st.error("No recording found. Record a clip before running ASR.")
        elif len(live_audio_bytes) <= 44:
            st.error("Recorded audio appears empty. Please try recording again.")
        else:
            try:
                lang_label = (
                    live_language_code if live_language_code else "auto"
                )
                with st.spinner(f"Running Whisper base on CPU (language: {lang_label})…"):
                    live_hypothesis, live_detected = _transcribe_live_audio(
                        live_audio_bytes,
                        language=live_language_code,
                        model_size="base",
                    )
                # Use detected language for WER normalization; fall back to "en"
                wer_lang = live_detected if live_detected not in ("unknown", None) else "en"
                live_metrics = (
                    _live_entry_metrics(live_reference, live_hypothesis, wer_lang)
                    if live_reference.strip()
                    else None
                )
                # Detect language mismatch: user forced a language but Whisper saw something else
                lang_mismatch = (
                    live_language_code is not None
                    and live_detected not in ("unknown", None)
                    and live_detected != live_language_code
                )
                st.session_state["live_asr_result"] = {
                    "selected_language": live_language_code,
                    "detected_language": live_detected,
                    "lang_mismatch": lang_mismatch,
                    "reference": live_reference,
                    "hypothesis": live_hypothesis,
                    "duration": live_duration,
                    "metrics": live_metrics,
                }
            except Exception as exc:
                st.error(f"ASR failed: {exc}")

    live_result = st.session_state.get("live_asr_result")
    if live_result:
        st.divider()
        st.markdown("#### Result")
        st.caption("Model: `Whisper base` on `CPU`")

        # Language info row
        _det = live_result.get("detected_language", "unknown")
        _sel = live_result.get("selected_language")
        if _sel is None:
            st.caption(f"Language detected by Whisper: **{_det}**")
        else:
            st.caption(f"Language: selected **{_sel}** | Whisper reported **{_det}**")

        if live_result.get("lang_mismatch"):
            st.warning(
                f"Language mismatch: you selected **{_sel}** but Whisper detected **{_det}**. "
                "The transcript may be less accurate."
            )

        if live_result["metrics"] is not None:
            metrics = live_result["metrics"]
            m1, m2, m3 = st.columns(3)
            if metrics.get("wer") is not None:
                m1.metric("WER", f"{metrics['wer']:.1%}")
            if metrics.get("mer") is not None:
                m2.metric("MER", f"{metrics['mer']:.1%}")
            m3.metric("Ref Words", str(metrics.get("ref_words", 0)))
            st.caption(
                f"Substitutions: **{metrics.get('substitutions', 0)}** | "
                f"Deletions: **{metrics.get('deletions', 0)}** | "
                f"Insertions: **{metrics.get('insertions', 0)}**"
            )
            st.caption(
                "Note: WER is meaningful only if the reference language matches the spoken language."
            )
        else:
            st.info("Enter a reference transcript to compute WER for the recorded clip.")

        st.markdown("**Hypothesis**")
        st.write(live_result["hypothesis"] or "—")

        st.markdown("**Reference**")
        st.write(live_result["reference"] or "—")
