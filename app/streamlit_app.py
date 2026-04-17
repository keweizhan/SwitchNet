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


# ─────────────────────────────────────────────────────────────────────────────
# Cached I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_manifest_cached(path: str) -> List[ManifestEntry]:
    return load_manifest(path)


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
        return float(librosa.get_duration(path=audio_path))
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


def _cue_rows_html(cues: List[SubtitleCue], switch_time: Optional[float]) -> str:
    """Render cues as an HTML table; inserts a highlighted switch separator row."""
    rows: List[str] = []
    switch_inserted = False
    for c in cues:
        if switch_time is not None and not switch_inserted and c.start >= switch_time:
            rows.append(
                '<tr style="background:#fff3cd;font-weight:bold;">'
                '<td colspan="3" style="text-align:center;padding:6px 8px;'
                'letter-spacing:0.05em;">⟵ ES → EN SWITCH ⟶</td></tr>'
            )
            switch_inserted = True
        badge = _lang_badge(c.source_language)
        text  = c.text.replace("\n", "<br>")
        rows.append(
            f'<tr><td style="white-space:nowrap;color:#888;font-size:0.78em;'
            f'padding:3px 6px;vertical-align:top;">{_fmt_time(c.start)}</td>'
            f'<td style="padding:3px 4px;vertical-align:top;">{badge}</td>'
            f'<td style="padding:3px 10px;">{text}</td></tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.88em;">'
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

tab_ref, tab_cmp, tab_wer = st.tabs(["📄 Reference", "⚖️ Comparison", "📊 WER Summary"])


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

            wav = _audio_wav_bytes(str(seg.audio_path))
            if wav:
                st.audio(wav, format="audio/wav")
            else:
                st.caption(f"_(audio unavailable: `{seg.audio_path}`)_")

            border_color = "#e74c3c" if seg.language == "es" else "#27ae60"
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:4px solid {border_color};'
                f'padding:10px 14px;border-radius:4px;font-size:1.05em;margin-top:4px;">'
                f'{seg.transcript or "(no transcript)"}'
                f"</div>",
                unsafe_allow_html=True,
            )

        if i < len(entry.segments) - 1:
            st.markdown(
                '<div style="text-align:center;margin:16px 0 12px;font-weight:bold;'
                'font-size:1.1em;color:#c0392b;letter-spacing:0.05em;">'
                "── ES → EN SWITCH ──"
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
        "The yellow row marks the ES→EN switch boundary."
    )

    ref_cues = _reference_cues_cached(entry.id, manifest_path)

    col_ref, col_w, col_wx = st.columns(3)

    with col_ref:
        st.markdown("#### 📄 Reference")
        st.markdown(_cue_rows_html(ref_cues, sw_time), unsafe_allow_html=True)

    with col_w:
        st.markdown("#### 🔵 Whisper")
        if whisper_path:
            w_cues = _model_cues_cached(entry.id, manifest_path, whisper_path)
            if w_cues:
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
                st.markdown(_cue_rows_html(wx_cues, sw_time), unsafe_allow_html=True)
                if wer_whisperx is not None:
                    st.caption(f"WER = **{wer_whisperx:.1%}**")
            else:
                st.caption("_(entry not found in selected JSONL)_")
        else:
            st.caption("_(no WhisperX JSONL selected)_")

    # ── per-segment text expanders ────────────────────────────────────────────
    w_records  = _load_jsonl(whisper_path)  if whisper_path  else {}
    wx_records = _load_jsonl(whisperx_path) if whisperx_path else {}
    w_rec  = w_records.get(entry.id)
    wx_rec = wx_records.get(entry.id)

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
