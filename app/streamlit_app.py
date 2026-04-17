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
  Reference   — ES audio + text | switch banner | EN audio + text
  Comparison  — Reference / Whisper / WhisperX cues side-by-side
  WER Summary — per-entry and overall WER/MER table + bar chart
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ── project root on path ──────────────────────────────────────────────────────
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
DEFAULT_ENTRY_HINT = "bilingual_es-en_0022"   # partial match

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwitchNet — Bilingual ASR Demo",
    page_icon="🎙️",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_manifest_cached(path: str) -> List[ManifestEntry]:
    return load_manifest(path)


@st.cache_data(show_spinner=False)
def _audio_wav_bytes(audio_path: str) -> Optional[bytes]:
    """Decode any audio file (opus/flac/wav) → 16-kHz mono WAV bytes."""
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
    """Load eval JSONL into {entry_id: record} dict."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Cue builders (no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _cues_from_record(record: dict, entry: ManifestEntry) -> List[SubtitleCue]:
    """Build SubtitleCues from an eval-JSONL record without writing any file."""
    seg_outputs = record.get("segment_outputs")
    if not seg_outputs:
        seg_outputs = [{
            "position":   0,
            "language":   entry.language,
            "reference":  entry.transcript,
            "hypothesis": record.get("hypothesis", ""),
        }]
    return build_segment_level_cues(
        seg_outputs,
        entry.segments,
        infer_timing_from_audio=True,
    )


def _reference_cues(entry: ManifestEntry) -> List[SubtitleCue]:
    """Build reference cues from manifest ground-truth segments."""
    cues: List[SubtitleCue] = []
    offset = 0.0
    for seg in entry.segments:
        try:
            import librosa
            duration = librosa.get_duration(path=seg.audio_path)
        except Exception:
            duration = 5.0
        text = seg.transcript or ""
        cues.append(SubtitleCue(
            start=offset,
            end=offset + duration,
            text=text,
            source_language=seg.language,
            timing_source="manifest",
        ))
        offset += duration
    return cues


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
    """Render cues as an HTML table with a highlighted switch row."""
    rows = []
    switch_inserted = False
    for c in cues:
        if switch_time is not None and not switch_inserted and c.start >= switch_time:
            rows.append(
                '<tr style="background:#fff3cd;font-weight:bold;">'
                '<td colspan="3" style="text-align:center;padding:4px 8px;">'
                "⟵ ES→EN SWITCH ⟶"
                "</td></tr>"
            )
            switch_inserted = True
        badge = _lang_badge(c.source_language)
        text  = c.text.replace("\n", "<br>")
        rows.append(
            f'<tr><td style="white-space:nowrap;color:#666;font-size:0.8em;'
            f'padding:2px 6px;">{_fmt_time(c.start)}</td>'
            f'<td style="padding:2px 4px;">{badge}</td>'
            f'<td style="padding:2px 8px;">{text}</td></tr>'
        )
    table = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em;">'
        + "".join(rows)
        + "</table>"
    )
    return table


def _switch_time(entry: ManifestEntry) -> Optional[float]:
    """Return the offset (seconds) where the first ES→EN switch occurs."""
    offset = 0.0
    prev_lang = None
    for seg in entry.segments:
        try:
            import librosa
            duration = librosa.get_duration(path=seg.audio_path)
        except Exception:
            duration = 5.0
        if prev_lang == "es" and seg.language == "en":
            return offset
        offset += duration
        prev_lang = seg.language
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _available_manifests() -> List[str]:
    return sorted(
        p.name for p in MANIFESTS_DIR.glob("bilingual_es-en_*.jsonl")
    )


def _available_jsonls(prefix: str = "") -> List[str]:
    names = [p.name for p in RESULTS_DIR.glob("*.jsonl") if prefix in p.name]
    return sorted(names)


with st.sidebar:
    st.title("🎙️ SwitchNet Demo")
    st.caption("Bilingual ES→EN ASR comparison")

    # ── manifest ──────────────────────────────────────────────────────────────
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

    # ── entry picker ──────────────────────────────────────────────────────────
    entry_ids = [e.id for e in bilingual_entries]
    default_entry_idx = next(
        (i for i, eid in enumerate(entry_ids) if DEFAULT_ENTRY_HINT in eid), 0
    )
    selected_id = st.selectbox("Entry", entry_ids, index=default_entry_idx,
                               format_func=lambda x: x.split("_", 3)[-1][:50])
    entry: ManifestEntry = next(e for e in bilingual_entries if e.id == selected_id)

    # ── Whisper JSONL ─────────────────────────────────────────────────────────
    whisper_choices = ["(none)"] + _available_jsonls("bilingual_es-en")
    # exclude wx_ files from whisper picker
    whisper_choices = [c for c in whisper_choices if not c.startswith("wx_")]
    w_default = (
        whisper_choices.index(DEFAULT_WHISPER)
        if DEFAULT_WHISPER in whisper_choices else 0
    )
    whisper_name = st.selectbox("Whisper JSONL", whisper_choices, index=w_default)
    whisper_path = (
        str(RESULTS_DIR / whisper_name) if whisper_name != "(none)" else None
    )

    # ── WhisperX JSONL ────────────────────────────────────────────────────────
    whisperx_choices = ["(none)"] + _available_jsonls("wx_bilingual_es-en")
    wx_default = (
        whisperx_choices.index(DEFAULT_WHISPERX)
        if DEFAULT_WHISPERX in whisperx_choices else 0
    )
    whisperx_name = st.selectbox("WhisperX JSONL", whisperx_choices, index=wx_default)
    whisperx_path = (
        str(RESULTS_DIR / whisperx_name) if whisperx_name != "(none)" else None
    )

    st.divider()
    st.caption(f"Streamlit {st.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# Load JSONL records
# ─────────────────────────────────────────────────────────────────────────────

whisper_records:  Dict[str, dict] = _load_jsonl(whisper_path)  if whisper_path  else {}
whisperx_records: Dict[str, dict] = _load_jsonl(whisperx_path) if whisperx_path else {}

whisper_rec  = whisper_records.get(entry.id)
whisperx_rec = whisperx_records.get(entry.id)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────

st.title("SwitchNet — Bilingual ASR Demo")
st.markdown(
    f"**Entry:** `{entry.id}`  \n"
    f"**Segments:** {len(entry.segments)} "
    f"({' → '.join(s.language.upper() for s in entry.segments)})"
)

sw_time = _switch_time(entry)
if sw_time is not None:
    st.info(f"ES→EN switch at **{_fmt_time(sw_time)}**")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_ref, tab_cmp, tab_wer = st.tabs(["📄 Reference", "⚖️ Comparison", "📊 WER Summary"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Reference
# ══════════════════════════════════════════════════════════════════════════════

with tab_ref:
    st.subheader("Ground-truth reference (per segment)")

    for i, seg in enumerate(entry.segments):
        lang_label = f"🇪🇸 Spanish (segment {i})" if seg.language == "es" else f"🇺🇸 English (segment {i})"
        with st.container():
            st.markdown(f"**{lang_label}**")
            wav = _audio_wav_bytes(str(seg.audio_path))
            if wav:
                st.audio(wav, format="audio/wav")
            else:
                st.caption(f"_(audio unavailable: {seg.audio_path})_")
            st.markdown(
                f'<div style="background:#f0f4ff;border-left:4px solid #4a90d9;'
                f'padding:8px 12px;border-radius:4px;font-size:1.05em;">'
                f'{seg.transcript or "(no transcript)"}'
                f"</div>",
                unsafe_allow_html=True,
            )

        if i < len(entry.segments) - 1:
            st.markdown(
                '<div style="text-align:center;margin:12px 0;font-weight:bold;'
                'color:#c0392b;">── ES → EN SWITCH ──</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_cmp:
    st.subheader("Cue-by-cue comparison")

    # Build cue sets
    ref_cues = _reference_cues(entry)

    col_ref, col_w, col_wx = st.columns(3)

    with col_ref:
        st.markdown("#### 📄 Reference")
        st.markdown(_cue_rows_html(ref_cues, sw_time), unsafe_allow_html=True)

    with col_w:
        st.markdown("#### 🔵 Whisper")
        if whisper_rec:
            with st.spinner("Building Whisper cues…"):
                w_cues = _cues_from_record(whisper_rec, entry)
            st.markdown(_cue_rows_html(w_cues, sw_time), unsafe_allow_html=True)
            # WER badge
            wer = whisper_rec.get("wer")
            if wer is not None:
                st.caption(f"WER = {wer:.1%}")
        else:
            st.caption("_(no Whisper JSONL selected or entry not found)_")

    with col_wx:
        st.markdown("#### 🟢 WhisperX")
        if whisperx_rec:
            with st.spinner("Building WhisperX cues…"):
                wx_cues = _cues_from_record(whisperx_rec, entry)
            st.markdown(_cue_rows_html(wx_cues, sw_time), unsafe_allow_html=True)
            wer = whisperx_rec.get("wer")
            if wer is not None:
                st.caption(f"WER = {wer:.1%}")
        else:
            st.caption("_(no WhisperX JSONL selected or entry not found)_")

    # ── per-segment text comparison ───────────────────────────────────────────
    if whisper_rec and whisperx_rec:
        st.divider()
        st.subheader("Per-segment hypothesis text")
        w_segs  = whisper_rec.get("segment_outputs", [])
        wx_segs = whisperx_rec.get("segment_outputs", [])
        n = max(len(w_segs), len(wx_segs), len(entry.segments))
        for i in range(n):
            ref_seg = entry.segments[i] if i < len(entry.segments) else None
            w_seg   = w_segs[i]  if i < len(w_segs)  else None
            wx_seg  = wx_segs[i] if i < len(wx_segs) else None
            lang    = ref_seg.language.upper() if ref_seg else "?"
            badge   = "🇪🇸" if lang == "ES" else "🇺🇸"

            with st.expander(f"{badge} Segment {i} [{lang}]", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Reference**")
                    st.write(ref_seg.transcript if ref_seg else "—")
                with c2:
                    st.markdown("**Whisper**")
                    st.write(w_seg["hypothesis"] if w_seg else "—")
                with c3:
                    st.markdown("**WhisperX**")
                    st.write(wx_seg["hypothesis"] if wx_seg else "—")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WER Summary
# ══════════════════════════════════════════════════════════════════════════════

with tab_wer:
    st.subheader("WER / MER summary")

    import pandas as pd

    def _summary_df(jsonl_name: Optional[str], label: str) -> Optional[pd.DataFrame]:
        if not jsonl_name or jsonl_name == "(none)":
            return None
        stem = Path(jsonl_name).stem
        summary_path = RESULTS_DIR / f"{stem}_summary.json"
        summary = _load_summary(str(summary_path))
        if not summary:
            return None
        rows = []
        for pe in summary.get("per_entry", []):
            rows.append({
                "backend": label,
                "id":      pe["id"].split("_", 3)[-1][:40],
                "WER":     round(pe.get("wer", float("nan")), 4),
                "MER":     round(pe.get("mer", float("nan")), 4),
                "ins":     pe.get("insertions", 0),
                "del":     pe.get("deletions", 0),
                "sub":     pe.get("substitutions", 0),
            })
        return pd.DataFrame(rows)

    df_w  = _summary_df(whisper_name,  "Whisper")
    df_wx = _summary_df(whisperx_name, "WhisperX")

    frames = [df for df in [df_w, df_wx] if df is not None]
    if not frames:
        st.info("Select at least one JSONL to see WER statistics.")
    else:
        combined = pd.concat(frames, ignore_index=True)

        # ── overall scorecard ─────────────────────────────────────────────────
        st.markdown("#### Overall")
        cards = st.columns(len(frames))
        for col, (lbl, grp) in zip(cards, combined.groupby("backend", sort=False)):
            overall_wer = grp["WER"].mean()
            overall_mer = grp["MER"].mean()
            col.metric(f"{lbl} — WER", f"{overall_wer:.1%}")
            col.metric(f"{lbl} — MER", f"{overall_mer:.1%}")

        # ── per-entry table ───────────────────────────────────────────────────
        st.markdown("#### Per-entry WER")

        if len(frames) == 2:
            # Pivot for easy side-by-side
            pivot = combined.pivot_table(
                index="id", columns="backend", values="WER"
            ).reset_index()
            if "Whisper" in pivot and "WhisperX" in pivot:
                pivot["Δ WER"] = pivot["WhisperX"] - pivot["Whisper"]
                pivot = pivot.sort_values("Δ WER")
                def _color_delta(v):
                    if pd.isna(v):
                        return ""
                    return "color: green" if v < 0 else ("color: red" if v > 0 else "")
                styler = pivot.style.format(
                    {"Whisper": "{:.1%}", "WhisperX": "{:.1%}", "Δ WER": "{:+.1%}"}
                )
                _map_fn = getattr(styler, "map", None) or styler.applymap
                st.dataframe(_map_fn(_color_delta, subset=["Δ WER"]), use_container_width=True)
            else:
                st.dataframe(pivot, use_container_width=True)
        else:
            tbl = combined[["backend", "id", "WER", "MER", "ins", "del", "sub"]]
            st.dataframe(tbl.style.format({"WER": "{:.1%}", "MER": "{:.1%}"}),
                         use_container_width=True)

        # ── WER bar chart ─────────────────────────────────────────────────────
        st.markdown("#### Per-entry WER comparison")
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            backends = combined["backend"].unique()
            all_ids  = combined["id"].unique()
            x = np.arange(len(all_ids))
            width = 0.35 / max(len(backends) - 1, 1) * 2  # dynamic width

            fig, ax = plt.subplots(figsize=(14, 4))
            colors = ["#4a90d9", "#27ae60"]
            for k, (backend, grp) in enumerate(combined.groupby("backend", sort=False)):
                wers = [
                    grp[grp["id"] == eid]["WER"].values[0]
                    if eid in grp["id"].values else float("nan")
                    for eid in all_ids
                ]
                offset = (k - (len(backends) - 1) / 2) * width
                ax.bar(x + offset, wers, width, label=backend, color=colors[k % len(colors)], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_ids, rotation=75, ha="right", fontsize=6)
            ax.set_ylabel("WER")
            ax.set_title("Per-entry WER — Whisper vs WhisperX")
            ax.legend()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not render bar chart: {exc}")
