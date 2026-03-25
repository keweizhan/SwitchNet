"""
subtitles.py — SRT subtitle export for SwitchNet bilingual pipeline.

Converts per-segment transcription output (from _transcribe_bilingual_oracle)
into SubtitleCue objects and writes standard .srt files.

Timing policy:
  1. Use segment.start / segment.end if end > start (real timing from manifest).
  2. Otherwise infer segment duration from the actual audio file via librosa.
  3. Build cue start times cumulatively in segment order, adding pause_s gaps.
  4. If neither timing source is available, fall back to 0-duration placeholder
     cues and emit a warning — do NOT crash.

Usage:
    from src.asr.subtitles import build_segment_level_cues, write_srt
    cues = build_segment_level_cues(segment_outputs, entry.segments)
    write_srt(cues, Path("results/subtitles/my_entry.srt"))
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubtitleCue:
    start: float          # seconds from audio start
    end: float            # seconds from audio start
    text: str             # hypothesis text for this cue
    source_language: Optional[str] = None   # "en" or "es" — where the audio came from


# ---------------------------------------------------------------------------
# SRT formatting helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert a float second offset to SRT timestamp format: HH:MM:SS,mmm

    >>> seconds_to_srt_timestamp(3661.5)
    '01:01:01,500'
    >>> seconds_to_srt_timestamp(0.0)
    '00:00:00,000'
    """
    seconds = max(0.0, seconds)
    millis = round(seconds * 1000)
    ms = millis % 1000
    total_s = millis // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_subtitle_text(text: str, add_period: bool = True) -> str:
    """
    Apply light, deterministic cleanup to a subtitle cue's text.

    Rules (applied in order):
      1. Strip leading/trailing whitespace.
      2. Collapse repeated internal whitespace to a single space.
      3. Capitalize the first alphabetical character in the string.
      4. If add_period=True and the text has no terminal punctuation (.!?),
         append a period.

    Does NOT reword, split, or merge cues. Safe to call on already-clean text.
    """
    if not text:
        return text

    # 1. Strip
    text = text.strip()
    if not text:
        return text

    # 2. Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)

    # 3. Capitalize the first letter (leaves numbers/punctuation at the
    #    start untouched, e.g. "...she said" → "...She said")
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break

    # 4. Add a period when no sentence-ending punctuation is present
    if add_period and text[-1] not in ".!?":
        text += "."

    return text


def write_srt(cues: List[SubtitleCue], output_path: str | Path, add_period: bool = True) -> None:
    """
    Write a list of SubtitleCue objects to a .srt file.

    Cues with empty text are skipped. Cues where end <= start are emitted with
    end = start + 0.001 so the SRT remains valid (most players handle this).
    clean_subtitle_text() is applied to every cue before writing.

    Args:
        add_period: Passed through to clean_subtitle_text — append a period to
            cues that lack terminal punctuation (default True).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    cue_index = 1
    for cue in cues:
        cleaned = clean_subtitle_text(cue.text, add_period=add_period)
        if not cleaned:
            continue
        cue_end = cue.end if cue.end > cue.start else cue.start + 0.001
        lines.append(str(cue_index))
        lines.append(f"{seconds_to_srt_timestamp(cue.start)} --> {seconds_to_srt_timestamp(cue_end)}")
        lines.append(cleaned)
        lines.append("")   # blank line between cues
        cue_index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Timing inference
# ---------------------------------------------------------------------------

def _get_audio_duration(audio_path: str) -> Optional[float]:
    """
    Return the duration of an audio file in seconds using librosa.
    Returns None if librosa is not installed or the file cannot be read.
    """
    try:
        import librosa
        return librosa.get_duration(path=audio_path)
    except Exception as exc:
        warnings.warn(f"Could not read duration from {audio_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_segment_level_cues(
    segment_outputs: List[dict],
    segments,                         # List[Segment] from ManifestEntry
    infer_timing_from_audio: bool = True,
) -> List[SubtitleCue]:
    """
    Build SubtitleCue objects from oracle-segment transcription output.

    Args:
        segment_outputs: List of dicts from _transcribe_bilingual_oracle, each with:
            {"position": int, "language": str, "reference": str, "hypothesis": str}
        segments: List[Segment] from ManifestEntry.segments — provides timing and
            audio_path for duration inference.
        infer_timing_from_audio: If True (default), fall back to reading the actual
            audio file duration when segment.start/end are both 0.0.

    Returns:
        List[SubtitleCue] in segment order, with cumulative timestamps.
    """
    if not segment_outputs:
        return []

    # Index segments by position for O(1) lookup.
    # segment_outputs and segments are parallel arrays (same order), but
    # we use position as the authoritative key in case they ever diverge.
    seg_by_pos = {i: seg for i, seg in enumerate(segments)}

    cues: List[SubtitleCue] = []
    cursor = 0.0  # running start time (seconds)

    for out in segment_outputs:
        pos = out["position"]
        seg = seg_by_pos.get(pos)

        # --- Determine this cue's duration ---
        duration: Optional[float] = None

        if seg is not None:
            if seg.end > seg.start:
                # Manifest has real timing — use it directly.
                # NOTE: we still use cursor for the cue start so that
                # pause_s gaps accumulate correctly even if the manifest
                # timestamps are absolute rather than zero-based.
                duration = seg.end - seg.start
            elif infer_timing_from_audio and seg.audio_path:
                # Manifest timing is missing (start==end==0.0).
                # Infer from the actual audio file.
                duration = _get_audio_duration(seg.audio_path)

        if duration is None or duration <= 0:
            warnings.warn(
                f"Segment position={pos} has no valid timing and audio inference "
                f"failed or was disabled. Cue will have zero duration."
            )
            duration = 0.0

        cue_start = cursor
        cue_end = cursor + duration

        cues.append(SubtitleCue(
            start=round(cue_start, 3),
            end=round(cue_end, 3),
            text=out["hypothesis"],
            source_language=out.get("language"),
        ))

        cursor = cue_end

        # Advance cursor by pause gap after this segment (before the next).
        if seg is not None and seg.pause_s > 0.0:
            cursor += seg.pause_s

    return cues
