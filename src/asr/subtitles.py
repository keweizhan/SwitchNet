"""
subtitles.py -- SRT subtitle export for SwitchNet bilingual pipeline.

Converts per-segment transcription output (from _transcribe_bilingual_oracle)
into SubtitleCue objects and writes standard .srt files.

Timing modes
------------
fallback (default / V1.0 behaviour):
  One SubtitleCue per bilingual manifest segment.
  Timing comes from segment.start/end if real, otherwise inferred from audio
  file duration via librosa.  Cue start times are built cumulatively.

whisper_segments (V1.1, opt-in via use_whisper_segments=True):
  Multiple SubtitleCues per bilingual manifest segment, one per Whisper-internal
  segment.  Requires _transcribe_bilingual_oracle to be called with
  return_whisper_segments=True so that segment_outputs["whisper_segments"] is
  populated.  Falls back to the single-cue path if whisper_segments is absent
  or empty for a given bilingual segment.

rule_split (V1.2, opt-in via apply_rule_based_splitting):
  Long cues are broken into shorter sub-cues using deterministic text rules,
  even when Whisper returns only one internal segment.  Text is split at
  sentence-ending punctuation first, then clause marks, then word midpoint.
  Sub-cue durations are distributed proportionally by word count.

Subtitle modes (V1.2)
---------------------
english (default):
  Every cue shows the English text only.  Spanish segments are expected to have
  already been translated to English by the Whisper translate task.

bilingual:
  English segments: English text only (no duplication).
  Spanish segments: source-language reference text on line 1, English
  translation on line 2.  Falls back to English-only if source_text is absent.

Usage:
    from src.asr.subtitles import build_segment_level_cues, write_srt
    cues = build_segment_level_cues(segment_outputs, entry.segments)
    write_srt(cues, Path("results/subtitles/my_entry.srt"))
"""

from __future__ import annotations

import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubtitleCue:
    start: float                           # seconds from audio start
    end: float                             # seconds from audio start
    text: str                              # hypothesis text (English or translated)
    source_language: Optional[str] = None  # "en" or "es" -- where the audio came from
    timing_source: str = "fallback"        # "whisper_segments", "fallback", or "*_split"
    source_text: Optional[str] = None      # original-language reference (V1.2 bilingual mode)
    is_intermediate_split: bool = False    # True for non-final chunks of a rule-split cue


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

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

    text = text.strip()
    if not text:
        return text

    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)

    # Capitalize first letter
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break

    # Add terminal period if missing
    if add_period and text[-1] not in ".!?":
        text += "."

    return text


def wrap_subtitle_text(text: str, max_chars_per_line: int = 42) -> str:
    """
    Wrap subtitle text so no line exceeds max_chars_per_line characters.

    Returns the original string unchanged if it already fits on one line.
    Does not split mid-word. Each Whisper cue is typically one sentence so
    the result is usually one or two lines.

    Args:
        text: already-cleaned cue text (may contain newlines from prior passes).
        max_chars_per_line: target line width (default 42, a common SRT limit).
    """
    if not text or len(text) <= max_chars_per_line:
        return text
    lines = textwrap.wrap(
        text,
        width=max_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


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


def write_srt(
    cues: List[SubtitleCue],
    output_path: str | Path,
    add_period: bool = True,
    max_chars_per_line: int = 42,
    subtitle_mode: str = "english",
) -> None:
    """
    Write a list of SubtitleCue objects to a .srt file.

    For each cue:
      - Text formatting is applied (clean, wrap).
      - In bilingual mode: Spanish cues show source_text on line 1 and the
        English translation on line 2.  English cues show one line only.
      - In english mode (default): all cues show the English text only.

    Cues with empty text after formatting are skipped.
    Cues where end <= start are emitted with end = start + 0.001 so the SRT
    remains valid.

    Args:
        add_period: append a period when no terminal punctuation present.
        max_chars_per_line: wrap lines longer than this many characters.
            Set to a large number (e.g. 999) to disable wrapping.
        subtitle_mode: "english" (default) or "bilingual".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    cue_index = 1
    for cue in cues:
        display_text = _format_cue_text(
            cue,
            add_period=add_period,
            max_chars_per_line=max_chars_per_line,
            subtitle_mode=subtitle_mode,
        )
        if not display_text.strip():
            continue

        cue_end = cue.end if cue.end > cue.start else cue.start + 0.001
        lines.append(str(cue_index))
        lines.append(
            f"{seconds_to_srt_timestamp(cue.start)} --> {seconds_to_srt_timestamp(cue_end)}"
        )
        lines.append(display_text)
        lines.append("")   # blank line between cues
        cue_index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _format_cue_text(
    cue: SubtitleCue,
    add_period: bool,
    max_chars_per_line: int,
    subtitle_mode: str,
) -> str:
    """
    Return the display text for one cue, applying cleanup, wrapping, and
    subtitle_mode logic.

    bilingual mode rules:
      - source_language == "es" and source_text is available:
          line 1 = cleaned source text (Spanish)
          line 2 = cleaned hypothesis (English translation)
      - otherwise (English source, or no source_text): English text only.
    """
    # Intermediate split chunks must not receive an added period -- only the
    # final chunk of a split cue may have one appended.
    effective_add_period = add_period and not cue.is_intermediate_split

    if (
        subtitle_mode == "bilingual"
        and cue.source_language == "es"
        and cue.source_text
        and cue.source_text.strip()
    ):
        src_line = clean_subtitle_text(cue.source_text, add_period=effective_add_period)
        hyp_line = clean_subtitle_text(cue.text, add_period=effective_add_period)
        src_wrapped = wrap_subtitle_text(src_line, max_chars_per_line)
        hyp_wrapped = wrap_subtitle_text(hyp_line, max_chars_per_line)
        return f"{src_wrapped}\n{hyp_wrapped}"

    cleaned = clean_subtitle_text(cue.text, add_period=effective_add_period)
    return wrap_subtitle_text(cleaned, max_chars_per_line)


# ---------------------------------------------------------------------------
# Rule-based cue splitting (V1.2)
# ---------------------------------------------------------------------------

# Short function / closed-class words that form awkward cue-final boundaries.
# If the last word of a proposed left-chunk matches one of these we scan for a
# better split point rather than cutting there.
_SPLIT_AVOID_FINAL = frozenset({
    "the", "a", "an", "to", "of", "and", "or", "in", "on",
    "at", "by", "for", "with", "as", "is", "it", "its",
})


def _find_midpoint_split(words: List[str]) -> int:
    """
    Return a split index *i* so that words[:i] and words[i:] form a natural
    boundary near the middle of the word list.

    Starting at the true midpoint, scans outward (alternating right/left) until
    it finds an index where the last word of the left chunk is not a short
    function word.  Falls back to the true midpoint if no better position
    exists within half the word count.
    """
    n = len(words)
    mid = n // 2

    for delta in range(n // 2 + 1):
        for i in ([mid] if delta == 0 else [mid + delta, mid - delta]):
            if 1 <= i <= n - 1:
                last = words[i - 1].lower().rstrip(".,!?;:")
                if last not in _SPLIT_AVOID_FINAL:
                    return i

    return mid  # absolute fallback


def split_text_into_chunks(
    text: str,
    max_chars: Optional[int] = None,
    max_words: Optional[int] = None,
) -> List[str]:
    """
    Split text into shorter chunks at natural boundaries.

    Split strategy (tried in order, stops at first that produces multiple pieces
    within the requested limits):
      1. Sentence-ending punctuation (. ! ?) followed by whitespace.
      2. Clause marks (, ; :) followed by whitespace.
      3. Word-count midpoint (last resort -- no preferred boundary).

    Splitting is recursive: if any piece is still over the limit after a first
    pass, the same rules are applied to that piece.

    Returns:
        List of non-empty strings.  Returns [text] unchanged if no limit is
        exceeded, if the text has two words or fewer, or if no split is possible.

    Notes:
      - Does not reword, merge, or reorder text.
      - No external dependencies beyond re (stdlib).
    """
    if max_chars is None and max_words is None:
        return [text]

    words = text.split()
    if len(words) <= 2:
        return [text]

    def _within_limits(s: str) -> bool:
        if max_chars is not None and len(s) > max_chars:
            return False
        if max_words is not None and len(s.split()) > max_words:
            return False
        return True

    if _within_limits(text):
        return [text]

    # Try split patterns in priority order
    for pattern in (r"(?<=[.!?])\s+", r"(?<=[,;:])\s+"):
        pieces = [p.strip() for p in re.split(pattern, text) if p.strip()]
        if len(pieces) >= 2:
            # Recursively check/split each piece
            result: List[str] = []
            for piece in pieces:
                result.extend(split_text_into_chunks(piece, max_chars, max_words))
            return result

    # Last resort: split near word-count midpoint, avoiding function-word boundaries
    mid = _find_midpoint_split(words)
    c1 = " ".join(words[:mid])
    c2 = " ".join(words[mid:])
    # Each half may still need splitting
    result = []
    result.extend(split_text_into_chunks(c1, max_chars, max_words))
    result.extend(split_text_into_chunks(c2, max_chars, max_words))
    return result


def split_long_cue_by_rules(
    cue: SubtitleCue,
    max_chars_per_cue: Optional[int] = None,
    max_words_per_cue: Optional[int] = None,
) -> List[SubtitleCue]:
    """
    Split a single SubtitleCue if its text exceeds the given limits.

    Returns [cue] unchanged if:
      - No limit is exceeded.
      - The text cannot be split (two words or fewer, no boundary found).

    Timing assignment after splitting:
      Sub-cue durations are proportional to each chunk's word count relative
      to the total.  This is a simple heuristic -- speech rate is assumed
      roughly constant within a cue.  Timestamps are placed contiguously
      with no gap between sub-cues.  The last sub-cue always ends at
      cue.end so floating-point drift does not accumulate.

    source_text is NOT propagated to sub-cues because it refers to the full
    source-language segment and cannot be cleanly partitioned.  In bilingual
    subtitle mode the sub-cues will fall back to English-only display.

    Args:
        max_chars_per_cue: split if len(cue.text) exceeds this.
        max_words_per_cue: split if word count exceeds this.
    """
    if max_chars_per_cue is None and max_words_per_cue is None:
        return [cue]

    chunks = split_text_into_chunks(
        cue.text,
        max_chars=max_chars_per_cue,
        max_words=max_words_per_cue,
    )
    if len(chunks) <= 1:
        return [cue]

    duration = cue.end - cue.start
    word_counts = [max(1, len(c.split())) for c in chunks]
    total_words = sum(word_counts)

    result: List[SubtitleCue] = []
    t = cue.start
    last_idx = len(chunks) - 1
    for i, (chunk, wc) in enumerate(zip(chunks, word_counts)):
        fraction = wc / total_words
        chunk_end = round(t + duration * fraction, 3)
        result.append(SubtitleCue(
            start=round(t, 3),
            end=chunk_end,
            text=chunk,
            source_language=cue.source_language,
            timing_source=cue.timing_source + "_split",
            source_text=None,               # see docstring
            is_intermediate_split=(i < last_idx),
        ))
        t = chunk_end

    # Correct any floating-point drift on the final cue
    result[-1].end = cue.end
    return result


def apply_rule_based_splitting(
    cues: List[SubtitleCue],
    max_chars_per_cue: Optional[int] = None,
    max_words_per_cue: Optional[int] = None,
) -> List[SubtitleCue]:
    """
    Apply rule-based splitting to every cue in a list.

    Cues that are already within the limits pass through unchanged.
    Cues that are split are replaced by their sub-cues in the same position.
    Overall cue order and timestamp monotonicity are preserved.

    This is a no-op (returns the original list) if both limits are None.

    Args:
        max_chars_per_cue: character count limit per cue text.
        max_words_per_cue: word count limit per cue text.
    """
    if max_chars_per_cue is None and max_words_per_cue is None:
        return cues

    result: List[SubtitleCue] = []
    for cue in cues:
        result.extend(
            split_long_cue_by_rules(cue, max_chars_per_cue, max_words_per_cue)
        )
    return result


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
# Whisper-segment cue builder (V1.1)
# ---------------------------------------------------------------------------

def build_whisper_segment_cues(
    whisper_segs: list,
    cursor_offset: float,
    source_language: Optional[str],
    max_cue_seconds: Optional[float] = None,
) -> List[SubtitleCue]:
    """
    Convert Whisper's internal segment list into SubtitleCue objects placed on
    the global bilingual timeline.

    Args:
        whisper_segs: result["segments"] from model.transcribe().
            Each dict has at minimum {"start": float, "end": float, "text": str}.
            Timestamps are 0-based relative to the audio fed to Whisper (i.e.
            one bilingual manifest segment's audio file or time-slice).
        cursor_offset: seconds added to every Whisper timestamp to convert from
            segment-local time to global timeline position.
        source_language: propagated to SubtitleCue.source_language ("en"/"es").
        max_cue_seconds: if set, clamp each cue's end so its duration does not
            exceed this value.  The text is not split -- only the end timestamp
            is moved.  Useful for preventing runaway long cues when Whisper
            groups multiple sentences into one internal segment.

    Returns:
        List[SubtitleCue] with timing_source="whisper_segments".
        Returns [] if whisper_segs is empty (caller should fall back to
        single-cue mode in that case).

    Note: source_text is not set here because Whisper segments do not carry
    per-segment reference text from the manifest.
    """
    cues: List[SubtitleCue] = []
    for ws in whisper_segs:
        text = ws.get("text", "").strip()
        if not text:
            continue
        start = round(cursor_offset + float(ws["start"]), 3)
        end   = round(cursor_offset + float(ws["end"]),   3)
        if max_cue_seconds is not None and (end - start) > max_cue_seconds:
            end = round(start + max_cue_seconds, 3)
        cues.append(SubtitleCue(
            start=start,
            end=end,
            text=text,
            source_language=source_language,
            timing_source="whisper_segments",
        ))
    return cues


# ---------------------------------------------------------------------------
# Main cue builder
# ---------------------------------------------------------------------------

def build_segment_level_cues(
    segment_outputs: List[dict],
    segments,                           # List[Segment] from ManifestEntry
    infer_timing_from_audio: bool = True,
    use_whisper_segments: bool = False,
    max_cue_seconds: Optional[float] = None,
) -> List[SubtitleCue]:
    """
    Build SubtitleCue objects from oracle-segment transcription output.

    Args:
        segment_outputs: List of dicts from _transcribe_bilingual_oracle.
            Base schema: {"position": int, "language": str,
                          "reference": str, "hypothesis": str}
            Extended schema (when return_whisper_segments=True was used):
            adds "whisper_segments": list[dict]
        segments: List[Segment] from ManifestEntry.segments -- provides timing
            and audio_path for duration inference.
        infer_timing_from_audio: If True (default), infer segment duration from
            audio file when segment.start/end are both 0.0.
        use_whisper_segments: If True, use Whisper's internal segments (from
            segment_outputs["whisper_segments"]) to build multiple short cues
            per bilingual manifest segment.  Falls back to one-cue-per-segment
            if the key is missing or the list is empty for a given segment.
        max_cue_seconds: passed through to build_whisper_segment_cues.
            No effect in fallback (single-cue) mode.

    Returns:
        List[SubtitleCue] in segment order, with monotonically increasing
        timestamps.

    Note (V1.2): the single-cue fallback path now populates source_text from
    out["reference"].  This enables bilingual subtitle display in write_srt
    without changes to downstream callers.
    """
    if not segment_outputs:
        return []

    # Index manifest segments by position for O(1) lookup.
    seg_by_pos = {i: seg for i, seg in enumerate(segments)}

    cues: List[SubtitleCue] = []
    cursor = 0.0  # running global start time (seconds)

    for out in segment_outputs:
        pos = out["position"]
        seg = seg_by_pos.get(pos)

        # ------------------------------------------------------------------
        # 1. Determine this bilingual segment's duration on the global timeline.
        #    This is used for cursor advancement regardless of cue-building mode.
        # ------------------------------------------------------------------
        duration: Optional[float] = None

        if seg is not None:
            if seg.end > seg.start:
                duration = seg.end - seg.start
            elif infer_timing_from_audio and seg.audio_path:
                duration = _get_audio_duration(seg.audio_path)

        if duration is None or duration <= 0:
            warnings.warn(
                f"Segment position={pos} has no valid timing and audio inference "
                f"failed or was disabled. Cue will have zero duration."
            )
            duration = 0.0

        # ------------------------------------------------------------------
        # 2. Build cue(s) for this bilingual segment.
        # ------------------------------------------------------------------
        w_segs = out.get("whisper_segments") if use_whisper_segments else None

        if w_segs:
            # Whisper-segment mode: multiple short cues per manifest segment.
            # cursor is the offset to add to all Whisper-relative timestamps.
            seg_cues = build_whisper_segment_cues(
                w_segs,
                cursor_offset=cursor,
                source_language=out.get("language"),
                max_cue_seconds=max_cue_seconds,
            )
            cues.extend(seg_cues)
        else:
            # Fallback (V1.0 / V1.2): one cue for the entire bilingual segment.
            # source_text is populated so bilingual mode can display both lines.
            cues.append(SubtitleCue(
                start=round(cursor, 3),
                end=round(cursor + duration, 3),
                text=out["hypothesis"],
                source_language=out.get("language"),
                timing_source="fallback",
                source_text=out.get("reference"),
            ))

        # ------------------------------------------------------------------
        # 3. Advance cursor by the full bilingual segment duration.
        #    Do this regardless of cue mode so pause_s gaps stay correct.
        # ------------------------------------------------------------------
        cursor += duration
        if seg is not None and seg.pause_s > 0.0:
            cursor += seg.pause_s

    return cues


# ---------------------------------------------------------------------------
# Reference output helpers  (no model required — use manifest ground truth)
# ---------------------------------------------------------------------------

def _parse_srt_texts(srt_path: Path) -> List[str]:
    """
    Parse an SRT file into a flat list of cue text strings.
    Returns [] if the file does not exist or cannot be parsed.
    """
    try:
        raw = Path(srt_path).read_text(encoding="utf-8")
    except OSError:
        return []
    texts: List[str] = []
    for block in re.split(r"\n{2,}", raw.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        texts.append("\n".join(lines[2:]).strip())
    return texts


def write_reference_srt(
    segments,
    output_path: str | Path,
    switch_marker: bool = True,
    max_chars_per_line: int = 42,
) -> None:
    """
    Write a reference SRT from manifest segments without running any model.

    Uses the ground-truth transcript from each Segment object.  If
    switch_marker=True, a brief [ES→EN] boundary cue is inserted at every
    language transition to make the switch point visually explicit in the SRT.

    Timing is taken from segment.start/end when non-zero, otherwise inferred
    from the segment's audio file via librosa.

    Args:
        segments:           List[Segment] from ManifestEntry.segments.
        output_path:        Destination .srt file path.
        switch_marker:      Insert a 0.3 s boundary cue at each language switch.
        max_chars_per_line: Line-wrap limit (default 42).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cues: List[SubtitleCue] = []
    cursor = 0.0
    prev_lang: Optional[str] = None

    for seg in segments:
        # Determine duration
        if seg.end > seg.start:
            duration = seg.end - seg.start
        else:
            inferred = _get_audio_duration(seg.audio_path) if seg.audio_path else None
            duration = inferred if (inferred and inferred > 0) else 1.0

        lang = seg.language

        # Insert switch-boundary marker cue when language changes
        if switch_marker and prev_lang is not None and lang != prev_lang:
            marker_dur = min(0.3, duration * 0.1)
            cues.append(SubtitleCue(
                start=round(cursor, 3),
                end=round(cursor + marker_dur, 3),
                text=f"[{prev_lang.upper()}\u2192{lang.upper()}]",
                source_language=None,
                timing_source="reference_marker",
            ))
            cursor += marker_dur

        cues.append(SubtitleCue(
            start=round(cursor, 3),
            end=round(cursor + duration, 3),
            text=seg.transcript,
            source_language=lang,
            timing_source="reference",
        ))
        cursor += duration
        if hasattr(seg, "pause_s") and seg.pause_s > 0:
            cursor += seg.pause_s
        prev_lang = lang

    write_srt(cues, output_path, add_period=False,
              max_chars_per_line=max_chars_per_line, subtitle_mode="english")


def write_reference_txt(segments, output_path: str | Path) -> None:
    """
    Write a plain-text reference file with explicit switch-point labeling.

    Output format::

        [ES]  por dar las señas de la taberna...

        ─── ES→EN SWITCH ───

        [EN]  AFTER AN APPRENTICESHIP ON A MERCHANT SHIP...

    Args:
        segments:    List[Segment] from ManifestEntry.segments.
        output_path: Destination .txt file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    prev_lang: Optional[str] = None

    for seg in segments:
        lang = seg.language
        if prev_lang is not None and lang != prev_lang:
            lines.append("")
            lines.append(
                f"\u2500\u2500\u2500 {prev_lang.upper()}\u2192{lang.upper()} SWITCH \u2500\u2500\u2500"
            )
            lines.append("")
        lines.append(f"[{lang.upper()}]  {seg.transcript}")
        prev_lang = lang

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_comparison_md(
    entry_id: str,
    output_path: str | Path,
    segments,
    srt_map: dict,
    manifest_path: Optional[str] = None,
) -> None:
    """
    Write a markdown comparison document for one bilingual demo case.

    Always includes the reference ground truth split by language segment.
    Reads any available model SRT files (Whisper, WhisperX) to include their
    output in the comparison table.

    Args:
        entry_id:      ManifestEntry.id — used for heading and code snippets.
        output_path:   Destination comparison.md path.
        segments:      List[Segment] from ManifestEntry (for reference text).
        srt_map:       Dict mapping model label -> Path|None, e.g.
                       {'Whisper': Path('whisper.srt'), 'WhisperX': None}.
        manifest_path: Optional source manifest path shown in the
                       'how to generate' code block.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_path.parent

    def _trunc(s: str, n: int = 140) -> str:
        return s if len(s) <= n else s[:n] + "\u2026"

    def _srt_full_text(label: str) -> str:
        p = srt_map.get(label)
        if p is None or not Path(p).exists():
            return "_not available_"
        texts = _parse_srt_texts(Path(p))
        # Filter out switch-marker cues like [ES→EN]
        texts = [t for t in texts if not re.match(r"^\[.{2}\u2192.{2}\]$", t)]
        return _trunc(" / ".join(texts)) if texts else "_empty_"

    # Compute approximate switch time using audio durations
    switch_time: Optional[float] = None
    cursor = 0.0
    prev_lang_: Optional[str] = None
    for seg in segments:
        dur = (seg.end - seg.start) if seg.end > seg.start else (
            (_get_audio_duration(seg.audio_path) or 1.0) if seg.audio_path else 1.0
        )
        if prev_lang_ is not None and seg.language != prev_lang_ and switch_time is None:
            switch_time = round(cursor, 1)
        cursor += dur
        prev_lang_ = seg.language

    md: List[str] = []
    md.append(f"# Switch-Point Demo: `{entry_id}`\n")

    # ── Reference ground truth ────────────────────────────────────────────────
    md.append("## Reference Ground Truth\n")
    for i, seg in enumerate(segments):
        md.append(f"**[{seg.language.upper()}]**  ")
        md.append(f"{seg.transcript}\n")
        if i + 1 < len(segments) and segments[i + 1].language != seg.language:
            sw_label = f"{seg.language.upper()}\u2192{segments[i+1].language.upper()}"
            switch_note = f" at ~{switch_time:.1f}s" if switch_time is not None else ""
            md.append(f"**\u2500\u2500\u2500 {sw_label} SWITCH{switch_note} \u2500\u2500\u2500**\n")

    # ── Model outputs table ───────────────────────────────────────────────────
    md.append("---\n")
    md.append("## Model Outputs\n")
    md.append("| Model | Transcript (condensed) |")
    md.append("|---|---|")
    ref_text = _trunc(" / ".join(seg.transcript for seg in segments))
    md.append(f"| **Reference** | {ref_text} |")
    for label in srt_map:
        md.append(f"| **{label}** | {_srt_full_text(label)} |")

    # ── File status ───────────────────────────────────────────────────────────
    md.append("\n---\n")
    md.append("## Files in This Directory\n")
    md.append("| File | Status |")
    md.append("|---|---|")
    for fname in ("reference.srt", "reference.txt"):
        tick = "\u2713" if (out_dir / fname).exists() else "\u2717"
        md.append(f"| `{fname}` | {tick} |")
    for label, path in srt_map.items():
        fname = label.lower() + ".srt"
        tick = "\u2713" if (path and Path(path).exists()) else "\u2717 not available"
        md.append(f"| `{fname}` | {tick} |")
    md.append("| `comparison.md` | \u2713 (this file) |")

    # ── How to generate whisper.srt ───────────────────────────────────────────
    if not (srt_map.get("Whisper") and Path(srt_map["Whisper"]).exists()):
        mfest = manifest_path or "data/manifests/bilingual_es-en_50.jsonl"
        md.append("\n---\n")
        md.append("## Generate `whisper.srt`\n")
        md.append("```bash")
        md.append("python scripts/export_subtitles.py \\")
        md.append(f"    --manifest   {mfest} \\")
        md.append(f"    --sample-id  {entry_id} \\")
        md.append(f"    --output-dir {out_dir} \\")
        md.append( "    --model      large-v3 \\")
        md.append( "    --translate-es \\")
        md.append( "    --subtitle-mode bilingual")
        md.append(f"# rename: mv \"{out_dir / entry_id}.srt\" \"{out_dir / 'whisper.srt'}\"")
        md.append("```")

    md.append("")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
