"""
build_manifests.py — Build SwitchNet JSONL manifests from raw datasets.

Supported sources:
  - Mozilla Common Voice (Spanish): cv-corpus-*/es/
  - LibriSpeech (English, for reference): LibriSpeech/test-clean/
  - MLS (Multilingual LibriSpeech, Spanish): mls_spanish/
  - Bilingual concat: given an EN manifest + ES manifest, interleave segments.

Usage examples:
  # Build Spanish manifest from MLS
  python scripts/build_manifests.py mls \
      --mls-root data/mls_spanish \
      --split    test \
      --output   data/manifests/es_mls_test.jsonl \
      --max      200

  # Build bilingual manifest by concatenating EN + ES utterances
  python scripts/build_manifests.py bilingual \
      --en-manifest data/manifests/en_librispeech_test.jsonl \
      --es-manifest data/manifests/es_mls_test.jsonl \
      --output      data/manifests/bilingual_concat.jsonl \
      --pairs       100
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import ManifestEntry, save_manifest


# ---------------------------------------------------------------------------
# Common Voice (Spanish) builder
# ---------------------------------------------------------------------------

def build_common_voice_manifest(
    cv_root: str | Path,
    split: str = "test",            # "test" | "dev" | "train"
    output_path: str | Path = "data/manifests/es_cv_test.jsonl",
    max_entries: Optional[int] = None,
    seed: int = 42,
) -> List[ManifestEntry]:
    """
    Build a manifest from Mozilla Common Voice data directory.

    Expected structure:
        cv_root/
          {split}.tsv        ¡û metadata
          clips/             ¡û audio files (.mp3)

    Common Voice audio is MP3. Whisper can handle MP3 directly.
    """
    cv_root = Path(cv_root)
    tsv_path = cv_root / f"{split}.tsv"
    clips_dir = cv_root / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"Common Voice TSV not found: {tsv_path}")

    entries = []
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Optionally subsample
    if max_entries and len(rows) > max_entries:
        random.seed(seed)
        rows = random.sample(rows, max_entries)

    skipped = 0
    for row in rows:
        clip_name = row.get("path", "")
        # CV sometimes stores filename without extension, sometimes with
        if not clip_name.endswith(".mp3"):
            clip_name = clip_name + ".mp3"
        audio_path = clips_dir / clip_name

        if not audio_path.exists():
            skipped += 1
            continue

        transcript = row.get("sentence", "").strip()
        if not transcript:
            skipped += 1
            continue

        uid = Path(clip_name).stem  # filename without extension
        entries.append(
            ManifestEntry(
                id=f"cv_es_{uid}",
                audio_path=str(audio_path.resolve()),
                language="es",
                transcript=transcript,
                # CV doesn't give duration in the TSV; leave None
                # You can populate this with librosa in a post-processing pass
                duration_s=None,
            )
        )

    print(f"Built {len(entries)} entries ({skipped} skipped) from {tsv_path}")
    save_manifest(entries, output_path)
    return entries


# ---------------------------------------------------------------------------
# LibriSpeech builder (mirrors your existing English baseline)
# ---------------------------------------------------------------------------

def build_librispeech_manifest(
    ls_root: str | Path,
    split: str = "test-clean",
    output_path: str | Path = "data/manifests/en_librispeech_test.jsonl",
    max_entries: Optional[int] = None,
) -> List[ManifestEntry]:
    """
    Build a manifest from a LibriSpeech split directory.

    Expected structure (standard LibriSpeech layout):
        ls_root/
          {split}/
            {speaker}/
              {chapter}/
                {speaker}-{chapter}-{utterance}.flac
                {speaker}-{chapter}.trans.txt
    """
    ls_root = Path(ls_root) / split
    if not ls_root.exists():
        raise FileNotFoundError(f"LibriSpeech split not found: {ls_root}")

    entries = []
    for trans_file in sorted(ls_root.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                uid, transcript = parts
                audio_path = chapter_dir / f"{uid}.flac"
                if not audio_path.exists():
                    continue
                entries.append(
                    ManifestEntry(
                        id=f"ls_{uid}",
                        audio_path=str(audio_path.resolve()),
                        language="en",
                        transcript=transcript.strip(),
                        duration_s=None,
                    )
                )

    if max_entries:
        entries = entries[:max_entries]

    print(f"Built {len(entries)} entries from {ls_root}")
    save_manifest(entries, output_path)
    return entries


# ---------------------------------------------------------------------------
# MLS (Multilingual LibriSpeech) Spanish builder
# ---------------------------------------------------------------------------

def build_mls_manifest(
    mls_root: str | Path,
    split: str = "test",
    output_path: str | Path = "data/manifests/es_mls_test.jsonl",
    max_entries: Optional[int] = None,
) -> List[ManifestEntry]:
    """
    Build a manifest from MLS (Multilingual LibriSpeech) Spanish data.

    Expected structure:
        mls_root/
          {split}/
            transcripts.txt        <- tab-separated: id \\t transcript
            audio/
              {speaker}/
                {chapter}/
                  {speaker}_{chapter}_{utt}.opus
    """
    mls_root = Path(mls_root)
    split_dir = mls_root / split
    transcripts_path = split_dir / "transcripts.txt"
    audio_dir = split_dir / "audio"

    if not transcripts_path.exists():
        raise FileNotFoundError(f"MLS transcripts not found: {transcripts_path}")

    entries = []
    skipped = 0
    with open(transcripts_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                skipped += 1
                continue
            uid, transcript = parts
            # uid format: {speaker_id}_{chapter_id}_{utterance_id}
            uid_parts = uid.split("_")
            if len(uid_parts) < 3:
                skipped += 1
                continue
            speaker_id, chapter_id = uid_parts[0], uid_parts[1]
            audio_path = audio_dir / speaker_id / chapter_id / f"{uid}.opus"
            if not audio_path.exists():
                skipped += 1
                continue
            entries.append(
                ManifestEntry(
                    id=f"mls_es_{uid}",
                    audio_path=str(audio_path.resolve()),
                    language="es",
                    transcript=transcript.strip(),
                    duration_s=None,
                )
            )
            if max_entries and len(entries) >= max_entries:
                break

    print(f"Built {len(entries)} entries ({skipped} skipped) from {transcripts_path}")
    save_manifest(entries, output_path)
    return entries


# ---------------------------------------------------------------------------
# Bilingual concat builder (MVP for bilingual merge)
# ---------------------------------------------------------------------------

def build_bilingual_concat_manifest(
    en_manifest: str | Path,
    es_manifest: str | Path,
    output_path: str | Path = "data/manifests/bilingual_concat.jsonl",
    n_pairs: int = 100,
    seed: int = 42,
    pattern: str = "mixed",
    pause_s: float = 0.0,
) -> List[ManifestEntry]:
    """
    Build a bilingual manifest by pairing EN and ES utterances.

    Each entry has audio_path="__multi__" (sentinel) and a segments list
    where each segment carries its own audio_path. The transcriber processes
    each segment file independently and merges the transcripts.

    Args:
        pattern:  Segment order / layout. Choices:
                    "en-es"     -- EN first, ES second (all pairs)
                    "es-en"     -- ES first, EN second (all pairs)
                    "mixed"     -- alternating en-es / es-en (default)
                    "en-es-en"  -- 3 segments: EN, ES, EN (uses 2 EN per pair)
                    "es-en-es"  -- 3 segments: ES, EN, ES (uses 2 ES per pair)
        pause_s:  Metadata-only gap (seconds) stored between segments.
                  Transcription ignores it now; reserved for future audio concat.
    """
    from src.data.manifest import load_manifest

    en_entries = load_manifest(en_manifest)
    es_entries = load_manifest(es_manifest)

    random.seed(seed)
    random.shuffle(en_entries)
    random.shuffle(es_entries)

    # For 3-segment patterns we consume more entries from one language pool.
    if pattern == "en-es-en":
        # Each pair uses 2 EN + 1 ES
        n = min(n_pairs, len(en_entries) // 2, len(es_entries))
        en_a = en_entries[:n]
        en_b = en_entries[n: 2 * n]
        es_a = es_entries[:n]
        group_iter = zip(en_a, es_a, en_b)
    elif pattern == "es-en-es":
        # Each pair uses 1 EN + 2 ES
        n = min(n_pairs, len(en_entries), len(es_entries) // 2)
        es_a = es_entries[:n]
        es_b = es_entries[n: 2 * n]
        en_a = en_entries[:n]
        group_iter = zip(es_a, en_a, es_b)
    else:
        n = min(n_pairs, len(en_entries), len(es_entries))
        group_iter = zip(en_entries[:n], es_entries[:n])

    def _seg(e, start: float) -> dict:
        """Build one segment dict for entry e starting at `start`."""
        end = start + (e.duration_s or 0.0)
        return {
            "start": start,
            "end": end,
            "language": e.language,
            "transcript": e.transcript,
            "audio_path": e.audio_path,
            "pause_s": pause_s,      # metadata; 0.0 means no pause
        }

    bilingual_entries = []
    for i, group in enumerate(group_iter):
        if pattern == "en-es-en":
            en1, es1, en2 = group
            ordered = [en1, es1, en2]
        elif pattern == "es-en-es":
            es1, en1, es2 = group
            ordered = [es1, en1, es2]
        elif pattern == "es-en":
            en_e, es_e = group
            ordered = [es_e, en_e]
        elif pattern == "en-es":
            en_e, es_e = group
            ordered = [en_e, es_e]
        else:  # mixed: alternate en-es / es-en
            en_e, es_e = group
            ordered = [en_e, es_e] if i % 2 == 0 else [es_e, en_e]

        # Build segments with cumulative time offsets
        segments = []
        cursor = 0.0
        for e in ordered:
            segments.append(_seg(e, cursor))
            cursor = segments[-1]["end"]

        combined_transcript = " ".join(e.transcript for e in ordered)
        total_dur = cursor or None

        entry = ManifestEntry(
            id=f"bilingual_{pattern}_{i:04d}_" + "_".join(e.id for e in ordered),
            audio_path="__multi__",
            language="bilingual",
            transcript=combined_transcript,
            duration_s=total_dur,
        )
        bilingual_entries.append((entry, segments))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry, segs in bilingual_entries:
            d = entry.to_dict()
            d["segments"] = segs
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Built {len(bilingual_entries)} bilingual ({pattern}) entries -> {output_path}")
    return [e for e, _ in bilingual_entries]


# ---------------------------------------------------------------------------
# Populate durations (optional post-processing pass)
# ---------------------------------------------------------------------------

def populate_durations(
    manifest_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Fill in missing duration_s fields using librosa.
    Run this once after building a manifest to enable RTF reporting.
    """
    try:
        import librosa
    except ImportError:
        print("librosa not installed ¡ª skipping duration population.")
        return

    from src.data.manifest import load_manifest, save_manifest

    entries = load_manifest(manifest_path)
    for i, entry in enumerate(entries):
        if entry.duration_s is None and entry.audio_path != "__multi__":
            try:
                dur = librosa.get_duration(path=entry.audio_path)
                entry.duration_s = round(dur, 3)
            except Exception as e:
                print(f"Warning: could not get duration for {entry.id}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(entries)}")

    out = output_path or manifest_path
    save_manifest(entries, out)
    print(f"Durations populated ¡ú {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SwitchNet manifest builder")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common Voice
    cv_p = sub.add_parser("cv", help="Build Spanish Common Voice manifest")
    cv_p.add_argument("--cv-root", required=True)
    cv_p.add_argument("--split",   default="test")
    cv_p.add_argument("--output",  default="data/manifests/es_cv_test.jsonl")
    cv_p.add_argument("--max",     type=int, default=None)

    # MLS Spanish
    mls_p = sub.add_parser("mls", help="Build MLS Spanish manifest")
    mls_p.add_argument("--mls-root", required=True)
    mls_p.add_argument("--split",    default="test")
    mls_p.add_argument("--output",   default="data/manifests/es_mls_test.jsonl")
    mls_p.add_argument("--max",      type=int, default=None)

    # LibriSpeech
    ls_p = sub.add_parser("librispeech", help="Build LibriSpeech manifest")
    ls_p.add_argument("--ls-root", required=True)
    ls_p.add_argument("--split",   default="test-clean")
    ls_p.add_argument("--output",  default="data/manifests/en_librispeech_test.jsonl")
    ls_p.add_argument("--max",     type=int, default=None)

    # Bilingual concat
    bi_p = sub.add_parser("bilingual", help="Build bilingual concat manifest")
    bi_p.add_argument("--en-manifest", required=True)
    bi_p.add_argument("--es-manifest", required=True)
    bi_p.add_argument("--output",      default="data/manifests/bilingual_concat.jsonl")
    bi_p.add_argument("--pairs",       type=int, default=100)
    bi_p.add_argument("--pattern",     default="mixed",
                      choices=["en-es", "es-en", "mixed", "en-es-en", "es-en-es"],
                      help="Segment order pattern (default: mixed)")
    bi_p.add_argument("--pause-s",     type=float, default=0.0,
                      help="Metadata pause between segments in seconds (default: 0)")

    # Duration population
    dur_p = sub.add_parser("durations", help="Populate duration_s fields")
    dur_p.add_argument("--manifest", required=True)
    dur_p.add_argument("--output",   default=None)

    args = parser.parse_args()

    if args.command == "mls":
        build_mls_manifest(args.mls_root, args.split, args.output, args.max)
    elif args.command == "cv":
        build_common_voice_manifest(args.cv_root, args.split, args.output, args.max)
    elif args.command == "librispeech":
        build_librispeech_manifest(args.ls_root, args.split, args.output, args.max)
    elif args.command == "bilingual":
        build_bilingual_concat_manifest(
            args.en_manifest, args.es_manifest, args.output, args.pairs,
            pattern=args.pattern, pause_s=args.pause_s,
        )
    elif args.command == "durations":
        populate_durations(args.manifest, args.output)


if __name__ == "__main__":
    main()