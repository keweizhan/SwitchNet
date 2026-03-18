"""
build_manifests.py ¡ª Build SwitchNet JSONL manifests from raw datasets.

Supported sources:
  - Mozilla Common Voice (Spanish): cv-corpus-*/es/
  - LibriSpeech (English, for reference): LibriSpeech/test-clean/
  - Bilingual concat: given an EN manifest + ES manifest, interleave segments.

Usage examples:
  # Build Spanish manifest from Common Voice
  python scripts/build_manifests.py cv \
      --cv-root  /data/cv-corpus-17.0/es \
      --split    test \
      --output   data/manifests/es_common_voice_test.jsonl \
      --max      200

  # Build bilingual manifest by concatenating EN + ES utterances
  python scripts/build_manifests.py bilingual \
      --en-manifest data/manifests/en_librispeech_test.jsonl \
      --es-manifest data/manifests/es_common_voice_test.jsonl \
      --output      data/manifests/bilingual_concat.jsonl \
      --pairs       100
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Optional

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
    with open(tsv_path, newline="", encoding="utf-8") as f:
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
# Bilingual concat builder (MVP for bilingual merge)
# ---------------------------------------------------------------------------

def build_bilingual_concat_manifest(
    en_manifest: str | Path,
    es_manifest: str | Path,
    output_path: str | Path = "data/manifests/bilingual_concat.jsonl",
    n_pairs: int = 100,
    seed: int = 42,
) -> List[ManifestEntry]:
    """
    Build a bilingual manifest by pairing EN and ES utterances.

    Strategy: concatenate audio files at the Python level is complex
    (requires ffmpeg/pydub). Instead, we create manifest entries that
    list BOTH segments as if they were sub-segments of a single logical
    "utterance". The bilingual transcriber in transcribe.py will process
    each segment file independently and merge the transcripts.

    Each bilingual entry uses a special audio_path format:
      "__multi__"   (sentinel ¡ª real paths are in the segments)

    The transcriber handles this via segment-level dispatch.
    """
    from src.data.manifest import load_manifest

    en_entries = load_manifest(en_manifest)
    es_entries = load_manifest(es_manifest)

    random.seed(seed)
    random.shuffle(en_entries)
    random.shuffle(es_entries)

    n = min(n_pairs, len(en_entries), len(es_entries))
    pairs = list(zip(en_entries[:n], es_entries[:n]))

    bilingual_entries = []
    for i, (en_e, es_e) in enumerate(pairs):
        # Alternate which language comes first (EN¡úES or ES¡úEN)
        if i % 2 == 0:
            first, second = en_e, es_e
        else:
            first, second = es_e, en_e

        # Build a pseudo-segment list using separate audio files
        # We use duration offsets: first=0¡údur_first, second=dur_first¡úend
        # Since we don't always have durations, use placeholder 0.0/1.0
        dur_first = first.duration_s or 0.0

        segments = [
            {
                "start": 0.0,
                "end": dur_first,
                "language": first.language,
                "transcript": first.transcript,
                # Non-standard field: actual audio file for this segment
                "_audio_path": first.audio_path,
            },
            {
                "start": dur_first,
                "end": dur_first + (second.duration_s or 0.0),
                "language": second.language,
                "transcript": second.transcript,
                "_audio_path": second.audio_path,
            },
        ]

        combined_transcript = first.transcript + " " + second.transcript

        entry = ManifestEntry(
            id=f"bilingual_{i:04d}_{first.id}_{second.id}",
            audio_path="__multi__",   # sentinel
            language="bilingual",
            transcript=combined_transcript,
            duration_s=(first.duration_s or 0) + (second.duration_s or 0) or None,
        )
        # Store raw segment dicts (including _audio_path) so transcriber can use them
        entry.segments = []  # will be set below via raw dict
        bilingual_entries.append((entry, segments))

    # Write manually to preserve _audio_path in segments
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry, segs in bilingual_entries:
            d = entry.to_dict()
            d["segments"] = segs
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Built {len(bilingual_entries)} bilingual entries ¡ú {output_path}")
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

    # Duration population
    dur_p = sub.add_parser("durations", help="Populate duration_s fields")
    dur_p.add_argument("--manifest", required=True)
    dur_p.add_argument("--output",   default=None)

    args = parser.parse_args()

    if args.command == "cv":
        build_common_voice_manifest(args.cv_root, args.split, args.output, args.max)
    elif args.command == "librispeech":
        build_librispeech_manifest(args.ls_root, args.split, args.output, args.max)
    elif args.command == "bilingual":
        build_bilingual_concat_manifest(args.en_manifest, args.es_manifest, args.output, args.pairs)
    elif args.command == "durations":
        populate_durations(args.manifest, args.output)


if __name__ == "__main__":
    main()