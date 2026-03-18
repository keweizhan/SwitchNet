"""
manifest.py — SwitchNet manifest loading and validation.

Manifest schema (JSONL, one record per line):
{
    "id": "unique-utterance-id",
    "audio_path": "/absolute/or/relative/path/to/audio.wav",
    "language": "en" | "es" | "bilingual",
    "transcript": "reference transcription",
    "duration_s": 4.32,              # optional but recommended
    "segments": [                    # required only for bilingual
        {"start": 0.0, "end": 2.1, "language": "en", "transcript": "hello world"},
        {"start": 2.1, "end": 4.32, "language": "es", "transcript": "hola mundo"}
    ]
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Segment:
    start: float
    end: float
    language: str
    transcript: str


@dataclass
class ManifestEntry:
    id: str
    audio_path: str
    language: str
    transcript: str
    duration_s: Optional[float] = None
    segments: List[Segment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        segments = [Segment(**s) for s in d.get("segments", [])]
        return cls(
            id=d["id"],
            audio_path=d["audio_path"],
            language=d["language"],
            transcript=d["transcript"],
            duration_s=d.get("duration_s"),
            segments=segments,
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "audio_path": self.audio_path,
            "language": self.language,
            "transcript": self.transcript,
        }
        if self.duration_s is not None:
            d["duration_s"] = self.duration_s
        if self.segments:
            d["segments"] = [
                {"start": s.start, "end": s.end, "language": s.language, "transcript": s.transcript}
                for s in self.segments
            ]
        return d

    def has_switch_points(self) -> bool:
        """True if this entry has annotated language-switch boundaries."""
        if len(self.segments) < 2:
            return False
        langs = [s.language for s in self.segments]
        return any(langs[i] != langs[i + 1] for i in range(len(langs) - 1))

    def switch_points(self) -> List[float]:
        """Return timestamps (seconds) where language switches occur."""
        pts = []
        for i in range(len(self.segments) - 1):
            if self.segments[i].language != self.segments[i + 1].language:
                pts.append(self.segments[i].end)
        return pts


def load_manifest(path: str | Path) -> List[ManifestEntry]:
    """Load a JSONL manifest file into a list of ManifestEntry objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ManifestEntry.from_dict(json.loads(line)))
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Manifest parse error at line {lineno}: {e}")
    return entries


def save_manifest(entries: List[ManifestEntry], path: str | Path) -> None:
    """Write a list of ManifestEntry objects to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} entries to {path}")


def filter_by_language(entries: List[ManifestEntry], language: str) -> List[ManifestEntry]:
    return [e for e in entries if e.language == language]


def manifest_stats(entries: List[ManifestEntry]) -> dict:
    """Quick summary stats for a manifest."""
    langs = {}
    total_dur = 0.0
    for e in entries:
        langs[e.language] = langs.get(e.language, 0) + 1
        if e.duration_s:
            total_dur += e.duration_s
    return {
        "total_entries": len(entries),
        "by_language": langs,
        "total_duration_hours": round(total_dur / 3600, 3),
        "switch_point_entries": sum(1 for e in entries if e.has_switch_points()),
    }