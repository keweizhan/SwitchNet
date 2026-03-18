"""
transcribe.py — Language-aware Whisper transcription for SwitchNet.

Handles:
  - English-only transcription  (language="en")
  - Spanish-only transcription  (language="es")
  - Bilingual concatenated mode (language="bilingual"):
      routes each segment to its own language pass, then merges.

Usage (CLI):
    python -m src.asr.transcribe \
        --manifest data/manifests/es_common_voice.jsonl \
        --output    data/manifests/es_common_voice_hyp.jsonl \
        --model     large-v3

Usage (API):
    from src.asr.transcribe import Transcriber
    t = Transcriber(model_size="large-v3")
    results = t.transcribe_manifest("data/manifests/es_common_voice.jsonl")
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import whisper

from src.data.manifest import ManifestEntry, load_manifest, save_manifest


# ---------------------------------------------------------------------------
# Core transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        self.model_size = model_size

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _transcribe_file(self, audio_path: str, language: str) -> str:
        """Transcribe a single audio file with a fixed language."""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            fp16=(self.device != "cpu"),
        )
        return result["text"].strip()

    def _transcribe_segment(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str,
    ) -> str:
        """
        Transcribe a time-bounded segment by slicing audio before feeding Whisper.
        Requires librosa for slice; falls back to full-file if not available.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                audio_path,
                sr=16000,
                offset=start,
                duration=end - start,
                mono=True,
            )
            # Whisper expects float32 numpy array at 16 kHz
            result = self.model.transcribe(
                audio.astype(np.float32),
                language=language,
                task="transcribe",
                fp16=(self.device != "cpu"),
            )
            return result["text"].strip()
        except ImportError:
            # Fallback: transcribe full file (less accurate for bilingual)
            return self._transcribe_file(audio_path, language)

    # ------------------------------------------------------------------
    # Per-entry dispatch
    # ------------------------------------------------------------------

    def transcribe_entry(self, entry: ManifestEntry) -> str:
        """
        Transcribe one manifest entry.
        - Monolingual (en/es): single Whisper call with that language.
        - Bilingual: per-segment routing, then join in order.
        """
        if entry.language in ("en", "es"):
            return self._transcribe_file(entry.audio_path, entry.language)

        if entry.language == "bilingual":
            if not entry.segments:
                raise ValueError(
                    f"Entry {entry.id} has language='bilingual' but no segments defined."
                )
            parts = []
            for seg in entry.segments:
                hyp = self._transcribe_segment(
                    entry.audio_path, seg.start, seg.end, seg.language
                )
                parts.append(hyp)
            return " ".join(parts)

        raise ValueError(f"Unknown language code: {entry.language!r}")

    # ------------------------------------------------------------------
    # Manifest-level runner
    # ------------------------------------------------------------------

    def transcribe_manifest(
        self,
        manifest_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_entries: Optional[int] = None,
    ) -> List[dict]:
        """
        Transcribe all entries in a manifest.

        Returns a list of dicts:
          {"id": ..., "reference": ..., "hypothesis": ..., "language": ..., "rtf": ...}

        If output_path is given, also writes a JSONL result file.
        """
        entries = load_manifest(manifest_path)
        if max_entries:
            entries = entries[:max_entries]

        results = []
        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry.id} ({entry.language})", end=" ... ", flush=True)
            t0 = time.time()
            try:
                hyp = self.transcribe_entry(entry)
                elapsed = time.time() - t0
                rtf = round(elapsed / entry.duration_s, 3) if entry.duration_s else None
                print(f"done ({elapsed:.1f}s)")
                results.append(
                    {
                        "id": entry.id,
                        "language": entry.language,
                        "reference": entry.transcript,
                        "hypothesis": hyp,
                        "rtf": rtf,
                    }
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append(
                    {
                        "id": entry.id,
                        "language": entry.language,
                        "reference": entry.transcript,
                        "hypothesis": "",
                        "error": str(exc),
                        "rtf": None,
                    }
                )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\nResults written to {output_path}")

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SwitchNet Whisper transcription")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--output",   required=True, help="Output JSONL results")
    parser.add_argument("--model",    default="large-v3", help="Whisper model size")
    parser.add_argument("--device",   default=None,       help="cuda | cpu (auto-detect if omitted)")
    parser.add_argument("--max",      type=int, default=None, help="Limit number of entries (debug)")
    args = parser.parse_args()

    transcriber = Transcriber(model_size=args.model, device=args.device)
    transcriber.transcribe_manifest(args.manifest, args.output, max_entries=args.max)


if __name__ == "__main__":
    main()