"""
transcribe.py — Language-aware Whisper transcription for SwitchNet.

Handles:
  - English-only transcription  (language="en")
  - Spanish-only transcription  (language="es")
  - Bilingual oracle mode       (language="bilingual", bilingual_mode="oracle_segments"):
      routes each segment to its own language pass, then merges.
  - Bilingual full-concat mode  (language="bilingual", bilingual_mode="full_concat"):
      concatenates all segment audio with real silence gaps (pause_s),
      then decodes the whole waveform in a single Whisper pass.
      Used for A2 pause-vs-no-pause experiments.

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
        preprocessor=None,
    ):
        """
        Args:
            preprocessor: optional callable (audio: np.ndarray, sr: int) -> np.ndarray.
                Applied to every audio array before it is fed to Whisper.
                Use src.audio.preprocess.get_preprocessor() to build one.
                None (default) means no preprocessing -- existing behaviour unchanged.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        self.model_size = model_size
        self.preprocessor = preprocessor  # (np.ndarray, int) -> np.ndarray | None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _load_file_audio(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        Load an audio file as a float32 numpy array and apply self.preprocessor.
        Called by _transcribe_file* when preprocessing is enabled.
        """
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        audio = audio.astype(np.float32)
        if self.preprocessor is not None:
            audio = self.preprocessor(audio, sr)
        return audio

    def _transcribe_file(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> str:
        """Transcribe a single audio file.

        Args:
            language: BCP-47 code ("en", "fr", …) or None for Whisper auto-detection.
            task:     "transcribe" (default) or "translate" (output English).

        If self.preprocessor is set, the audio is loaded, preprocessed, and passed
        as a numpy array.  Otherwise the file path is passed directly (faster path).
        """
        if self.preprocessor is not None:
            audio = self._load_file_audio(audio_path)
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        else:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        return result["text"].strip()

    def _transcribe_file_ex(
        self, audio_path: str, language: Optional[str] = None, task: str = "transcribe"
    ) -> tuple:
        """Like _transcribe_file but also returns the detected/used language code.

        Returns:
            (text: str, detected_language: str)
        """
        if self.preprocessor is not None:
            audio = self._load_file_audio(audio_path)
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        else:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        text = result["text"].strip()
        detected = result.get("language") or language or "unknown"
        return text, detected

    def _transcribe_file_with_segments(
        self, audio_path: str, language: str, task: str = "transcribe"
    ) -> tuple:
        """
        Like _transcribe_file but also returns Whisper's internal segment list.

        Returns:
            (text: str, whisper_segments: list[dict])
            Each dict in whisper_segments has at minimum:
              {"start": float, "end": float, "text": str}
            Timestamps are relative to the start of audio_path (0-based).
            Returns (text, []) if Whisper found no speech.
        """
        if self.preprocessor is not None:
            audio = self._load_file_audio(audio_path)
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        else:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
        return result["text"].strip(), result.get("segments", [])

    def _transcribe_segment(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str,
        task: str = "transcribe",
    ) -> str:
        """
        Transcribe a time-bounded segment by slicing audio before feeding Whisper.
        Requires librosa for slice; falls back to full-file if not available.

        Args:
            task: Whisper task -- "transcribe" (default) or "translate" (output English).
        """
        try:
            import librosa

            audio, _ = librosa.load(
                audio_path,
                sr=16000,
                offset=start,
                duration=end - start,
                mono=True,
            )
            audio = audio.astype(np.float32)
            if self.preprocessor is not None:
                audio = self.preprocessor(audio, 16000)
            # Whisper expects float32 numpy array at 16 kHz
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
            return result["text"].strip()
        except ImportError:
            # Fallback: transcribe full file (less accurate for bilingual)
            return self._transcribe_file(audio_path, language, task=task)

    def _transcribe_segment_with_segments(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str,
        task: str = "transcribe",
    ) -> tuple:
        """
        Like _transcribe_segment but also returns Whisper's internal segment list.

        Returns:
            (text: str, whisper_segments: list[dict])
            Timestamps in whisper_segments are relative to the sliced audio (0-based),
            not to the parent file's start offset.
            Falls back to _transcribe_file_with_segments if librosa is unavailable.
        """
        try:
            import librosa

            audio, _ = librosa.load(
                audio_path,
                sr=16000,
                offset=start,
                duration=end - start,
                mono=True,
            )
            audio = audio.astype(np.float32)
            if self.preprocessor is not None:
                audio = self.preprocessor(audio, 16000)
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                fp16=(self.device != "cpu"),
            )
            return result["text"].strip(), result.get("segments", [])
        except ImportError:
            return self._transcribe_file_with_segments(audio_path, language, task=task)

    # ------------------------------------------------------------------
    # Bilingual mode: oracle_segments  (original behavior)
    # ------------------------------------------------------------------

    def _transcribe_bilingual_oracle(
        self,
        entry: ManifestEntry,
        task_map: Optional[dict] = None,
        return_whisper_segments: bool = False,
    ) -> tuple:
        """
        Transcribe a bilingual entry segment-by-segment with oracle language routing.

        Each segment is decoded independently with its known language forced.
        pause_s is NOT used -- this mode ignores inter-segment gaps.

        Args:
            task_map: Per-language Whisper task override.
                Default {"en": "transcribe", "es": "transcribe"} preserves
                existing behaviour.  For subtitle export pass
                {"en": "transcribe", "es": "translate"} to get English output
                from Spanish segments.
            return_whisper_segments: If True, each dict in segment_outputs will
                include a "whisper_segments" key containing the raw Whisper
                internal segment list for that bilingual segment.  Timestamps
                inside each Whisper segment are relative to that segment's own
                audio (0-based).  Used by subtitle export for fine-grained cue
                splitting.  Default False preserves existing behaviour exactly.

        Returns:
            (joined_hypothesis: str,
             segment_outputs: list[dict])   # one dict per segment

        segment_outputs schema (return_whisper_segments=False):
            {"position": int, "language": str, "reference": str, "hypothesis": str}

        segment_outputs schema (return_whisper_segments=True):
            {"position": int, "language": str, "reference": str, "hypothesis": str,
             "whisper_segments": list[dict]}
        """
        if not entry.segments:
            raise ValueError(
                f"Entry {entry.id} has language='bilingual' but no segments defined."
            )
        if task_map is None:
            task_map = {"en": "transcribe", "es": "transcribe"}

        parts: list[str] = []
        segment_outputs: list[dict] = []
        for i, seg in enumerate(entry.segments):
            task = task_map.get(seg.language, "transcribe")
            whisper_segs: list = []

            if seg.audio_path:
                # Bilingual-concat: each segment is its own complete audio file
                if return_whisper_segments:
                    hyp, whisper_segs = self._transcribe_file_with_segments(
                        seg.audio_path, seg.language, task=task
                    )
                else:
                    hyp = self._transcribe_file(seg.audio_path, seg.language, task=task)
            else:
                # Bilingual-interleaved: segments are time-slices of one file
                if return_whisper_segments:
                    hyp, whisper_segs = self._transcribe_segment_with_segments(
                        entry.audio_path, seg.start, seg.end, seg.language, task=task
                    )
                else:
                    hyp = self._transcribe_segment(
                        entry.audio_path, seg.start, seg.end, seg.language, task=task
                    )

            parts.append(hyp)
            out: dict = {
                "position": i,
                "language": seg.language,
                "reference": seg.transcript,
                "hypothesis": hyp,
            }
            if return_whisper_segments:
                out["whisper_segments"] = whisper_segs
            segment_outputs.append(out)

        return " ".join(parts), segment_outputs

    # ------------------------------------------------------------------
    # Bilingual mode: full_concat  (A2 pause-vs-no-pause mode)
    # ------------------------------------------------------------------

    def _load_segment_audio(self, seg, entry: ManifestEntry, sr: int = 16000):
        """
        Load a single segment's audio as a float32 numpy array at `sr` Hz mono.

        - If seg.audio_path is set, load the whole file.
        - Otherwise slice [seg.start, seg.end) from entry.audio_path.
        """
        import librosa  # deferred import so monolingual paths stay lightweight

        if seg.audio_path:
            audio, _ = librosa.load(seg.audio_path, sr=sr, mono=True)
        else:
            audio, _ = librosa.load(
                entry.audio_path,
                sr=sr,
                mono=True,
                offset=seg.start,
                duration=seg.end - seg.start,
            )
        audio = audio.astype(np.float32)
        if self.preprocessor is not None:
            audio = self.preprocessor(audio, sr)
        return audio

    def _transcribe_full_concat(self, entry: ManifestEntry) -> tuple:
        """
        Transcribe a bilingual entry in ONE Whisper pass over the full
        concatenated waveform.

        For each segment:
          1. Load / slice the segment audio.
          2. Append it to the running waveform.
          3. After the segment (before the next one) insert seg.pause_s seconds
             of zero-valued silence.  pause_s == 0.0 → no silence gap.

        The resulting waveform is decoded with language=None so Whisper can
        handle the code-switched audio without being locked to a single language.

        Returns:
            (hypothesis: str, [])   # empty segment list — no per-seg oracle here
        """
        if not entry.segments:
            raise ValueError(
                f"Entry {entry.id} has language='bilingual' but no segments defined."
            )

        SR = 16000
        chunks: list[np.ndarray] = []

        for i, seg in enumerate(entry.segments):
            chunks.append(self._load_segment_audio(seg, entry, SR))

            # Insert silence AFTER this segment (i.e. before the next segment).
            # pause_s is the gap that appears between this segment and the next.
            if i < len(entry.segments) - 1 and seg.pause_s > 0.0:
                silence_len = int(SR * seg.pause_s)
                chunks.append(np.zeros(silence_len, dtype=np.float32))

        full_audio = np.concatenate(chunks)

        # Do NOT force a language so Whisper's multilingual decoder can handle
        # language switches across the 30-second chunk boundaries.
        result = self.model.transcribe(
            full_audio,
            language=None,
            task="transcribe",
            fp16=(self.device != "cpu"),
        )
        hyp = result["text"].strip()
        return hyp, []

    # ------------------------------------------------------------------
    # Bilingual dispatcher
    # ------------------------------------------------------------------

    def _transcribe_bilingual(
        self,
        entry: ManifestEntry,
        bilingual_mode: str = "oracle_segments",
        task_map: Optional[dict] = None,
    ) -> tuple:
        """Dispatch to the correct bilingual decoding mode."""
        if bilingual_mode == "full_concat":
            return self._transcribe_full_concat(entry)
        # Default / "oracle_segments"
        return self._transcribe_bilingual_oracle(entry, task_map=task_map)

    def transcribe_entry(
        self,
        entry: ManifestEntry,
        bilingual_mode: str = "oracle_segments",
    ) -> str:
        """
        Transcribe one manifest entry.
        - Monolingual (en/es): single Whisper call with that language.
        - Bilingual: dispatched by bilingual_mode.
        """
        if entry.language in ("en", "es"):
            return self._transcribe_file(entry.audio_path, entry.language)

        if entry.language == "bilingual":
            hyp, _ = self._transcribe_bilingual(entry, bilingual_mode=bilingual_mode)
            return hyp

        raise ValueError(f"Unknown language code: {entry.language!r}")

    # ------------------------------------------------------------------
    # Manifest-level runner
    # ------------------------------------------------------------------

    def transcribe_manifest(
        self,
        manifest_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_entries: Optional[int] = None,
        bilingual_mode: str = "oracle_segments",
        task_map: Optional[dict] = None,
    ) -> List[dict]:
        """
        Transcribe all entries in a manifest.

        Args:
            bilingual_mode: How to decode bilingual entries.
                "oracle_segments" — per-segment decoding with forced language (default).
                "full_concat"     — concatenate all segment audio with pause_s silence,
                                    decode in one Whisper pass (A2 experiment mode).
            task_map: Per-language Whisper task for oracle_segments mode.
                Default None → {"en": "transcribe", "es": "transcribe"}.
                Pass {"en": "transcribe", "es": "translate"} for English subtitle output.

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
                seg_outputs = None
                if entry.language == "bilingual":
                    hyp, seg_outputs = self._transcribe_bilingual(
                        entry, bilingual_mode=bilingual_mode, task_map=task_map
                    )
                else:
                    hyp = self.transcribe_entry(entry)
                elapsed = time.time() - t0
                rtf = round(elapsed / entry.duration_s, 3) if entry.duration_s else None
                print(f"done ({elapsed:.1f}s)")
                record: dict = {
                    "id": entry.id,
                    "language": entry.language,
                    "reference": entry.transcript,
                    "hypothesis": hyp,
                    "rtf": rtf,
                }
                if seg_outputs is not None:
                    record["segment_outputs"] = seg_outputs
                if entry.language == "bilingual":
                    record["bilingual_mode"] = bilingual_mode
                results.append(record)
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
            with open(output_path, "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--bilingual-mode",
        default="oracle_segments",
        choices=["oracle_segments", "full_concat"],
        help=(
            "Bilingual decoding strategy: "
            "'oracle_segments' (default) decodes each segment with forced language; "
            "'full_concat' concatenates all audio with pause_s silence and runs one "
            "Whisper pass (A2 experiment mode)."
        ),
    )
    args = parser.parse_args()

    transcriber = Transcriber(model_size=args.model, device=args.device)
    transcriber.transcribe_manifest(
        args.manifest,
        args.output,
        max_entries=args.max,
        bilingual_mode=args.bilingual_mode,
    )


if __name__ == "__main__":
    main()