"""
transcribe_whisperx.py — WhisperX-based transcription for SwitchNet.

Mirrors the Transcriber interface from transcribe.py but uses WhisperX as
the backend.  Output format is identical to Transcriber.transcribe_manifest()
so results can be evaluated by the same evaluate_results() call and aggregated
by aggregate_results.py without any changes to those files.

Key differences vs. the openai-whisper Transcriber:
  - WhisperX uses batch decoding (faster on GPU).
  - Audio must be pre-loaded as a float32 numpy array at 16 kHz.
  - An extra "backend": "whisperx" field is added to each result record
    (ignored by evaluate_results).

Install:
    pip install whisperx
    # GPU: also requires torch with CUDA and, optionally, pyannote.audio for diarization.

Usage (API):
    from src.asr.transcribe_whisperx import WhisperXTranscriber
    t = WhisperXTranscriber(model_size="large-v3", device="cuda")
    t.transcribe_manifest(
        "data/manifests/bilingual_es-en_100.jsonl",
        output_path="results/wx_bilingual_es-en_100_oracle.jsonl",
    )
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.data.manifest import ManifestEntry, load_manifest


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio as a float32 numpy array at `sr` Hz mono.
    Delegates to whisperx.load_audio (uses ffmpeg internally) so the same
    codecs supported by openai-whisper are available here.
    """
    import whisperx  # deferred so import errors surface at call-site
    return whisperx.load_audio(audio_path)  # always returns float32 at 16 kHz


# ---------------------------------------------------------------------------
# WhisperX transcriber
# ---------------------------------------------------------------------------

class WhisperXTranscriber:
    """
    Drop-in replacement for Transcriber that uses WhisperX as the ASR backend.

    Public interface is intentionally kept identical to Transcriber so that
    run_eval_whisperx.py can call transcribe_manifest() and then hand the
    output directly to evaluate_results() without modification.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "whisperx is not installed. Run: pip install whisperx\n"
                "GPU users also need torch with CUDA support."
            ) from None

        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"Loading WhisperX {model_size} on {device} ({compute_type})...")
        self.model = whisperx.load_model(model_size, device, compute_type=compute_type)
        self.device = device
        self.model_size = model_size
        self._wx = whisperx

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _transcribe_audio(self, audio: np.ndarray, language: Optional[str]) -> str:
        """
        Run WhisperX inference on a pre-loaded float32 array.
        Returns the full hypothesis string (all segments joined).
        """
        result = self.model.transcribe(audio, batch_size=16, language=language)
        return " ".join(s["text"].strip() for s in result.get("segments", [])).strip()

    def _transcribe_file(self, audio_path: str, language: str) -> str:
        """Transcribe a single audio file with a fixed language."""
        audio = _load_audio(audio_path)
        return self._transcribe_audio(audio, language)

    # ------------------------------------------------------------------
    # Bilingual mode: oracle_segments
    # ------------------------------------------------------------------

    def _transcribe_bilingual_oracle(self, entry: ManifestEntry) -> tuple:
        """
        Oracle-segment mode: each segment decoded independently with its
        known language forced.  Mirrors Transcriber._transcribe_bilingual_oracle().

        Returns:
            (joined_hypothesis: str, segment_outputs: list[dict])
        """
        if not entry.segments:
            raise ValueError(
                f"Entry {entry.id} has language='bilingual' but no segments defined."
            )

        parts: list[str] = []
        segment_outputs: list[dict] = []

        for i, seg in enumerate(entry.segments):
            # Bilingual-concat entries: each segment is a separate audio file.
            audio_path = seg.audio_path if seg.audio_path else entry.audio_path
            hyp = self._transcribe_file(audio_path, seg.language)
            parts.append(hyp)
            segment_outputs.append({
                "position": i,
                "language": seg.language,
                "reference": seg.transcript,
                "hypothesis": hyp,
            })

        return " ".join(parts), segment_outputs

    # ------------------------------------------------------------------
    # Bilingual mode: full_concat
    # ------------------------------------------------------------------

    def _transcribe_full_concat(self, entry: ManifestEntry) -> tuple:
        """
        Full-concat mode: concatenate all segment audio with pause_s silence
        gaps and decode in one WhisperX pass with language=None.
        Mirrors Transcriber._transcribe_full_concat().

        Returns:
            (hypothesis: str, [])   # empty segment list — no per-seg oracle
        """
        if not entry.segments:
            raise ValueError(
                f"Entry {entry.id} has language='bilingual' but no segments defined."
            )

        SR = 16000
        chunks: list[np.ndarray] = []

        for i, seg in enumerate(entry.segments):
            audio_path = seg.audio_path if seg.audio_path else entry.audio_path
            chunks.append(_load_audio(audio_path, SR))
            if i < len(entry.segments) - 1 and seg.pause_s > 0.0:
                chunks.append(np.zeros(int(SR * seg.pause_s), dtype=np.float32))

        full_audio = np.concatenate(chunks)
        hyp = self._transcribe_audio(full_audio, language=None)
        return hyp, []

    # ------------------------------------------------------------------
    # Manifest-level runner
    # ------------------------------------------------------------------

    def transcribe_manifest(
        self,
        manifest_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_entries: Optional[int] = None,
        bilingual_mode: str = "oracle_segments",
    ) -> List[dict]:
        """
        Transcribe all entries in a manifest using WhisperX.

        Output format is identical to Transcriber.transcribe_manifest() so
        results can be passed directly to evaluate_results().  Each record
        includes an extra "backend": "whisperx" field (ignored by evaluate_results).

        Args:
            manifest_path:  Path to input JSONL manifest.
            output_path:    If given, write results as JSONL to this path.
            max_entries:    Cap number of entries (useful for debug/smoke tests).
            bilingual_mode: "oracle_segments" (default) or "full_concat".
        """
        entries = load_manifest(manifest_path)
        if max_entries:
            entries = entries[:max_entries]

        results: List[dict] = []

        for i, entry in enumerate(entries):
            print(f"[{i+1}/{len(entries)}] {entry.id} ({entry.language})", end=" ... ", flush=True)
            t0 = time.time()
            try:
                seg_outputs = None
                if entry.language == "bilingual":
                    if bilingual_mode == "full_concat":
                        hyp, seg_outputs = self._transcribe_full_concat(entry)
                    else:
                        hyp, seg_outputs = self._transcribe_bilingual_oracle(entry)
                else:
                    hyp = self._transcribe_file(entry.audio_path, entry.language)

                elapsed = time.time() - t0
                rtf = round(elapsed / entry.duration_s, 3) if entry.duration_s else None
                print(f"done ({elapsed:.1f}s)")

                record: dict = {
                    "id": entry.id,
                    "language": entry.language,
                    "reference": entry.transcript,
                    "hypothesis": hyp,
                    "rtf": rtf,
                    "backend": "whisperx",
                }
                if seg_outputs is not None:
                    record["segment_outputs"] = seg_outputs
                if entry.language == "bilingual":
                    record["bilingual_mode"] = bilingual_mode
                results.append(record)

            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "id": entry.id,
                    "language": entry.language,
                    "reference": entry.transcript,
                    "hypothesis": "",
                    "error": str(exc),
                    "rtf": None,
                    "backend": "whisperx",
                })

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\nResults written to {output_path}")

        return results
