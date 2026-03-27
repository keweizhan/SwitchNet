"""
preprocess.py -- Simple, explainable audio preprocessing for SwitchNet ablations.

Provides three conditions for controlled comparison before Whisper transcription:

  raw          No processing. Audio is passed to Whisper unchanged (baseline).

  normalize    Peak amplitude normalization: scale audio so the loudest sample
               reaches +/-1.0.  Removes gross level differences between files
               without changing spectral shape.

  preemphasis  First-order high-pass pre-emphasis filter (coef=0.97).
               Boosts high-frequency content relative to low.  A standard
               speech preprocessing step that can improve ASR for speakers
               whose recordings have low high-frequency energy.
               Equation: y[n] = x[n] - 0.97 * x[n-1]

Usage:
    from src.audio.preprocess import get_preprocessor
    preprocessor = get_preprocessor("preemphasis")  # or "normalize" or "raw"
    audio_out = preprocessor(audio_in, sr=16000)    # audio_in: float32 numpy array

    # None is returned for "raw" -- callers should treat None as identity.
    preprocessor = get_preprocessor("raw")   # -> None
    if preprocessor is not None:
        audio = preprocessor(audio, sr)

All functions:
  - accept float32 numpy arrays at any sample rate
  - return float32 numpy arrays of the same length
  - are deterministic and stateless
  - depend only on numpy (no scipy, no external models)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

# Supported mode names -- used for CLI choices validation.
MODES = ("raw", "normalize", "preemphasis")


# ---------------------------------------------------------------------------
# Preprocessing functions
# Each has signature: (audio: np.ndarray, sr: int) -> np.ndarray
# ---------------------------------------------------------------------------

def normalize_amplitude(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Peak amplitude normalization.

    Scales the waveform so the maximum absolute sample value is 1.0.
    If the input is silent (all zeros), it is returned unchanged.

    This is the simplest level normalization.  It removes gross loudness
    differences between recordings but does not change spectral shape.
    Whisper already applies its own internal normalization, so the practical
    effect is mainly to standardize very quiet or clipped inputs.
    """
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        return (audio / peak).astype(np.float32)
    return audio


def apply_preemphasis(audio: np.ndarray, sr: int, coef: float = 0.97) -> np.ndarray:
    """
    First-order FIR pre-emphasis filter.

    y[0] = x[0]
    y[n] = x[n] - coef * x[n-1]  for n > 0

    This is a standard speech preprocessing step.  It applies a gentle
    high-pass tilt to the spectrum, boosting frequencies above roughly
    sr * (1 - coef) / (2*pi) Hz.  At coef=0.97 and sr=16000 this is
    approximately 77 Hz, so essentially all speech energy is passed but
    the low-frequency rolloff is flattened.

    Practical effect: compensates for the natural spectral tilt of voiced
    speech (more energy at low frequencies) and can improve recognition of
    fricatives and other high-frequency consonants.

    Args:
        coef: pre-emphasis coefficient (default 0.97, typical range 0.95-0.99).
    """
    if len(audio) < 2:
        return audio
    out = audio.copy()
    out[1:] = audio[1:] - coef * audio[:-1]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_preprocessor(mode: str) -> Optional[Callable[[np.ndarray, int], np.ndarray]]:
    """
    Return a preprocessing callable for the given mode, or None for "raw".

    Args:
        mode: one of "raw", "normalize", "preemphasis".

    Returns:
        A callable (audio, sr) -> audio for non-raw modes.
        None for "raw" -- callers should treat None as identity (no-op).

    Raises:
        ValueError for unrecognised modes.
    """
    if mode == "raw":
        return None
    if mode == "normalize":
        return normalize_amplitude
    if mode == "preemphasis":
        return apply_preemphasis
    raise ValueError(
        f"Unknown preprocessing mode {mode!r}. "
        f"Choose from: {', '.join(MODES)}"
    )
