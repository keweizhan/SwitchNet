"""
prepare_clean_demo_data.py — Filter bilingual manifest to entries that are
fully available locally for demo use.

An entry is kept when:
  1. Every segment's audio_path exists on the local filesystem
     (resolved via the same four-strategy logic used in the Streamlit app).
  2. The entry ID appears in the Whisper result JSONL.
  3. The entry ID appears in the WhisperX result JSONL.

The filtered manifest and matching JSONL slices are written to the output
paths you specify (directories are created automatically).

Usage
-----
python scripts/prepare_clean_demo_data.py \\
    --manifest            data/manifests/bilingual_es-en_50.jsonl \\
    --whisper-jsonl       results/bilingual_es-en_50_large_cpu.jsonl \\
    --whisperx-jsonl      results/wx_bilingual_es-en_50_large_cpu.jsonl \\
    --out-manifest        data/demo/bilingual_es-en_clean_demo.jsonl \\
    --out-whisper-jsonl   results/demo/bilingual_es-en_clean_whisper.jsonl \\
    --out-whisperx-jsonl  results/demo/bilingual_es-en_clean_whisperx.jsonl \\
    --max-entries 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Audio-path resolution (mirrors logic in streamlit_app.py)
# ---------------------------------------------------------------------------

def resolve_audio_path(path_str: str) -> Path | None:
    """Try four strategies to locate an audio file regardless of OS / machine.

    Strategy 1 — direct: path exists as-is.
    Strategy 2 — data/ anchor (forward slashes): normalise separators, find
                  the first occurrence of "data/" and rebuild from repo root.
    Strategy 3 — data\\ anchor (back slashes): same with Windows separator.
    Strategy 4 — SWITCHNET_DATA_ROOT env var: replace the leading drive / UNC
                  prefix up to the first "data" component.

    Returns the resolved ``Path`` on the first hit, or ``None`` if no candidate
    exists locally.
    """
    p = Path(path_str)
    if p.exists():
        return p

    # Normalise to forward slashes for anchor search
    norm = path_str.replace("\\", "/")

    for anchor in ("data/", "data\\"):
        idx = path_str.find(anchor)
        if idx == -1:
            idx = norm.find(anchor.replace("\\", "/"))
        if idx != -1:
            rel = norm[idx:]          # e.g. "data/LibriSpeech/..."
            candidate = REPO_ROOT / Path(rel)
            if candidate.exists():
                return candidate

    # SWITCHNET_DATA_ROOT override
    data_root = os.environ.get("SWITCHNET_DATA_ROOT")
    if data_root:
        norm2 = path_str.replace("\\", "/")
        idx = norm2.find("data/")
        if idx != -1:
            rel = norm2[idx:]
            candidate = Path(data_root) / Path(rel)
            if candidate.exists():
                return candidate

    return None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl_lines(path: Path) -> list[dict]:
    """Return all non-empty JSON lines as dicts."""
    lines: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def _load_jsonl_index(path: Path) -> dict[str, dict]:
    """Return {entry_id: record} for a results JSONL."""
    return {r["id"]: r for r in _load_jsonl_lines(path)}


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def _all_audio_present(entry: dict) -> tuple[bool, list[str]]:
    """Check every segment audio_path.  Returns (ok, list_of_missing)."""
    segments = entry.get("segments", [])
    if not segments:
        # Monolingual: check top-level audio_path
        ap = entry.get("audio_path", "")
        if ap == "__multi__" or not ap:
            return False, [ap]
        resolved = resolve_audio_path(ap)
        if resolved is None:
            return False, [ap]
        return True, []

    missing: list[str] = []
    for seg in segments:
        ap = seg.get("audio_path", "")
        if not ap:
            missing.append("(empty)")
            continue
        if resolve_audio_path(ap) is None:
            missing.append(ap)
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary_from_jsonl(
    jsonl_path: Path,
    manifest_path: Optional[Path] = None,
) -> dict:
    """Compute a *_summary.json-compatible dict from a results JSONL file.

    Delegates entirely to ``evaluate_results`` from ``src.asr.evaluate``,
    which computes corpus WER/MER, per-entry metrics, per-language breakdowns,
    and switch-point WER (when a manifest is provided).

    The returned dict has the same shape as files written by ``run_eval.py``:
    ``overall``, ``per_language``, ``per_entry``, ``switch_point``, …
    """
    from src.asr.evaluate import evaluate_results

    return evaluate_results(
        results_path=jsonl_path,
        manifest_path=manifest_path,
        output_path=None,   # we write it ourselves with ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a bilingual manifest to locally-available entries."
    )
    parser.add_argument("--manifest",            required=True,
                        help="Source JSONL manifest (bilingual_es-en_50.jsonl)")
    parser.add_argument("--whisper-jsonl",        required=True,
                        help="Whisper results JSONL")
    parser.add_argument("--whisperx-jsonl",       required=True,
                        help="WhisperX results JSONL")
    parser.add_argument("--out-manifest",         required=True,
                        help="Output manifest JSONL (e.g. data/demo/...)")
    parser.add_argument("--out-whisper-jsonl",    required=True,
                        help="Output Whisper JSONL slice")
    parser.add_argument("--out-whisperx-jsonl",   required=True,
                        help="Output WhisperX JSONL slice")
    parser.add_argument("--max-entries",          type=int, default=50,
                        help="Maximum entries to keep (default: 50)")
    args = parser.parse_args()

    manifest_path   = REPO_ROOT / args.manifest
    whisper_path    = REPO_ROOT / args.whisper_jsonl
    whisperx_path   = REPO_ROOT / args.whisperx_jsonl
    out_manifest    = REPO_ROOT / args.out_manifest
    out_whisper     = REPO_ROOT / args.out_whisper_jsonl
    out_whisperx    = REPO_ROOT / args.out_whisperx_jsonl

    for p in (manifest_path, whisper_path, whisperx_path):
        if not p.exists():
            sys.exit(f"ERROR: input file not found: {p}")

    # Create output directories
    for p in (out_manifest, out_whisper, out_whisperx):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print(f"Loading manifest       : {manifest_path.name}")
    manifest_lines = _load_jsonl_lines(manifest_path)
    print(f"Loading Whisper JSONL  : {whisper_path.name}")
    whisper_idx    = _load_jsonl_index(whisper_path)
    print(f"Loading WhisperX JSONL : {whisperx_path.name}")
    whisperx_idx   = _load_jsonl_index(whisperx_path)

    print(
        f"\nSource: {len(manifest_lines)} manifest entries, "
        f"{len(whisper_idx)} Whisper records, "
        f"{len(whisperx_idx)} WhisperX records\n"
    )

    kept:    list[dict] = []
    skipped: list[tuple[str, str]] = []

    for entry in manifest_lines:
        eid = entry.get("id", "")

        # Filter 1: in Whisper results
        if eid not in whisper_idx:
            skipped.append((eid, "missing from Whisper JSONL"))
            continue

        # Filter 2: in WhisperX results
        if eid not in whisperx_idx:
            skipped.append((eid, "missing from WhisperX JSONL"))
            continue

        # Filter 3: all audio files present locally
        ok, missing = _all_audio_present(entry)
        if not ok:
            reason = "missing audio: " + "; ".join(
                Path(m).name if m else "(empty)" for m in missing[:3]
            )
            skipped.append((eid, reason))
            continue

        kept.append(entry)
        if len(kept) >= args.max_entries:
            print(f"  [limit] reached --max-entries={args.max_entries}, stopping early.")
            break

    # Report
    print(f"Kept   : {len(kept)} entries")
    print(f"Skipped: {len(skipped)} entries")
    if skipped:
        for eid, reason in skipped:
            short = eid.split("_", 3)[-1][:50] if eid else eid
            print(f"  SKIP  {short!r:<52}  ({reason})")

    if not kept:
        sys.exit("No entries passed all filters — nothing written.")

    # Write outputs
    def _write_jsonl(path: Path, lines: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in lines:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote  : {path}  ({len(lines)} lines)")

    _write_jsonl(out_manifest, kept)
    _write_jsonl(out_whisper,  [whisper_idx[e["id"]] for e in kept])
    _write_jsonl(out_whisperx, [whisperx_idx[e["id"]] for e in kept])

    # Generate summary JSON files
    print()
    for out_jsonl, label in [(out_whisper, "Whisper"), (out_whisperx, "WhisperX")]:
        out_summary = out_jsonl.with_name(out_jsonl.stem + "_summary.json")
        summary = _compute_summary_from_jsonl(out_jsonl, manifest_path=out_manifest)
        summary["source"] = "summary_generated_from_jsonl"
        summary["backend"] = label.lower()
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        n = summary["overall"]["n_utterances"]
        w = summary["overall"]["wer"]
        m = summary["overall"]["mer"]
        print(f"Summary: {out_summary}  (n={n}, WER={w:.4f}, MER={m:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
