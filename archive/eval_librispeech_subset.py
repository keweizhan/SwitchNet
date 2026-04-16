import re
import argparse
import subprocess
import sys
from pathlib import Path
import csv
from datetime import datetime


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # keep letters/digits/space/apostrophe
    text = re.sub(r"\s+", " ", text).strip()
    return text

def srt_to_text(srt_path: Path) -> str:
    lines = []
    for line in srt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        lines.append(line)
    return " ".join(lines)

def wer(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    # word-level Levenshtein
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len(r)][len(h)] / max(1, len(r))

UTT_RE = re.compile(r"^\d+-\d+-\d+$")  # e.g., 61-70968-0000

def load_trans(trans_path):
    m = {}
    for line in trans_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # Case A: ID first (LibriSpeech common format)
        if UTT_RE.match(parts[0]):
            utt_id = parts[0]
            transcript = " ".join(parts[1:])
        # Case B: ID last (some other datasets)
        elif UTT_RE.match(parts[-1]):
            utt_id = parts[-1]
            transcript = " ".join(parts[:-1])
        else:
            # can't detect, skip
            continue

        m[utt_id] = transcript
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="e.g., data/LibriSpeech/test-clean/61/70968")
    ap.add_argument("--model", default="base")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--out", default="eval_outputs")
    ap.add_argument("--report", default=None, help="CSV report path (default: <out>/report.csv)")
    ap.add_argument("--summary", default=None, help="Summary txt path (default: <out>/summary.txt)")
    args = ap.parse_args()

    d = Path(args.dir)
    outdir = Path(args.out)
    report_path = Path(args.report) if args.report else (outdir / "report.csv")
    summary_path = Path(args.summary) if args.summary else (outdir / "summary.txt")

    rows = []
    n_total = 0
    n_scored = 0
    n_skipped = 0

    outdir.mkdir(parents=True, exist_ok=True)

    trans_files = list(d.glob("*.trans.txt"))
    if not trans_files:
        raise FileNotFoundError(f"No *.trans.txt found in {d}")
    trans_map = load_trans(trans_files[0])

    flacs = sorted(d.glob("*.flac"))[: args.max]
    if not flacs:
        raise FileNotFoundError(f"No .flac found in {d}")


    wers = []
    for flac in flacs:
        n_total += 1
        utt_id = flac.stem  # e.g., 61-70968-0000
        ref = trans_map.get(utt_id)
        if ref is None:
            n_skipped += 1
            print(f"[skip] no ref for {utt_id}")
            continue

        srt_path = outdir / f"{utt_id}.srt"
        cmd = [
            sys.executable, "transcribe.py",
            str(flac), str(srt_path),
            "--model", args.model,
            "--language", "en"
        ]
        subprocess.run(cmd, check=True)

        hyp = srt_to_text(srt_path)
        w = wer(ref, hyp)
        n_scored += 1
        wers.append(w)
        
        rows.append({
            "utt_id": utt_id,
            "wer": w,
            "ref": ref,
            "hyp": hyp,
            "audio": str(flac)
        })
        print(f"{utt_id}  WER={w:.3f}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}\n")

    avg = None
    if wers:
        avg = sum(wers) / len(wers)
        print(f"AVG WER over {len(wers)} utt: {avg:.3f}")

    # ---- write CSV report ----
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "wer", "audio", "ref", "hyp"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ---- write summary ----
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"dir: {args.dir}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"max_requested: {args.max}\n")
        f.write(f"total_seen: {n_total}\n")
        f.write(f"scored: {n_scored}\n")
        f.write(f"skipped_no_ref: {n_skipped}\n")
        f.write(f"avg_wer: {avg if avg is not None else 'NA'}\n")
        if avg is not None:
            f.write(f"approx_word_accuracy(1-wer): {1-avg:.3f}\n")

    print(f"\nSaved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
