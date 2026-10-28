"""Turn a training checkpoint into a released model directory (profanity/models/current) and pick its threshold.

    python profanity/release.py [--checkpoint saved/.../epoch=1-step=2638.ckpt]

1. Converts the Lightning checkpoint with upstream's convert_weights.py (default: the newest run's last epoch).
2. Copies the base model's config and tokenizer into hf/, so the model loads offline.
3. Picks the threshold on the DEV set: the best F1 among thresholds that refuse at most 2% of innocent dev handles
   and none of words/must_allow.txt.
4. Writes profanity.json: checkpoint file, threshold, its dev scores, the dataset hash, the source checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detoxify import profanity as U  # noqa: E402

HERE = ROOT / "profanity"
BASE = HERE / "base" / "toxic-bert"
OUT = HERE / "models" / "current"
RUNS = ROOT / "saved" / "Profanity_compact_text_toxic_bert" / "lightning_logs"
FPR_CAP = 0.02   # at most 2% of innocent dev texts flagged


def newest_checkpoint() -> Path:
    ckpts = sorted(RUNS.glob("version_*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit(f"no checkpoint under {RUNS}: train first (python profanity/train.py)")
    return ckpts[-1]


def dataset_hash() -> str:
    h = hashlib.sha256()
    for split in ("train", "dev", "test"):
        h.update((HERE / "dataset" / f"{split}.tsv").read_bytes())
    return h.hexdigest()


def read(split: str) -> list[tuple[str, int]]:
    rows = []
    for ln in (HERE / "dataset" / f"{split}.tsv").read_text().splitlines()[1:]:
        h, label, *_ = ln.split("\t")
        rows.append((h, int(label)))
    return rows


def pick_threshold(scores: list[float], labels: list[int], floor: float) -> tuple[float, float, float]:
    """(threshold, F1, FPR): best F1 with FPR <= FPR_CAP among thresholds above `floor` (the highest must-allow
    score, so none of those is flagged)."""
    negatives = sum(1 for y in labels if y == 0) or 1
    best = (0.99, -1.0, 0.0)
    for i in range(1, 100):
        t = i / 100
        if t <= floor:
            continue
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, labels) if s < t and y)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if fp / negatives <= FPR_CAP and f1 > best[1]:
            best = (t, f1, fp / negatives)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=None)
    a = ap.parse_args()
    src = a.checkpoint or newest_checkpoint()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "hf").mkdir(parents=True)
    subprocess.run([sys.executable, str(ROOT / "convert_weights.py"), "--checkpoint", str(src),
                    "--save_to", str(OUT / "model.ckpt")], check=True, cwd=ROOT)
    for extra in OUT.glob("model-*.ckpt"):   # convert_weights.py also writes a hash-named copy; keep one file
        extra.unlink()
    for f in ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(BASE / f, OUT / "hf" / f)

    # threshold: provisional meta so the model loads, then score dev and the must-allow texts
    (OUT / U.META).write_text(json.dumps({"checkpoint": "model.ckpt", "threshold": 1.0}))
    m = U.Profanity(OUT)
    dev = read("dev")
    dev_scores = batched(m, [h for h, _ in dev])
    allow_max = max(batched(m, U.must_allow()))
    t, f1, fpr = pick_threshold(dev_scores, [y for _, y in dev], allow_max)
    meta = {
        "checkpoint": "model.ckpt",
        "threshold": t,
        "dev_f1": round(f1, 4),
        "dev_fpr": round(fpr, 4),
        "dev_fpr_cap": FPR_CAP,
        "must_allow_max_score": round(allow_max, 4),
        "base": (BASE / "REVISION").read_text().strip(),
        "dataset_sha256": dataset_hash(),
        "source_checkpoint": str(src.relative_to(ROOT)),
        "config": "configs/Profanity_compact_text_toxic_bert.json",
    }
    (OUT / U.META).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def batched(m: U.Profanity, handles: list[str], n: int = 256) -> list[float]:
    out: list[float] = []
    for i in range(0, len(handles), n):
        out.extend(m.scores(handles[i:i + n]))
    return out


if __name__ == "__main__":
    main()
