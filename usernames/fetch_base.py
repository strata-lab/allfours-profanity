"""Fetch the base model, unitary/toxic-bert (Apache-2.0; Detoxify's "original" model), at a pinned revision into
usernames/base/toxic-bert, which configs/Username_moderation_toxic_bert.json names as its model_type. Pinning it
here keeps training reproducible without changing upstream's trainer, which loads model_type by name.

    python usernames/fetch_base.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "unitary/toxic-bert"
REVISION = "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"
DEST = Path(__file__).resolve().parent / "base" / "toxic-bert"

if __name__ == "__main__":
    snapshot_download(REPO, revision=REVISION, local_dir=DEST,
                      allow_patterns=["config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json",
                                      "special_tokens_map.json"])
    (DEST / "REVISION").write_text(f"{REPO}@{REVISION}\n")
    print(f"{REPO}@{REVISION} -> {DEST}")
