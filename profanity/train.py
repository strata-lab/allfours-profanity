"""Run upstream's trainer (train.py, unchanged) on the compact-text profanity config.

    python profanity/train.py            # = python train.py -c configs/Profanity_compact_text_toxic_bert.json -e 2 --num_workers 0

Upstream trains with Trainer(deterministic=True). On Apple-silicon GPUs (MPS) one op in BERT's backward pass has no
deterministic implementation and raises, so on MPS this launcher makes PyTorch WARN about such ops instead (every other
op stays deterministic). On CUDA or CPU it changes nothing. Extra arguments are passed through to train.py.
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # upstream's train.py imports src.* and reads config paths relative to the root
sys.path.insert(0, str(ROOT))

if torch.backends.mps.is_available():
    _use = torch.use_deterministic_algorithms
    torch.use_deterministic_algorithms = lambda mode, *, warn_only=False: _use(mode, warn_only=True)

import train  # noqa: E402

sys.argv = [sys.argv[0], "-c", "configs/Profanity_compact_text_toxic_bert.json", "-e", "2", "--num_workers", "0", *sys.argv[1:]]
train.cli_main()
