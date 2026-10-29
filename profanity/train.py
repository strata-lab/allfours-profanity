"""Run upstream's trainer (train.py, unchanged) on the compact-text profanity config.

    python profanity/train.py            # = python train.py -c configs/Profanity_compact_text_toxic_bert.json -e 2 --num_workers 0

Upstream trains with Trainer(deterministic=True). On a GPU (Apple MPS or CUDA) an op in BERT's backward pass may have
no deterministic implementation and raise, so on a GPU this launcher makes PyTorch WARN about such ops instead (every
other op stays deterministic). On CUDA, deterministic cuBLAS also needs CUBLAS_WORKSPACE_CONFIG, set here unless
already set. On CPU it changes nothing. Extra arguments are passed through to train.py.
"""
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # read when CUDA's cuBLAS starts; harmless elsewhere

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # upstream's train.py imports src.* and reads config paths relative to the root
sys.path.insert(0, str(ROOT))

if torch.backends.mps.is_available() or torch.cuda.is_available():
    _use = torch.use_deterministic_algorithms
    torch.use_deterministic_algorithms = lambda mode, *, warn_only=False: _use(mode, warn_only=True)

import train  # noqa: E402

sys.argv = [sys.argv[0], "-c", "configs/Profanity_compact_text_toxic_bert.json", "-e", "2", "--num_workers", "0", *sys.argv[1:]]
train.cli_main()
