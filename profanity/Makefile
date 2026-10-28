# Profanity in compact text (fork addition). Run from the repo root as `make -C profanity <target>`, or from here.
# Every target runs upstream's tooling where there is one (train.py, convert_weights.py) and only adds the profanity data.
ROOT := $(abspath ..)
PY ?= $(ROOT)/.venv/bin/python
BUCKET ?= gs://allfours-moderator-models
CLOUDSDK_PYTHON ?= /opt/homebrew/bin/python3.12
export CLOUDSDK_PYTHON

.PHONY: help setup base dataset train release eval test publish
help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup: ## Create .venv with upstream's dev extras (pytorch-lightning, datasets, ...) and transformers 4.x
	cd $(ROOT) && uv venv -q --python 3.12 .venv && uv pip install -q --python .venv -e ".[dev]" "transformers>=4.57,<5" "safetensors>=0.6"

base: ## Fetch the base model, unitary/toxic-bert at a pinned revision, into base/toxic-bert
	$(PY) fetch_base.py

dataset: ## Rebuild dataset/ from the term and word lists (seeded; after changing a list)
	$(PY) build_dataset.py

train: ## Fine-tune with upstream's train.py on configs/Profanity_compact_text_toxic_bert.json (2 epochs, ~20 min on an M-series Mac)
	$(PY) train.py

release: ## Convert the newest checkpoint with upstream's convert_weights.py into models/current and pick the threshold on dev (none of words/must_allow.txt flagged)
	$(PY) release.py

eval: ## Write reports/eval.md from models/current; fails if any must-allow or hand-written innocent text is flagged
	$(PY) evaluate.py

test: ## The profanity tests (tests/test_profanity.py); the model check runs once models/current exists
	cd $(ROOT) && $(PY) -m pytest -q -rs tests/test_profanity.py

publish: eval ## Upload models/current to $(BUCKET) (current/ and a copy by dataset hash); prints the hashes a consumer pins (All Fours: allfours-moderator/models.sha256)
	@ver="$$($(PY) -c 'import json; print(json.load(open("models/current/profanity.json"))["dataset_sha256"][:12])')"; \
	gcloud storage cp -r models/current/* "$(BUCKET)/$$ver/" && gcloud storage cp -r models/current/* "$(BUCKET)/current/" && \
	echo "published $$ver. Pin these hashes (All Fours: allfours-moderator/models.sha256):" && \
	cd models/current && shasum -a 256 model.ckpt profanity.json hf/*
