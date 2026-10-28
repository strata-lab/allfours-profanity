# Profanity in compact text: the addition to Detoxify

This fork extends Detoxify to one more kind of text: **compact text**. That means handles, tags and short
identifiers with no spaces, such as `FuckYou_69`, `Fvck_U` or `Bumba_Clot`. It also adds the **Trinidad and Tobago /
Caribbean** terms that comment datasets don't cover.

It decides one thing: **is this text profane?** It has no rules about what makes a valid username, tag or slug in
any particular product. Each caller applies its own rules (length, characters, uniqueness) and uses this as one
check among them, so the same model serves any project. The first caller is the card game All Fours: its moderation
service ([shottah/allfours](https://github.com/shottah/allfours), `allfours-moderator/`) asks it about every username
change.

Upstream's design is kept as it is. Nothing in `train.py`, `convert_weights.py`, `detoxify/detoxify.py`, the existing
configs or the existing data loaders changes. The fork only adds:

| Added | What |
| --- | --- |
| `configs/Profanity_compact_text_toxic_bert.json` | A training config for upstream's `train.py`: toxic-bert, six heads, compact-text data |
| `src/data_loaders.py` → `ProfanityData` | A dataset class (appended) that reads `dataset/*.tsv`; only `toxicity` is labelled, the other heads are masked (-1) |
| `detoxify/profanity.py` | The normaliser, the whole-word term match, and `Profanity`: `check(texts)` → `Verdict(profane, score, term)` |
| `profanity/` | This folder: term lists, word lists, eval sets, dataset, scripts, report |
| `tests/test_profanity.py` | Tests for all of the above |

```python
from detoxify.profanity import Profanity

p = Profanity("profanity/models/current")
p.check(["FuckYou_69", "Scunthorpe"])
# [Verdict(profane=True, score=0.99, term='fuck you'), Verdict(profane=False, score=0.01, term=None)]
```

## Why compact text needs normalising

Models trained on comments do badly on compact text as written, and `reports/eval.md` measures by how much. So the
model is trained and run on a text's **readings**, joined into one string (`detoxify.profanity.text`):

- split on `_`, digits and camelCase, then lowercase: `FuckYou_69` → `fuck you | fuckyou`;
- leetspeak back to letters: `0→o 1→i 3→e 4→a 5→s 7→t @→a`. A digit is read both ways: `69` in `FuckYou_69` is a
  number, the `1` in `Sh1t` is a letter;
- `xX…Xx` decoration is dropped.

A text is **profane** when either of these is true:

- the model's `toxicity` score reaches the release's threshold;
- a listed term is a **whole word** of it: a single word, adjacent words run together, or the whole text run
  together. Terms are never matched as substrings, so `Scunthorpe`, `Cocktail` and `Grasshopper` are not flagged.

## Data (licences in `LICENSES.md`)

| Path | What |
| --- | --- |
| `terms/ldnoobw_en.txt` | LDNOOBW English list, CC BY 4.0, unmodified |
| `terms/caribbean_tt.txt` | Trinidad and Tobago / Caribbean terms, approved by the All Fours owner 2026-10-28 |
| `terms/exclusions.txt` | LDNOOBW entries not used, and why |
| `words/` | Innocent vocabulary: everyday handle words, first names, a public-domain dictionary sample |
| `words/must_allow.txt` | Innocent texts that must **never** be flagged. Any project may add its own; today it holds All Fours' generated names |
| `eval/` | Hand-written held-out texts, innocent and profane. Never trained on |
| `dataset/` | The train/dev/test sets, built by `build_dataset.py` (seeded) |

The training texts are synthetic:

- **Profane** texts render each term the way people write handles: casing, number tails, prefix and suffix words,
  `xX…Xx`, leet, and spellings the normaliser does not undo (`fvck`, `phuck`, doubled letters, dropped vowels).
- **Innocent** texts use the same styles.
- **Hard negatives** are dictionary words that contain a short term (cocktail, assassin, therapist).

The terms are split 80/10/10, so the test set includes terms the model never saw. The hand-written sets are kept out
of training. `must_allow.txt` is always a training negative, and the release threshold is picked above every score
on it.

## Workflow (`make -C profanity <target>`)

| Target | What it does |
| --- | --- |
| `setup` | `.venv` with upstream's dev extras, plus transformers 4.x |
| `base` | Fetch `unitary/toxic-bert` at the pinned revision into `base/`. The config names it by path, so upstream's trainer needs no change |
| `dataset` | Rebuild `dataset/` |
| `train` | Upstream's `train.py` on the config, 2 epochs. On Apple GPUs (MPS), `train.py` here turns one non-deterministic op into a warning; nothing else differs |
| `release` | Upstream's `convert_weights.py` on the newest checkpoint into `models/current/`, then pick the threshold on dev: best F1, at most 2% of innocent dev texts flagged, no must-allow text flagged |
| `eval` | Write `reports/eval.md`; fails if a must-allow or hand-written innocent text is flagged |
| `test` | `tests/test_profanity.py` |
| `publish` | Upload `models/current` to `gs://allfours-moderator-models` and print the hashes a consumer pins |

Weights (`base/`, `models/`) are never committed.
