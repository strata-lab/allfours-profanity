# Game usernames: the All Fours addition to Detoxify (ALL-181)

This fork extends Detoxify to one more kind of text: **game usernames**. These are short handles with no spaces,
3–20 characters of `[A-Za-z0-9_]`, such as `FuckYou_69`, `Fvck_U` or `Bumba_Clot`. It also adds the **Trinidad and
Tobago / Caribbean** terms that comment datasets don't cover. All Fours' moderation service
([shottah/allfours](https://github.com/shottah/allfours), `allfours-moderator/`) vendors this repo as a submodule and
asks it about every rename.

Upstream's design is kept as it is. Nothing in `train.py`, `convert_weights.py`, `detoxify/detoxify.py`, the existing
configs or the existing data loaders changes. The fork only adds:

| Added | What |
| --- | --- |
| `configs/Username_moderation_toxic_bert.json` | A training config for upstream's `train.py`: toxic-bert, six heads, username data |
| `src/data_loaders.py` → `UsernameData` | A dataset class (appended) that reads `dataset/*.tsv`; only `toxicity` is labelled, the other heads are masked (-1) |
| `detoxify/usernames.py` | The handle normaliser, the whole-word term match, the game's generated names, and `UsernameModerator` (the verdict) |
| `usernames/` | This folder: term lists, word lists, eval sets, dataset, scripts, report |
| `tests/test_usernames.py` | Tests for all of the above |

## Why handles need normalising

Models trained on comments do badly on handles as written, and `reports/eval.md` measures by how much. So the model
is trained and run on a handle's **readings**, joined into one string (`detoxify.usernames.text`):

- split on `_`, digits and camelCase, then lowercase: `FuckYou_69` → `fuck you | fuckyou`;
- leetspeak back to letters: `0→o 1→i 3→e 4→a 5→s 7→t @→a`. A digit is read both ways: `69` in `FuckYou_69` is a
  number, the `1` in `Sh1t` is a letter;
- `xX…Xx` decoration is dropped.

A handle is refused when **either** of these is true:

- the model's `toxicity` score reaches the release's threshold;
- a listed term is a **whole word** of it: a single word, adjacent words run together, or the whole handle run
  together. Terms are never matched as substrings, so `Scunthorpe`, `Cocktail` and `Grasshopper` pass.

## Data (licences in `LICENSES.md`)

| Path | What |
| --- | --- |
| `terms/ldnoobw_en.txt` | LDNOOBW English list, CC BY 4.0, unmodified |
| `terms/caribbean_tt.txt` | Trinidad and Tobago / Caribbean terms, approved by the All Fours owner 2026-10-28 |
| `terms/exclusions.txt` | LDNOOBW entries not used, and why |
| `words/` | Innocent vocabulary: everyday handle words, first names, a public-domain dictionary sample |
| `eval/` | Hand-written held-out handles, innocent and offensive. Never trained on |
| `dataset/` | The train/dev/test sets, built by `build_dataset.py` (seeded) |

The training handles are synthetic:

- **Offensive** handles render each term the way players write them: casing, number tails, prefix and suffix words,
  `xX…Xx`, leet, and spellings the normaliser does not undo (`fvck`, `phuck`, doubled letters, dropped vowels).
- **Innocent** handles use the same styles.
- **Hard negatives** are dictionary words that contain a short term (cocktail, assassin, therapist).

The terms are split 80/10/10, so the test set includes terms the model never saw. The hand-written sets are kept out
of training. Every name All Fours' first-launch generator can make is a **training** negative, on purpose: that name
must never be refused. The generator's word lists are read from the game's `game/account/username_gen.gd`, found in
the superproject, in a sibling `../allfours` checkout, or at `$ALLFOURS_USERNAME_GEN`.

## Workflow (`make -C usernames <target>`)

| Target | What it does |
| --- | --- |
| `setup` | `.venv` with upstream's dev extras, plus transformers 4.x |
| `base` | Fetch `unitary/toxic-bert` at the pinned revision into `base/`. The config names it by path, so upstream's trainer needs no change |
| `dataset` | Rebuild `dataset/` |
| `train` | Upstream's `train.py` on the username config, 2 epochs. On Apple GPUs (MPS), `train.py` here turns one non-deterministic op into a warning; nothing else differs |
| `release` | Upstream's `convert_weights.py` on the newest checkpoint into `models/current/`, then pick the threshold on dev: best F1, at most 2% of innocent dev handles refused, no generated name refused |
| `eval` | Write `reports/eval.md`; fails if a generated or hand-written innocent name is refused |
| `test` | `tests/test_usernames.py` |
| `publish` | Upload `models/current` to `gs://allfours-moderator-models` and print the hashes All Fours pins |

Weights (`base/`, `models/`) are never committed.
