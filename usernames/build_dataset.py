"""Build the training, dev and test sets of handles from the term lists and word lists (seeded: same files, same rows).

    python usernames/build_dataset.py                                   # writes usernames/dataset/{train,dev,test}.tsv
    python usernames/build_dataset.py --sample-web2 /usr/share/dict/words   # refresh words/web2_sample.txt (rarely)

Each row is ``handle <TAB> label <TAB> group <TAB> term``; label 1 = offensive.

* Offensive handles render a listed term the way players write them: camelCase or underscores, upper/lower/title
  case, a number tail (69, 420, 88), a prefix or suffix word (BigX, XLord), "xX..Xx", and obfuscations the
  normaliser does NOT undo (u->v, ck->k, f->ph, a doubled letter, a dropped vowel) as well as leet it does.
* Innocent handles render everyday words, first names and dictionary words the same ways, leet included, so the
  model cannot learn "leet means bad". **Hard negatives** are dictionary words that contain a short term
  (cocktail, assassin, therapist): the Scunthorpe problem.
* The terms are split 80/10/10. A dev/test term's handles never appear in training ("unseen_term" rows measure
  how the model generalises to a term it was never shown); training terms also get fresh renderings in dev/test
  ("seen_term"). Words are split the same way.
* Held out of all three: the hand-written eval sets (usernames/eval), so they are an honest test.
* Every name the game's generator can make (detoxify.usernames.generated_names) is a TRAINING negative: the first-launch name must never be
  refused, so that is a guarantee the model is trained to, not a held-out measurement (the report says so).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere: the repo root holds detoxify/
from detoxify import usernames as U   # noqa: E402
from detoxify.usernames import forms  # noqa: E402

DATA = Path(__file__).resolve().parent   # usernames/
SEED = 181
VALID = re.compile(r"^[A-Za-z0-9_]{3,20}$")
PREFIXES = ["Big", "Lil", "Mr", "Ms", "The", "Da", "Real", "Its", "Im", "Ur", "Yo", "King", "Queen", "Captain"]
SUFFIXES = ["Lord", "King", "Boy", "Gyal", "Man", "Face", "Head", "Master", "Pro", "Gamer", "Lover", "Queen", "Ting"]
TAILS = ["69", "420", "666", "88", "1", "7", "13", "99", "2026", "868", "123", "007"]
LEET_OUT = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def _bucket(s: str) -> str:
    """train / dev / test, by a stable hash, 80/10/10."""
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def _case(words: list[str], rng: random.Random) -> list[str]:
    style = rng.random()
    if style < 0.45:
        return [w.capitalize() for w in words]
    if style < 0.7:
        return [w.lower() for w in words]
    if style < 0.85:
        return [w.upper() for w in words]
    return [w.capitalize() if i % 2 == 0 else w.lower() for i, w in enumerate(words)]


def _obfuscate(word: str, rng: random.Random, harsh: bool) -> str:
    """Spell `word` the way a player dodging a filter might (harsh) or just stylise it (leet only)."""
    w = word
    if rng.random() < 0.35:   # leet, which the normaliser undoes
        w = "".join(LEET_OUT[c] if c in LEET_OUT and rng.random() < 0.5 else c for c in w)
    if harsh:
        r = rng.random()
        if r < 0.12:
            w = w.replace("u", "v", 1)
        elif r < 0.22:
            w = w.replace("ck", "k", 1) if "ck" in w else w.replace("c", "k", 1)
        elif r < 0.28:
            w = w.replace("f", "ph", 1)
        elif r < 0.40 and len(w) > 3:
            i = rng.randrange(1, len(w))
            w = w[:i] + w[i - 1] + w[i:]               # a doubled letter
        elif r < 0.50 and len(w) > 4:
            vowels = [i for i, c in enumerate(w) if c in "aeiou" and i > 0]
            if vowels:
                i = rng.choice(vowels)
                w = w[:i] + w[i + 1:]                    # a dropped vowel
        elif r < 0.55:
            w = w.replace("s", "z", 1)
        elif r < 0.60 and w.endswith("er"):
            w = w[:-2] + "a"
    return w


def render(words: list[str], rng: random.Random, harsh: bool) -> str | None:
    """One handle from `words`, or None if it would not fit the username rules."""
    ws = [_obfuscate(w, rng, harsh) for w in words]
    r = rng.random()
    if r < 0.25:
        ws = [rng.choice(PREFIXES).lower()] + ws
    elif r < 0.45:
        ws = ws + [rng.choice(SUFFIXES).lower()]
    ws = _case(ws, rng)
    joiner = "_" if rng.random() < 0.35 else ""
    handle = joiner.join(ws)
    r = rng.random()
    if r < 0.30:
        handle += ("_" if rng.random() < 0.4 else "") + rng.choice(TAILS)
    elif r < 0.36:
        handle = "xX" + handle + "Xx"
    elif r < 0.40:
        handle = "_".join(handle)                      # F_U_C_K
    return handle if VALID.match(handle) else None


def _renderings(words: list[str], n: int, rng: random.Random, harsh: bool) -> list[str]:
    out: set[str] = set()
    for _ in range(n * 6):
        if len(out) >= n:
            break
        h = render(words, rng, harsh)
        if h is not None:
            out.add(h)
    return sorted(out)


def word_lines(name: str) -> list[str]:
    out: list[str] = []
    for ln in (DATA / "words" / name).read_text(encoding="utf-8").splitlines():
        if not ln.startswith("#"):
            out.extend(w.lower() for w in ln.split() if w.isalpha())
    return out


def eval_handles() -> set[str]:
    out: set[str] = set()
    for name in ("innocent.txt", "offensive.txt"):
        out.update(ln.strip() for ln in (DATA / "eval" / name).read_text().splitlines()
                   if ln.strip() and not ln.startswith("#"))
    return out


def _held_out_forms() -> set[str]:
    """Readings of every handle kept out of training: the hand-written eval sets."""
    return {f.replace(" ", "") for h in eval_handles() for f in forms(h)}


def _is_term_word(word: str, keys: set[str]) -> bool:
    """`word` is a listed term or an inflection of one (cocks, fucking, bitches)."""
    k = U.key_of(word)
    if k in keys:
        return True
    for suffix in ("s", "es", "er", "ers", "ing", "ed", "y", "ie", "ies", "in"):
        if k.endswith(suffix) and k[: -len(suffix)] in keys:
            return True
    return False


def _contains_long_term(word: str, keys: set[str]) -> bool:
    return any(len(k) >= 5 and k in word for k in keys)


def build(seed: int = SEED) -> dict[str, list[tuple[str, int, str, str]]]:
    rng = random.Random(seed)
    term_list = U.load_terms()
    keys = {t.key for t in term_list}
    held = _held_out_forms()
    rows: dict[str, list[tuple[str, int, str, str]]] = {"train": [], "dev": [], "test": []}

    def add(split: str, handle: str, label: int, group: str, term: str) -> None:
        if label == 0 and any(f.replace(" ", "") in held for f in forms(handle)):
            return
        if label == 1 and handle in held:
            return
        rows[split].append((handle, label, group, term))

    # offensive: every term, in its own split; training terms also get fresh renderings in dev/test
    for t in term_list:
        words = t.text.split()
        split = _bucket("term:" + t.key)
        for h in _renderings(words, 22 if split == "train" else 12, rng, harsh=True):
            add(split, h, 1, "seen_term" if split == "train" else "unseen_term", t.text)
        if split == "train":
            for h in _renderings(words, 3, random.Random(f"{seed}:{t.key}"), harsh=True):
                add(rng.choice(("dev", "test")), h, 1, "seen_term", t.text)

    # innocent: handle words + first names + dictionary words, alone, in pairs, and with a hard-negative word
    handle_words = sorted(set(w for w in word_lines("handle_words.txt") if not _is_term_word(w, keys)))
    names = sorted(set(w for w in word_lines("propernames.txt") if not _is_term_word(w, keys)))
    dictionary = sorted(set(w for w in word_lines("web2_sample.txt")
                            if not _is_term_word(w, keys) and not _contains_long_term(w, keys)))
    hard = sorted(w for w in dictionary if any(len(k) <= 4 and k in w for k in keys))
    plain = [w for w in dictionary if w not in set(hard)]

    def innocent(pool: list[str], group: str, n_each: int, pair_with: list[str] | None = None) -> None:
        for w in pool:
            split = _bucket("word:" + w)
            for _ in range(n_each):
                ws = [w] if pair_with is None or rng.random() < 0.4 else (
                    [w, rng.choice(pair_with)] if rng.random() < 0.5 else [rng.choice(pair_with), w])
                h = render(ws, rng, harsh=False)
                if h is not None and not any(U.key_of(x) in keys for x in ws):
                    add(split, h, 0, group, "")

    innocent(hard, "hard_negative", 5, handle_words)
    innocent(handle_words, "negative", 25, handle_words + names)
    innocent(names, "negative", 3, handle_words)
    innocent(plain, "negative", 2, handle_words)

    # the generator's names: every one, plain and with collision digits, always in training
    for h in U.generated_names():
        rows["train"].append((h, 0, "generator", ""))

    for split in rows:   # one row per handle; a handle that came out both ways keeps the offensive label
        best: dict[str, tuple[str, int, str, str]] = {}
        for r in rows[split]:
            if r[0] not in best or r[1] > best[r[0]][1]:
                best[r[0]] = r
        rows[split] = sorted(best.values())
    return rows


def write(rows: dict[str, list[tuple[str, int, str, str]]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        with open(out / f"{split}.tsv", "w", encoding="utf-8") as f:
            f.write("handle\tlabel\tgroup\tterm\n")
            for r in rs:
                f.write("\t".join(map(str, r)) + "\n")


def read(path: Path) -> list[tuple[str, int, str, str]]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        h, label, group, term = ln.split("\t")
        rows.append((h, int(label), group, term))
    return rows


def sample_web2(src: Path, n: int = 12000, seed: int = SEED) -> list[str]:
    """A deterministic sample of a web2 word list: lowercase words of 3-10 letters, plus every word that contains a
    short (<= 4 letter) term, the hard negatives. Offensive words themselves are filtered at build time."""
    words = sorted({w.strip() for w in src.read_text().splitlines() if re.fullmatch(r"[a-z]{3,10}", w.strip())})
    keys = {t.key for t in U.load_terms()}
    hard = [w for w in words if any(len(k) <= 4 and k in w for k in keys)]
    rest = [w for w in words if w not in set(hard)]
    return sorted(set(hard) | set(random.Random(seed).sample(rest, n)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA / "dataset")
    ap.add_argument("--sample-web2", type=Path, help="refresh data/words/web2_sample.txt from this word list")
    a = ap.parse_args(argv)
    if a.sample_web2:
        words = sample_web2(a.sample_web2)
        (DATA / "words" / "web2_sample.txt").write_text(
            "# A deterministic sample of Webster's Second International (1934; public domain), see usernames/LICENSES.md.\n"
            + "\n".join(words) + "\n")
        print(f"web2_sample.txt: {len(words)} words")
        return
    rows = build()
    write(rows, a.out)
    for split, rs in rows.items():
        pos = sum(r[1] for r in rs)
        print(f"{split}: {len(rs)} handles ({pos} offensive, {len(rs) - pos} innocent)", file=sys.stderr)


if __name__ == "__main__":
    main()
