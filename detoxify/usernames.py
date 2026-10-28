"""Game usernames (All Fours fork addition, ALL-181): decide whether a short handle may be a player's username.

A username is 3-20 characters of ``[A-Za-z0-9_]`` with no spaces (``FuckYou_69``, ``Fvck_U``, ``Bumba_Clot``), and
models trained on comments do badly on handles as written. So a handle is normalised before the model sees it:

* split on ``_``, on digits and on camelCase, then lowercase (``FuckYou_69`` -> ``fuck you``);
* leetspeak back to letters: ``0->o 1->i 3->e 4->a 5->s 7->t @->a``. A digit is ambiguous (``69`` in ``FuckYou_69`` is a
  number, ``1`` in ``Sh1t`` is a letter), so both readings are kept;
* ``xX...Xx`` decoration around a handle is dropped.

``text(handle)`` joins every reading into the one string the model is trained and run on (``fuck you | fuckyou``),
so training uses upstream's trainer unchanged (one text per example).

A handle is refused when the model's toxicity score reaches the threshold, OR a listed term (usernames/terms) is a
whole word of it. Terms are never matched as substrings, so "Scunthorpe", "Cocktail" and "Grasshopper" pass.

Scoring loads a checkpoint converted by upstream's ``convert_weights.py`` through upstream's
``get_model_and_tokenizer`` (see UsernameModerator).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS_DIR = ROOT / "usernames" / "terms"
TERM_FILES = ("ldnoobw_en.txt", "caribbean_tt.txt")
META = "usernames.json"   # next to the converted checkpoint: threshold, checkpoint file name, provenance

# --- normalising a handle ---------------------------------------------------------------------------------------

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
# A word boundary inside a run of letters: aB (lower to upper), or ABc (the last capital of an acronym starts a word).
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[^A-Za-z]+")
# xXnameXx: the x's around a handle are decoration, not part of a word.
_XX = re.compile(r"^[xX]{2,}_?(.+?)_?[xX]{2,}$")
# A run of leet digits with a letter on both sides (Sh1t, A55Hole), or at the very start before a letter (5hit).
# A run at the end (GoldenJack4321, FuckYou_69) is a number tail, not a letter.
_LEET_RUN = re.compile(r"(?<=[A-Za-z])[013457@]+(?=[A-Za-z])|^[013457@]+(?=[A-Za-z])")


def leet(s: str) -> str:
    return s.translate(LEET)


def split_words(handle: str) -> list[str]:
    """Words of `handle`, original case kept: split on anything that is not a letter, then on camelCase."""
    out: list[str] = []
    for chunk in _SEPARATORS.split(handle):
        out.extend(w for w in _CAMEL.split(chunk) if w)
    return out


def tokens(handle: str) -> list[str]:
    return [w.lower() for w in split_words(handle)]


def _leet_inside(handle: str) -> str:
    """Map the leet digits that sit inside a word; leave number tails alone. A mapped letter is lowercase unless the
    letters around it are capitals, so camelCase still splits where it should (Sh1tHead -> ShitHead)."""

    def sub(m: re.Match) -> str:
        mapped = leet(m.group())
        before = handle[m.start() - 1] if m.start() > 0 else ""
        after = handle[m.end()] if m.end() < len(handle) else ""
        return mapped.upper() if before.isupper() and after.isupper() else mapped

    return _LEET_RUN.sub(sub, handle)


def forms(handle: str) -> list[str]:
    """Every reading of `handle`, sorted and unique: digits as separators and as leet, without and with the xX..Xx
    decoration, each as words (``fuck you``) and run together (``fuckyou``)."""
    seen: set[str] = set()
    m = _XX.match(handle)
    bases = (handle, m.group(1)) if m else (handle,)
    for variant in [v for b in bases for v in (b, _leet_inside(b))]:
        words = tokens(variant)
        if words:
            seen.add(" ".join(words))
            seen.add("".join(words))
    return sorted(seen) if seen else [handle.lower()]


def text(handle: str) -> str:
    """The one string the model is trained and run on: every reading of the handle."""
    return " | ".join(forms(handle))


# --- the term lists ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    text: str     # as listed, lowercase ("kiss me ass")
    key: str      # letters only, run together ("kissmeass"): what a handle's words are compared with
    source: str   # the file it came from


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def key_of(s: str) -> str:
    return re.sub(r"[^a-z]", "", leet(s.lower()))


def load_terms(terms_dir: Path = TERMS_DIR) -> list[Term]:
    """Both lists, less the entries in exclusions.txt."""
    skip = {ln.split("\t")[0].strip() for ln in _lines(terms_dir / "exclusions.txt")}
    out: dict[str, Term] = {}
    for name in TERM_FILES:
        for t in _lines(terms_dir / name):
            t = t.lower()
            key = key_of(t)
            if t in skip or len(key) < 2:
                continue
            out.setdefault(key, Term(t, key, name))
    return sorted(out.values(), key=lambda t: t.key)


class Matcher:
    """Whole-word term match over every reading of a handle: each word, each run of adjacent words ("kiss me ass"),
    and the handle run together. Never a substring."""

    def __init__(self, terms: list[Term]):
        self.by_key = {t.key: t for t in terms}

    def match(self, handle: str) -> Term | None:
        for form in forms(handle):
            words = form.split(" ")
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    hit = self.by_key.get("".join(words[i:j]))
                    if hit is not None:
                        return hit
        return None


# --- the game's generated names ---------------------------------------------------------------------------------


def find_username_gen() -> Path:
    """All Fours' game/account/username_gen.gd: $ALLFOURS_USERNAME_GEN, else the superproject (this repo vendored as
    allfours/allfours-moderator/allfours-profanity), else a sibling checkout (~/Github/allfours)."""
    env = os.environ.get("ALLFOURS_USERNAME_GEN")
    if env:
        return Path(env)
    candidates = [ROOT.parents[1] / "game/account/username_gen.gd", ROOT.parent / "allfours/game/account/username_gen.gd"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("game/account/username_gen.gd not found: set ALLFOURS_USERNAME_GEN to the allfours "
                            f"checkout's copy (looked in {', '.join(map(str, candidates))})")


def generator_words(gd_file: Path | None = None) -> tuple[list[str], list[str]]:
    """The ADJECTIVES and NOUNS arrays of username_gen.gd, read from the GDScript itself."""
    path = gd_file or find_username_gen()
    src = path.read_text(encoding="utf-8")

    def array(name: str) -> list[str]:
        m = re.search(name + r"\s*:\s*PackedStringArray\s*=\s*\[(.*?)\]", src, re.S)
        if m is None:
            raise ValueError(f"{path}: no {name} array")
        return re.findall(r'"([^"]+)"', m.group(1))

    adjectives, nouns = array("ADJECTIVES"), array("NOUNS")
    if not adjectives or not nouns:
        raise ValueError(f"{path}: empty word list")
    return adjectives, nouns


def generated_names(gd_file: Path | None = None) -> list[str]:
    """Every ADJECTIVE + NOUN, plus each with collision-retry digit tails (the game appends 10..9999) chosen to cover
    every digit and the leet digits next to a letter."""
    adjectives, nouns = generator_words(gd_file)
    names = [a + n for a in adjectives for n in nouns]
    tails = ("10", "13", "34", "57", "69", "88", "420", "666", "1337", "4321", "7777", "9999")
    return names + [n + tails[i % len(tails)] for i, n in enumerate(names)] + [n + "69" for n in names]


# --- the decision -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    score: float
    term: str | None   # the listed term the handle contains as a word, if any


class UsernameModerator:
    """A released model directory (usernames/release.py writes it): model.ckpt (upstream's converted format), hf/
    (config + tokenizer, for offline loading) and usernames.json (threshold, provenance).

    It loads through upstream's ``get_model_and_tokenizer`` with ``model_type`` pointed at hf/: the checkpoint was
    trained from the local base path (usernames/base), which does not exist where the model is served."""

    def __init__(self, model_dir: Path | str, device: str = "cpu", terms_dir: Path = TERMS_DIR):
        import torch

        from detoxify.detoxify import get_model_and_tokenizer

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / META).read_text())
        self.threshold = float(self.meta["threshold"])
        loaded = torch.load(model_dir / self.meta["checkpoint"], map_location=device, weights_only=False)
        args = dict(loaded["config"]["arch"]["args"], model_type=str(model_dir / "hf"))
        self.model, self.tokenizer = get_model_and_tokenizer(**args, state_dict=loaded["state_dict"],
                                                             huggingface_config_path=str(model_dir / "hf"))
        self.model.to(device).eval()
        self.device = device
        self.index = loaded["config"]["dataset"]["args"]["classes"].index("toxicity")
        self.torch = torch
        self.matcher = Matcher(load_terms(terms_dir))

    def scores(self, handles: list[str]) -> list[float]:
        with self.torch.no_grad():
            enc = self.tokenizer([text(h) for h in handles], return_tensors="pt", truncation=True,
                                 padding=True).to(self.device)
            return self.torch.sigmoid(self.model(**enc)[0][:, self.index]).tolist()

    def check(self, handles: list[str]) -> list[Verdict]:
        out = []
        for h, s in zip(handles, self.scores(handles)):
            hit = self.matcher.match(h)
            out.append(Verdict(allowed=s < self.threshold and hit is None, score=s, term=hit.text if hit else None))
        return out
