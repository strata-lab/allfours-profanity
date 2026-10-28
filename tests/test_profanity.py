"""Profanity in compact text (fork addition): the normaliser, the whole-word term match, and the guarantee that no
must-allow text (profanity/words/must_allow.txt) is flagged.

The model check runs when a released model is present (profanity/models/current, or $ALLFOURS_PROFANITY_MODEL), and is
SKIPPED, loudly, otherwise. Run: pytest tests/test_profanity.py
"""
import os
from pathlib import Path

import pytest

from detoxify import profanity as U
from detoxify.profanity import forms, leet, split_words, text, tokens

MODEL = Path(os.environ.get("ALLFOURS_PROFANITY_MODEL", Path(__file__).resolve().parents[1] / "profanity/models/current"))


def test_split_on_underscore_digits_and_camel_case():
    assert split_words("FuckYou_69") == ["Fuck", "You"]
    assert split_words("Brave_Soca") == ["Brave", "Soca"]
    assert split_words("GoldenJack4321") == ["Golden", "Jack"]
    assert split_words("HTMLParser") == ["HTML", "Parser"]
    assert split_words("xXslayerXx") == ["x", "Xslayer", "Xx"]
    assert split_words("lowercase") == ["lowercase"]
    assert split_words("123") == []


def test_leet_maps_digits_and_at_back_to_letters():
    assert leet("f4ck") == "fack"
    assert leet("sh1t") == "shit"
    assert leet("b1tch") == "bitch"
    assert leet("a55") == "ass"
    assert leet("7est") == "test"
    assert leet("3vil") == "evil"
    assert leet("p0rn") == "porn"
    assert leet("@ss") == "ass"
    assert leet("ok") == "ok"


def test_tokens_lowercase_and_drop_number_tails():
    assert tokens("FuckYou_69") == ["fuck", "you"]
    assert tokens("Fvck_U") == ["fvck", "u"]


def test_forms_try_digits_both_as_separators_and_as_leet():
    # a number tail is a separator; a digit inside a word is leet
    assert "fuck you" in forms("FuckYou_69")
    assert "shit head" in forms("Sh1tHead")
    assert "bitch" in forms("B1tch")
    assert "ass hole" in forms("A55Hole") or "asshole" in forms("A55Hole")
    # an innocent name keeps a clean form
    assert "golden jack" in forms("GoldenJack4321")
    assert "cocktail" in forms("Cocktail")


def test_forms_are_unique_and_never_empty_for_a_real_name():
    f = forms("Lucky_Ace")
    assert f == sorted(set(f))
    assert f
    assert forms("12345") == ["12345"]


def test_forms_strip_the_xx_decoration():
    assert "cunt" in forms("xXcuntXx")
    assert "slayer" in forms("xXSlayerXx")
    assert "xxx" in forms("xxx")       # no decoration to strip: kept whole


def test_text_joins_every_reading():
    assert text("FuckYou_69") == "fuck you | fuckyou"


M = U.Matcher(U.load_terms())


def test_both_lists_load_and_exclusions_apply():
    keys = {t.key for t in U.load_terms()}
    assert "fuck" in keys and "bumbaclot" in keys and "kissmeass" in keys and "battyboy" in keys
    assert "xx" not in keys          # handle decoration, excluded
    assert "xxx" in keys


def test_whole_words_match_through_case_leet_and_separators():
    for h in ("FuckYou_69", "fuck_you", "SH1T", "Bumba_Clot", "Kiss_Me_Ass", "KissMeAss", "BattyBoy", "xXcuntXx",
              "A55Hole", "F_U_C_K"):
        assert M.match(h) is not None, h


def test_substrings_do_not_match():
    for h in ("Cocktail", "Scunthorpe", "Grasshopper", "Dickens", "Assassin", "Therapist", "Brass_Band", "Totem",
              "Titanic", "Peacock", "xXSlayerXx", "Lol_Lol", "Bullet", "Coolio", "Rascal"):
        assert M.match(h) is None, h


def test_must_allow_list_loads_and_has_no_listed_term_as_a_word():
    allow = U.must_allow()
    assert "GoldenJack" in allow and len(allow) == len(set(allow))
    m = U.Matcher(U.load_terms())
    assert [(t, m.match(t).text) for t in allow if m.match(t)] == []


@pytest.mark.skipif(not (MODEL / U.META).exists(), reason=f"no released model at {MODEL}")
def test_the_model_flags_no_must_allow_text():
    allow = U.must_allow()
    flagged = [(t, round(v.score, 3)) for t, v in zip(allow, U.Profanity(MODEL).check(allow)) if v.profane]
    assert flagged == []
