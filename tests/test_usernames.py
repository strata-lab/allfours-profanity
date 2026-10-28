"""Game usernames (All Fours fork addition, ALL-181): the handle normaliser, the whole-word term match, and the
guarantee that every name All Fours' first-launch generator can make is allowed.

The model check runs when a released model is present (usernames/models/current, or $ALLFOURS_PROFANITY_MODEL), and is
SKIPPED, loudly, otherwise. Run: pytest tests/test_usernames.py
"""
import os
from pathlib import Path

import pytest

from detoxify import usernames as U
from detoxify.usernames import forms, leet, split_words, text, tokens

MODEL = Path(os.environ.get("ALLFOURS_PROFANITY_MODEL", Path(__file__).resolve().parents[1] / "usernames/models/current"))


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


def test_generator_lists_parse():
    adjectives, nouns = U.generator_words()
    assert "Golden" in adjectives and "Jack" in nouns
    names = U.generated_names()
    assert len(names) == 3 * len(adjectives) * len(nouns)


def test_no_generated_name_contains_a_listed_term_as_a_word():
    m = U.Matcher(U.load_terms())
    assert [(n, m.match(n).text) for n in U.generated_names() if m.match(n)] == []


@pytest.mark.skipif(not (MODEL / U.META).exists(), reason=f"no released model at {MODEL}")
def test_the_model_allows_every_generated_name():
    names = U.generated_names()
    refused = [(n, round(v.score, 3)) for n, v in zip(names, U.UsernameModerator(MODEL).check(names)) if not v.allowed]
    assert refused == []
