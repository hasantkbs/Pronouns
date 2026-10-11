# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.sentence_corrector import SentenceCorrector

REAL_LEXICON = ROOT / "data" / "lexicon" / "tr_50k.txt"

# Siklik sirasina gore azalan, gercek dosyayla ayni bicim
_SMALL_LEXICON = """\
bir 1000
ben 990
şu 950
geldim 900
iyi 880
aldım 800
aldim 700
su 600
bugün 500
okula 400
yaram 300
kır 200
kir 100
"""


@pytest.fixture
def lexicon(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text(_SMALL_LEXICON, encoding="utf-8")
    return path


def _corrector(lexicon, vocabulary=("yardım", "kampanya", "geri bildirim")):
    return SentenceCorrector(list(vocabulary), lexicon)


def test_known_words_are_kept(lexicon):
    assert _corrector(lexicon).correct("ben bugün okula geldim") == "Ben bugün okula geldim."


def test_unknown_word_is_corrected_to_nearest(lexicon):
    assert _corrector(lexicon).correct("gldim") == "Geldim."


def test_ascii_variant_is_never_a_correction(lexicon):
    # "aldm" -> "aldım" ve "aldim" ikisine de 1 uzakta; ASCII varyanti aday olamaz
    assert _corrector(lexicon).correct("aldm") == "Aldım."


def test_ascii_variant_is_still_recognised(lexicon):
    # "su" ve "kir" gecerli kelimeler: daha sik "şu"/"kır" olsa da dokunulmaz
    assert _corrector(lexicon).correct("su kir") == "Su kir."


def test_user_vocabulary_wins_a_tie(lexicon):
    # "yardm": "yardım" (Furkan) ve "yaram" (genel) ikisi de 1 uzakta
    assert _corrector(lexicon).correct("yardm") == "Yardım."


def test_word_without_close_candidate_is_kept(lexicon):
    assert _corrector(lexicon).correct("xqzvw") == "Xqzvw."


def test_short_unknown_word_is_kept(lexicon):
    assert _corrector(lexicon).correct("zq bir") == "Zq bir."


def test_multi_word_vocabulary_entry_is_known(lexicon):
    assert _corrector(lexicon).correct("geri bildirim") == "Geri bildirim."


def test_question_particle_gets_question_mark(lexicon):
    assert _corrector(lexicon).correct("iyi misin") == "İyi misin?"


def test_question_word_gets_question_mark(lexicon):
    assert _corrector(lexicon).correct("ne aldım") == "Ne aldım?"


def test_dotted_capital_i(lexicon):
    assert _corrector(lexicon).correct("iyi") == "İyi."


def test_whisper_style_input_is_normalised(lexicon):
    assert _corrector(lexicon).correct("Ben bugün okula geldim.") == "Ben bugün okula geldim."


def test_proper_noun_with_apostrophe_is_kept(lexicon):
    assert _corrector(lexicon).correct("ben istanbul'a geldim,") == "Ben istanbul'a geldim."
    assert _corrector(lexicon).correct("İstanbul’a geldim") == "İstanbul'a geldim."


def test_digits_are_kept(lexicon):
    assert _corrector(lexicon).correct("3 bir") == "3 bir."


def test_empty_or_punctuation_only_gives_empty(lexicon):
    corrector = _corrector(lexicon)
    assert corrector.correct("") == ""
    assert corrector.correct(None) == ""
    assert corrector.correct(" ... !") == ""


def test_missing_lexicon_disables_word_correction(tmp_path):
    corrector = SentenceCorrector(["okul"], tmp_path / "yok.txt")
    # Yalnizca Furkan sozlugune zorla yaklastirma yapilmaz ("okula" -> "okul" olmaz)
    assert corrector.correct("okula gldim") == "Okula gldim."


@pytest.mark.skipif(not REAL_LEXICON.exists(), reason="data/lexicon/tr_50k.txt yok")
@pytest.mark.parametrize("sentence, expected", [
    ("ben bugün okula geldim", "Ben bugün okula geldim."),
    ("yarın arkadaşımla sinemaya gideceğiz", "Yarın arkadaşımla sinemaya gideceğiz."),
    ("su istiyorum", "Su istiyorum."),
    ("gldim", "Geldim."),
    ("kampnya", "Kampanya."),
    ("tşekkür ederim", "Teşekkür ederim."),
    ("su kir isim", "Su kir isim."),
])
def test_real_lexicon_regressions(sentence, expected):
    corrector = SentenceCorrector(["okul", "geldi", "istiyor", "yakın", "arkadaş", "kampanya"], REAL_LEXICON)
    assert corrector.correct(sentence) == expected


# --- Son inceleme bulgulari: anlami degistiren duzeltmeler ---

def test_negation_is_never_added(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text("kaşınmıyor 100\n", encoding="utf-8")
    # "kaşınıyor" -> "kaşınmıyor" tek harf ekleme olsa da olumsuzluk ekler
    assert SentenceCorrector([], path).correct("kaşınıyor") == "Kaşınıyor."


def test_substitution_is_never_applied(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text("ilacını 100\n", encoding="utf-8")
    # "ilacımı" -> "ilacını" tek harf degistirme; "benim" -> "senin" olur
    assert SentenceCorrector([], path).correct("ilacımı") == "İlacımı."


def test_capitalised_word_mid_sentence_is_kept(lexicon):
    assert _corrector(lexicon).correct("dün Ahmet geldim") == "Dün Ahmet geldim."


@pytest.mark.parametrize("sentence, expected", [
    ("3,5 kilo", "3,5 kilo."),
    ("saat 14:30", "Saat 14:30."),
    ("%50 bir", "%50 bir."),
    ("bir 3.", "Bir 3."),
])
def test_numbers_keep_their_punctuation(lexicon, sentence, expected):
    assert _corrector(lexicon).correct(sentence) == expected


@pytest.mark.skipif(not REAL_LEXICON.exists(), reason="data/lexicon/tr_50k.txt yok")
@pytest.mark.parametrize("sentence, expected", [
    ("ağrımıyor", "Ağrımıyor."),
    ("kaşınmıyor", "Kaşınmıyor."),
    ("üşümüyorum", "Üşümüyorum."),
    ("susamadım", "Susamadım."),
    ("acıkmadım", "Acıkmadım."),
    ("ilacımı almadım", "İlacımı almadım."),
    ("yorulmadım", "Yorulmadım."),
    ("ben Furkan", "Ben Furkan."),
    ("Furkan geldi", "Furkan geldi."),
    ("Bursa güzel", "Bursa güzel."),
    ("Antalya", "Antalya."),
    ("Emine geldi", "Emine geldi."),
])
def test_real_lexicon_never_changes_meaning(sentence, expected):
    corrector = SentenceCorrector(["okul", "geldi", "istiyor", "kampanya", "yardım"], REAL_LEXICON)
    assert corrector.correct(sentence) == expected
