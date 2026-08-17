"""
test_normalizer.py — unit tests for ArabicNormalizer.

Tests cover:
* alef variant normalisation
* ya / alif maqsura normalisation
* tatweel removal
* diacritic preservation by default
* optional diacritic stripping
* whitespace cleanup
* mode-dependent punctuation handling
"""
import pytest

from sarftok.config import SarfTokConfig
from sarftok.normalizer import ArabicNormalizer, normalize_text


class TestAlefNormalisation:
    def test_alef_with_hamza_above(self):
        assert normalize_text("\u0623\u0647\u0644", mode="classical_strict") == "\u0627\u0647\u0644"

    def test_alef_with_hamza_below(self):
        assert normalize_text("\u0625\u0633\u0644\u0627\u0645", mode="classical_strict") == "\u0627\u0633\u0644\u0627\u0645"

    def test_alef_with_madda(self):
        assert normalize_text("\u0622\u062F\u0627\u0628", mode="classical_strict") == "\u0627\u062F\u0627\u0628"

    def test_alef_wasla(self):
        assert normalize_text("\u0671\u0644\u0643\u062A\u0627\u0628", mode="classical_strict") == "\u0627\u0644\u0643\u062A\u0627\u0628"


class TestYaNormalisation:
    def test_alif_maqsura_to_ya(self):
        # ى → ي
        result = normalize_text("\u0639\u0644\u0649", mode="classical_strict")
        assert "\u064A" in result  # contains ya
        assert "\u0649" not in result  # no alif maqsura

    def test_farsi_yeh_to_ya(self):
        result = normalize_text("\u06CC", mode="classical_strict")
        assert result == "\u064A"


class TestTatweel:
    def test_tatweel_removed(self):
        result = normalize_text("\u0643\u0640\u0640\u062A\u0627\u0628", mode="classical_strict")
        assert "\u0640" not in result
        assert len(result) == 4  # كتاب


class TestDiacritics:
    def test_diacritics_preserved_by_default(self):
        text_with_harakat = "\u0643\u064E\u062A\u064E\u0628\u064E"
        result = normalize_text(text_with_harakat, mode="classical_strict")
        assert "\u064E" in result  # fatha preserved

    def test_diacritics_stripped_when_requested(self):
        text_with_harakat = "\u0643\u064E\u062A\u064E\u0628\u064E"
        result = normalize_text(text_with_harakat, mode="classical_strict", strip_diacritics=True)
        assert "\u064E" not in result
        assert result == "\u0643\u062A\u0628"


class TestWhitespace:
    def test_multiple_spaces_collapsed(self):
        result = normalize_text("كتب   الطالب", mode="classical_strict")
        assert "   " not in result
        assert "كتب" in result

    def test_leading_trailing_stripped(self):
        result = normalize_text("  كتب  ", mode="classical_strict")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_newlines_replaced(self):
        result = normalize_text("كتب\nالدرس", mode="classical_strict")
        assert "\n" not in result


class TestModes:
    def test_classical_strict_removes_western_punct(self):
        result = normalize_text("كتب! الدرس.", mode="classical_strict")
        assert "!" not in result
        assert "." not in result

    def test_classical_soft_keeps_arabic_punct(self):
        result = normalize_text("كتب، الدرس؟", mode="classical_soft")
        assert "،" in result
        assert "؟" in result

    def test_msa_soft_keeps_arabic_punct(self):
        result = normalize_text("كتب؟ الدرس", mode="msa_soft")
        assert "؟" in result


class TestEmptyInput:
    def test_empty_string(self):
        assert normalize_text("", mode="classical_strict") == ""

    def test_whitespace_only(self):
        assert normalize_text("   ", mode="classical_strict") == ""


class TestNormalizerClass:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown normalisation mode"):
            ArabicNormalizer(mode="invalid")

    def test_normalise_method(self):
        n = ArabicNormalizer(mode="classical_strict")
        result = n.normalise("\u0623\u0647\u0644")
        assert result == "\u0627\u0647\u0644"

    def test_classical_config_preserves_alef_and_ya_by_default(self):
        config = SarfTokConfig()
        n = ArabicNormalizer(
            mode=config.norm_mode,
            strip_diacritics=config.strip_diacritics,
            normalise_alef=config.normalise_alef,
            normalise_ya=config.normalise_ya,
        )
        assert n.normalise("أعلى") == "أعلى"
