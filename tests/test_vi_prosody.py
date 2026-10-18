"""Chính tả + nhịp ngắt tiếng Việt trước G2P (vieneu_utils.vi_prosody)."""

import pytest

from vieneu_utils.vi_prosody import add_phrase_breaks, finalize_chunk, fix_spelling
from vieneu_utils.phonemize_text import (
    normalize_to_chunks_v3_with_gaps,
    phonemize_batch,
    phonemize_text,
)


@pytest.mark.parametrize("old,new", [
    ("họa", "hoạ"), ("Tai họa", "Tai hoạ"), ("đòa", "đoà"), ("lõa", "loã"),
    ("khỏe", "khoẻ"), ("tùy", "tuỳ"), ("qúy", "quý"),
    ("họat", "hoạt"), ("tòan", "toàn"), ("tóan", "toán"),
    ("thưở", "thuở"), ("Thưở", "Thuở"),
])
def test_fix_spelling_moves_tone_to_where_g2p_reads_it(old, new):
    assert fix_spelling(old) == new


@pytest.mark.parametrize("word", [
    "của", "mùa", "người", "khuya", "quyết", "giường", "được", "khuỷu",
    "hoà", "thuỷ", "Đắk", "hôm nay",
])
def test_fix_spelling_leaves_correct_words_alone(word):
    assert fix_spelling(word) == word


def test_fix_spelling_currency_per_unit_and_honorifics():
    assert fix_spelling("giá 3.500.000đ/chỉ") == "giá 3.500.000 đồng một chỉ"
    assert fix_spelling("Mr. Smith gặp Dr. Lee.") == "Mr Smith gặp Dr Lee."


def test_hoa_is_one_syllable_after_fix():
    assert phonemize_text("tai họa") == phonemize_text("tai hoạ")


def test_phrase_break_before_connective_in_long_clause():
    s = ("hắn đứng đó lặng lẽ nhìn về phía chân trời xa xăm nơi mặt trời đang dần "
         "khuất sau những dãy núi trùng điệp mà lòng thì nặng trĩu bao nỗi niềm khó tả.")
    assert add_phrase_breaks(s) == (
        "hắn đứng đó lặng lẽ nhìn về phía chân trời xa xăm, nơi mặt trời đang dần "
        "khuất sau những dãy núi trùng điệp, mà lòng thì nặng trĩu bao nỗi niềm khó tả.")


def test_phrase_break_keeps_pairs_together():
    s = ("hắn lặng lẽ ngồi bên cửa sổ suốt cả buổi chiều hôm ấy sau khi nhận "
         "được bức thư cuối cùng của nàng.")
    assert "hôm ấy, sau khi nhận" in add_phrase_breaks(s)


@pytest.mark.parametrize("s", [
    "anh đi đi nhưng nhớ về sớm.",                                 # mệnh đề ngắn
    "ta sẽ đi về nơi mà ngày xưa hai đứa từng hẹn ước dưới gốc cây đa đầu làng.",
    "cô ấy đã cố gắng rất nhiều, nhưng vẫn không thể nào quên được anh.",
    "suốt những năm tháng dài đằng đẵng ấy cô vẫn chờ mà thôi.",  # vế phải quá ngắn
])
def test_phrase_break_skips_short_or_bound_connectives(s):
    assert add_phrase_breaks(s) == s


def test_phrase_breaks_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VIENEU_PHRASE_BREAKS", "0")
    s = ("cô ấy đã cố gắng rất nhiều suốt những năm tháng qua nhưng cuối cùng "
         "vẫn không thể nào quên được anh.")
    assert add_phrase_breaks(s) == s


def test_finalize_chunk_keeps_short_sentences_inside():
    assert finalize_chunk("khó tả. gió thổi. lá rơi.") == "khó tả. gió thổi. lá rơi."
    assert finalize_chunk("một câu không dấu") == "một câu không dấu."
    # Dấu cuối vẫn do punc_norm quyết trên cả chunk như trước.
    assert finalize_chunk("không!") == "không."


def test_chunks_and_phonemes_keep_sentence_periods():
    chunks, _ = normalize_to_chunks_v3_with_gaps("Tai họa ập đến. Gió thổi. Lá rơi.")
    assert chunks == ["tai hoạ ập đến. gió thổi. lá rơi."]
    ph = phonemize_batch(["tai hoạ ập đến. gió thổi. lá rơi."], skip_normalize=True)[0]
    assert ph.count(".") == 3 and "," not in ph
    assert phonemize_text("Gió thổi. Lá rơi. Hắn đi rồi.").count(".") == 3
