"""Sửa chính tả và nhịp ngắt cho tiếng Việt TRƯỚC khi vào sea-g2p.

Ba việc, đều chạy trên text (không đụng model):

* :func:`fix_spelling` — đưa dấu thanh về đúng chỗ G2P đọc được. G2P của sea-g2p
  chỉ nhận chắc kiểu đặt dấu "mới" cho ``oa/oe/uy`` (``hoạ``, ``loè``, ``tuỳ``):
  kiểu cũ chỉ đúng khi từ có sẵn trong từ điển, còn lại bị tách làm hai âm
  (``đòa`` -> "đo à") hoặc đọc sai hẳn (``họa`` -> "hô a"). Dấu đặt lệch trong
  âm tiết đóng (``họat`` thay vì ``hoạt``) cũng vậy. Kèm vài lỗi gõ thường gặp.
* :func:`add_phrase_breaks` — chèn dấu phẩy trước từ nối trong mệnh đề dài không
  có dấu ngắt nào, để model lấy hơi như người đọc thay vì đọc một mạch 5-6 giây.
* :func:`finalize_chunk` — chốt dấu câu cuối chunk như ``punc_norm`` nhưng KHÔNG
  để nó gộp các câu ngắn giữa chunk thành dấu phẩy ("Gió thổi. Lá rơi." ->
  "gió thổi, lá rơi.") — mất nhịp nghỉ cuối câu.

Tắt chèn ngắt nhịp bằng ``VIENEU_PHRASE_BREAKS=0``.
"""

from __future__ import annotations

import os
import re
import unicodedata

from sea_g2p import punc_norm

# Phiên bản giọng đọc: TĂNG mỗi khi thay đổi làm audio sinh ra khác đi (chuẩn hoá
# text, ngắt nhịp, model…). Báo ở ``/health`` để bên dùng (story-services) gộp
# vào khoá cache và sinh lại audio đã lưu. 1 = trước khi có module này.
VOICE_REVISION = 3

# ─── Dấu thanh ───────────────────────────────────────────────────────────────

_TONES = "̣̀́̉̃"            # huyền sắc hỏi ngã nặng
_VOWEL_BASE = frozenset("aeiouy")                     # nguyên âm (dạng NFD, bỏ mũ/móc)
_CODA = re.compile(r"(?:ch|nh|ng|[cmnpt])$")
# Chỉ chạm tới từ có dấu thanh: phần còn lại của văn bản đi thẳng qua.
_TONED_WORD = re.compile(r"\w*[À-ỹ]\w*")

# Lỗi gõ cố định (khoá viết thường, so sau khi đã NFC).
_TYPOS = {
    "thưở": "thuở",
}


def _vowel_positions(d: str) -> list[int]:
    """Vị trí các ký tự nguyên âm trong chuỗi NFD ``d`` (bỏ qua dấu phụ)."""
    return [i for i, c in enumerate(d) if c.lower() in _VOWEL_BASE]


def _fix_tone_word(word: str) -> str:
    d = unicodedata.normalize("NFD", word)
    tone_at = [i for i, c in enumerate(d) if c in _TONES]
    if len(tone_at) != 1:
        return word
    t = tone_at[0]
    vowels = _vowel_positions(d)
    if len(vowels) < 2:
        return word
    # Nguyên âm đang mang dấu thanh: ký tự gốc gần nhất phía trước dấu.
    carrier = max(v for v in vowels if v < t) if any(v < t for v in vowels) else None
    if carrier is None:
        return word
    base = "".join(c for c in d if not unicodedata.combining(c)).lower()
    last = vowels[-1]
    if carrier == last:
        return word
    if _CODA.search(base):
        # Âm tiết đóng: dấu thanh luôn nằm trên nguyên âm cuối ("hoạt", "quyết").
        target = last
    else:
        # Âm tiết mở: chỉ ba vần oa/oe/uy có hai kiểu đặt dấu; còn lại ("của",
        # "mùa", "người") dấu nằm trên nguyên âm trước là đúng, không đụng.
        tail = base[vowels[-2]:]
        if tail not in ("oa", "oe", "uy") or carrier != vowels[-2]:
            return word
        if vowels[-2] > 0 and base[vowels[-2] - 1] in _VOWEL_BASE:
            return word                     # "khuya", "ngoay"…: không phải vần đó
        target = last
    # Bỏ dấu thanh khỏi chỗ cũ, gắn ngay sau ký tự gốc + dấu mũ/móc của nguyên âm đích.
    chars = list(d)
    tone = chars.pop(t)
    if t < target:
        target -= 1
    j = target + 1
    while j < len(chars) and unicodedata.combining(chars[j]):
        j += 1
    chars.insert(j, tone)
    return unicodedata.normalize("NFC", "".join(chars))


def _fix_word(m: re.Match) -> str:
    w = m.group(0)
    typo = _TYPOS.get(w.lower())
    if typo is not None:
        return typo if w.islower() else typo.capitalize()
    return _fix_tone_word(w)


# Tiền tệ trước "/đơn vị": "3.500.000đ/chỉ" bị đọc thành "… địa chỉhỉ".
_CURRENCY_PER = re.compile(r"(\d)\s*(?:đ|₫|vnđ|vnd)\s*/\s*", re.IGNORECASE)


# Dấu chấm của danh xưng viết tắt không phải hết câu: "Mr. Smith" bị tách câu và
# nghỉ giữa "Mister" với "Smith". Bỏ chấm thì normalizer vẫn đọc "Mister".
_HONORIFIC_DOT = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr)\.(?=\s)")


def fix_spelling(text: str) -> str:
    """Chuẩn hoá chính tả để G2P không đọc méo (xem docstring module)."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _HONORIFIC_DOT.sub(r"\1", text)
    text = _CURRENCY_PER.sub(r"\1 đồng một ", text)
    return _TONED_WORD.sub(_fix_word, text)


# ─── Ngắt nhịp trong mệnh đề dài ─────────────────────────────────────────────

# Từ mở đầu một mệnh đề mới: chèn "," TRƯỚC chúng khi mệnh đề đang dài. Cặp hai
# từ được xử lý như một từ ("sau khi", "bởi vì"). Cố ý KHÔNG có "và", "thì",
# "rồi", "nên" đứng một mình: hay nằm trong cụm danh từ/động từ ("cha và mẹ",
# "ăn rồi", "nên làm"), chèn ngắt vào đó nghe như vấp.
_BREAK_PAIRS = frozenset([
    ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("cho", "đến"),
    ("bởi", "vì"), ("bởi", "lẽ"), ("cho", "nên"), ("vì", "vậy"), ("vì", "thế"),
    ("do", "đó"), ("mặc", "dù"), ("để", "rồi"), ("thế", "nhưng"), ("tuy", "nhiên"),
])
_BREAK_WORDS = frozenset("nhưng song mà nơi khi vì để khiến dù tuy nếu".split())
# Từ đứng TRƯỚC khiến từ nối thành một phần của cụm khác: "về nơi", "mỗi khi",
# "cái mà", "chỉ để", "là vì"…  -> không ngắt.
_NO_BREAK_AFTER = frozenset(
    "về ở tại đến tới khắp mọi những các một nơi mỗi có lúc là chính chỉ cái điều "
    "người kẻ thứ ai gì đâu sao chứ thì vừa đã đang sẽ không chẳng chưa".split()
)
_BREAK_STRIP = "\"'“”‘’()[]«»…"
_PUNCT_END = frozenset(",.;:!?…")

# Mệnh đề trái phải dài ít nhất ngần này từ mới ngắt, phần còn lại sau điểm ngắt
# (tới dấu câu kế tiếp) ít nhất ngần này từ: không đẻ ra mẩu cụt hai ba chữ.
MIN_LEFT_WORDS = 8
MIN_RIGHT_WORDS = 4

_TOKEN = re.compile(r"<en>.*?</en>|<\|emotion_\d+\|>|\S+", re.IGNORECASE | re.DOTALL)


def phrase_breaks_enabled() -> bool:
    return os.environ.get("VIENEU_PHRASE_BREAKS", "1").strip().lower() not in ("0", "false", "no", "off")


def _key(tok: str) -> str:
    return tok.strip(_BREAK_STRIP).lower()


def add_phrase_breaks(sentence: str) -> str:
    """Chèn "," trước từ nối khi mệnh đề đã dài >= ``MIN_LEFT_WORDS`` từ không dấu.

    Chạy trên câu ĐÃ normalize (chữ thường, số đã đọc thành chữ). Không chạm vào
    câu đã có đủ dấu ngắt: chỉ đếm từ kể từ dấu câu gần nhất.
    """
    if not sentence or not phrase_breaks_enabled():
        return sentence
    toks = _TOKEN.findall(sentence)
    n = len(toks)
    if n < MIN_LEFT_WORDS + MIN_RIGHT_WORDS:
        return sentence
    keys = [_key(t) for t in toks]
    ends = [bool(t) and t[-1] in _PUNCT_END for t in toks]
    # right_run[i] = số từ từ i tới hết mệnh đề (gồm cả từ mang dấu câu).
    right_run = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        right_run[i] = 1 if ends[i] else right_run[i + 1] + 1
    out: list[str] = []
    run = 0                                   # số từ của mệnh đề đang mở
    for i, tok in enumerate(toks):
        k = keys[i]
        nxt = keys[i + 1] if i + 1 < n else ""
        prev = keys[i - 1] if i > 0 else ""
        starts = (k, nxt) in _BREAK_PAIRS or (k in _BREAK_WORDS and (prev, k) not in _BREAK_PAIRS)
        if (starts and out and run >= MIN_LEFT_WORDS and right_run[i] >= MIN_RIGHT_WORDS
                and prev not in _NO_BREAK_AFTER and not ends[i - 1] and not tok.startswith("<")):
            out[-1] += ","
            run = 0
        out.append(tok)
        run = 0 if ends[i] else run + 1
    return " ".join(out)


# ─── Chốt dấu câu cuối chunk ─────────────────────────────────────────────────

# Đầu chunk (tham lam) tới dấu kết câu CUỐI CÙNG còn câu theo sau.
_LAST_SENTENCE = re.compile(r"^(.*[.!?])\s+(\S.*)$", re.DOTALL)


def finalize_chunk(chunk: str) -> str:
    """``punc_norm`` chỉ cho câu CUỐI chunk; các câu phía trước giữ nguyên dấu.

    ``punc_norm`` trên cả chunk gộp câu ngắn giữa chunk thành dấu phẩy
    ("khó tả. gió thổi. lá rơi." -> "khó tả, gió thổi, lá rơi."), nên chuỗi câu
    ngắn dồn dập trong văn kể chuyện bị đọc liền như một câu kê.
    """
    whole = punc_norm(chunk)
    m = _LAST_SENTENCE.match(chunk.strip())
    if not m or not whole or whole[-1] not in ".!?":
        return whole
    # Dấu cuối lấy đúng như punc_norm quyết trên CẢ chunk (luật câu ngắn tính
    # trên toàn chunk), phần trước giữ nguyên.
    tail = m.group(2).rstrip().rstrip(".,;:!?…").rstrip()
    return m.group(1) + " " + tail + whole[-1]
