"""
src/text_utils.py

Shared text helpers for the Serbian semantic search project.

Serbian is written in two scripts, Cyrillic and Latin, that map one-to-one
apart from three digraphs: љ↔lj, њ↔nj, џ↔dž. Cyrillic→Latin is deterministic
(every Cyrillic letter has exactly one Latin equivalent); the reverse isn't
(a Latin "lj" could be the single letter љ or two separate letters). We
therefore pick Latin as our canonical form everywhere downstream.

The same helpers run at ingest time (collect.py transliterates downloaded
articles to Latin) and at search time (keyword queries get normalized the
same way so a Latin query matches a Latin-stored document).
"""
from __future__ import annotations

import re

# Digraphs: handled with a plain str.replace pre-pass because str.translate
# only maps single characters.
_DIGRAPHS = {
    "Љ": "Lj", "љ": "lj",
    "Њ": "Nj", "њ": "nj",
    "Џ": "Dž", "џ": "dž",
}

# Single-character mapping for the rest of the alphabet, both cases.
_SINGLES = str.maketrans({
    "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Ђ":"Đ","Е":"E","Ж":"Ž","З":"Z",
    "И":"I","Ј":"J","К":"K","Л":"L","М":"M","Н":"N","О":"O","П":"P","Р":"R",
    "С":"S","Т":"T","Ћ":"Ć","У":"U","Ф":"F","Х":"H","Ц":"C","Ч":"Č","Ш":"Š",
    "а":"a","б":"b","в":"v","г":"g","д":"d","ђ":"đ","е":"e","ж":"ž","з":"z",
    "и":"i","ј":"j","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r",
    "с":"s","т":"t","ћ":"ć","у":"u","ф":"f","х":"h","ц":"c","ч":"č","ш":"š",
})


def transliterate(text: str) -> str:
    """
    Deterministically transliterate Serbian Cyrillic to Latin.
    Non-Cyrillic characters pass through unchanged, so mixed-script input
    (Cyrillic article body containing Latin names like "Nikola") is safe.
    """
    for cyr, lat in _DIGRAPHS.items():
        text = text.replace(cyr, lat)
    return text.translate(_SINGLES)


# Reverse mapping, used only to resolve human-written Latin config strings
# (like category names in collect.py) back to the Cyrillic form the sr.wiki
# API needs. Technically ambiguous for digraphs ("lj" could be two letters or
# one), but all practical Serbian uses the digraph reading.
_DIGRAPHS_REV = {
    "LJ": "Љ", "Lj": "Љ", "lj": "љ",
    "NJ": "Њ", "Nj": "Њ", "nj": "њ",
    "DŽ": "Џ", "Dž": "Џ", "dž": "џ",
}

_SINGLES_REV = str.maketrans({
    "A":"А","B":"Б","V":"В","G":"Г","D":"Д","Đ":"Ђ","E":"Е","Ž":"Ж","Z":"З",
    "I":"И","J":"Ј","K":"К","L":"Л","M":"М","N":"Н","O":"О","P":"П","R":"Р",
    "S":"С","T":"Т","Ć":"Ћ","U":"У","F":"Ф","H":"Х","C":"Ц","Č":"Ч","Š":"Ш",
    "a":"а","b":"б","v":"в","g":"г","d":"д","đ":"ђ","e":"е","ž":"ж","z":"з",
    "i":"и","j":"ј","k":"к","l":"л","m":"м","n":"н","o":"о","p":"п","r":"р",
    "s":"с","t":"т","ć":"ћ","u":"у","f":"ф","h":"х","c":"ц","č":"ч","š":"ш",
})


def to_cyrillic(text: str) -> str:
    """Reverse direction, see docstring on _DIGRAPHS_REV for the caveat."""
    for lat, cyr in _DIGRAPHS_REV.items():
        text = text.replace(lat, cyr)
    return text.translate(_SINGLES_REV)


# Short Serbian stop-word list (post-normalize, Latin, lowercase).
# Not exhaustive — BM25 just benefits from dropping the most frequent noise.
SERBIAN_STOPWORDS = frozenset([
    "i", "ili", "ali", "a", "da", "ne", "u", "na", "za", "sa", "se", "si",
    "je", "su", "bio", "bila", "bili", "biti", "ima", "nema", "od", "do",
    "po", "o", "ka", "kod", "iz", "pre", "posle", "između", "preko", "kroz",
    "ovaj", "ova", "ovo", "ovi", "ove", "taj", "ta", "to", "ti", "te",
    "on", "ona", "ono", "oni", "one", "mi", "vi", "ja",
    "li", "što", "ko", "kako", "kada", "zašto", "gde", "koji", "koja", "koje",
    "sve", "svi", "svaki", "svaka", "svako", "nije", "jedan", "jedna", "jedno",
    "neki", "neka", "neko", "ništa", "nešto", "još", "već", "samo", "tako", "tu",
    "no", "iako", "mada", "ako", "jer", "pa",
    # Common English leakage from Wikipedia infoboxes
    "the", "and", "or", "of", "in", "to",
])


def normalize(text: str) -> str:
    """
    Transliterate → lowercase → strip everything that isn't a Latin or
    Serbian-diacritic letter → collapse whitespace. Output is one or more
    whitespace-separated word-shaped tokens.
    """
    text = transliterate(text).lower()
    text = re.sub(r"[^a-zđžćčš\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Normalize, split on whitespace, drop stop-words and single-char tokens.
    This is what BM25 consumes.
    """
    return [t for t in normalize(text).split()
            if t not in SERBIAN_STOPWORDS and len(t) > 1]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]

    # Smoke test — exercise all three digraphs plus a mixed-script sentence.
    sample = "Њ, Џек и Љиљана живе у Београду. Nikola je naučnik!"
    print("Original:   ", sample)
    print("transliterate: ", transliterate(sample))
    print("normalize:  ", normalize(sample))
    print("tokenize:   ", tokenize(sample))
