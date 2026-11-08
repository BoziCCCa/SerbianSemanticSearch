#Serbian Cyrillic<->Latin transliteration, normalize, tokenize.
from __future__ import annotations

import re

#Three Cyrillic letters map to two-character Latin forms.
_DIGRAPHS = {
    "Љ": "Lj", "љ": "lj",
    "Њ": "Nj", "њ": "nj",
    "Џ": "Dž", "џ": "dž",
}

#Everything else is one-to-one.
_SINGLES = str.maketrans({
    "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Ђ":"Đ","Е":"E","Ж":"Ž","З":"Z",
    "И":"I","Ј":"J","К":"K","Л":"L","М":"M","Н":"N","О":"O","П":"P","Р":"R",
    "С":"S","Т":"T","Ћ":"Ć","У":"U","Ф":"F","Х":"H","Ц":"C","Ч":"Č","Ш":"Š",
    "а":"a","б":"b","в":"v","г":"g","д":"d","ђ":"đ","е":"e","ж":"ž","з":"z",
    "и":"i","ј":"j","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r",
    "с":"s","т":"t","ћ":"ć","у":"u","ф":"f","х":"h","ц":"c","ч":"č","ш":"š",
})


def transliterate(text: str) -> str:
    #Digraphs first, then the single-char table.
    for cyr, lat in _DIGRAPHS.items():
        text = text.replace(cyr, lat)
    return text.translate(_SINGLES)


#Reverse mapping. Digraphs make this ambiguous in general.
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
    "s":"с","t":"т","ć":"ћ","u":"у","f":"ф","h":"х","c":"ц","č":"ч","š":"š",
})


def to_cyrillic(text: str) -> str:
    for lat, cyr in _DIGRAPHS_REV.items():
        text = text.replace(lat, cyr)
    return text.translate(_SINGLES_REV)


#Short stop-word list (post-normalize, Latin, lowercase).
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
    "the", "and", "or", "of", "in", "to",
])


def normalize(text: str) -> str:
    #Transliterate, lowercase, strip punctuation.
    text = transliterate(text).lower()
    text = re.sub(r"[^a-zđžćčš\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split()
            if t not in SERBIAN_STOPWORDS and len(t) > 1]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]

    sample = "Њ, Џек и Љиљана живе у Београду. Nikola je naučnik!"
    print("Original:   ", sample)
    print("transliterate:", transliterate(sample))
    print("normalize:  ", normalize(sample))
    print("tokenize:   ", tokenize(sample))
