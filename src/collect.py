#Download sr.wikipedia articles, transliterate, save to disk.
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests
import wikipediaapi
from tqdm import tqdm

from text_utils import transliterate, to_cyrillic

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]


#Config
USER_AGENT = "SerbianSemanticSearchHomework/1.0"
API_URL = "https://sr.wikipedia.org/w/api.php"
LANG = "sr"

ARTICLES_PER_BUCKET = 32
OVERFETCH = 3
MIN_CHARS = 1500
REQUEST_DELAY = 0.2

#Latin for readability; API needs Cyrillic cmtitles.
CATEGORIES_LAT: dict[str, list[str]] = {
    "istorija":    ["Kategorija:Istorija Srbije", "Kategorija:Srpski vladari", "Kategorija:Bitke"],
    "sport":       ["Kategorija:Srpski fudbaleri", "Kategorija:Srpski košarkaši", "Kategorija:Srpski teniseri"],
    "geografija":  ["Kategorija:Gradovi u Srbiji", "Kategorija:Planine u Srbiji", "Kategorija:Reke u Srbiji"],
    "nauka":       ["Kategorija:Srpski naučnici", "Kategorija:Hemijski elementi", "Kategorija:Fizika"],
    "muzika":      ["Kategorija:Srpski muzičari", "Kategorija:Srpski kompozitori", "Kategorija:Muzički instrumenti"],
    "film":        ["Kategorija:Srpski filmovi", "Kategorija:Srpski glumci", "Kategorija:Srpski reditelji", "Kategorija:Jugoslovenski filmovi", "Kategorija:Akcioni filmovi", "Kategorija:Horor filmovi", "Kategorija:Komični filmovi", "Kategorija:Dramski filmovi", "Kategorija:Animirani filmovi"],
    "tehnologija": ["Kategorija:Programski jezici", "Kategorija:Operativni sistemi", "Kategorija:Računarstvo"],
    "hrana":       ["Kategorija:Srpska kuhinja", "Kategorija:Piće", "Kategorija:Hrana"],
}

CATEGORIES: dict[str, list[str]] = {
    slug: [to_cyrillic(seed) for seed in seeds]
    for slug, seeds in CATEGORIES_LAT.items()
}

#Section headers that mark the end of real article content.
STOP_SECTIONS = [
    "Vidi još", "Napomene", "Reference", "Literatura", "Izvori", "Spoljašnje veze",
]


def api_get(**params) -> dict:
    params["format"] = "json"
    r = requests.get(API_URL, params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return r.json()


def iter_category_members(category: str, kind: str):
    #kind = "page" (articles, ns=0) or "subcat" (sub-categories, ns=14).
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": 0 if kind == "page" else 14,
        "cmlimit": 500,
    }
    while True:
        data = api_get(**params)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)


def gather_candidate_titles(seeds: list[str], target: int) -> list[str]:
    #Try seeds, descend one level if needed.
    seen: set[str] = set()
    titles: list[str] = []

    def take_from(category: str) -> bool:
        for t in iter_category_members(category, "page"):
            if t in seen:
                continue
            seen.add(t)
            titles.append(t)
            if len(titles) >= target:
                return True
        return False

    for seed in seeds:
        if take_from(seed):
            return titles
        for sub in iter_category_members(seed, "subcat"):
            if take_from(sub):
                return titles
    return titles


def slugify(title: str) -> str:
    #\w matches Cyrillic; script is preserved.
    return re.sub(r"[^\w]+", "_", title.lower()).strip("_") or "article"


#Matches section headers, with or without =.
_STOP_PATTERN = re.compile(
    r"(?m)^(?:=+\s*)?(?:" + "|".join(re.escape(s) for s in STOP_SECTIONS) + r")(?:\s*=+)?\s*$"
)


def clean_article_text(text: str) -> str:
    cut = _STOP_PATTERN.search(text)
    if cut:
        text = text[: cut.start()]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[^\]]*uredi[^\]]*\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_disambiguation(text: str) -> bool:
    head = text[:400].lower()
    return "višeznačna odrednica" in head or "disambiguation" in head


CSV_FIELDS = ["id", "title", "category", "url", "char_count", "filename"]


def load_existing(meta_path: Path, raw_dir: Path) -> dict[str, dict]:
    if not meta_path.exists():
        return {}
    rows: dict[str, dict] = {}
    with meta_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (raw_dir / row["filename"]).exists():
                rows[row["title"]] = row
    return rows


def write_metadata(meta_path: Path, rows: dict[str, dict]) -> None:
    #UTF-8-sig so Excel double-click renders diacritics correctly.
    with meta_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in sorted(rows.values(), key=lambda r: int(r["id"])):
            w.writerow(row)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = root / "data" / "metadata.csv"

    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language=LANG,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    rows = load_existing(meta_path, raw_dir)
    next_id = max((int(r["id"]) for r in rows.values()), default=0) + 1

    for cat_slug, seeds in CATEGORIES.items():
        print(f"\n{cat_slug}:")
        kept = sum(1 for r in rows.values() if r["category"] == cat_slug)
        if kept >= ARTICLES_PER_BUCKET:
            print(f"  already have {kept}/{ARTICLES_PER_BUCKET}, skipping")
            continue

        candidates = gather_candidate_titles(seeds, target=ARTICLES_PER_BUCKET * OVERFETCH)
        print(f"  {len(candidates)} candidate titles")

        bar = tqdm(candidates, desc=cat_slug, unit="art")
        for raw_title in bar:
            if kept >= ARTICLES_PER_BUCKET:
                break
            #Dedup on the Latin form that will be stored.
            title = transliterate(raw_title)
            if title in rows:
                continue

            page = wiki.page(raw_title)
            if not page.exists():
                continue

            #Transliterate first; downstream works on Latin only.
            text = clean_article_text(transliterate(page.text or ""))
            if len(text) < MIN_CHARS or looks_like_disambiguation(text):
                continue

            fname = f"{cat_slug}_{slugify(title)}.txt"
            (raw_dir / fname).write_text(text, encoding="utf-8")
            rows[title] = {
                "id": next_id,
                "title": title,
                "category": cat_slug,
                "url": page.fullurl,
                "char_count": len(text),
                "filename": fname,
            }
            next_id += 1
            kept += 1
            bar.set_postfix(kept=kept)
            time.sleep(REQUEST_DELAY)
        bar.close()
        print(f"  kept {kept} / {ARTICLES_PER_BUCKET} for {cat_slug}")

    write_metadata(meta_path, rows)
    print(f"\nTotal articles: {len(rows)}  →  {meta_path}")


if __name__ == "__main__":
    main()
