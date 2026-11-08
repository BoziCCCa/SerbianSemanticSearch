#BM25 + TF-IDF over the Latin corpus.
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import normalize, tokenize


@dataclass
class SearchHit:
    article_id: int
    title: str
    category: str
    filename: str
    score: float
    snippet: str


def _article_snippet(text: str, limit: int = 220) -> str:
    snippet = " ".join(text.split())
    return snippet[:limit] + ("…" if len(snippet) > limit else "")


def _load_corpus(meta_path: Path, raw_dir: Path) -> list[dict]:
    with meta_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        text = (raw_dir / row["filename"]).read_text(encoding="utf-8")
        row["text"] = text
        row["snippet"] = _article_snippet(text)
    return rows


class BM25Index:
    def __init__(self, articles: list[dict]):
        self.articles = articles
        self.bm25 = BM25Okapi([tokenize(a["text"]) for a in articles])

    @classmethod
    def build(cls, meta_path: Path, raw_dir: Path) -> "BM25Index":
        return cls(_load_corpus(meta_path, raw_dir))

    def search(self, query: str, k: int = 5,
               category: str | None = None) -> list[SearchHit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)

        #Rank first, then filter — preserves scores.
        order = np.argsort(scores)[::-1]
        hits: list[SearchHit] = []
        for idx in order:
            a = self.articles[idx]
            if category and a["category"] != category:
                continue
            if scores[idx] <= 0:
                break
            hits.append(SearchHit(
                article_id=int(a["id"]),
                title=a["title"],
                category=a["category"],
                filename=a["filename"],
                score=float(scores[idx]),
                snippet=a["snippet"],
            ))
            if len(hits) >= k:
                break
        return hits


class TfidfIndex:
    def __init__(self, articles: list[dict]):
        self.articles = articles
        docs = [normalize(a["text"]) for a in articles]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-zđžćčš]+",
            min_df=2,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    @classmethod
    def build(cls, meta_path: Path, raw_dir: Path) -> "TfidfIndex":
        return cls(_load_corpus(meta_path, raw_dir))

    def search(self, query: str, k: int = 5,
               category: str | None = None) -> list[SearchHit]:
        q_vec = self.vectorizer.transform([normalize(query)])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(scores)[::-1]
        hits: list[SearchHit] = []
        for idx in order:
            a = self.articles[idx]
            if category and a["category"] != category:
                continue
            if scores[idx] <= 0:
                break
            hits.append(SearchHit(
                article_id=int(a["id"]),
                title=a["title"],
                category=a["category"],
                filename=a["filename"],
                score=float(scores[idx]),
                snippet=a["snippet"],
            ))
            if len(hits) >= k:
                break
        return hits


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]

    root = Path(__file__).resolve().parents[1]
    meta = root / "data" / "metadata.csv"
    raw = root / "data" / "raw"

    query = " ".join(sys.argv[1:]) or "Nikola Tesla"
    print(f"Query: {query!r}")

    print("\n== BM25 ==")
    for h in BM25Index.build(meta, raw).search(query, k=5):
        print(f"  [{h.score:6.2f}] ({h.category}) {h.title}")

    print("\n== TF-IDF ==")
    for h in TfidfIndex.build(meta, raw).search(query, k=5):
        print(f"  [{h.score:.3f}] ({h.category}) {h.title}")
