# Serbian Semantic Search — Report

University NLP homework. End-to-end semantic search over Serbian Wikipedia
articles with a comparison against classic keyword search.

For the full code walkthrough, embedded plots, 20-query sweep, overlap
matrices, precision@5 bars, and latency charts, see
`notebooks/analysis.ipynb`. This file is a 1–2 page prose summary.

## Data collection

The collector (`src/collect.py`) uses the MediaWiki API against
`sr.wikipedia.org` to walk a hand-picked set of seed categories in eight
topic buckets:

| Bucket        | Articles |
|---------------|---------:|
| istorija      | 32 |
| sport         | 32 |
| geografija    | 32 |
| nauka         | 32 |
| muzika        | 32 |
| film          | 12 |
| tehnologija   | 32 |
| hrana         | 32 |
| **total**     | **236** |

Each bucket has 2–3 seed categories. The walker descends one level into
sub-categories if a seed doesn't yield enough direct articles. Filtering
rules drop:

* disambiguation pages (detected via the Serbian marker *"višeznačna
  odrednica"*),
* stubs below 1500 characters,
* titles already saved to disk (the script is idempotent).

Text is cleaned to strip reference markers (`[1]`), edit markers
(`[uredi | uredi izvor]`), and trailing housekeeping sections
(*Reference*, *Literatura*, *Spoljašnje veze*, …) before being saved as
UTF-8 `.txt` files alongside a `metadata.csv` row (id, title, category,
URL, char count, filename).

**Film is thinner than other buckets** (12/32). The Serbian WP film genre
categories are dominated by one-paragraph stubs that fail the 1500-char
minimum. Documented as a known limitation.

## Embedding model and chunking

**Model:** `intfloat/multilingual-e5-base` (278 M parameters, 768-dim
output). Picked for three reasons:

1. First-class multilingual coverage including Serbian in both scripts.
2. Permissive license (MIT), free to download via sentence-transformers.
3. Small enough to run on CPU in reasonable time (~5 min to embed 1355
   chunks on a Ryzen 9600X; a GPU would reduce this to seconds).

E5 was trained with asymmetric prefixes — the ingest script tags every
chunk with `"passage: "` and the query path uses `"query: "`. Skipping
these silently cuts retrieval quality.

**Chunking:** articles are split into ~350-word windows with 50 words of
overlap. This is driven by E5's 512-token input cap — longer articles
would be silently truncated to just the lead section. 1355 chunks come
out of 236 articles (mean ≈ 5.7 chunks per article).

## Architecture

**Qdrant** stores one vector per chunk plus a payload of article metadata
(`article_id`, `title`, `category`, `filename`, `chunk_index`,
`chunk_text`). Collection uses cosine distance (vectors are
L2-normalized at encode time) and payload indices on `article_id` and
`category` for cheap filtering.

**Grouping.** Semantic search returns the top five *distinct articles*
by querying with `query_points_groups(group_by="article_id",
group_size=1)`. Qdrant does the per-article best-chunk reduction
server-side, so we never need the client to deduplicate.

**Honest note on vector DBs.** At 236 articles / 1355 chunks, a plain
NumPy brute-force cosine would be just as fast as Qdrant. We chose
Qdrant anyway for:

1. **Persistence** across restarts (data lives in a Docker named volume),
2. **Metadata filtering** (`category == "hrana"` is a one-line change),
3. **Scalability** — the same code works at 1 M chunks with HNSW
   indexing.

All Qdrant interaction is isolated in `src/vector_store.py` so other
scripts don't import `qdrant_client` directly. Swapping backends is a
one-file change.

## Keyword search and transliteration

Serbian is written in both Cyrillic and Latin. Transliteration is
deterministic in one direction (Cyrillic → Latin) because every Cyrillic
letter has exactly one Latin equivalent. The reverse is ambiguous
(a Latin "lj" could be the single letter *љ* or two separate letters).

We therefore picked **Latin as the canonical form** everywhere
downstream. The collector transliterates every downloaded article to
Latin before saving. Search-time queries are also transliterated via
`text_utils.transliterate`, so a Cyrillic query like *"Никола Тесла"* and
a Latin query *"Nikola Tesla"* produce byte-identical results.

**BM25 index** (`src/keyword_search.py:BM25Index`) runs
`rank_bm25.BM25Okapi` on tokenized full articles (not chunks). Tokens
pass through the same normalize/stop-word pipeline queries use. BM25
operates on whole articles because length normalization already handles
the long-doc problem and we'd just re-group chunks back to articles
anyway.

**TF-IDF baseline** (`TfidfIndex`) uses scikit-learn's
`TfidfVectorizer` + cosine similarity. Different weighting curve from
BM25; included because the assignment asked for a second lexical
baseline.

## Results

Full tables and plots are in `notebooks/analysis.ipynb`; the headline
numbers across 20 handpicked Serbian queries:

| Metric                           | Semantic | BM25 | TF-IDF |
|----------------------------------|---------:|-----:|-------:|
| Mean precision@5 (vs intended category) | see notebook § 7 |      |        |
| Mean latency (ms, CPU)           | ~70–100 ms | ~1–2 ms | ~2–3 ms |
| Mean top-5 overlap with semantic | —        | § 6 | § 6 |

(Numbers filled in by the executed notebook — see `comparison_latency.csv`
and the plot in section 8.)

### Where semantic search wins

* **Paraphrase** — `"glavni grad Srbije"` finds *Beograd* even though the
  article never literally says "glavni grad Srbije"; BM25 has to rely on
  indirect co-occurrence.
* **Conceptual queries** — `"jezik za skriptovanje"` surfaces articles
  about scripting languages (Lua, Python entries) by concept, not by
  substring.

### Where BM25 wins

* **Exact title / exact phrase** — `"Burek"` or `"austrougarski ultimatum
  Srbiji 1913"` are pinned at #1 because the match is unambiguous.
* **Head-word queries** where TF scoring dominates and semantic can
  get distracted by chunks that discuss the topic inside unrelated
  articles.

### Where both struggle

* **Out-of-corpus** — queries like `"veštačka inteligencija"` return
  weak, off-topic hits from both methods because no relevant article
  exists. Scores collapse near zero (BM25) or near 0.5 (semantic cosine).
* **Heavily misspelled queries** — both degrade, semantic less so because
  E5's sub-word tokenizer catches partial matches.

### Illustrative case: "Nikola Tesla"

Our corpus has no article about Tesla the physicist, but Belgrade's
*Nikola Tesla Institute* appears in many unrelated bios. BM25 ranks the
engineer **Vladan Vučković** at #1 because he worked at that institute
for 40 years and the string "Nikola Tesla" appears 8× in his article.
Semantic search also returns Vučković in the top 5, but at a lower
confidence; it does not confuse the institute with the inventor.

## Limitations

* **Precision@5 as a relevance proxy.** We treat the query's "intended
  category" as the only acceptable answer. Cross-category articles can
  legitimately be correct answers; the metric underestimates both
  methods' real precision.
* **236 articles with one thin bucket.** The film bucket at 12/32
  biases any metric involving film-targeted queries.
* **20 queries.** Larger evaluation sets would give tighter precision
  intervals.
* **No ground-truth relevance labels.** We never saw a human mark each
  (query, article) pair as relevant / not; everything is proxy.

## Improvements worth exploring

1. **Hybrid search.** Qdrant has native fusion of dense and sparse
   (BM25-style) signals. Likely closes the gap on exact-title queries
   while keeping semantic strength on paraphrase.
2. **Larger E5 variant.** `multilingual-e5-large` has ~2× the parameters
   and usually measurably better retrieval.
3. **CLASSLA lemmatization.** Serbian is morphologically rich.
   Lemmatizing (`fudbaleri` → `fudbaler`) would merge inflected forms
   and improve BM25 recall.
4. **Doubling the corpus.** 500+ articles with a healthier film bucket
   would make the comparison tighter.
