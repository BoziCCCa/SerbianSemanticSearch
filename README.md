# Serbian Semantic Search

University NLP homework: semantic search over 236 Serbian Wikipedia
articles using a multilingual embedding model and Qdrant, compared
side-by-side with classic keyword search (BM25 and TF-IDF).

Full written analysis with plots: `notebooks/analysis.ipynb`.
Prose summary: `report.md`.

## What's in this repo

```
docker-compose.yml          Qdrant service only
.env.example                QDRANT_HOST/PORT/COLLECTION, EMBEDDING_MODEL
requirements.txt            pinned Python deps, grouped by pipeline step
src/
  collect.py                download + clean articles from sr.wikipedia.org
  text_utils.py             Cyrillic↔Latin + normalize/tokenize/stopwords
  vector_store.py           all Qdrant calls live here (one import, one wrapper)
  embed.py                  chunk, embed with E5, upsert to Qdrant
  keyword_search.py         BM25Okapi + TF-IDF indices
  search.py                 interactive CLI: semantic + BM25 side by side
  main.py                   batch runner over 20 handpicked queries
data/
  raw/                      one UTF-8 .txt per article (produced by collect.py)
  metadata.csv              id, title, category, url, char_count, filename
results/                    embeddings.npy, embedding_ids.csv, comparison_*.csv
notebooks/
  analysis.ipynb            comparison plots, tables, conclusions
report.md                   1–2 page prose summary
```

## Prerequisites

* Docker Desktop (for Qdrant)
* Anaconda or Miniconda (for the Python environment)

## Step 1 — Start Qdrant

Qdrant is the only service that runs in Docker. The Python code runs on
the host so the embedding model can use a host GPU if present.

```bash
docker compose up -d
docker compose ps          # expect "Up … (healthy)"
```

Open the dashboard at <http://localhost:6333/dashboard>.

Stopping and wiping:

```bash
docker compose down          # stop, keep data
docker compose down -v       # stop AND delete the named volume (fresh start)
```

## Step 2 — Python environment

```bash
conda create -n serbian-search python=3.11 -y
conda activate serbian-search
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to override the default Qdrant
host, port, or collection name.

## Step 3 — Collect the corpus

```bash
python src/collect.py
```

Downloads ~236 articles across 8 topic buckets from `sr.wikipedia.org`,
cleans them, transliterates to Latin, writes one `.txt` file per article
to `data/raw/` and appends rows to `data/metadata.csv`. Takes ~5–10 min
on first run; subsequent runs skip articles that already exist on disk.

To expand or change the corpus, edit `CATEGORIES_LAT` at the top of
`src/collect.py`.

## Step 4 — Embed and index

```bash
python src/embed.py
```

Loads `intfloat/multilingual-e5-base` (~450 MB download on first run),
splits each article into ~350-word chunks with 50-word overlap, embeds
each chunk, and upserts the whole set into the Qdrant `serbian_wiki`
collection. Also exports `results/embeddings.npy` + `embedding_ids.csv`
so the raw vectors can be inspected without the DB.

Takes ~5 min on CPU, seconds on GPU.

## Step 5 — Try it interactively

**One-shot query:**

```bash
python src/search.py "glavni grad Srbije"
```

**REPL:**

```bash
python src/search.py
> nikola tesla
> jezik za skriptovanje
> exit
```

Queries can be either Cyrillic or Latin — the transliteration layer
handles both. Each query prints the top 5 semantic hits and the top 5
BM25 hits as two labeled sections.

## Step 6 — Batch comparison

```bash
python src/main.py
```

Runs 20 handpicked Serbian queries through all three backends (semantic,
BM25, TF-IDF) and prints side-by-side top-5 tables. Useful for a quick
feel; the real analysis is in the notebook.

## Step 7 — Read the analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

Already fully executed with plots and tables embedded — just open and
scroll. Covers:

1. Corpus overview (per-bucket counts, length histogram)
2. Full 20-query sweep across semantic + BM25 + TF-IDF
3. Top-5 overlap between methods
4. Precision@5 by query type
5. Latency comparison
6. Case studies (paraphrase, Cyrillic/Latin, misspelled, out-of-corpus)
7. Conclusions, limitations, improvements

Section 11 saves 4 CSV summaries to `results/`.

If you re-run the notebook, select the **`Python (serbian-search)`**
kernel registered during the env setup.

## Troubleshooting

**`ModuleNotFoundError`** — you're not in the `serbian-search` env.
`conda activate serbian-search`.

**`Qdrant not reachable at localhost:6333`** — the container isn't
running. `docker compose up -d`.

**Cyrillic characters display as `Å¡` / `Ä‡` in PowerShell** — the
Windows cp1252 console doesn't auto-detect UTF-8. Either add
`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` to your
`$PROFILE`, or open the files in VS Code / Jupyter (which read UTF-8
natively). Our `metadata.csv` is written with a UTF-8 BOM so Excel
double-click works.

**`cp1252` crash when a script tries to print Cyrillic** — all our
scripts call `sys.stdout.reconfigure("utf-8")` at the top, so this only
happens if you add new print statements that bypass that or run on a
very old Python.
