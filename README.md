# The Financial RAG Challenge

A Retrieval-Augmented Generation (RAG) system that answers financial questions
from U.S. Treasury Bulletins, comparing a **Baseline** pipeline against an
**Engineered (optimized)** pipeline across six retriever/generator metrics.

Data source: the [databricks/officeqa](https://github.com/databricks/officeqa)
benchmark (U.S. Treasury Bulletin corpus + `officeqa_full.csv` answer key),
restricted to **2022–2025**.

---

## TL;DR — run it

```bash
pip install -r requirements.txt
python run.py
```

That single command prepares the data, chunks it two ways, builds both
retrievers, evaluates all six metrics, and writes everything to `results/`.
It runs **end-to-end with no API key and no dataset access** using bundled mock
data and a deterministic offline LLM fallback. To use real Claude generation and
the real (gated) corpus, set two environment variables — see
[Configuration](#configuration).

Headline result (mock corpus, offline backend, K=5):

| Metric | Baseline | Engineered | Δ |
| --- | --- | --- | --- |
| Hit Rate@5 | 0.76 | 1.00 | +0.24 |
| MRR | 0.55 | 1.00 | +0.45 |
| Recall@5 | 0.38 | 0.98 | +0.60 |
| Groundedness | 1.00 | 1.00 | 0.00 |
| Factual Accuracy | 0.40 | 0.96 | +0.56 |
| Hallucination Rate | 0.00 | 0.00 | 0.00 |

---

## Architecture

```
data_prep.py  ->  chunking.py  ->  retrieval.py  ->  generation.py  ->  evaluate.py
  (load +          (baseline /       (embed +          (answer from       (6 metrics
   filter)          engineered)       FAISS + meta)      context)           + judge)
                                   \___________________ run.py ___________________/
```

| Module | Responsibility |
| --- | --- |
| `config.py` | All tunable knobs (years, K, models, chunk sizes, tolerance). |
| `data_prep.py` | Download real HF data **or** generate a mock corpus; filter both corpus and CSV to 2022–2025 with an **auditable** drop log. |
| `chunking.py` | Baseline (fixed, table-blind) vs Engineered (table-aware, overlapping) chunkers; tags every chunk with Year/Month. |
| `retrieval.py` | Sentence-Transformer embeddings + FAISS index; Baseline (pure semantic) vs Engineered (Year/Month **pre-filter**) retrievers. |
| `generation.py` | Baseline (simple) vs Engineered (grounded/cite/refuse) prompts to Claude; deterministic offline extractor fallback. |
| `evaluate.py` | Computes the 3 retriever + 3 generator metrics; LLM-as-judge for groundedness/hallucination. |
| `reward.py` | **Official** databricks/officeqa scorer, vendored (Apache-2.0) and used verbatim for Factual Accuracy at ±1%. |
| `run.py` | Orchestrates everything and writes `results/`. |

### Vector DB choice — FAISS (`IndexFlatIP`)

- **Exact** inner-product search over L2-normalized vectors (= cosine). No ANN
  approximation error to confound a Baseline-vs-Engineered comparison.
- In-memory, no server, trivial to rebuild — ideal for a 4-year slice
  (~48 docs / a few hundred chunks).
- Metadata filtering is done **in code** by restricting the candidate set before
  ranking, which is transparent and easy to explain. (ChromaDB's built-in `where`
  filter is convenient but heavier and hides that step; at this scale FAISS +
  a metadata array is simpler and fully offline.)

### Chunking strategy

- **Baseline:** fixed ~128-word windows, **no overlap, table-blind** — it will cut
  a Markdown table mid-row, orphaning `Total receipts | 408,587` from its heading.
- **Engineered:** ~512-word windows with 64-word overlap, and **table-aware** — a
  Markdown table (heading + all its rows) is kept as one atomic chunk so a figure
  is never separated from its label. ("Tokens" are approximated by whitespace
  words; the embedding model applies its own sub-word tokenizer at encode time.)

### Metadata usage (the core "engineered" lever)

Every chunk is tagged with **Year** and **Month**, parsed from the source filename
`treasury_bulletin_{YEAR}_{MONTH}.txt` — the only date signal the real dataset
carries. The Engineered retriever parses a Year/Month out of the *question* and
**pre-filters** the candidate chunks to that period before ranking (falling back
to a full search when the question names no period). Because every monthly
bulletin shares identical table labels, this is what lets it distinguish
"March 2024" from "March 2023".

---

## The six metrics

**Retriever (at K=5):**
- **Hit Rate@5** = queries with a gold-file chunk in the top 5 / total queries.
- **MRR** = mean of 1 / (rank of the first gold-file chunk).
- **Recall@5** = relevant chunks retrieved / relevant chunks in the DB, averaged
  over queries (relevant = any chunk from a gold source file). This is the literal
  snippet-level reading of the rubric; it is bounded by `K / (relevant chunks per
  query)`, so absolute values are modest — the Baseline-vs-Engineered *gap* is the
  point.

**Generator:**
- **Factual Accuracy** = answers matching the CSV within ±1%, scored by the
  **official `reward.py`** (`score_answer(gold, pred, tolerance=0.01)`).
- **Groundedness** = supported claims / total claims.
- **Hallucination Rate** = fabricated claims / total claims.

Groundedness/Hallucination use a deterministic **LLM-as-judge** (Claude, temp 0):
each answer is split into atomic claims, each labeled *supported* (entailed by the
retrieved context) and/or *fabricated* (asserts a specific number absent from the
context). Without an API key, a documented offline heuristic is used instead (a
numeric answer is supported iff it appears verbatim in the retrieved context).

---

## Configuration

Copy `.env.example` and fill in whichever you have (or export in your shell):

| Variable | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | Enables **real Claude** generation + judging (`claude-haiku-4-5`, temp 0). Unset → deterministic **offline** fallback; the pipeline still runs. |
| `HF_TOKEN` | Enables the **real gated corpus**. Requires prior access approval at [huggingface.co/datasets/databricks/officeqa](https://huggingface.co/datasets/databricks/officeqa). Unset → bundled **mock** corpus. |

**Swapping to real data is just setting `HF_TOKEN` and re-running — no code
changes.** `data_prep.py` writes real and mock data into the identical on-disk
layout, so chunking/retrieval/evaluation are source-agnostic.

---

## Outputs (`results/`)

- `results.csv` / `results.md` — the metrics table above.
- `per_question_baseline.csv` / `per_question_engineered.csv` — full audit trail
  (retrieved files, prediction, per-question scores).
- `analysis.md` — the three written analyses (Bottleneck / Metadata Fix / Scaling),
  regenerated with live numbers each run.

---

## Findings (summary — full text in `results/analysis.md`)

1. **The Bottleneck** is *retrieval*, shown most clearly by **MRR (0.55)** paired
   with **Factual Accuracy (0.40)**: the baseline generator is faithful to what it
   is handed (Groundedness 1.00), so wrong answers come from ranking the *wrong
   month's* identically-labeled table first — not from bad generation.
2. **The Metadata Fix** moved **retrieval** metrics far more than generation
   metrics (MRR/Recall jump toward 1.0; Groundedness/Hallucination barely move).
   Factual Accuracy rises as a downstream consequence of handing the generator the
   correct month.
3. **Scaling** from 4 years to the full 1939–2025 archive (~697 bulletins) breaks
   the **brute-force in-memory FAISS index** first (O(N) scan + must fit in RAM),
   then single-pass chunking; and pure semantic search gets *worse* as ~86
   near-identical "March receipts" rows compete — making the metadata pre-filter
   essential. Fix: an approximate index (FAISS IVF/HNSW) or a metadata-partitioned
   vector DB.

---

## Assumptions & flags (ambiguous parts, called out)

- **Repo path corrected.** The prompt's `databrickslabs/OfficeQA` does not exist;
  the real repo is `databricks/officeqa`. Its large files (CSVs + corpus) moved to
  a **gated** Hugging Face dataset in May 2026, which is why real data needs an
  `HF_TOKEN` and prior access approval.
- **Date metadata comes from the filename.** The CSV has no explicit year/month
  column; Year/Month are parsed from `source_files` / the filename, exactly as the
  real schema intends.
- **CSV filtering rule.** A row is kept only if **all** of its source files fall in
  2022–2025 **and** exist in the filtered corpus; partially-out-of-range rows are
  dropped (they can't be answered from the filtered corpus). Drops are logged with
  reasons. The mock data deliberately includes two out-of-range rows to exercise
  this.
- **Multi-source questions** count **any** listed gold file as correct for Hit/MRR.
  The Engineered retriever's naive regex period-parser picks the *first* month a
  question mentions, which is why a two-month question is the one engineered miss
  (0.96, not 1.00) — an honest limitation, not a bug.
- **Offline backend caveat.** With no API key, the generator is purely extractive
  (copies numbers out of retrieved context), so Groundedness ≈ 1.0 and
  Hallucination ≈ 0.0 **by construction** — those two metrics become discriminating
  only under the real Claude backend. The retrieval-driven Factual Accuracy gap is
  the headline in both modes.
- **"Tokens" ≈ words** for chunk sizing (documented approximation; keeps the
  project dependency-light and offline).

---

## Reproduce

```bash
pip install -r requirements.txt        # numpy, pandas, sentence-transformers, faiss-cpu, anthropic, ...
python data_prep.py                    # (optional) inspect data prep + filter log
python chunking.py                     # (optional) inspect chunk stats
python retrieval.py                    # (optional) see baseline vs engineered retrieval
python run.py                          # full evaluation -> results/
```

Licenses: this project's code is provided for the course assignment; `reward.py`
is from databricks/officeqa (Apache-2.0) and retained with its original header.
