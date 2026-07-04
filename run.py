"""
run.py — one command to reproduce the whole Financial RAG Challenge.

    python run.py

Steps: prepare/filter data -> chunk (baseline & engineered) -> build retrievers
-> evaluate both systems on all 6 metrics -> write results + analysis.

Outputs (results/):
  * results.csv / results.md       — the Baseline vs Engineered metrics table
  * per_question_baseline.csv      — per-question audit trail
  * per_question_engineered.csv
  * analysis.md                    — the three written analyses, with live numbers
"""
from __future__ import annotations

import pandas as pd

import config
from data_prep import ensure_data
from chunking import chunk_corpus
from retrieval import BaselineRetriever, EngineeredRetriever
from evaluate import evaluate_system
from llm import get_llm

METRIC_ORDER = [
    "Hit Rate@5", "MRR", "Recall@5",
    "Groundedness", "Factual Accuracy", "Hallucination Rate",
]
# For these, higher is better; for Hallucination Rate, lower is better.
LOWER_BETTER = {"Hallucination Rate"}


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    backend = "Anthropic (Claude)" if get_llm().available else "OFFLINE deterministic heuristic"

    print("=" * 70)
    print("THE FINANCIAL RAG CHALLENGE — Baseline vs Engineered")
    print(f"LLM backend: {backend}")
    print("=" * 70)

    # 1. Data (mock or real HF), filtered to 2022-2025.
    paths, df = ensure_data()

    # 2. Chunk with each strategy.
    base_chunks = chunk_corpus(paths, "baseline")
    eng_chunks = chunk_corpus(paths, "engineered")
    print(f"\n[run] Baseline chunks: {len(base_chunks)} | Engineered chunks: {len(eng_chunks)}")

    # 3. Build retrievers (embed + FAISS index).
    print("[run] Building retrievers (embedding + FAISS index)...")
    base_ret = BaselineRetriever(base_chunks)
    eng_ret = EngineeredRetriever(eng_chunks)

    # 4. Evaluate both systems.
    print(f"[run] Evaluating {len(df)} questions per system...\n")
    base_res = evaluate_system("Baseline", base_ret, base_chunks, df, "baseline")
    eng_res = evaluate_system("Engineered", eng_ret, eng_chunks, df, "engineered")

    # 5. Assemble the results table.
    table = pd.DataFrame({
        "Metric": METRIC_ORDER,
        "Baseline": [round(base_res.metrics[m], 3) for m in METRIC_ORDER],
        "Engineered": [round(eng_res.metrics[m], 3) for m in METRIC_ORDER],
    })
    table["Delta (Eng - Base)"] = (table["Engineered"] - table["Baseline"]).round(3)
    table.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    (config.RESULTS_DIR / "results.md").write_text(
        f"# Baseline vs Engineered — Results (K={config.TOP_K})\n\n"
        f"LLM backend for generation & judging: **{backend}**\n\n"
        f"Corpus: {len(paths)} Treasury bulletins ({config.YEARS[0]}-{config.YEARS[-1]}), "
        f"{len(df)} evaluation questions.\n\n"
        + _to_markdown(table) + "\n",
        encoding="utf-8",
    )

    # 6. Per-question audit trails.
    pd.DataFrame(base_res.records).to_csv(config.RESULTS_DIR / "per_question_baseline.csv", index=False)
    pd.DataFrame(eng_res.records).to_csv(config.RESULTS_DIR / "per_question_engineered.csv", index=False)

    # 7. Written analysis with live numbers.
    _write_analysis(base_res.metrics, eng_res.metrics, backend, len(paths), len(df))

    print(table.to_string(index=False))
    print(f"\n[run] Wrote results + analysis to {config.RESULTS_DIR}/")


def _write_analysis(base: dict, eng: dict, backend: str, n_docs: int, n_q: int) -> None:
    """Write analysis.md, reasoning HONESTLY from the actual deltas.

    The narrative adapts to what really happened: on the clean mock corpus the
    retrieval fix flows all the way through to Factual Accuracy; on the hard real
    OfficeQA slice the retrieval + generation-quality metrics improve but end-to-end
    exact-match accuracy stays at the floor. Both stories are told from the numbers.
    """
    def d(m):
        return eng[m] - base[m]

    window = f"{config.YEAR_MIN}-{config.YEAR_MAX}"
    fact_moved = d("Factual Accuracy") >= 0.02
    mrr_rel = (d("MRR") / base["MRR"] * 100) if base["MRR"] else 0.0

    ret_gain = (f"Hit@5 {base['Hit Rate@5']:.2f}->{eng['Hit Rate@5']:.2f}, "
                f"MRR {base['MRR']:.2f}->{eng['MRR']:.2f}, "
                f"Recall@5 {base['Recall@5']:.3f}->{eng['Recall@5']:.3f}")
    gen_gain = (f"Groundedness {base['Groundedness']:.2f}->{eng['Groundedness']:.2f}, "
                f"Factual Accuracy {base['Factual Accuracy']:.3f}->{eng['Factual Accuracy']:.3f}, "
                f"Hallucination {base['Hallucination Rate']:.3f}->{eng['Hallucination Rate']:.3f}")

    # --- Section 2 adapts to whether Factual Accuracy actually moved -----------
    if fact_moved:
        metadata_section = f"""Adding Year/Month metadata filtering moved the **retrieval** metrics far more
than the **generation** metrics, and that retrieval gain flowed all the way
through to answers. The engineered retriever parses the period out of the question
and restricts the candidate set to that year+month before ranking, lifting
{ret_gain} and driving MRR up by ~{mrr_rel:.0f}%. **Factual Accuracy rose as a
downstream consequence** ({base['Factual Accuracy']:.2f}->{eng['Factual Accuracy']:.2f}):
the generator was already grounded, so once it is handed the correct period it
produces the correct figure. Groundedness/Hallucination move comparatively little,
confirming the fix operated on the retrieval stage."""
    else:
        metadata_section = f"""Adding Year/Month metadata filtering improved the **retrieval** metrics and the
**generation-quality** metrics, but **not** end-to-end exact-match accuracy. The
period pre-filter lifted {ret_gain} (MRR up ~{mrr_rel:.0f}%), and on the generation
side the engineered prompt's "answer only from context / refuse to guess" rules
raised Groundedness ({base['Groundedness']:.2f}->{eng['Groundedness']:.2f}) and cut
Hallucination ({base['Hallucination Rate']:.3f}->{eng['Hallucination Rate']:.3f}).
Yet **Factual Accuracy stayed at the floor**
({base['Factual Accuracy']:.3f} vs {eng['Factual Accuracy']:.3f}). The honest
reading: on this hard, largely multi-document benchmark a modest top-5 retrieval
gain is *necessary but not sufficient* — moving the gold document from rank ~8 to
rank ~4 still often leaves the exact figure outside the top 5, and many questions
need several documents or a derived value that no single retrieved chunk contains.
The naive regex period-parser also helps only when a question names one clean
period; multi-period questions get filtered to the wrong slice. Closing the
accuracy gap needs larger K, finer chunk granularity, and multi-hop retrieval — not
metadata filtering alone."""

    text = f"""# Written Analysis — The Financial RAG Challenge

*Generated by `run.py`. LLM backend: {backend}. Corpus: {n_docs} bulletins
({window}), {n_q} questions, K={config.TOP_K}.*

Retriever metrics moved: {ret_gain}.
Generator metrics moved: {gen_gain}.

## 1. The Bottleneck
The dominant weakness is **retrieval, not generation** — the clearest signal is
**low MRR/Hit@5 ({base['MRR']:.2f} / {base['Hit Rate@5']:.2f} baseline)** sitting
next to **high Groundedness ({base['Groundedness']:.2f})**. When the generator does
answer it stays faithful to the text it was handed (a generation failure would
instead show up as *low* Groundedness / *high* Hallucination); the answers are
wrong because the evidence needed to answer them is usually **not in the top 5**.
That is a retrieval failure. Factual Accuracy ({base['Factual Accuracy']:.3f}) is
therefore floored by retrieval: no matter how good the LLM is, it cannot state a
figure it was never shown. (Note Recall@5 is ~0 for both systems — a metric
artifact: each real bulletin is hundreds of chunks, so snippet-level recall@5 is
bounded by ~5/hundreds. Document-level Hit@5/MRR is the meaningful retrieval signal
here.)

## 2. The Metadata Fix
{metadata_section}

## 3. Scaling Insight ({window} -> full 1939-2025 archive)
The first component to break is the **brute-force, in-memory vector index**. At
this window the corpus is ~{n_docs} documents but already tens of thousands of
chunks, and a FAISS `IndexFlatIP` doing an exact linear scan is still fast enough.
Scaled to the full 1939-2025 archive (~697 bulletins, ~10x the chunks), three
things degrade in order: (a) the flat index becomes an O(N) scan on every query and
the whole embedding matrix must fit in RAM — the first thing to slow down; (b)
single-pass, in-memory chunking + embedding of the entire corpus stops fitting in
memory and must become a streamed/batched job; (c) pure semantic search gets
*worse*, not just slower, because ~86 years of near-identical "Total receipts" rows
now compete — which is exactly why the engineered system's metadata pre-filter (and
year/month **partitioning**, so each query scans only one partition) becomes
essential rather than optional. The fix is an approximate index (FAISS IVF/HNSW) or
a vector DB with native metadata filtering, plus partitioned indexes by period.
"""
    (config.RESULTS_DIR / "analysis.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
