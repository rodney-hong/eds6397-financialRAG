"""
retrieval.py — embed chunks, index them in FAISS, and retrieve top-K.

Vector DB choice: FAISS (IndexFlatIP).
  Why FAISS over Chroma here:
    * IndexFlatIP does EXACT inner-product (== cosine, on L2-normalized vectors)
      search — no ANN approximation error to muddy a Baseline-vs-Engineered
      metric comparison.
    * Zero server, pure in-memory, trivial to rebuild — ideal for a 4-year slice.
    * Metadata filtering is done transparently in-code by restricting the
      candidate set, which is easy to explain in a write-up. (Chroma's built-in
      `where` filter is convenient but hides that step and adds a heavier
      dependency; at this scale we don't need it.)

Two retrievers:
  * BaselineRetriever  — pure semantic top-K over ALL chunks. No metadata.
  * EngineeredRetriever — parses a Year/Month out of the question and PRE-FILTERS
    the candidate chunks to that period before ranking. If the question names no
    period (or the filter is empty), it gracefully falls back to a full search.
"""
from __future__ import annotations

import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from chunking import Chunk
from data_prep import MONTH_NAMES

# question -> (year, month) parsing for the Engineered retriever.
# Match any plausible 4-digit year; if the parsed year has no chunks the store's
# search() simply falls back to a full (unfiltered) search, so being liberal here
# is safe.
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MONTH_RE = re.compile(
    r"\b(" + "|".join(MONTH_NAMES[i] for i in range(1, 13)) + r")\b", re.IGNORECASE
)
_MONTH_TO_NUM = {name.lower(): num for num, name in MONTH_NAMES.items()}


def parse_query_period(question: str) -> tuple[int | None, int | None]:
    """Pull an explicit (year, month) out of a question, if present."""
    y = _YEAR_RE.search(question)
    m = _MONTH_RE.search(question)
    year = int(y.group(1)) if y else None
    month = _MONTH_TO_NUM[m.group(1).lower()] if m else None
    return year, month


# --- Embedding model loader (with graceful fallback) --------------------------
def _load_model(name: str) -> tuple[SentenceTransformer, str]:
    """Load a SentenceTransformer, falling back to the baseline model if needed."""
    try:
        return SentenceTransformer(name), name
    except Exception as e:  # offline / model unavailable
        print(f"[retrieval] Could not load '{name}' ({e}); "
              f"falling back to '{config.BASELINE_EMBED_MODEL}'.")
        return SentenceTransformer(config.BASELINE_EMBED_MODEL), config.BASELINE_EMBED_MODEL


class VectorStore:
    """A FAISS-backed store of embedded chunks plus parallel metadata arrays."""

    def __init__(self, embed_model: str):
        self.model, self.model_name = _load_model(embed_model)
        self.chunks: list[Chunk] = []
        self.years = np.array([], dtype=np.int32)
        self.months = np.array([], dtype=np.int32)
        self.emb: np.ndarray | None = None      # (N, d), L2-normalized
        self.index: faiss.Index | None = None

    def build(self, chunks: list[Chunk]) -> "VectorStore":
        self.chunks = chunks
        texts = [c.text for c in chunks]
        emb = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=64,
        ).astype(np.float32)
        self.emb = emb
        self.years = np.array([c.year for c in chunks], dtype=np.int32)
        self.months = np.array([c.month for c in chunks], dtype=np.int32)
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product on unit vectors = cosine
        self.index.add(emb)
        return self

    def _encode_query(self, question: str) -> np.ndarray:
        return self.model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def search(self, question: str, k: int,
               year: int | None = None, month: int | None = None
               ) -> list[tuple[Chunk, float]]:
        """Top-K by cosine similarity, optionally pre-filtered to year/month.

        When a year (and optionally month) is given, ranking runs only over the
        chunks from that period; otherwise the full FAISS index is searched.
        """
        q = self._encode_query(question)

        if year is not None:
            mask = self.years == year
            if month is not None:
                mask &= self.months == month
            cand = np.nonzero(mask)[0]
            if cand.size:  # only use the filter if it actually selects something
                sims = self.emb[cand] @ q[0]
                order = np.argsort(-sims)[:k]
                return [(self.chunks[cand[i]], float(sims[i])) for i in order]

        # Unfiltered path: exact FAISS search over everything.
        scores, idx = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]


class BaselineRetriever:
    """Pure semantic similarity, top-K, NO metadata filtering."""

    def __init__(self, chunks: list[Chunk]):
        self.store = VectorStore(config.BASELINE_EMBED_MODEL).build(chunks)

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[Chunk]:
        return [c for c, _ in self.store.search(question, k)]


class EngineeredRetriever:
    """Metadata-aware: pre-filter to the question's Year/Month, then rank."""

    def __init__(self, chunks: list[Chunk]):
        self.store = VectorStore(config.ENGINEERED_EMBED_MODEL).build(chunks)

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[Chunk]:
        year, month = parse_query_period(question)
        return [c for c, _ in self.store.search(question, k, year=year, month=month)]


if __name__ == "__main__":
    from data_prep import ensure_data
    from chunking import chunk_corpus

    paths, df = ensure_data()
    base = BaselineRetriever(chunk_corpus(paths, "baseline"))
    eng = EngineeredRetriever(chunk_corpus(paths, "engineered"))

    q = df.iloc[3]["question"]
    print(f"\nQ: {q}")
    print(f"gold: {df.iloc[3]['source_files']}")
    print("baseline top-5 files:  ", [c.source_file for c in base.retrieve(q)])
    print("engineered top-5 files:", [c.source_file for c in eng.retrieve(q)])
