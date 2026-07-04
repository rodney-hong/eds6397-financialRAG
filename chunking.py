"""
chunking.py — turn Treasury .txt documents into retrievable chunks.

Two strategies, so the Baseline vs Engineered comparison isolates the effect of
chunking quality:

  * Baseline (chunk_baseline): fixed-size, ~128-word windows, NO overlap, and
    completely table-blind. It happily cuts a Markdown table in half, orphaning
    the "Total receipts | 408,587" row from its "Table I" heading.

  * Engineered (chunk_engineered): ~512-word target with 64-word overlap, and
    TABLE-AWARE. A Markdown table (its heading + all its | rows) is kept together
    as one atomic chunk so a figure is never separated from its row label or the
    table title. Prose is packed into overlapping windows.

Every chunk carries Year/Month metadata (parsed from the source filename), plus
its source_file and a stable chunk_id. That metadata is what the Engineered
retriever filters on.

"Tokens" here are approximated by whitespace-delimited words. This keeps the
project dependency-light and fully offline; the sentence-embedding model applies
its own sub-word tokenizer at encode time. Word-count sizing is a documented
approximation, not an exact BPE count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import config
from data_prep import parse_year_month


@dataclass
class Chunk:
    text: str
    source_file: str
    year: int
    month: int
    chunk_id: str
    kind: str = "prose"  # "prose" or "table" (engineered only), for transparency
    meta: dict = field(default_factory=dict)


def _word_count(s: str) -> int:
    return len(s.split())


# --- Baseline: fixed-size, no overlap, table-blind ---------------------------
def chunk_baseline(text: str, source_file: str,
                   size: int = config.BASELINE_CHUNK_TOKENS) -> list[Chunk]:
    """Split into fixed ~`size`-word chunks with no overlap and no structure awareness."""
    y, m = parse_year_month(source_file)
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, len(words), size):
        piece = " ".join(words[i:i + size])
        if not piece.strip():
            continue
        chunks.append(Chunk(
            text=piece, source_file=source_file, year=y, month=m,
            chunk_id=f"{source_file}::base::{len(chunks)}",
        ))
    return chunks


# --- Engineered: table-aware, overlapping ------------------------------------
def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _segment_document(text: str) -> list[tuple[str, str]]:
    """Split raw text into ordered ("table"|"prose", block_text) segments.

    A table block is a run of consecutive "|"-lines, prefixed with the nearest
    preceding heading/label line so the table's title travels with it.
    """
    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    i = 0
    prose_buf: list[str] = []

    def flush_prose():
        if prose_buf:
            joined = "\n".join(prose_buf).strip()
            if joined:
                segments.append(("prose", joined))
            prose_buf.clear()

    while i < len(lines):
        if _is_table_line(lines[i]):
            # Grab the whole contiguous table.
            j = i
            table_lines = []
            while j < len(lines) and (_is_table_line(lines[j]) or not lines[j].strip()):
                if lines[j].strip():
                    table_lines.append(lines[j])
                j += 1
            # Attach the nearest non-empty preceding line (usually the table heading)
            # that we buffered as prose, so the table keeps its title.
            heading = ""
            if prose_buf:
                for k in range(len(prose_buf) - 1, -1, -1):
                    if prose_buf[k].strip():
                        heading = prose_buf[k].strip()
                        break
            flush_prose()
            block = (heading + "\n" if heading else "") + "\n".join(table_lines)
            segments.append(("table", block.strip()))
            i = j
        else:
            prose_buf.append(lines[i])
            i += 1
    flush_prose()
    return segments


def chunk_engineered(text: str, source_file: str,
                     size: int = config.ENGINEERED_CHUNK_TOKENS,
                     overlap: int = config.ENGINEERED_CHUNK_OVERLAP) -> list[Chunk]:
    """Table-aware chunking with overlapping prose windows.

    Tables become their own atomic chunks (never split mid-row). Prose is packed
    into ~`size`-word windows with `overlap`-word overlap between consecutive
    windows so context isn't lost at chunk boundaries.
    """
    y, m = parse_year_month(source_file)
    chunks: list[Chunk] = []

    def emit(piece: str, kind: str):
        if piece.strip():
            chunks.append(Chunk(
                text=piece.strip(), source_file=source_file, year=y, month=m,
                chunk_id=f"{source_file}::eng::{len(chunks)}", kind=kind,
            ))

    for kind, block in _segment_document(text):
        if kind == "table":
            # Keep the table intact even if it exceeds `size` — splitting a table
            # is exactly the failure we're avoiding.
            emit(block, "table")
            continue
        # Prose: overlapping word windows.
        words = block.split()
        if _word_count(block) <= size:
            emit(block, "prose")
            continue
        step = max(1, size - overlap)
        for i in range(0, len(words), step):
            emit(" ".join(words[i:i + size]), "prose")
            if i + size >= len(words):
                break
    return chunks


# --- Convenience: chunk a whole corpus ---------------------------------------
def chunk_corpus(corpus_paths: list[Path], strategy: str) -> list[Chunk]:
    """Chunk every file with the given strategy ('baseline' or 'engineered')."""
    fn = chunk_baseline if strategy == "baseline" else chunk_engineered
    out: list[Chunk] = []
    for p in corpus_paths:
        out.extend(fn(p.read_text(encoding="utf-8"), p.name))
    return out


if __name__ == "__main__":
    from data_prep import ensure_data
    paths, _ = ensure_data()
    base = chunk_corpus(paths, "baseline")
    eng = chunk_corpus(paths, "engineered")
    print(f"Baseline chunks:   {len(base)} (avg words "
          f"{sum(_word_count(c.text) for c in base) / len(base):.0f})")
    print(f"Engineered chunks: {len(eng)}  "
          f"({sum(c.kind == 'table' for c in eng)} table, "
          f"{sum(c.kind == 'prose' for c in eng)} prose)")
    # Show that the baseline chops a table, while engineered keeps it whole.
    print("\n--- example engineered TABLE chunk ---")
    print(next(c for c in eng if c.kind == "table").text)
