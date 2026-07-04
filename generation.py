"""
generation.py — turn retrieved chunks + a question into an answer.

Two prompt templates, matching the two systems:

  * Baseline (SIMPLE): dump the retrieved context and ask for the answer. No
    grounding guardrails.

  * Engineered (GROUNDED): explicitly instruct the model to answer ONLY from the
    retrieved context, cite the source file, and refuse to guess a number that
    isn't present. This is the generation-side half of the "engineered" system.

Both templates ask the model to emit a machine-readable
`<FINAL_ANSWER>...</FINAL_ANSWER>` tag, which the official reward.py scorer knows
how to extract — so Factual Accuracy measures CORRECTNESS, not answer verbosity.

Every answer is returned as GeneratedAnswer(answer_text, final_answer, context),
where `context` is the exact retrieved text the judge later checks claims against.

Offline fallback (no ANTHROPIC_API_KEY): a deterministic keyword→table-row
extractor reads the answer straight out of the retrieved chunks. This lets the
whole pipeline run without a key; with a key you get real Claude generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config
from chunking import Chunk
from llm import get_llm
from reward import extract_final_answer

# --- Offline extractor: question keyword -> the table-row label to read --------
# Order matters: "outlays exceed receipts" must map to the deficit row, so the
# deficit/exceed check comes before the plain outlays/receipts checks.
_LABELS = [
    ("public debt", "Total public debt outstanding, end of period"),
    ("exceed", "Budget deficit"),
    ("deficit", "Budget deficit"),
    ("outlays", "Total Federal budget outlays"),
    ("receipts", "Total Federal budget receipts"),
]
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")  # must start with a digit (not a stray comma)


@dataclass
class GeneratedAnswer:
    answer_text: str       # full model output (used by the groundedness judge)
    final_answer: str      # extracted concise answer (used by factual-accuracy)
    context: str           # concatenated retrieved chunk text (the "sources")


def _format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks with their source file, for the prompt and the judge."""
    return "\n\n".join(
        f"[source: {c.source_file}]\n{c.text}" for c in chunks
    )


def _offline_answer(question: str, chunks: list[Chunk]) -> str:
    """Deterministic extractive answer: read the queried figure from retrieved text.

    Finds the relevant row label, then returns the FIRST number appearing after it
    (the Amount column). Works whether the chunk preserved the table structure
    (engineered) or flattened it into one line (baseline) — and, crucially, it
    reads from whichever document the retriever actually returned, so a baseline
    that retrieved the wrong month produces a confidently-wrong number rather than
    silence.
    """
    label = next((lab for key, lab in _LABELS if key in question.lower()), None)
    if label is None:
        return ""
    for c in chunks:
        low = c.text.lower()
        pos = low.find(label.lower())
        if pos == -1:
            continue
        after = c.text[pos + len(label):]
        m = _NUM_RE.search(after)
        if m:
            return m.group(0).replace(",", "")
    return ""


# --- Prompt templates ---------------------------------------------------------
_BASELINE_SYSTEM = (
    "You answer questions about U.S. Treasury records using the provided context."
)


def _baseline_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in one short sentence, then output the value on its own line as "
        "<FINAL_ANSWER>value</FINAL_ANSWER>."
    )


_ENGINEERED_SYSTEM = (
    "You are a careful financial analyst answering questions from U.S. Treasury "
    "Bulletins. Follow these rules strictly:\n"
    "1. Use ONLY the retrieved context below — never prior knowledge or guesses.\n"
    "2. Every figure you state must appear verbatim in the context.\n"
    "3. Cite the source file (e.g. treasury_bulletin_2024_03.txt) you used.\n"
    "4. If the answer is not in the context, say 'Not found in the provided context' "
    "and do not invent a number."
)


def _engineered_prompt(question: str, context: str) -> str:
    return (
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in one sentence, cite the source file, and then output the value "
        "on its own line as <FINAL_ANSWER>value</FINAL_ANSWER>. If the value is not "
        "in the context, put <FINAL_ANSWER>Not found</FINAL_ANSWER>."
    )


def generate(question: str, chunks: list[Chunk], system: str) -> GeneratedAnswer:
    """Generate an answer for one system ('baseline' or 'engineered')."""
    context = _format_context(chunks)
    llm = get_llm()

    if system == "baseline":
        sys_prompt, user_prompt = _BASELINE_SYSTEM, _baseline_prompt(question, context)
    else:
        sys_prompt, user_prompt = _ENGINEERED_SYSTEM, _engineered_prompt(question, context)

    if llm.available:
        text = llm.complete(sys_prompt, user_prompt, config.GEN_MAX_TOKENS)
        final = extract_final_answer(text) or text
    else:
        # Deterministic offline fallback (no API key).
        value = _offline_answer(question, chunks)
        if value:
            text = (f"Based on the retrieved context, the value is {value} "
                    f"(source: {chunks[0].source_file}). <FINAL_ANSWER>{value}</FINAL_ANSWER>")
            final = value
        else:
            text = "Not found in the provided context. <FINAL_ANSWER>Not found</FINAL_ANSWER>"
            final = "Not found"

    return GeneratedAnswer(answer_text=text, final_answer=final, context=context)


if __name__ == "__main__":
    from data_prep import ensure_data
    from chunking import chunk_corpus
    from retrieval import BaselineRetriever, EngineeredRetriever

    paths, df = ensure_data()
    base = BaselineRetriever(chunk_corpus(paths, "baseline"))
    eng = EngineeredRetriever(chunk_corpus(paths, "engineered"))
    row = df.iloc[0]
    q, gold = row["question"], row["answer"]
    print(f"Q: {q}\ngold answer: {gold}\n")
    ba = generate(q, base.retrieve(q), "baseline")
    ea = generate(q, eng.retrieve(q), "engineered")
    print("BASELINE  final:", ba.final_answer)
    print("ENGINEERED final:", ea.final_answer)
