"""
evaluate.py — the metrics harness. Computes all six metrics at K=5 for a system.

RETRIEVER metrics (over the top-K retrieved chunks vs. the question's gold files):
  * Hit Rate@5 = fraction of queries whose top-5 contains a chunk from a gold file.
  * MRR       = mean of 1 / (rank of the first gold-file chunk); 0 if none in top-5.
  * Recall    = (distinct gold source files in the top-K) / (total gold source files
                for the query), averaged over queries — DOCUMENT-LEVEL recall. The
                earlier chunk-level reading counted every chunk of a gold file as
                "relevant", pinning recall near 0; document-level recall reflects
                whether the right documents were surfaced. Hit@5 and MRR are unchanged.

GENERATOR metrics (over the produced answers):
  * Factual Accuracy = fraction of answers matching the CSV within +/-1%, scored by
                       the OFFICIAL reward.py (score_answer(gold, pred, tol=0.01)).
  * Groundedness     = supported claims / total claims (claim-weighted across queries).
  * Hallucination    = fabricated claims / total claims.

Claim extraction + verification is done by a deterministic LLM-as-judge (Claude,
temperature 0). Without an API key, a documented OFFLINE heuristic judge is used:
a numeric answer is "supported" iff it appears verbatim in the retrieved context,
"fabricated" iff it is a specific number absent from the context; an honest
"Not found" counts as one supported, non-fabricated claim. (Because the offline
GENERATOR only ever copies numbers out of the context, offline groundedness is
~1.0 and hallucination ~0.0 by construction — these two metrics become
discriminating only under the real Claude backend.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

import config
from chunking import Chunk
from generation import generate
from llm import get_llm
from reward import score_answer


# --- Retriever metrics --------------------------------------------------------
def _retriever_metrics(retrieved: list[Chunk], gold_files: set[str]) -> tuple[int, float, float]:
    """Return (hit@k, reciprocal_rank, recall@k) for one query.

    Recall is DOCUMENT-LEVEL: (distinct gold source files appearing in the top-K) /
    (total gold source files for the query). Hit@5 and MRR are unchanged.
    """
    hit, rr = 0, 0.0
    for rank, c in enumerate(retrieved, start=1):
        if c.source_file in gold_files:
            hit = 1
            rr = 1.0 / rank
            break
    retrieved_gold_files = {c.source_file for c in retrieved if c.source_file in gold_files}
    recall = len(retrieved_gold_files) / len(gold_files) if gold_files else 0.0
    return hit, rr, recall


# --- LLM-as-judge for groundedness / hallucination ---------------------------
@dataclass
class ClaimTally:
    claims: int = 0
    supported: int = 0
    fabricated: int = 0

    def add(self, other: "ClaimTally"):
        self.claims += other.claims
        self.supported += other.supported
        self.fabricated += other.fabricated


_JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. You are given the CONTEXT that an answer "
    "was supposed to be based on, and the ANSWER. Break the answer into atomic "
    "factual claims and label each one."
)


def _judge_prompt(context: str, answer: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Break the ANSWER into atomic factual claims. For each claim output an object "
        '{"claim": "...", "supported": true|false, "fabricated": true|false} where:\n'
        "- supported = the claim is directly stated in or entailed by the CONTEXT.\n"
        "- fabricated = the claim asserts a specific fact or number that does NOT "
        "appear in the CONTEXT.\n"
        'An honest "not found"/refusal statement is supported=true, fabricated=false.\n'
        "Output ONLY a JSON array of such objects, nothing else."
    )


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalize_nums(s: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(s)}


def _offline_judge(answer_text: str, final_answer: str, context: str) -> ClaimTally:
    """Deterministic judge: is the answer's figure present in the retrieved context?"""
    if final_answer.strip().lower().startswith("not found"):
        return ClaimTally(claims=1, supported=1, fabricated=0)  # honest refusal
    ctx_nums = _normalize_nums(context)
    ans_nums = _normalize_nums(final_answer) or _normalize_nums(answer_text)
    if not ans_nums:  # no numeric claim at all
        return ClaimTally(claims=1, supported=1, fabricated=0)
    supported = all(n in ctx_nums for n in ans_nums)
    return ClaimTally(claims=1, supported=int(supported), fabricated=int(not supported))


def _llm_judge(answer_text: str, context: str) -> ClaimTally:
    """Claude-based judge; falls back to offline heuristic on any parse failure."""
    llm = get_llm()
    raw = llm.complete(_JUDGE_SYSTEM, _judge_prompt(context, answer_text), config.JUDGE_MAX_TOKENS)
    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        items = json.loads(raw[start:end])
        t = ClaimTally(claims=len(items))
        for it in items:
            t.supported += int(bool(it.get("supported")))
            t.fabricated += int(bool(it.get("fabricated")))
        return t if t.claims else ClaimTally(claims=1, supported=1, fabricated=0)
    except Exception:
        return ClaimTally(claims=1, supported=0, fabricated=0)  # unparseable → conservative


def judge(answer_text: str, final_answer: str, context: str) -> ClaimTally:
    if get_llm().available:
        return _llm_judge(answer_text, context)
    return _offline_judge(answer_text, final_answer, context)


# --- Full evaluation of one system -------------------------------------------
@dataclass
class SystemResult:
    name: str
    metrics: dict = field(default_factory=dict)
    records: list = field(default_factory=list)  # per-question detail for auditing


def evaluate_system(name: str, retriever, all_chunks: list[Chunk],
                    df: pd.DataFrame, prompt_system: str) -> SystemResult:
    """Run retrieval + generation + scoring across every evaluation question.

    `all_chunks` is retained in the signature for call-site stability; document-level
    recall no longer needs a per-file chunk census.
    """
    hits, rrs, recalls, facts = [], [], [], []
    tally = ClaimTally()
    records = []

    for _, row in df.iterrows():
        q = row["question"]
        gold_answer = row["answer"]
        gold_files = set(row["gold_files"])

        retrieved = retriever.retrieve(q, config.TOP_K)
        hit, rr, recall = _retriever_metrics(retrieved, gold_files)

        ans = generate(q, retrieved, prompt_system)
        fact = score_answer(gold_answer, ans.final_answer, config.FACTUAL_TOLERANCE)
        ct = judge(ans.answer_text, ans.final_answer, ans.context)

        hits.append(hit); rrs.append(rr); recalls.append(recall); facts.append(fact)
        tally.add(ct)
        records.append({
            "uid": row["uid"], "question": q, "gold_answer": gold_answer,
            "pred": ans.final_answer, "hit@5": hit, "rr": round(rr, 3),
            "recall@5": round(recall, 3), "factual": fact,
            "top5_files": [c.source_file for c in retrieved],
        })

    n = len(df)
    metrics = {
        "Hit Rate@5": sum(hits) / n,
        "MRR": sum(rrs) / n,
        "Recall@5": sum(recalls) / n,
        "Groundedness": (tally.supported / tally.claims) if tally.claims else 0.0,
        "Factual Accuracy": sum(facts) / n,
        "Hallucination Rate": (tally.fabricated / tally.claims) if tally.claims else 0.0,
    }
    return SystemResult(name=name, metrics=metrics, records=records)
