"""
Central configuration for the Financial RAG Challenge.

Everything tunable lives here so the Baseline vs Engineered comparison is
auditable at a glance. Nothing here should need editing to switch from the
mock corpus to the real (gated) Hugging Face data — that is driven purely by
whether HF_TOKEN is set in the environment (see data_prep.py).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no dependency).

    Existing environment variables win, so an explicitly exported value is never
    overridden. Called automatically on import so `HF_TOKEN` / `ANTHROPIC_API_KEY`
    placed in .env are picked up by the pipeline.
    """
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val:
            os.environ.setdefault(key, val)


load_env()
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # transformed .txt Treasury bulletins land here
CSV_PATH = DATA_DIR / "officeqa_full.csv"  # answer key (mock or real)
RESULTS_DIR = ROOT / "results"

# --- Timeframe filter ---------------------------------------------------------
# The analysis window. Both the corpus files AND the CSV rows are filtered to
# [YEAR_MIN, YEAR_MAX]; a row is kept only if EVERY one of its source files falls
# inside the window (see data_prep.py).
#
# The assignment specifies 2022-2025, but the real OfficeQA benchmark is a
# cross-decade set: only 3/246 questions fall entirely within 2022-2025. We
# therefore widen the REAL-data window to 2010-2025 (40 questions) so the
# Baseline-vs-Engineered metrics comparison is statistically meaningful. This
# deviation, and its rationale, are documented in the README. (Set both to
# 2022/2025 to honour the strict spec; the offline mock demo is unaffected.)
YEAR_MIN = 2010
YEAR_MAX = 2025
YEARS = tuple(range(YEAR_MIN, YEAR_MAX + 1))  # inclusive; membership used by filters

# --- Retrieval knobs ----------------------------------------------------------
TOP_K = 5  # both metric sets are computed at K=5

# Baseline: a small, general-purpose sentence embedding model, no metadata use.
BASELINE_EMBED_MODEL = "all-MiniLM-L6-v2"          # 384-d, ~90MB
# Engineered: a stronger retrieval-tuned model. Falls back to the baseline model
# automatically if it can't be downloaded (see retrieval.py).
ENGINEERED_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, retrieval-tuned

# --- Chunking knobs -----------------------------------------------------------
# "tokens" here are approximated by whitespace words (documented in chunking.py).
BASELINE_CHUNK_TOKENS = 128     # fixed-size, no overlap, table-blind
ENGINEERED_CHUNK_TOKENS = 512   # target size, table-aware
ENGINEERED_CHUNK_OVERLAP = 64   # overlap between engineered prose chunks

# --- Generation / judging -----------------------------------------------------
# Fast, cheap, deterministic (temperature 0) Claude model for both answering and
# LLM-as-judge. Overridable via env for experimentation.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEN_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 1024

# Factual-accuracy tolerance handed to the official reward.py scorer.
# 0.01 == "answer matches the CSV within +/-1%", exactly as the rubric asks.
FACTUAL_TOLERANCE = 0.01
