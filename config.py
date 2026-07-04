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
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # transformed .txt Treasury bulletins land here
CSV_PATH = DATA_DIR / "officeqa_full.csv"  # answer key (mock or real)
RESULTS_DIR = ROOT / "results"

# --- Timeframe filter ---------------------------------------------------------
# The assignment restricts everything to these four recent years. Both the
# corpus files AND the CSV rows are filtered to this range; a row is kept only
# if EVERY one of its source files falls inside the range (see data_prep.py).
YEARS = (2022, 2023, 2024, 2025)

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
