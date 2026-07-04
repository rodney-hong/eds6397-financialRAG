"""
smoke_test.py — small end-to-end check against the REAL data + REAL Claude.

Downloads just the answer CSV and a handful of real Treasury .txt files from the
gated Hugging Face dataset, then runs a few questions through BOTH the baseline
and engineered systems using the live Claude backend. Purpose: confirm HF_TOKEN
and ANTHROPIC_API_KEY are picked up and the real data downloads correctly, before
running the full pipeline at scale.

    python smoke_test.py

Nothing is written to results/; this only prints.
"""
from __future__ import annotations

import os

import config  # importing loads .env into os.environ

from data_prep import parse_year_month, split_source_files, MONTH_NAMES
from chunking import chunk_corpus
from retrieval import BaselineRetriever, EngineeredRetriever
from generation import generate
from llm import get_llm
from reward import score_answer

N_QUESTIONS = 3
N_DISTRACTOR_FILES = 6  # extra same-period docs so retrieval has something to confuse
SMOKE_DIR = config.DATA_DIR / "_smoke_corpus"
HF_REPO = "databricks/officeqa"
HF_TXT_PREFIX = "treasury_bulletins_parsed/transformed"


def _mask(v: str) -> str:
    return f"set (...{v[-4:]})" if v else "MISSING"


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "")
    print("=" * 68)
    print("SMOKE TEST — real data + real Claude")
    print(f"  HF_TOKEN          : {_mask(token)}")
    print(f"  ANTHROPIC_API_KEY : {_mask(os.environ.get('ANTHROPIC_API_KEY',''))}")
    print("=" * 68)
    if not token:
        raise SystemExit("HF_TOKEN not found — cannot download the gated dataset.")

    llm = get_llm()  # prints which backend is active
    if not llm.available:
        raise SystemExit("Anthropic backend not available — check ANTHROPIC_API_KEY.")

    # 1. Download the answer key and pick a few easy, single-source, in-range rows.
    print("\n[1/4] Downloading officeqa_full.csv ...")
    csv_local = hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                filename="officeqa_full.csv", token=token)
    df = pd.read_csv(csv_local, dtype=str).fillna("")
    print(f"      CSV loaded: {len(df)} total rows. Columns: {list(df.columns)}")

    df["gold_files"] = df["source_files"].apply(split_source_files)

    def in_range_single(row) -> bool:
        files = row["gold_files"]
        if len(files) != 1:
            return False
        ym = parse_year_month(files[0])
        return ym is not None and ym[0] in config.YEARS

    cand = df[df.apply(in_range_single, axis=1)]
    if "difficulty" in cand.columns and (cand["difficulty"] == "easy").any():
        cand = cand[cand["difficulty"] == "easy"]
    chosen = cand.head(N_QUESTIONS)
    if len(chosen) == 0:
        raise SystemExit("No in-range single-source rows found in the real CSV.")
    print(f"      Selected {len(chosen)} question(s) in {config.YEARS}.")

    # 2. Figure out which real .txt files to download: the referenced ones + a few
    #    same-period distractors so baseline retrieval has competitors.
    needed = {f for files in chosen["gold_files"] for f in files}
    years = {parse_year_month(f)[0] for f in needed}
    for _, row in df.iterrows():
        for f in row["gold_files"]:
            ym = parse_year_month(f)
            if ym and ym[0] in years and f not in needed:
                needed.add(f)
        if len(needed) >= len(chosen) + N_DISTRACTOR_FILES:
            break

    # 3. Download those .txt files.
    print(f"\n[2/4] Downloading {len(needed)} real corpus file(s) ...")
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for name in sorted(needed):
        try:
            p = hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                filename=f"{HF_TXT_PREFIX}/{name}", token=token)
            dest = SMOKE_DIR / name
            text = open(p, encoding="utf-8", errors="ignore").read()
            dest.write_text(text, encoding="utf-8")
            local_paths.append(dest)
            print(f"      {name:32s} {len(text):>8,d} chars")
        except Exception as e:
            print(f"      {name:32s} SKIPPED ({type(e).__name__}: {e})")
    # Keep only questions whose gold file actually downloaded.
    have = {p.name for p in local_paths}
    chosen = chosen[chosen["gold_files"].apply(lambda fs: all(f in have for f in fs))]
    print(f"      {len(local_paths)} files on disk; {len(chosen)} question(s) answerable.")

    # 4. Build retrievers over the downloaded docs and answer with real Claude.
    print("\n[3/4] Building baseline + engineered retrievers over the sample ...")
    base_ret = BaselineRetriever(chunk_corpus(local_paths, "baseline"))
    eng_ret = EngineeredRetriever(chunk_corpus(local_paths, "engineered"))

    print("\n[4/4] Answering with the real Claude backend:\n")
    for _, row in chosen.iterrows():
        q, gold = row["question"], row["answer"]
        gold_files = set(row["gold_files"])
        b_chunks, e_chunks = base_ret.retrieve(q), eng_ret.retrieve(q)
        b_ans = generate(q, b_chunks, "baseline")
        e_ans = generate(q, e_chunks, "engineered")

        print("-" * 68)
        print(f"Q ({row.get('difficulty','?')}): {q}")
        print(f"gold answer : {gold}   [gold file(s): {sorted(gold_files)}]")
        print(f"  BASELINE   top file: {b_chunks[0].source_file if b_chunks else '-'}")
        print(f"             answer  : {b_ans.final_answer!r}  "
              f"factual={score_answer(gold, b_ans.final_answer, config.FACTUAL_TOLERANCE)}")
        print(f"  ENGINEERED top file: {e_chunks[0].source_file if e_chunks else '-'}")
        print(f"             answer  : {e_ans.final_answer!r}  "
              f"factual={score_answer(gold, e_ans.final_answer, config.FACTUAL_TOLERANCE)}")
    print("-" * 68)
    print("\nSmoke test complete. Both tokens work and real Claude answered. "
          "Safe to run `python run.py` at scale.")


if __name__ == "__main__":
    main()
