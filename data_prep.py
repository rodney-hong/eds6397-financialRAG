"""
data_prep.py — get the corpus + answer key onto disk, filtered to 2022-2025.

Two sources, one on-disk layout:

  * REAL data (if HF_TOKEN is set): downloads databricks/officeqa from the gated
    Hugging Face dataset — the transformed .txt Treasury bulletins and
    officeqa_full.csv. Swapping to real data is JUST setting HF_TOKEN and
    re-running; no code changes.

  * MOCK data (default): generates a small, self-consistent Treasury-style
    corpus + a matching answer CSV so the whole pipeline runs end-to-end today.
    Answers are baked into the documents, so retrieval and generation are
    actually gradeable.

Both paths produce:
    data/corpus/treasury_bulletin_YYYY_MM.txt
    data/officeqa_full.csv   (columns: uid, question, answer, source_docs,
                              source_files, difficulty)

Year/Month for every document come from the filename (treasury_bulletin_{YEAR}_{MONTH}.txt)
— that filename is the ONLY source of date metadata in the real dataset too,
which is exactly what the Engineered retriever filters on.

Filtering rule (auditable, printed at runtime):
  * A corpus file is kept iff its year is in config.YEARS.
  * A CSV row is kept iff ALL of its source files parse to a year in config.YEARS
    AND every referenced file survived the corpus filter. Rows with any
    out-of-range or missing source file are DROPPED (they can't be answered from
    the filtered corpus), and the reason is reported.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config

# --- Filename <-> (year, month) ----------------------------------------------
_FNAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})", re.IGNORECASE)

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def parse_year_month(name: str) -> tuple[int, int] | None:
    """Extract (year, month) from a filename/path, or None if it doesn't match."""
    m = _FNAME_RE.search(str(name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def split_source_files(cell: str) -> list[str]:
    """Split a CSV source_files cell into individual filenames.

    Real data may list multiple files; be liberal about the delimiter (comma,
    semicolon, or whitespace) and keep only *.txt basenames.
    """
    if not isinstance(cell, str):
        return []
    tokens = re.split(r"[;,\s]+", cell.strip())
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        base = Path(t).name
        if base.lower().endswith(".txt"):
            out.append(base)
    return out


# --- Public entry point -------------------------------------------------------
def ensure_data() -> tuple[list[Path], pd.DataFrame]:
    """Make sure filtered corpus + CSV exist on disk; return (corpus_paths, df).

    Uses real HF data when HF_TOKEN is set, otherwise generates mock data.
    """
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_TOKEN"):
        print("[data_prep] HF_TOKEN found -> attempting REAL databricks/officeqa download.")
        _download_real_data()
    else:
        print("[data_prep] No HF_TOKEN -> generating MOCK Treasury corpus + CSV.")
        _generate_mock_data()

    corpus_paths = _filter_corpus()
    df = _filter_csv(corpus_paths)
    return corpus_paths, df


# --- Filtering ----------------------------------------------------------------
def _filter_corpus() -> list[Path]:
    """Keep only corpus .txt files whose year is in config.YEARS."""
    all_txt = sorted(config.CORPUS_DIR.glob("treasury_bulletin_*.txt"))
    kept, dropped = [], []
    for p in all_txt:
        ym = parse_year_month(p.name)
        if ym and ym[0] in config.YEARS:
            kept.append(p)
        else:
            dropped.append(p.name)
    print(f"[data_prep] Corpus files: {len(all_txt)} found, "
          f"{len(kept)} kept in {config.YEARS}, {len(dropped)} out-of-range dropped.")
    if dropped:
        print(f"[data_prep]   dropped (out of range): {dropped}")
    return kept


def _filter_csv(corpus_paths: list[Path]) -> pd.DataFrame:
    """Keep only rows fully answerable from the filtered corpus. Report drops."""
    df = pd.read_csv(config.CSV_PATH, dtype=str).fillna("")
    kept_files = {p.name for p in corpus_paths}

    keep_rows, reasons = [], {}
    for idx, row in df.iterrows():
        files = split_source_files(row.get("source_files", ""))
        if not files:
            reasons[row.get("uid", idx)] = "no parseable source_files"
            continue
        years = [parse_year_month(f) for f in files]
        if any(ym is None for ym in years):
            reasons[row.get("uid", idx)] = "unparseable source filename"
            continue
        if not all(ym[0] in config.YEARS for ym in years):
            bad = [f for f, ym in zip(files, years) if ym[0] not in config.YEARS]
            reasons[row.get("uid", idx)] = f"source file(s) out of range: {bad}"
            continue
        missing = [f for f in files if f not in kept_files]
        if missing:
            reasons[row.get("uid", idx)] = f"source file(s) not in corpus: {missing}"
            continue
        keep_rows.append(idx)

    kept = df.loc[keep_rows].reset_index(drop=True)
    print(f"[data_prep] CSV rows: {len(df)} total, {len(kept)} kept, {len(reasons)} dropped.")
    for uid, why in list(reasons.items())[:10]:
        print(f"[data_prep]   dropped row {uid}: {why}")
    if len(reasons) > 10:
        print(f"[data_prep]   ... and {len(reasons) - 10} more.")
    # Attach parsed gold source-file list for downstream use.
    kept["gold_files"] = kept["source_files"].apply(split_source_files)
    return kept


# --- Real data (gated Hugging Face) -------------------------------------------
def _download_real_data() -> None:
    """Download the real corpus + officeqa_full.csv from Hugging Face.

    Requires prior access approval on https://huggingface.co/datasets/databricks/officeqa
    and a valid HF_TOKEN. To avoid pulling the whole (hundreds of files) corpus,
    we download ONLY the .txt files referenced by CSV rows that fall entirely
    inside the [YEAR_MIN, YEAR_MAX] window. Files land in the same on-disk layout
    the mock generator uses, so the rest of the pipeline is source-agnostic.
    """
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    repo = "databricks/officeqa"
    prefix = "treasury_bulletins_parsed/transformed"

    # 1. Answer key.
    csv_local = hf_hub_download(
        repo_id=repo, repo_type="dataset",
        filename="officeqa_full.csv", token=token,
    )
    df = pd.read_csv(csv_local, dtype=str).fillna("")
    df.to_csv(config.CSV_PATH, index=False)

    # 2. Which corpus files do the in-window rows actually need?
    needed: set[str] = set()
    for _, row in df.iterrows():
        files = split_source_files(row.get("source_files", ""))
        yms = [parse_year_month(f) for f in files]
        if files and all(ym and ym[0] in config.YEARS for ym in yms):
            needed.update(files)

    # 3. Download exactly those files.
    n = 0
    for name in sorted(needed):
        try:
            local = hf_hub_download(
                repo_id=repo, repo_type="dataset",
                filename=f"{prefix}/{name}", token=token,
            )
            (config.CORPUS_DIR / name).write_text(
                Path(local).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
            )
            n += 1
        except Exception as e:
            print(f"[data_prep]   could not fetch {name}: {type(e).__name__}: {e}")
    print(f"[data_prep] Downloaded real data: {len(df)} CSV rows total, "
          f"{n}/{len(needed)} in-window corpus files "
          f"(window {config.YEAR_MIN}-{config.YEAR_MAX}).")


# --- Mock data generator ------------------------------------------------------
@dataclass
class _DocFigures:
    receipts: int
    outlays: int
    deficit: int   # outlays - receipts (a shortfall); shown explicitly in a table row
    public_debt: int


def _figures_for(year: int, month: int, rng: random.Random) -> _DocFigures:
    """Deterministic, distinct-per-document financial figures (millions USD)."""
    receipts = rng.randint(280_000, 520_000)
    outlays = receipts + rng.randint(10_000, 90_000)   # always a deficit, for realism
    public_debt = rng.randint(28_000_000, 36_000_000)
    return _DocFigures(receipts, outlays, outlays - receipts, public_debt)


def _render_doc(year: int, month: int, f: _DocFigures) -> str:
    """Render one Treasury-bulletin-style .txt with Markdown tables.

    Table row LABELS are identical across every month/year — only the numbers and
    the header period differ. That is deliberate: it makes pure semantic search
    (Baseline) confuse months, which is precisely what Year/Month metadata
    filtering (Engineered) is meant to fix.
    """
    mname = MONTH_NAMES[month]
    return f"""# Treasury Bulletin — {mname} {year}

United States Department of the Treasury. This issue summarizes Federal fiscal
operations and the public debt for the reporting period ending {mname} {year}.
All amounts are in millions of dollars unless otherwise noted.

## Table I.—Summary of Federal Fiscal Operations

| Item | Amount |
| --- | --- |
| Total Federal budget receipts | {f.receipts:,} |
| Total Federal budget outlays | {f.outlays:,} |
| Budget deficit (outlays less receipts) | {f.deficit:,} |

Receipts for the period reflect individual income taxes, corporation income
taxes, social insurance taxes, and other collections. Outlays include payments
for national defense, income security, health, and net interest.

## Table II.—Public Debt Outstanding

| Item | Amount |
| --- | --- |
| Total public debt outstanding, end of period | {f.public_debt:,} |
| Debt held by the public | {int(f.public_debt * 0.78):,} |
| Intragovernmental holdings | {int(f.public_debt * 0.22):,} |

The total public debt outstanding at the end of {mname} {year} is shown above.
This bulletin is prepared by the Bureau of the Fiscal Service.
"""


def _generate_mock_data() -> None:
    """Write the mock corpus + officeqa_full.csv (idempotent, deterministic)."""
    rng = random.Random(6397)  # course-code seed :)

    # In-range documents: the mock demo uses the most recent 4 years of the
    # window (keeps the offline demo small regardless of how wide the real window
    # is) with every month => lots of same-looking distractors.
    mock_years = list(config.YEARS)[-4:]
    in_range = [(y, m) for y in mock_years for m in range(1, 13)]
    # Out-of-range documents (just outside the window) to prove the filter drops them.
    out_of_range = [(config.YEAR_MIN - 1, 12), (config.YEAR_MAX + 1, 1)]

    figures: dict[tuple[int, int], _DocFigures] = {}
    for (y, m) in in_range + out_of_range:
        f = _figures_for(y, m, rng)
        figures[(y, m)] = f
        path = config.CORPUS_DIR / f"treasury_bulletin_{y}_{m:02d}.txt"
        path.write_text(_render_doc(y, m, f), encoding="utf-8")

    # --- Build the answer CSV --------------------------------------------------
    rows = []
    uid = 0

    def add(question, answer, files, difficulty):
        nonlocal uid
        uid += 1
        rows.append({
            "uid": f"mock-{uid:03d}",
            "question": question,
            "answer": str(answer),
            # source_docs mimics the real Fraser-archive URL column (unused downstream).
            "source_docs": ";".join(
                f"https://fraser.stlouisfed.org/title/treasury-bulletin-407/{f}" for f in files
            ),
            "source_files": ";".join(files),
            "difficulty": difficulty,
        })

    # Sample ~6 questions per year, spread across months and question types, so the
    # number of (LLM-backed) evaluations stays bounded while the corpus is rich.
    qtypes = ["receipts", "outlays", "public_debt", "deficit"]
    for y in mock_years:
        chosen_months = rng.sample(range(1, 13), 6)
        for i, m in enumerate(chosen_months):
            f = figures[(y, m)]
            mname = MONTH_NAMES[m]
            fname = f"treasury_bulletin_{y}_{m:02d}.txt"
            qt = qtypes[i % len(qtypes)]
            if qt == "receipts":
                add(f"What were total Federal budget receipts, in millions of dollars, "
                    f"reported in the {mname} {y} Treasury Bulletin?",
                    f.receipts, [fname], "easy")
            elif qt == "outlays":
                add(f"What were total Federal budget outlays, in millions of dollars, in "
                    f"{mname} {y}?", f.outlays, [fname], "easy")
            elif qt == "public_debt":
                add(f"What was the total public debt outstanding at the end of "
                    f"{mname} {y}, in millions of dollars?", f.public_debt, [fname], "easy")
            else:  # deficit — a "hard" question (requires the right period AND the deficit row)
                add(f"By how many millions of dollars did outlays exceed receipts in "
                    f"{mname} {y}?", f.deficit, [fname], "hard")

    # One multi-source in-range question (Hit/MRR count ANY listed source file as correct).
    y = mock_years[-1]
    ma, mb = 3, 9
    fb = figures[(y, mb)]
    add(f"Total Federal budget receipts were reported for both {MONTH_NAMES[ma]} {y} and "
        f"{MONTH_NAMES[mb]} {y}. What were the receipts, in millions, for {MONTH_NAMES[mb]} {y}?",
        fb.receipts,
        [f"treasury_bulletin_{y}_{ma:02d}.txt", f"treasury_bulletin_{y}_{mb:02d}.txt"],
        "hard")

    # Two rows that MUST be dropped by the CSV filter (exercise the audit path).
    (lo_y, lo_m), (hi_y, hi_m) = out_of_range
    add(f"What were total Federal budget receipts in {MONTH_NAMES[lo_m]} {lo_y}?",
        figures[(lo_y, lo_m)].receipts,
        [f"treasury_bulletin_{lo_y}_{lo_m:02d}.txt"], "easy")   # below window -> dropped
    add(f"What were total Federal budget receipts in {MONTH_NAMES[hi_m]} {hi_y}?",
        figures[(hi_y, hi_m)].receipts,
        [f"treasury_bulletin_{hi_y}_{hi_m:02d}.txt"], "easy")   # above window -> dropped

    pd.DataFrame(rows).to_csv(config.CSV_PATH, index=False)
    print(f"[data_prep] Generated mock corpus: {len(in_range)} in-range + "
          f"{len(out_of_range)} out-of-range docs, {len(rows)} CSV rows "
          f"(2 designed to be filtered out).")


if __name__ == "__main__":
    paths, frame = ensure_data()
    print(f"\nReady: {len(paths)} corpus files, {len(frame)} evaluation questions.")
    print(frame[["uid", "question", "answer", "source_files", "difficulty"]].head(8).to_string(index=False))
