"""Merge OpenAI Batch output JSONL files into gpt_labels.parquet.

Usage:
  python src/merge_batch_output.py file1.jsonl file2.jsonl ...
  python src/merge_batch_output.py "e:/download/batch_xxx_output.jsonl" "e:/download/batch_yyy_output.jsonl"

Filters out labels for news after 2024-12-31.
"""
from pathlib import Path
from typing import List

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
RAW_RAVENPACK_PATH = DATA_DIR / "ravenpack_dj_equities.parquet"
INTRADAY_STORY_PATH = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"
INTERIM_PATH = DATA_DIR / "interim" / "gpt_labels.parquet"
CUTOFF_DATE = "2024-12-31"


def _parse_label(raw_text: str) -> str:
    first = (raw_text or "").strip().split()[0].upper()
    return first if first in {"YES", "NO", "UNKNOWN"} else "UNKNOWN"


def _label_to_score(label: str) -> float:
    return {"YES": 1.0, "NO": -1.0, "UNKNOWN": 0.0}.get(label, 0.0)


def parse_jsonl(path: Path) -> List[dict]:
    """Parse batch output JSONL; return list of {rp_story_id, label, raw_response}."""
    import json

    rows = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        if not cid or rec.get("error"):
            continue
        resp = rec.get("response") or {}
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        raw = (choices[0]["message"]["content"]) if choices else ""
        label = _parse_label(raw)
        rows.append({
            "rp_story_id": cid,
            "label": label,
            "raw_response": raw,
        })
    return rows


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python merge_batch_output.py file1.jsonl file2.jsonl ...")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"Error: {p} not found")
            sys.exit(1)

    # Parse all JSONL files
    all_rows = []
    for p in paths:
        rows = parse_jsonl(p)
        all_rows.extend(rows)
        print(f"Parsed {p.name}: {len(rows):,} labels")

    if not all_rows:
        print("No labels to merge.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df = new_df.drop_duplicates(subset=["rp_story_id"], keep="last")

    # Get ticker, headline from ravenpack
    rp_raw = pd.read_parquet(RAW_RAVENPACK_PATH, columns=["rp_story_id", "headline", "ticker"])
    rp_raw = rp_raw.dropna(subset=["headline"]).drop_duplicates(subset=["rp_story_id"])
    rp_raw["rp_story_id"] = rp_raw["rp_story_id"].astype(str)
    new_df["rp_story_id"] = new_df["rp_story_id"].astype(str)
    new_df = new_df.merge(rp_raw, on="rp_story_id", how="left")
    new_df = new_df.dropna(subset=["ticker"])

    # Get date from ravenpack_intraday_story
    rp_story = pd.read_parquet(INTRADAY_STORY_PATH, columns=["rp_story_id", "ticker", "date"])
    rp_story = rp_story.drop_duplicates(subset=["rp_story_id", "ticker"])
    rp_story["rp_story_id"] = rp_story["rp_story_id"].astype(str)
    new_df = new_df.merge(rp_story, on=["rp_story_id", "ticker"], how="inner")
    new_df["date"] = pd.to_datetime(new_df["date"])

    # Filter out news after 2024-12-31
    new_df = new_df[new_df["date"] <= CUTOFF_DATE].copy()
    print(f"After date filter (<= {CUTOFF_DATE}): {len(new_df):,} labels")

    # Build gpt_labels format
    new_df = new_df.assign(
        score=new_df["label"].map(_label_to_score),
        model="gpt-4o-mini",
        created_at="",
        prompt_version="v1",
    )
    new_df = new_df[
        ["rp_story_id", "ticker", "headline", "label", "score", "model", "created_at", "raw_response", "prompt_version"]
    ]

    # Load existing, get dates, filter
    if INTERIM_PATH.exists():
        existing = pd.read_parquet(INTERIM_PATH)
        existing["rp_story_id"] = existing["rp_story_id"].astype(str)
        # Get date for existing labels
        rp_story = pd.read_parquet(INTRADAY_STORY_PATH, columns=["rp_story_id", "ticker", "date"])
        rp_story = rp_story.drop_duplicates(subset=["rp_story_id", "ticker"])
        rp_story["rp_story_id"] = rp_story["rp_story_id"].astype(str)
        ex_merged = existing.merge(rp_story, on=["rp_story_id", "ticker"], how="left")
        ex_merged["date"] = pd.to_datetime(ex_merged["date"])
        existing = ex_merged[ex_merged["date"].isna() | (ex_merged["date"] <= CUTOFF_DATE)].drop(columns=["date"])
        # Drop rows we're replacing with new
        existing = existing[~existing["rp_story_id"].isin(new_df["rp_story_id"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["rp_story_id"], keep="last")
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(INTERIM_PATH, index=False)
    print(f"Saved {INTERIM_PATH} | total rows: {len(combined):,}")


if __name__ == "__main__":
    main()
