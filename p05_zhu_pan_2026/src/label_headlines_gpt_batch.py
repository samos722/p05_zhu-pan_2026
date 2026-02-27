"""Label headlines using OpenAI Batch API (faster, ~50% cheaper).
Same API key as streaming. Results in hours instead of days.

Usage:
  python src/label_headlines_gpt_batch.py           # submit one 50k batch
  python src/label_headlines_gpt_batch.py --all     # submit all remaining (~22 batches)
  python src/label_headlines_gpt_batch.py --fetch   # fetch results of running/completed batches
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import polars as pl
import openai

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
RAW_RAVENPACK_PATH = DATA_DIR / "ravenpack_dj_equities.parquet"
INTRADAY_STORY_PATH = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"
INTERIM_PATH = DATA_DIR / "interim" / "gpt_labels.parquet"
BATCH_STATE_PATH = DATA_DIR / "interim" / "batch_jobs.json"

OPENAI_API_KEY = config("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

PROMPT_TEMPLATE = """Forget all your previous instructions. Pretend you are a financial expert. You are
a financial expert with stock recommendation experience. Answer \"YES\" if good
news, \"NO\" if bad news, or \"UNKNOWN\" if uncertain in the first line. Then
elaborate with one short and concise sentence on the next line. Is this headline
good or bad for the stock price of {company} in the short term?
Headline: {headline}"""

MAX_REQUESTS_PER_BATCH = 50_000


IN_FLIGHT_STATUSES = {"validating", "in_progress", "finalizing", "completed"}


def _get_in_flight_ids() -> set:
    """rp_story_ids in batches not yet merged (in-progress, completed, etc). Excludes failed/expired."""
    state = _load_batch_state()
    ids = set()
    for s in state:
        if s.get("status") in IN_FLIGHT_STATUSES or (s.get("status") != "merged" and s.get("status") not in {"failed", "expired", "cancelled"}):
            meta = s.get("meta") or {}
            ids.update(meta.keys())
    return ids


def _get_unlabeled_df() -> pd.DataFrame:
    stories = pl.read_parquet(INTRADAY_STORY_PATH, columns=["rp_story_id"])
    valid_ids = set(stories["rp_story_id"].unique().to_list())

    df = pd.read_parquet(RAW_RAVENPACK_PATH, columns=["rp_story_id", "headline", "ticker"])
    df = df.dropna(subset=["headline"]).drop_duplicates(subset=["rp_story_id"])
    df = df[df["rp_story_id"].isin(valid_ids)]

    if INTERIM_PATH.exists():
        done = pd.read_parquet(INTERIM_PATH)
        df = df[~df["rp_story_id"].isin(set(done["rp_story_id"]))]

    in_flight = _get_in_flight_ids()
    if in_flight:
        df = df[~df["rp_story_id"].isin(in_flight)]
        print(f"Excluding {len(in_flight):,} headlines already in submitted/in-progress batches.")

    return df.reset_index(drop=True)


def _parse_label(raw_text: str) -> str:
    first = (raw_text or "").strip().split()[0].upper()
    return first if first in {"YES", "NO", "UNKNOWN"} else "UNKNOWN"


def _label_to_score(label: str) -> float:
    return {"YES": 1.0, "NO": 0.0, "UNKNOWN": 0.5}.get(label, 0.5)


def _load_batch_state() -> list:
    if BATCH_STATE_PATH.exists():
        return json.loads(BATCH_STATE_PATH.read_text())
    return []


def _save_batch_state(state: list) -> None:
    BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATE_PATH.write_text(json.dumps(state, indent=2))


def build_and_submit_batch(n: int = MAX_REQUESTS_PER_BATCH) -> str | None:
    sync_batch_statuses()
    df = _get_unlabeled_df()
    if len(df) == 0:
        print("No unlabeled headlines.")
        return None

    df = df.head(n)
    rows = df.to_dict("records")

    jsonl_path = DATA_DIR / "interim" / "batch_input.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            rp = str(row["rp_story_id"])
            headline = str(row["headline"])
            ticker = str(row.get("ticker") or "")
            company = ticker or "the company"
            prompt = PROMPT_TEMPLATE.format(headline=headline.strip(), company=company.strip())

            req = {
                "custom_id": rp,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "You are a careful financial news classifier."},
                        {"role": "user", "content": prompt},
                    ],
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"Uploading {len(rows):,} requests...")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    meta = {row["rp_story_id"]: row for row in rows}
    state = _load_batch_state()
    state.append({
        "batch_id": batch.id,
        "count": len(rows),
        "meta": meta,
        "status": batch.status,
    })
    _save_batch_state(state)

    print(f"Submitted batch {batch.id} ({len(rows):,} requests)")
    return batch.id


def sync_batch_statuses() -> None:
    """Update all batch statuses from API (so failed/expired are marked correctly)."""
    state = _load_batch_state()
    for s in state:
        if s.get("status") == "merged":
            continue
        try:
            b = client.batches.retrieve(s["batch_id"])
            s["status"] = b.status
        except Exception:
            pass
    _save_batch_state(state)


def fetch_and_merge_results() -> int:
    state = _load_batch_state()
    merged = 0

    for i, s in enumerate(state):
        if s.get("status") in ("completed", "finalizing"):
            bid = s["batch_id"]
            try:
                b = client.batches.retrieve(bid)
                if b.status != "completed":
                    print(f"  Batch {bid} status: {b.status}")
                    continue
                out_id = b.output_file_id
                if not out_id:
                    continue
                content = client.files.content(out_id).read().decode()
                meta = s.get("meta", {})

                rows_out = []
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    rec = json.loads(line)
                    cid = rec.get("custom_id")
                    if cid not in meta or rec.get("error"):
                        continue
                    m = meta[cid]
                    resp = rec.get("response") or {}
                    body = resp.get("body") or {}
                    choices = body.get("choices") or []
                    raw = (choices[0]["message"]["content"]) if choices else ""
                    label = _parse_label(raw)
                    rows_out.append({
                        "rp_story_id": cid,
                        "ticker": m.get("ticker", ""),
                        "headline": m.get("headline", ""),
                        "label": label,
                        "score": _label_to_score(label),
                        "model": "gpt-4o-mini",
                        "created_at": "",
                        "raw_response": raw,
                        "prompt_version": "v1",
                    })

                existing = pd.read_parquet(INTERIM_PATH) if INTERIM_PATH.exists() else pd.DataFrame()
                new_df = pd.DataFrame(rows_out)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["rp_story_id"], keep="last")
                INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(INTERIM_PATH, index=False)
                merged += len(rows_out)
                s["status"] = "merged"
                _save_batch_state(state)
                print(f"  Merged batch {bid}: {len(rows_out):,} labels")
            except Exception as e:
                print(f"  Error fetching {bid}: {e}")

    return merged


def cancel_all_in_progress() -> None:
    """Cancel all batches with status in_progress or validating."""
    sync_batch_statuses()
    state = _load_batch_state()
    to_cancel = [s for s in state if s.get("status") in ("in_progress", "validating")]
    if not to_cancel:
        print("No in-progress or validating batches to cancel.")
        return
    for s in to_cancel:
        bid = s["batch_id"]
        try:
            b = client.batches.cancel(bid)
            s["status"] = b.status  # cancelling -> cancelled
            print(f"Cancelled {bid}")
        except Exception as e:
            print(f"Failed to cancel {bid}: {e}")
    _save_batch_state(state)
    print(f"Initiated cancel for {len(to_cancel)} batch(es). Status will become 'cancelled' in a few minutes.")


def poll_and_fetch(sleep_sec: int = 300) -> None:
    state = _load_batch_state()
    pending = [s for s in state if s.get("status") not in ("completed", "merged", "finalizing")]

    for s in pending:
        bid = s["batch_id"]
        b = client.batches.retrieve(bid)
        s["status"] = b.status
        print(f"  Batch {bid}: {b.status}")
    _save_batch_state(state)

    if any(s.get("status") == "completed" for s in state):
        fetch_and_merge_results()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Submit all remaining (multiple 50k batches)")
    p.add_argument("--max-batches", type=int, default=999, help="Max batches to submit per run (default 999). Use 2-4 to avoid token limit.")
    p.add_argument("--fetch", action="store_true", help="Fetch results of completed batches only")
    p.add_argument("--poll", action="store_true", help="Poll status and fetch if any completed")
    p.add_argument("--cancel", action="store_true", help="Cancel all in-progress batches")
    args = p.parse_args()

    if args.cancel:
        cancel_all_in_progress()
        return

    if args.fetch:
        sync_batch_statuses()
        n = fetch_and_merge_results()
        print(f"Merged {n:,} new labels.")
        return

    if args.poll:
        poll_and_fetch()
        return

    if args.all:
        total = 0
        for _ in range(args.max_batches):
            bid = build_and_submit_batch()
            if bid is None:
                break
            total += MAX_REQUESTS_PER_BATCH
            print(f"Submitted. Total this run: {total:,}. Run --fetch when batches complete.")
        if total > 0:
            print(f"Tip: Submit more with --all --max-batches 4 after in-progress batches complete.")
        return

    build_and_submit_batch()
    print("Run: python src/label_headlines_gpt_batch.py --fetch  (after batches complete, ~24h)")


if __name__ == "__main__":
    main()
