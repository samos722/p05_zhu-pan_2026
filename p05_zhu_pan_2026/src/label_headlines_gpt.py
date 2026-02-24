from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import openai

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
RAW_RAVENPACK_PATH = DATA_DIR / "ravenpack_dj_equities.parquet"
INTERIM_PATH = DATA_DIR / "interim" / "gpt_labels.parquet"

OPENAI_API_KEY = config("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY


@dataclass
class GPTLabel:
    rp_story_id: str
    ticker: str
    headline: str
    label: str  # YES / NO / UNKNOWN
    score: float
    model: str
    created_at: str  # ISO timestamp
    prompt_version: str = "v1"


PROMPT_TEMPLATE = """Forget all your previous instructions. Pretend you are a financial expert. You are
a financial expert with stock recommendation experience. Answer \"YES\" if good
news, \"NO\" if bad news, or \"UNKNOWN\" if uncertain in the first line. Then
elaborate with one short and concise sentence on the next line. Is this headline
good or bad for the stock price of {company} in the short term?
Headline: {headline}"""


def build_prompt(headline: str, company: str) -> str:
    return PROMPT_TEMPLATE.format(
        headline=headline.strip(),
        company=company.strip(),
    )


def parse_label(raw_text: str) -> str:
    first = (raw_text or "").strip().split()[0].upper()
    if first not in {"YES", "NO", "UNKNOWN"}:
        return "UNKNOWN"
    return first


def label_to_score(label: str) -> float:
    mapping = {"YES": 1.0, "NO": 0.0, "UNKNOWN": 0.5}
    return mapping.get(label, 0.5)


def load_already_labeled() -> pd.DataFrame:
    if INTERIM_PATH.exists():
        return pd.read_parquet(INTERIM_PATH)
    return pd.DataFrame(
        columns=[
            "rp_story_id",
            "headline",
            "label",
            "score",
            "model",
            "created_at",
            "prompt_version",
            "raw_response",
        ]
    )


def sample_headlines(n: int = 300) -> pd.DataFrame:
    """Sample unique headlines by rp_story_id from raw RavenPack data."""
    df = pd.read_parquet(RAW_RAVENPACK_PATH, columns=["rp_story_id", "headline", "ticker"])
    df = df.dropna(subset=["headline"]).drop_duplicates(subset=["rp_story_id"])

    existing = load_already_labeled()
    if not existing.empty:
        done_ids = set(existing["rp_story_id"])
        df = df[~df["rp_story_id"].isin(done_ids)]

    if len(df) == 0:
        return df

    n_sample = min(n, len(df))
    return df.sample(n=n_sample, random_state=0)


def call_gpt_on_headlines(
    df_sample: pd.DataFrame,
    model: str = "gpt-3.5-turbo",
) -> List[GPTLabel]:
    results: List[GPTLabel] = []
    for _, row in df_sample.iterrows():
        rp_story_id = str(row["rp_story_id"])
        headline = str(row["headline"])
        ticker = str(row.get("ticker", "") or "")
        company = ticker if ticker else "the company"
        prompt = build_prompt(headline, company)

        resp = openai.ChatCompletion.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful financial news classifier.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = resp["choices"][0]["message"]["content"]
        label = parse_label(raw_text)
        score = label_to_score(label)

        results.append(
            GPTLabel(
                rp_story_id=rp_story_id,
                ticker=ticker,
                headline=headline,
                label=label,
                score=score,
                model=model,
                created_at=datetime.utcnow().isoformat(),
                prompt_version="v1",
            )
        )
    return results


def save_results(new_labels: List[GPTLabel]) -> None:
    existing = load_already_labeled()
    if not new_labels:
        return

    df_new = pd.DataFrame([l.__dict__ for l in new_labels])
    # raw_response was not stored in dataclass; add empty column for now
    df_new["raw_response"] = pd.NA

    combined = pd.concat([existing, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["rp_story_id"], keep="last")

    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(INTERIM_PATH, index=False)


def main(sample_size: int = 300) -> None:
    df_sample = sample_headlines(sample_size)
    if df_sample.empty:
        print("No new headlines to label (all rp_story_id already in interim file).")
        return

    print(f"Labeling {len(df_sample):,} headlines with GPT...")
    labels = call_gpt_on_headlines(df_sample)
    save_results(labels)
    print(f"Saved labels for {len(labels):,} headlines to {INTERIM_PATH}")


if __name__ == "__main__":
    main(sample_size=300)

