"""Step 5: Compute Initial Reaction, Drift, and Portfolio Performance.

Methodology (following the paper):

Intraday news (9:30-16:00 ET):
    Initial Reaction = (mid_t15 - prev_close) / prev_close
    Drift            = (close_next_day - close_today) / close_today

Overnight news (before 9:00 or after 16:00):
    Initial Reaction = (open_today - prev_close) / prev_close
    Drift            = (close_today - open_today) / open_today

Portfolios (daily rebalanced, equal-weighted):
    Long-Short : buy positive - sell negative (both legs >= 2 firms)
    Long-Only  : buy positive predictions
    Short-Only : sell negative predictions

Outputs:
    _data/clean/event_returns.parquet   (story-level event returns)
    _data/clean/portfolio_daily.parquet (daily portfolio returns + summary)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
GPT_LABELS_PATH = DATA_DIR / "interim" / "gpt_labels.parquet"
INTRADAY_STORY_PATH = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"
TAQ_MINUTE_PATH = DATA_DIR / "clean" / "taq_nbbo_minute.parquet"
CRSP_PATH = DATA_DIR / "CRSP_daily_stock.parquet"

OUT_EVENT_PATH = DATA_DIR / "clean" / "event_returns.parquet"
OUT_PORTFOLIO_PATH = DATA_DIR / "clean" / "portfolio_daily.parquet"


# ---------------------------------------------------------------------------
# 1. Load & prepare CRSP daily prices with prev/next close
# ---------------------------------------------------------------------------

def load_crsp_prices() -> pl.DataFrame:
    df = pl.read_parquet(
        CRSP_PATH, columns=["date", "ticker", "openprc", "closeprc"]
    )
    df = df.with_columns(
        pl.col("date").cast(pl.Date).alias("date"),
        pl.col("closeprc").abs().alias("closeprc"),
        pl.col("openprc").abs().alias("openprc"),
    )
    df = df.sort(["ticker", "date"])
    df = df.with_columns(
        pl.col("closeprc").shift(1).over("ticker").alias("prev_close"),
        pl.col("closeprc").shift(-1).over("ticker").alias("close_next"),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Build story-level event returns
# ---------------------------------------------------------------------------

def compute_event_returns() -> pl.DataFrame:
    # --- GPT labels ---
    labels = pl.read_parquet(GPT_LABELS_PATH)
    labels = labels.filter(
        pl.col("ticker").is_not_null() & (pl.col("ticker") != "")
    )
    labels = labels.select([
        "rp_story_id", "ticker", "headline", "label", "score",
    ])

    # --- Intraday stories (with t15) ---
    stories = pl.read_parquet(
        INTRADAY_STORY_PATH,
        columns=["rp_story_id", "ticker", "date", "is_intraday", "t15"],
    )

    # Join: labels <-> stories
    events = labels.join(stories, on=["rp_story_id", "ticker"], how="inner")

    # --- Prepare t15 for TAQ join: strip timezone to naive datetime ---
    events = events.with_columns(
        pl.col("t15").dt.replace_time_zone(None).cast(pl.Datetime("ns")).alias("t15_naive"),
    )

    # --- TAQ minute mid prices ---
    taq = pl.read_parquet(TAQ_MINUTE_PATH)
    taq = taq.with_columns(pl.col("date").cast(pl.Date).alias("taq_date"))

    # Join events with TAQ to get mid price at t+15
    events = events.join(
        taq.select([
            pl.col("taq_date"),
            pl.col("ticker"),
            pl.col("minute_ts"),
            pl.col("mid").alias("mid_t15"),
        ]),
        left_on=["date", "ticker", "t15_naive"],
        right_on=["taq_date", "ticker", "minute_ts"],
        how="left",
    )

    # --- CRSP daily prices ---
    crsp = load_crsp_prices()

    events = events.join(
        crsp.select(["date", "ticker", "openprc", "closeprc", "prev_close", "close_next"]),
        on=["date", "ticker"],
        how="left",
    )

    # --- Compute returns ---
    events = events.with_columns(
        # Intraday Initial Reaction
        pl.when(pl.col("is_intraday"))
        .then((pl.col("mid_t15") - pl.col("prev_close")) / pl.col("prev_close"))
        .otherwise((pl.col("openprc") - pl.col("prev_close")) / pl.col("prev_close"))
        .alias("initial_reaction"),

        # Drift
        pl.when(pl.col("is_intraday"))
        .then((pl.col("close_next") - pl.col("closeprc")) / pl.col("closeprc"))
        .otherwise((pl.col("closeprc") - pl.col("openprc")) / pl.col("openprc"))
        .alias("drift"),
    )

    OUT_EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.write_parquet(OUT_EVENT_PATH)

    n_matched = events.filter(pl.col("mid_t15").is_not_null()).height
    print(f"[event_returns] {events.height} stories, {n_matched} matched TAQ t15 price")
    return events


# ---------------------------------------------------------------------------
# 3. Firm-day aggregation
# ---------------------------------------------------------------------------

def aggregate_firm_day(events: pl.DataFrame) -> pl.DataFrame:
    """Aggregate multiple headlines per (ticker, date) to firm-day sentiment."""
    firm_day = (
        events.group_by(["ticker", "date"])
        .agg(
            pl.col("score").mean().alias("avg_score"),
            pl.col("label").count().alias("n_headlines"),
            pl.col("initial_reaction").mean().alias("initial_reaction"),
            pl.col("drift").mean().alias("drift"),
        )
    )
    firm_day = firm_day.with_columns(
        pl.when(pl.col("avg_score") > 0.5).then(pl.lit("positive"))
        .when(pl.col("avg_score") < 0.5).then(pl.lit("negative"))
        .otherwise(pl.lit("neutral"))
        .alias("sentiment"),
    )
    return firm_day


# ---------------------------------------------------------------------------
# 4. Portfolio construction & performance
# ---------------------------------------------------------------------------

def build_portfolios(firm_day: pl.DataFrame) -> pl.DataFrame:
    """Build daily Long-Short, Long-Only, Short-Only portfolio returns."""
    dates = firm_day["date"].unique().sort()
    rows = []

    for date_val in dates:
        day = firm_day.filter(pl.col("date") == date_val)
        pos = day.filter(pl.col("sentiment") == "positive")
        neg = day.filter(pl.col("sentiment") == "negative")
        n_pos, n_neg = pos.height, neg.height
        n_neutral = day.filter(pl.col("sentiment") == "neutral").height

        ir_long = pos["initial_reaction"].drop_nulls().mean() if n_pos > 0 else None
        ir_short = neg["initial_reaction"].drop_nulls().mean() if n_neg > 0 else None
        dr_long = pos["drift"].drop_nulls().mean() if n_pos > 0 else None
        dr_short = neg["drift"].drop_nulls().mean() if n_neg > 0 else None

        # Long-Short only when both legs have >= 2 firms
        if n_pos >= 2 and n_neg >= 2:
            ir_ls = (ir_long or 0) - (ir_short or 0)
            dr_ls = (dr_long or 0) - (dr_short or 0)
        else:
            ir_ls = None
            dr_ls = None

        rows.append({
            "date": date_val,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_neutral": n_neutral,
            "ir_long_only": ir_long,
            "ir_short_only": -ir_short if ir_short is not None else None,
            "ir_long_short": ir_ls,
            "drift_long_only": dr_long,
            "drift_short_only": -dr_short if dr_short is not None else None,
            "drift_long_short": dr_ls,
        })

    portfolio = pl.DataFrame(rows)
    OUT_PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    portfolio.write_parquet(OUT_PORTFOLIO_PATH)
    return portfolio


def summarize(portfolio: pl.DataFrame, firm_day: pl.DataFrame) -> None:
    """Print summary statistics matching the paper's table format."""
    n_firm_day = firm_day.height
    n_trading_days = portfolio.height

    metrics = [
        ("ir_long_short", "Initial Reaction", "Long-Short"),
        ("ir_long_only", "Initial Reaction", "Long-Only"),
        ("ir_short_only", "Initial Reaction", "Short-Only"),
        ("drift_long_short", "Drift", "Long-Short"),
        ("drift_long_only", "Drift", "Long-Only"),
        ("drift_short_only", "Drift", "Short-Only"),
    ]

    print("\n" + "=" * 70)
    print("Intraday News — Portfolio Performance (GPT-labeled)")
    print("=" * 70)

    for col, category, portfolio_name in metrics:
        vals = portfolio[col].drop_nulls().to_numpy()
        if len(vals) == 0:
            print(f"\n  {category} | {portfolio_name}: no data")
            continue

        hit_rate = np.mean(vals > 0) * 100
        mean_ret = np.mean(vals) * 100
        std_ret = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        sharpe = (np.mean(vals) / std_ret * math.sqrt(252)) if std_ret > 0 else float("nan")

        print(f"\n  {category} | {portfolio_name}:")
        print(f"    Hit Rate     : {hit_rate:.1f}%")
        print(f"    Mean Return  : {mean_ret:.4f}% daily")
        if category == "Drift":
            print(f"    Sharpe Ratio : {sharpe:.2f} (annualized)")
        print(f"    Trading Days : {len(vals)}")

    print(f"\n  Firm-Day Observations: {n_firm_day:,}")
    print(f"  Trading Days (total): {n_trading_days}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 5: Computing event returns and portfolio performance...\n")

    events = compute_event_returns()
    firm_day = aggregate_firm_day(events)
    portfolio = build_portfolios(firm_day)
    summarize(portfolio, firm_day)


if __name__ == "__main__":
    main()
