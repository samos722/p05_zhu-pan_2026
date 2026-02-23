from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))


RAW_PATH = DATA_DIR / "ravenpack_dj_equities.parquet"
OUT_PATH = DATA_DIR / "clean" / "news_firmday.parquet"


def _trading_date_expr() -> pl.Expr:
    """Map UTC timestamps to (approximate) US trading dates.

    - Convert UTC -> US/Eastern (naive treated as UTC)
    - If time >= 16:00, assign to next calendar day (overnight); else same day.
    """
    ts = pl.col("timestamp_utc")
    ts_local = ts.dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    base_date = ts_local.dt.date()
    hour = ts_local.dt.hour() + ts_local.dt.minute() / 60.0
    return pl.when(hour >= 16.0).then(base_date + pl.duration(days=1)).otherwise(base_date)


def clean_ravenpack_firmday(
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """
    Aggregate RavenPack Dow Jones news to firm-day level.

    - Keep: timestamp_utc, headline, ticker, cusip, rp_story_id, relevance, event_similarity_days
    - Convert timestamp_utc (UTC) to an approximate US trading date
    - Group by (ticker, date) and aggregate headlines and metadata
    """
    df = pl.read_parquet(raw_path)

    cols = [
        "timestamp_utc",
        "headline",
        "ticker",
        "cusip",
        "rp_story_id",
        "relevance",
        "event_similarity_days",
    ]
    df = df.select(cols).filter(pl.col("ticker").is_not_null())
    df = df.with_columns(_trading_date_expr().alias("date"))

    grouped = (
        df.group_by(["ticker", "date"])
        .agg(
            pl.col("cusip").first(),
            pl.col("relevance").mean(),
            pl.col("event_similarity_days").max(),
            pl.col("headline").alias("headlines"),
            pl.col("timestamp_utc").alias("timestamps_utc"),
            pl.col("rp_story_id").alias("story_ids"),
        )
        .with_columns(pl.col("headlines").list.len().alias("n_headlines"))
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.write_parquet(out_path)
    return grouped


if __name__ == "__main__":
    df_firmday = clean_ravenpack_firmday()
    print(f"Saved: {OUT_PATH} | rows={len(df_firmday):,}")
