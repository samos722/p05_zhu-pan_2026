from pathlib import Path

import polars as pl
from rapidfuzz.distance import DamerauLevenshtein

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

RAW_PATH = DATA_DIR / "ravenpack_dj_equities.parquet"
OUT_PATH = DATA_DIR / "clean" / "news_firmday.parquet"
OUT_PATH_STORY = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"


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


def build_news_intraday_story(
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH_STORY,
) -> pl.DataFrame:
    """
    Build story-level intraday news table for TAQ join.

    Fields:
    - rp_story_id
    - date (trading day)
    - ticker
    - timestamp_utc
    - timestamp_et (ET, for reference)
    - is_intraday (9:30–16:00 ET)
    - t15 (timestamp_et + 15 min, floored to minute) — use for TAQ minute join

    Note: ~31M rows, so expect ~1–2 min. Uses LazyFrame for projection pushdown.
    """
    df = (
        pl.scan_parquet(raw_path)
        .select(["rp_story_id", "timestamp_utc", "ticker"])
        .filter(pl.col("ticker").is_not_null())
    )

    # UTC -> ET
    ts = pl.col("timestamp_utc")
    ts_et = ts.dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    df = df.with_columns(ts_et.alias("timestamp_et"))

    # trading date
    base_date = ts_et.dt.date()
    hour = ts_et.dt.hour() + ts_et.dt.minute() / 60.0 + ts_et.dt.second() / 3600.0
    date_expr = pl.when(hour >= 16.0).then(base_date + pl.duration(days=1)).otherwise(base_date)
    df = df.with_columns(date_expr.alias("date"))

    # is_intraday: 9:30–16:00 ET (market hours, 16:00 exclusive)
    # Use strftime since dt.hour/dt.minute can misbehave with timezone-aware datetimes
    h = pl.col("timestamp_et").dt.strftime("%H").cast(pl.Int32)
    m = pl.col("timestamp_et").dt.strftime("%M").cast(pl.Int32)
    minute_of_day = h * 60 + m
    is_intraday = (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)
    df = df.with_columns(is_intraday.alias("is_intraday"))

    # t15 = timestamp_et + 15 min, floored to minute
    t15 = (pl.col("timestamp_et") + pl.duration(minutes=15)).dt.truncate("1m")
    df = df.with_columns(t15.alias("t15"))

    df = df.collect()
    return df


def _headline_similarity(a: str, b: str) -> float:
    """Optimal String Alignment (Restricted Damerau-Levenshtein) similarity in [0, 1]."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return DamerauLevenshtein.normalized_similarity(a, b)


def deduplicate_similar_headlines(
    df_story: pl.DataFrame,
    raw_path: Path = RAW_PATH,
    sim_threshold: float = 0.6,
) -> pl.DataFrame:
    """
    Remove duplicate/similar headlines per (ticker, date) per Lopez-Lira, Tang, Zhu (2025).

    Uses Optimal String Alignment (Restricted Damerau-Levenshtein); removes subsequent
    headlines with similarity > sim_threshold to any already-kept headline.
    """
    raw = pl.read_parquet(raw_path, columns=["rp_story_id", "headline"])
    df = df_story.join(raw, on="rp_story_id", how="inner")
    # sort by timestamp for "subsequent" order
    df = df.sort(["ticker", "date", "timestamp_et"])

    keep_ids = []
    for _keys, sub in df.group_by(["ticker", "date"]):
        sub = sub.sort("timestamp_et")
        rows = sub.iter_rows(named=True)
        kept_headlines: list[str] = []
        for row in rows:
            h = str(row.get("headline") or "")
            rp_id = row["rp_story_id"]
            is_dup = False
            for kh in kept_headlines:
                if _headline_similarity(h, kh) > sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_ids.append(rp_id)
                kept_headlines.append(h)

    keep_set = set(keep_ids)
    out = df_story.filter(pl.col("rp_story_id").is_in(keep_set))
    return out


if __name__ == "__main__":
    df_firmday = clean_ravenpack_firmday()
    print(f"Saved: {OUT_PATH} | rows={len(df_firmday):,}")
    df_story = build_news_intraday_story()
    n_before = len(df_story)
    df_story = deduplicate_similar_headlines(df_story)
    n_after = len(df_story)
    OUT_PATH_STORY.parent.mkdir(parents=True, exist_ok=True)
    df_story.write_parquet(OUT_PATH_STORY)
    n_intra = df_story.filter(pl.col("is_intraday")).height
    print(f"Saved: {OUT_PATH_STORY} | rows={n_after:,} (dropped {n_before - n_after:,} similar, intraday={n_intra:,})")
