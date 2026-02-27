from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import polars as pl
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")  # 2021-10-01, align with CRSP/RavenPack
END_DATE = config("END_DATE")  # 2024-05-31
CRSP_PATH = DATA_DIR / "CRSP_daily_stock.parquet"
NEWS_FIRMDAY_PATH = DATA_DIR / "clean" / "news_firmday.parquet"
NEWS_INTRADAY_STORY_PATH = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"
BATCH_SIZE = 400  # symbols per SQL query to avoid IN-clause limits


def load_symbols_intraday_news(
    dates: Iterable[str],
    crsp_path: Path = CRSP_PATH,
    news_path: Path = NEWS_INTRADAY_STORY_PATH,
) -> List[str]:
    """
    Load tickers that have intraday news (9:30-16:00 ET) on the given dates.
    Use for TAQ pull when computing Intraday Initial Reaction (price at t+15min).
    """
    dates_list = list(dates)
    if not dates_list:
        return []
    target_dates = [date.fromisoformat(d) for d in dates_list]

    crsp = pl.read_parquet(crsp_path, columns=["ticker"])
    crsp_syms = set(crsp["ticker"].unique().to_list())

    news = pl.read_parquet(news_path, columns=["ticker", "date", "is_intraday"])
    news = news.filter(pl.col("date").is_in(target_dates) & pl.col("is_intraday"))
    news_syms = set(news["ticker"].unique().to_list())

    return sorted(news_syms & crsp_syms)


def load_symbols_intraday_news_for_date(
    date_str: str,
    crsp_syms: set,
    news_path: Path = NEWS_INTRADAY_STORY_PATH,
) -> List[str]:
    """Per-day variant: return tickers with intraday news on a single date."""
    target = date.fromisoformat(date_str)
    news = pl.read_parquet(news_path, columns=["ticker", "date", "is_intraday"])
    news = news.filter((pl.col("date") == target) & pl.col("is_intraday"))
    news_syms = set(news["ticker"].unique().to_list())
    return sorted(news_syms & crsp_syms)


def load_symbols_with_news(
    dates: Iterable[str],
    crsp_path: Path = CRSP_PATH,
    news_path: Path = NEWS_FIRMDAY_PATH,
) -> List[str]:
    """
    Load tickers that have news on the given trading days, intersected with CRSP.
    Reduces TAQ pull workload by only fetching stocks with news.
    """
    dates_list = list(dates)
    if not dates_list:
        return []
    target_dates = [date.fromisoformat(d) for d in dates_list]

    crsp = pl.read_parquet(crsp_path, columns=["ticker"])
    crsp_syms = set(crsp["ticker"].unique().to_list())

    news = pl.read_parquet(news_path, columns=["ticker", "date"])
    news = news.filter(pl.col("date").is_in(target_dates))
    news_syms = set(news["ticker"].unique().to_list())

    return sorted(news_syms & crsp_syms)


def load_symbols_from_crsp(
    crsp_path: Path = CRSP_PATH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """Load unique tickers from CRSP daily parquet, optionally filtered by date range."""
    df = pl.read_parquet(crsp_path, columns=["date", "ticker"])
    if start_date is not None:
        df = df.filter(pl.col("date") >= pl.lit(start_date))
    if end_date is not None:
        df = df.filter(pl.col("date") <= pl.lit(end_date))
    return sorted(df["ticker"].unique().to_list())


def _taqm_lib_for_date(date_str: str) -> str:
    yyyy = pd.to_datetime(date_str).year
    return f"taqm_{yyyy}"


def _nbbo_table_for_date(date_str: str) -> str:
    d = pd.to_datetime(date_str)
    return f"complete_nbbo_{d.strftime('%Y%m%d')}"


def pull_TAQ_intraday_nbbo(
    dates: Iterable[str],
    symbols: Iterable[str],
    wrds_username: str = WRDS_USERNAME,
    out_dir: Optional[Path] = None,
    per_day_symbols: Optional[dict] = None,
) -> int:
    """
    Pull TAQM NBBO **aggregated to minute-level** from WRDS.

    SQL-side aggregation: for each (sym_root, sym_suffix, minute), take the
    last best_bid/best_ask and compute mid = (best_bid + best_ask) / 2.
    This reduces data volume by ~100x compared to raw tick-level pulls.

    Saves one parquet per date under DATA_DIR/taqm_nbbo/.

    Parameters
    ----------
    dates : iterable of 'YYYY-MM-DD'
    symbols : iterable of sym_root tickers (fallback if per_day_symbols is None)
    per_day_symbols : optional dict {date_str: [sym_list]} for per-day filtering
    """
    if out_dir is None:
        out_dir = DATA_DIR / "taqm_nbbo"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = list(symbols)
    db = wrds.Connection(wrds_username=wrds_username)

    total_rows = 0
    try:
        for date_str in dates:
            out_path = out_dir / f"nbbo_{date_str}.parquet"
            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue

            day_symbols = per_day_symbols[date_str] if per_day_symbols else symbols
            if not day_symbols:
                print(f"[SKIP] {date_str}: no symbols")
                continue

            lib = _taqm_lib_for_date(date_str)
            table = _nbbo_table_for_date(date_str)

            chunks: List[pd.DataFrame] = []
            skipped = False
            for i in range(0, len(day_symbols), BATCH_SIZE):
                batch = day_symbols[i : i + BATCH_SIZE]
                syms_sql = ", ".join([f"'{s}'" for s in batch])
                sql = f"""
                    WITH ranked AS (
                        SELECT
                            date,
                            sym_root,
                            sym_suffix,
                            date + date_trunc('minute', time_m) AS minute_ts,
                            best_bid,
                            best_ask,
                            ROW_NUMBER() OVER (
                                PARTITION BY sym_root, sym_suffix,
                                             date_trunc('minute', time_m)
                                ORDER BY time_m DESC
                            ) AS rn
                        FROM {lib}.{table}
                        WHERE sym_root IN ({syms_sql})
                          AND best_bid > 0 AND best_ask > 0
                          AND time_m >= '09:30:00' AND time_m < '16:00:00'
                    )
                    SELECT
                        date,
                        sym_root,
                        sym_suffix,
                        minute_ts,
                        (best_bid + best_ask) / 2.0 AS mid
                    FROM ranked
                    WHERE rn = 1
                """
                try:
                    df_batch = db.raw_sql(sql)
                except Exception as e:
                    if "UndefinedTable" in str(type(e).__name__) or "UndefinedTable" in str(e):
                        print(f"[SKIP] {date_str}: table {lib}.{table} not found (holiday?)")
                        skipped = True
                        break
                    raise
                if len(df_batch) > 0:
                    chunks.append(df_batch)
                print(
                    f"  {date_str} batch {i // BATCH_SIZE + 1} "
                    f"({len(batch)} syms) -> {len(df_batch):,} rows"
                )

            if skipped:
                continue

            empty_cols = ["date", "sym_root", "sym_suffix", "minute_ts", "mid"]
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=empty_cols)
            df.to_parquet(out_path, index=False)
            n = len(df)
            total_rows += n
            print(f"[SAVE] {out_path} rows={n:,}")

    finally:
        db.close()

    return total_rows


def _default_dates() -> List[str]:
    """US trading days in [START_DATE, END_DATE], aligned with CRSP/RavenPack."""
    dr = pd.bdate_range(
        start=pd.Timestamp(START_DATE),
        end=pd.Timestamp(END_DATE),
        freq="B",
    )
    return [d.strftime("%Y-%m-%d") for d in dr]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pull TAQ NBBO (minute-level) from WRDS")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pull all CRSP symbols",
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Pull only symbols with intraday news (9:30-16:00 ET), per-day filtered.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help=f"Trading dates (YYYY-MM-DD). Default: all trading days in [{START_DATE.date()}, {END_DATE.date()}]",
    )
    args = parser.parse_args()

    dates = args.dates if args.dates is not None else _default_dates()

    if args.intraday:
        crsp = pl.read_parquet(CRSP_PATH, columns=["ticker"])
        crsp_syms = set(crsp["ticker"].unique().to_list())
        per_day = {}
        for d in dates:
            per_day[d] = load_symbols_intraday_news_for_date(d, crsp_syms)
        total_syms = len(set().union(*per_day.values())) if per_day else 0
        print(f"[mode] Intraday news, per-day filtered | {total_syms} unique symbols across {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(dates=dates, symbols=[], per_day_symbols=per_day)
    elif args.all:
        symbols = load_symbols_from_crsp(start_date=min(dates), end_date=max(dates))
        print(f"[mode] All CRSP symbols | {len(symbols):,} symbols, {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(dates=dates, symbols=symbols)
    else:
        symbols = load_symbols_with_news(dates)
        print(f"[mode] News-filtered | {len(symbols):,} symbols, {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(dates=dates, symbols=symbols)
    print("Total rows:", n)


if __name__ == "__main__":
    main()