"""Pull TAQ NBBO minute-level data from WRDS for stocks with intraday news.

Filters by dates and tickers with news (from ravenpack_intraday_story).
Output: _data/taqm_nbbo/nbbo_YYYY-MM-DD.parquet per date. Run on WRDS login node.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import os

import pandas as pd
import wrds
# 绕过 settings/decouple，避免 segfault（decouple 在某些环境下会崩溃）
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _env_path in (_SCRIPT_DIR / ".env", _PROJECT_ROOT / ".env"):
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT / "_data")))
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "samoszhu")
START_DATE = pd.Timestamp("2021-10-01")
END_DATE = pd.Timestamp("2024-12-31")
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

    crsp = pd.read_parquet(crsp_path, columns=["ticker"])
    crsp_syms = set(crsp["ticker"].unique().tolist())

    news = pd.read_parquet(news_path, columns=["ticker", "date", "is_intraday"])
    news = news[news["date"].isin(target_dates) & news["is_intraday"]]
    news_syms = set(news["ticker"].unique().tolist())

    return sorted(news_syms & crsp_syms)


def load_symbols_intraday_news_for_date(
    date_str: str,
    crsp_syms: set,
    news_path: Path = NEWS_INTRADAY_STORY_PATH,
) -> List[str]:
    """Per-day variant: return tickers with intraday news on a single date."""
    target = date.fromisoformat(date_str)
    news = pd.read_parquet(news_path, columns=["ticker", "date", "is_intraday"])
    news = news[(news["date"] == target) & news["is_intraday"]]
    news_syms = set(news["ticker"].unique().tolist())
    return sorted(news_syms & crsp_syms)


def load_per_day_symbols_intraday(
    dates: List[str],
    crsp_syms: set,
    news_path: Path = NEWS_INTRADAY_STORY_PATH,
) -> Dict[str, List[str]]:
    """Load intraday symbols for all dates in one news file read."""
    target_dates = [date.fromisoformat(d) for d in dates]
    news = pd.read_parquet(news_path, columns=["ticker", "date", "is_intraday"])
    news = news[news["date"].isin(target_dates) & news["is_intraday"]]
    result = {}
    for d in dates:
        target = date.fromisoformat(d)
        day_syms = set(news[news["date"] == target]["ticker"].unique().tolist())
        result[d] = sorted(day_syms & crsp_syms)
    return result


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

    crsp = pd.read_parquet(crsp_path, columns=["ticker"])
    crsp_syms = set(crsp["ticker"].unique().tolist())

    news = pd.read_parquet(news_path, columns=["ticker", "date"])
    news = news[news["date"].isin(target_dates)]
    news_syms = set(news["ticker"].unique().tolist())

    return sorted(news_syms & crsp_syms)


def load_symbols_from_crsp(
    crsp_path: Path = CRSP_PATH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """Load unique tickers from CRSP daily parquet, optionally filtered by date range."""
    df = pd.read_parquet(crsp_path, columns=["date", "ticker"])
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    return sorted(df["ticker"].unique().tolist())


def _taqm_lib_for_date(date_str: str) -> str:
    """Return TAQM schema name for year, e.g. taqm_2024."""
    yyyy = pd.to_datetime(date_str).year
    return f"taqm_{yyyy}"


def _nbbo_table_for_date(date_str: str) -> str:
    """Return NBBO table name for date, e.g. complete_nbbo_20241015."""
    d = pd.to_datetime(date_str)
    return f"complete_nbbo_{d.strftime('%Y%m%d')}"


def _pull_single_date(
    date_str: str,
    day_symbols: List[str],
    wrds_username: str,
    out_dir: Path,
    batch_size: int,
) -> tuple:
    """Pull one date's NBBO. For ProcessPoolExecutor. Returns (date_str, row_count) or (date_str, -1) if skipped."""
    out_path = out_dir / f"nbbo_{date_str}.parquet"
    if out_path.exists():
        return (date_str, -1)
    if not day_symbols:
        return (date_str, -1)
    try:
        db = wrds.Connection(wrds_username=wrds_username)
    except Exception:
        return (date_str, -2)
    lib = f"taqm_{pd.to_datetime(date_str).year}"
    table = f"complete_nbbo_{pd.to_datetime(date_str).strftime('%Y%m%d')}"
    chunks = []
    try:
        for i in range(0, len(day_symbols), batch_size):
            batch = day_symbols[i : i + batch_size]
            syms_sql = ", ".join([f"'{s}'" for s in batch])
            sql = f"""
                WITH ranked AS (
                    SELECT date, sym_root, sym_suffix,
                        date + date_trunc('minute', time_m) AS minute_ts,
                        best_bid, best_ask,
                        ROW_NUMBER() OVER (
                            PARTITION BY sym_root, sym_suffix, date_trunc('minute', time_m)
                            ORDER BY time_m DESC
                        ) AS rn
                    FROM {lib}.{table}
                    WHERE sym_root IN ({syms_sql})
                      AND best_bid > 0 AND best_ask > 0
                      AND time_m >= '09:30:00' AND time_m < '16:00:00'
                )
                SELECT date, sym_root, sym_suffix, minute_ts,
                       (best_bid + best_ask) / 2.0 AS mid
                FROM ranked WHERE rn = 1
            """
            try:
                df_batch = db.raw_sql(sql)
            except Exception as e:
                if "UndefinedTable" in str(type(e).__name__) or "UndefinedTable" in str(e):
                    db.close()
                    return (date_str, -1)
                raise
            if len(df_batch) > 0:
                chunks.append(df_batch)
        db.close()
    except Exception:
        db.close()
        raise
    empty_cols = ["date", "sym_root", "sym_suffix", "minute_ts", "mid"]
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=empty_cols)
    df.to_parquet(out_path, index=False)
    return (date_str, len(df))


def pull_TAQ_intraday_nbbo(
    dates: Iterable[str],
    symbols: Iterable[str],
    wrds_username: str = WRDS_USERNAME,
    out_dir: Optional[Path] = None,
    per_day_symbols: Optional[dict] = None,
    workers: int = 1,
    batch_size: int = BATCH_SIZE,
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

    dates_list = list(dates)
    symbols = list(symbols)
    day_symbols_map = (
        {d: per_day_symbols.get(d, symbols) for d in dates_list}
        if per_day_symbols is not None
        else {d: symbols for d in dates_list}
    )

    if workers > 1:
        total_rows = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _pull_single_date,
                    d,
                    day_symbols_map[d],
                    wrds_username,
                    out_dir,
                    batch_size,
                ): d
                for d in dates_list
            }
            for fut in as_completed(futures):
                date_str, n = fut.result()
                if n >= 0:
                    total_rows += n
                    print(f"[SAVE] {out_dir / f'nbbo_{date_str}.parquet'} rows={n:,}")
        return total_rows

    db = wrds.Connection(wrds_username=wrds_username)
    total_rows = 0
    try:
        for date_str in dates_list:
            out_path = out_dir / f"nbbo_{date_str}.parquet"
            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue

            day_symbols = day_symbols_map[date_str]
            if not day_symbols:
                print(f"[SKIP] {date_str}: no symbols")
                continue

            lib = _taqm_lib_for_date(date_str)
            table = _nbbo_table_for_date(date_str)

            chunks: List[pd.DataFrame] = []
            skipped = False
            for i in range(0, len(day_symbols), batch_size):
                batch = day_symbols[i : i + batch_size]
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
                    f"  {date_str} batch {i // batch_size + 1} "
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel date workers (default 1). Use >1 for intraday pull.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Symbols per SQL batch (default {BATCH_SIZE})",
    )
    args = parser.parse_args()

    dates = args.dates if args.dates is not None else _default_dates()

    if args.intraday:
        crsp = pd.read_parquet(CRSP_PATH, columns=["ticker"])
        crsp_syms = set(crsp["ticker"].unique().tolist())
        per_day = load_per_day_symbols_intraday(dates, crsp_syms)
        total_syms = len(set().union(*per_day.values())) if per_day else 0
        print(f"[mode] Intraday news, per-day filtered | {total_syms} unique symbols across {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(
            dates=dates, symbols=[], per_day_symbols=per_day,
            workers=args.workers, batch_size=args.batch_size,
        )
    elif args.all:
        symbols = load_symbols_from_crsp(start_date=min(dates), end_date=max(dates))
        print(f"[mode] All CRSP symbols | {len(symbols):,} symbols, {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(
            dates=dates, symbols=symbols, workers=args.workers, batch_size=args.batch_size
        )
    else:
        symbols = load_symbols_with_news(dates)
        print(f"[mode] News-filtered | {len(symbols):,} symbols, {len(dates)} date(s)")
        n = pull_TAQ_intraday_nbbo(
            dates=dates, symbols=symbols, workers=args.workers, batch_size=args.batch_size
        )
    print("Total rows:", n)


if __name__ == "__main__":
    main()