from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")


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
) -> int:
    """
    Pull TAQM NBBO (minute/sub-minute quotes) from tables like:
      taqm_2024.complete_nbbo_20240102

    Saves one parquet per date under DATA_DIR/taqm_nbbo/.

    Parameters
    ----------
    dates : iterable of 'YYYY-MM-DD'
    symbols : iterable of sym_root tickers (e.g. ['AAPL','MSFT'])
    """
    if out_dir is None:
        out_dir = DATA_DIR / "taqm_nbbo"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = list(symbols)
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")

    db = wrds.Connection(wrds_username=wrds_username)

    total_rows = 0
    try:
        for date_str in dates:
            lib = _taqm_lib_for_date(date_str)
            table = _nbbo_table_for_date(date_str)

            # Build IN (...) safely
            syms_sql = ", ".join([f"'{s}'" for s in symbols])

            sql = f"""
                select
                    date,
                    time_m,
                    time_m_nano,
                    sym_root,
                    sym_suffix,
                    best_bid,
                    best_bidsizeshares,
                    best_ask,
                    best_asksizeshares
                from {lib}.{table}
                where sym_root in ({syms_sql})
            """
            df = db.raw_sql(sql)

            out_path = out_dir / f"nbbo_{date_str}.parquet"
            df.to_parquet(out_path, index=False)
            print(df.head())
            print(f"[SAVE] {out_path} rows={len(df):,} (from {lib}.{table})")

            total_rows += len(df)

    finally:
        db.close()

    return total_rows


def main():
    # start tiny
    dates = ["2024-01-02"]
    symbols = ["A", "AAPL", "MSFT"]
    n = pull_TAQ_intraday_nbbo(dates=dates, symbols=symbols)
    print("Total rows:", n)


if __name__ == "__main__":
    main()