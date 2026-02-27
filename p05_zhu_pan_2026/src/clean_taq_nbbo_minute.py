"""
Step A: Combine per-date minute-level NBBO parquets into a single panel.

pull_TAQ_intraday.py now does minute-level aggregation on the WRDS side,
so this script only needs to concat, build ticker, and write out.

Output: _data/clean/taq_nbbo_minute.parquet
Fields: date, ticker, minute_ts, mid
"""
from pathlib import Path

import pandas as pd
import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
NBBO_DIR = DATA_DIR / "taqm_nbbo"
OUT_PATH = DATA_DIR / "clean" / "taq_nbbo_minute.parquet"


def clean_taq_nbbo_minute(
    nbbo_dir: Path = NBBO_DIR,
    out_path: Path = OUT_PATH,
) -> pl.DataFrame:
    """
    Combine per-date minute-level NBBO parquets.
    Each file already has: date, sym_root, sym_suffix, minute_ts, mid.
    """
    files = sorted(nbbo_dir.glob("nbbo_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No nbbo_*.parquet in {nbbo_dir}")

    dfs = []
    for f in files:
        pdf = pd.read_parquet(f)
        if pdf.empty:
            continue
        df = pl.from_pandas(pdf)

        # ticker = sym_root (+ sym_suffix if non-null/non-empty)
        if "sym_suffix" in df.columns:
            df = df.with_columns(
                pl.when(
                    pl.col("sym_suffix").is_not_null()
                    & (pl.col("sym_suffix").cast(pl.Utf8) != "")
                )
                .then(pl.col("sym_root").cast(pl.Utf8) + pl.col("sym_suffix").cast(pl.Utf8))
                .otherwise(pl.col("sym_root"))
                .alias("ticker"),
            )
        else:
            df = df.with_columns(pl.col("sym_root").alias("ticker"))

        df = df.select(["date", "ticker", "minute_ts", "mid"]).drop_nulls("mid")
        dfs.append(df)

    out = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


if __name__ == "__main__":
    df = clean_taq_nbbo_minute()
    print(f"Saved: {OUT_PATH} | rows={len(df):,}")
