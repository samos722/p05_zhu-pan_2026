from pathlib import Path

import polars as pl

from settings import config

DATA_DIR = Path(config("DATA_DIR"))


RAW_PATH = DATA_DIR / "CRSP_daily_stock.parquet"
OUT_PATH = DATA_DIR / "clean" / "crsp_daily.parquet"


def clean_crsp_daily(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pl.DataFrame:
    """
    Clean CRSP daily stock data for downstream use.

    Steps (minimum set):
    - Filter to common stocks (shrcd in 10, 11) and main exchanges (exchcd in 1, 2, 3)
    - Fix price sign: use abs(closeprc)
    - Compute market capitalization: mktcap = abs(closeprc) * shrout
    """
    df = pl.read_parquet(raw_path)

    df = (
        df.filter(pl.col("shrcd").is_in([10, 11]) & pl.col("exchcd").is_in([1, 2, 3]))
        .with_columns(pl.col("closeprc").abs().alias("closeprc"))
        .with_columns((pl.col("closeprc") * pl.col("shrout")).alias("mktcap"))
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df


if __name__ == "__main__":
    df_clean = clean_crsp_daily()
    print(f"Saved: {OUT_PATH} | rows={len(df_clean):,}")
