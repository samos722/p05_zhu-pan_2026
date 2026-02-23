from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")  # 2021-10-01
END_DATE = config("END_DATE")  # 2024-05-31


def pull_ravenpack_dj(
    start_date=START_DATE,
    end_date=END_DATE,
    wrds_username=WRDS_USERNAME,
    relevance_threshold=100,
    min_similarity_days=90,
    out_path=None,
):
    """
    Pull RavenPack RPA 1.0 Equities (Dow Jones + PR Edition) headlines data.
    Writes year-by-year to parquet to avoid memory overflow.

    Parameters
    ----------
    start_date, end_date, wrds_username, relevance_threshold, min_similarity_days
    out_path : Path or str
        Output parquet path. Required.

    Returns
    -------
    int
        Total rows written.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    ts_start = f"{start_str}T00:00:00"
    ts_end = f"{end_str}T23:59:59"

    db = wrds.Connection(wrds_username=wrds_username)
    writer = None
    total = 0

    for year in range(start_dt.year, end_dt.year + 1):
        y_start = f"{year}-01-01" if year > start_dt.year else start_str
        y_end = f"{year}-12-31" if year < end_dt.year else end_str
        tbl = f"ravenpack_dj.rpa_djpr_equities_{year}"
        sql = f"""
            select e.rp_entity_id, e.rpa_date_utc, e.timestamp_utc, e.rp_story_id,
                   e.relevance, e.event_similarity_days, e.source_name, e.headline,
                   m.ticker, m.cusip
            from {tbl} e
            left join (
                select rp_entity_id, ticker, cusip
                from ravenpack_common.wrds_rpa_company_mappings
            ) m on e.rp_entity_id = m.rp_entity_id
            where e.rpa_date_utc between '{y_start}'::date and '{y_end}'::date
              and e.relevance = {relevance_threshold}
              and e.event_similarity_days > {min_similarity_days}
              and e.timestamp_utc between '{ts_start}' and '{ts_end}'
        """
        df_y = db.raw_sql(sql, date_cols=["timestamp_utc", "rpa_date_utc"])
        df_y["date"] = df_y["timestamp_utc"].dt.date

        table = pa.Table.from_pandas(df_y, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total += len(df_y)
        print(f"  {year}: {len(df_y):,} rows")

    db.close()
    writer.close()
    return total


if __name__ == "__main__":
    path = DATA_DIR / "ravenpack_dj_equities.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = pull_ravenpack_dj(start_date=START_DATE, end_date=END_DATE, out_path=path)
    print(f"Saved: {path} | rows={total:,}")
