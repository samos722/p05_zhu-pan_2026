from datetime import datetime
from pathlib import Path

import pandas as pd
import wrds
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE") # 2021-10-01
END_DATE = config("END_DATE") # 2024-05-31


def pull_CRSP_daily(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pull CRSP daily stock data following the paper's definition.

    Universe:
    - Common stocks: shrcd in (10, 11)
    - Exchanges: NYSE, NASDAQ, AMEX (exchcd in 1,2,3)
    - Variables: date, permno, ret (daily stock return), open price, close price, volume

    Parameters
    ----------
    start_date : str or datetime-like
        Sample start date (e.g. '2021-10-01')
    end_date : str or datetime-like
        Sample end date (e.g. '2024-05-31')
    wrds_username : str
        WRDS username

    Returns
    -------
    pandas.DataFrame
        Clean CRSP daily panel indexed by (permno, date)
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    db = wrds.Connection(wrds_username=wrds_username)

    sql = f"""
        select
            d.date,
            d.permno,
            n.ticker,
            n.comnam,
            n.shrcd,
            n.exchcd,
            d.ret,
            d.openprc,
            d.prc as closeprc,
            d.vol
        from crsp.dsf as d
        join crsp.dsenames as n
          on d.permno = n.permno
         and n.namedt <= d.date
         and d.date <= n.nameendt
        where d.date between '{start_date}' and '{end_date}'
          and n.shrcd in (10, 11)
          and n.exchcd in (1, 2, 3)
    """

    df = db.raw_sql(sql, date_cols=["date"])
    db.close()

    return df


if __name__ == "__main__":
    df_msf = pull_CRSP_daily(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "CRSP_daily_stock.parquet"
    df_msf.to_parquet(path)