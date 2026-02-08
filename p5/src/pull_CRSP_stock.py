import os
from pathlib import Path
import pandas as pd
import wrds
from dotenv import load_dotenv

load_dotenv()
Path("_data").mkdir(exist_ok=True)

def main():
    user = os.getenv("WRDS_USERNAME")
    pwd = os.getenv("WRDS_PASSWORD")

    if not user or not pwd:
        raise RuntimeError("Missing WRDS_USERNAME / WRDS_PASSWORD in .env")

    db = wrds.Connection(wrds_username=user, wrds_password=pwd)

    sql = """
    select permno, date, ret, prc
    from crsp.dsf
    where date between '2022-01-01' and '2022-12-31'
    limit 5000
    """

    df = db.raw_sql(sql, date_cols=["date"])
    df.to_parquet("_data/crsp_daily.parquet", index=False)

    print("Saved _data/crsp_daily.parquet")

if __name__ == "__main__":
    main()
