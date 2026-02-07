from pathlib import Path
import pandas as pd
import plotly.express as px

DATA_DIR = Path("_data")
OUT_DIR = Path("_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def chart_crsp():
    df = pd.read_parquet(DATA_DIR / "CRSP_monthly_stock.parquet")


    permno = df["permno"].dropna().iloc[0]
    d = df.loc[df["permno"] == permno, ["date", "altprc"]].dropna().sort_values("date")

    fig = px.line(d, x="date", y="altprc", title=f"CRSP: Price over time (permno={int(permno)})")
    fig.show()
    fig.write_html(OUT_DIR / "crsp_price_timeseries.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    chart_crsp()