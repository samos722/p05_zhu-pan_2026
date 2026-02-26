from pathlib import Path
import pandas as pd
import plotly.express as px

DATA_DIR = Path("_data")
OUT_DIR = Path("_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def chart_crsp():
    df = pd.read_parquet(DATA_DIR / "CRSP_daily_stock.parquet")


    permno = df["permno"].dropna().iloc[0]
    d = df.loc[df["permno"] == permno, ["date", "openprc"]].dropna().sort_values("date")

    fig = px.line(d, x="date", y="openprc", title=f"CRSP: Price over time (permno={int(permno)})")
    fig.show()
    fig.write_html(OUT_DIR / "crsp_price_timeseries.html", include_plotlyjs="cdn")



def chart_ravenpack(ticker="PRD.LN"):
    df = pd.read_parquet(
        DATA_DIR / "ravenpack_dj_equities.parquet",
        columns=["ticker", "rpa_date_utc", "relevance"], 
    )

    d = (
        df.loc[
            (df["ticker"] == ticker)
            & (df["rpa_date_utc"] >= "2021-01-01")
            & (df["rpa_date_utc"] <= "2021-06-30"),
            ["rpa_date_utc", "relevance"],
        ]
        .dropna()
        .groupby("rpa_date_utc", as_index=False)
        .mean()
        .sort_values("rpa_date_utc")
    )

    fig = px.line(
        d,
        x="rpa_date_utc",
        y="relevance",
        title=f"RavenPack: Daily Avg Relevance ({ticker}, 6-month sample)",
    )
    fig.show()
    fig.write_html(OUT_DIR / f"ravenpack_relevance_{ticker.replace('.', '_')}.html",
                   include_plotlyjs="cdn")


if __name__ == "__main__":
    chart_crsp()
    chart_ravenpack()