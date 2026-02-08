import pandas as pd
import plotly.express as px
from pathlib import Path

Path("_output").mkdir(exist_ok=True)

def main():
    # load CRSP data
    df = pd.read_parquet("_data/crsp_daily.parquet")

    # drop missing returns
    df = df.dropna(subset=["ret"])

    # simple exploratory plot: histogram of returns
    fig = px.histogram(
        df,
        x="ret",
        nbins=100,
        title="CRSP Daily Returns (Exploratory)"
    )

    fig.write_html("_output/crsp_returns.html", include_plotlyjs="cdn")
    print("Saved _output/crsp_returns.html")

if __name__ == "__main__":
    main()
