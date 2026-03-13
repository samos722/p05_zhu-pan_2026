from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = Path("_data")
OUTPUT_DIR = Path("_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# 1 CRSP Exploration
# -------------------------------

def explore_crsp():
    df = pd.read_parquet(DATA_DIR / "clean" / "crsp_daily.parquet")

    cols = ["ret", "openprc", "closeprc", "vol", "mktcap"]
    # rename columns for better presentation
    df = df.rename(columns={
        "ret": "Return",
        "openprc": "Open Price",
        "closeprc": "Close Price",
        "vol": "Volume",
        "mktcap": "Market Cap"
    })
    stats = df[["Return", "Open Price", "Close Price", "Volume", "Market Cap"]].describe().T

    stats = stats[["mean", "std", "min", "max"]]

    stats.to_latex(
        OUTPUT_DIR / "crsp_summary_stats.tex",
        float_format="%.4f",
        caption="Summary statistics of CRSP daily stock data used in the replication.",
        label="tab:crsp_summary"
    )


# -------------------------------
# 2 TAQ Exploration
# -------------------------------

def explore_taq():

    taq = pd.read_parquet(DATA_DIR / "clean" / "taq_nbbo_minute.parquet")

    # choose a sample ticker and date
    sample = taq[(taq["ticker"] == taq["ticker"].iloc[0]) &
                 (taq["date"] == taq["date"].iloc[0])]

    sample = sample.sort_values("minute_ts")

    plt.figure(figsize=(10,5))
    plt.plot(sample["minute_ts"], sample["mid"])

    plt.title("Example Intraday Price Path (TAQ Data)")
    plt.xlabel("Time of Day")
    plt.ylabel("Mid Price")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())

    plt.xticks(rotation=45)

    plt.savefig(OUTPUT_DIR / "taq_intraday_midprice_example.png")
    plt.close()


# -------------------------------
# 3 RavenPack Exploration
# -------------------------------

def explore_ravenpack():
    ravenpack = pd.read_parquet(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet")

    ravenpack["date"] = pd.to_datetime(ravenpack["date"])
    daily_counts = ravenpack.groupby("date").size()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_counts.index, daily_counts.values)
    plt.title("Number of RavenPack News Events per Day")
    plt.xlabel("Date")
    plt.ylabel("News Count")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "ravenpack_news_per_day.png")
    plt.close()


# -------------------------------
# 4 GPT Labels Exploration
# -------------------------------

def explore_gpt():

    gpt = pd.read_parquet(DATA_DIR / "interim" / "gpt_labels.parquet")

    label_counts = gpt["label"].value_counts()

    plt.figure(figsize=(6,4))
    label_counts.plot(kind="bar")

    plt.title("Distribution of GPT Sentiment Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "gpt_label_distribution.png")
    plt.close()


# -------------------------------
# Run all explorations
# -------------------------------

def main():
    explore_crsp()
    explore_taq()
    explore_ravenpack()
    explore_gpt()
    print("Data exploration figures generated.")


if __name__ == "__main__":
    main()