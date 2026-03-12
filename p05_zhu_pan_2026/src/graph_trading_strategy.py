from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from settings import config

DATA_DIR = Path(config("DATA_DIR")).resolve()

def _divide_intraday_overnight():
    """Split intraday portfolio returns into intraday and overnight components."""
    gpt_headlines = pd.read_parquet(DATA_DIR / "interim" / "gpt_labels.parquet")
    ravenpack = pd.read_parquet(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet")

    gpt_headlines = gpt_headlines.merge(
        ravenpack,
        on=["rp_story_id", "ticker"],
        how="left"
    )

    intraday_headlines = gpt_headlines[gpt_headlines["is_intraday"] == True].copy()
    overnight_headlines = gpt_headlines[gpt_headlines["is_intraday"] == False].copy()

    return intraday_headlines, overnight_headlines


def _merge_overnight(overnight_headlines):
    """Merge overnight headlines with stock prices and compute tradable returns."""
    daily_stock = pd.read_parquet(DATA_DIR / "clean" / "crsp_daily.parquet")

    daily_stock["date"] = pd.to_datetime(daily_stock["date"])
    overnight_headlines = overnight_headlines.copy()
    overnight_headlines["date"] = pd.to_datetime(overnight_headlines["date"])

    daily_stock_unique = (
        daily_stock
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="first")
        .copy()
    )

    daily_stock_unique["next_open"] = (
        daily_stock_unique.groupby("ticker")["openprc"].shift(-1)
    )
    daily_stock_unique["next_close"] = (
        daily_stock_unique.groupby("ticker")["closeprc"].shift(-1)
    )
    daily_stock_unique["prev_close"] = (
        daily_stock_unique.groupby("ticker")["closeprc"].shift(1)
    )

    cols_to_keep = [
        "exchcd", "ticker", "date", "ret", "openprc", "closeprc", "vol",
        "next_open", "next_close", "prev_close", "mktcap"
    ]

    overnight_headlines = overnight_headlines.merge(
        daily_stock_unique[cols_to_keep],
        on=["ticker", "date"],
        how="left"
    )

    cols = [
        "exchcd", "ticker", "date", "headline", "label", "openprc", "closeprc",
        "timestamp_et", "next_open", "next_close", "prev_close", "mktcap"
    ]
    overnight_headlines_cleaned = overnight_headlines[cols].copy().dropna()

    overnight_headlines_cleaned["timestamp_et"] = pd.to_datetime(
        overnight_headlines_cleaned["timestamp_et"]
    )

    overnight_headlines_cleaned["before_open"] = (
        overnight_headlines_cleaned["timestamp_et"].dt.time < pd.to_datetime("09:00").time()
    )
    overnight_headlines_cleaned["after_close"] = (
        overnight_headlines_cleaned["timestamp_et"].dt.time > pd.to_datetime("16:00").time()
    )

    overnight_headlines_cleaned["same_day_return"] = (
        (overnight_headlines_cleaned["closeprc"] - overnight_headlines_cleaned["openprc"])
        / overnight_headlines_cleaned["openprc"]
    )
    overnight_headlines_cleaned["next_day_return"] = (
        (overnight_headlines_cleaned["next_close"] - overnight_headlines_cleaned["next_open"])
        / overnight_headlines_cleaned["next_open"]
    )

    return overnight_headlines_cleaned


def value_weight_market_portfolio():
    """Compute cumulative return of a value-weighted market portfolio."""
    daily_stock = pd.read_parquet(DATA_DIR / "clean" / "crsp_daily.parquet").copy()

    daily_stock["date"] = pd.to_datetime(daily_stock["date"])

    # keep needed columns
    daily_stock = daily_stock[["ticker", "date", "ret", "mktcap"]].copy()

    # clean
    daily_stock = daily_stock.dropna(subset=["ticker", "date", "ret", "mktcap"])
    daily_stock = daily_stock.sort_values(["ticker", "date"])

    # if ret may be stored as string, force numeric
    daily_stock["ret"] = pd.to_numeric(daily_stock["ret"], errors="coerce")
    daily_stock["mktcap"] = pd.to_numeric(daily_stock["mktcap"], errors="coerce")
    daily_stock = daily_stock.dropna(subset=["ret", "mktcap"])

    # lag market cap to avoid look-ahead bias
    daily_stock["lag_mktcap"] = daily_stock.groupby("ticker")["mktcap"].shift(1)

    # keep only rows with valid lagged weights
    daily_stock = daily_stock.dropna(subset=["lag_mktcap"])

    # daily total lagged market cap
    total_lag_mktcap = daily_stock.groupby("date")["lag_mktcap"].transform("sum")

    # value weights
    daily_stock["weight"] = daily_stock["lag_mktcap"] / total_lag_mktcap

    # weighted market return each day
    market_daily = (
        daily_stock.groupby("date")
        .apply(lambda x: np.sum(x["weight"] * x["ret"]))
        .rename("vwret")
        .sort_index()
    )

    # cumulative value of $1 invested
    market_cumret = (1 + market_daily).cumprod()

    return market_cumret




def long_short_strategy(df):
    df = df.copy()
    df = df[df["before_open"] | df["after_close"]].copy()
    df = df[df["label"].isin(["YES", "NO"])].copy()

    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    df["date"] = pd.to_datetime(df["date"])

    firmday = (
        df.groupby(["date", "ticker", "label"], as_index=False)["event_return"]
        .mean()
    )

    long_df = (
        firmday[firmday["label"] == "YES"]
        .groupby("date")["event_return"]
        .mean()
        .rename("long_ret")
    )

    short_df = (
        firmday[firmday["label"] == "NO"]
        .groupby("date")["event_return"]
        .mean()
        .rename("short_ret")
    )

    daily = pd.concat([long_df, short_df], axis=1).fillna(0).sort_index()
    daily["ls_ret"] = 3 * daily["long_ret"] - 2 * daily["short_ret"]

    return (1 + daily["ls_ret"]).cumprod()


def long_short_not_small(df):

    df = df.copy()
    df = df[df["before_open"] | df["after_close"]].copy()
    df = df[df["label"].isin(["YES", "NO"])].copy()

    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    df = df[df["prev_close"] > 5].copy()

    nyse_bp = (
        df[df["exchcd"] == 1]
        .groupby("date")["mktcap"]
        .quantile(0.20)
        .rename("nyse_bp")
    )


    df = df.merge(nyse_bp, on="date", how="left")
    df = df[df["mktcap"] > df["nyse_bp"]].copy()


    df["signal"] = df["label"].map({"YES": 1, "NO": -1})

    firmday = (
        df.groupby(["date", "ticker"])
        .agg(
            signal=("signal", "mean"),
            event_return=("event_return", "mean")
        )
        .reset_index()
    )

    long_ret = firmday[firmday["signal"] > 0].groupby("date")["event_return"].mean()
    short_ret = firmday[firmday["signal"] < 0].groupby("date")["event_return"].mean()

    daily = pd.concat([long_ret, short_ret], axis=1).fillna(0)
    daily.columns = ["long", "short"]
    daily["ls_ret"] = 3 * daily["long"] - 2 * daily["short"]

    return (1 + daily["ls_ret"]).cumprod()


def long_short_greater_5(df):
    df = df.copy()
    df = df[df["before_open"] | df["after_close"]].copy()
    df = df[df["label"].isin(["YES", "NO"])].copy()

    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    df = df[df["prev_close"] > 5].copy()
    df["signal"] = df["label"].map({"YES": 1, "NO": -1})

    firmday = (
        df.groupby(["date", "ticker"])
        .agg(
            signal=("signal", "mean"),
            event_return=("event_return", "mean")
        )
        .reset_index()
    )

    long_ret = firmday[firmday["signal"] > 0].groupby("date")["event_return"].mean()
    short_ret = firmday[firmday["signal"] < 0].groupby("date")["event_return"].mean()

    daily = pd.concat([long_ret, short_ret], axis=1).fillna(0)
    daily.columns = ["long", "short"]
    daily["ls_ret"] = 3 * daily["long"] - 2 * daily["short"]

    return (1 + daily["ls_ret"]).cumprod()



def long_short_top_percentile(df, pct=0.10):
    """
    Long top pct of signals and short bottom pct of signals within each date.
    Example: pct=0.10 means top 10% long, bottom 10% short.
    """
    df = df.copy()
    df = df[df["before_open"] | df["after_close"]].copy()
    df = df[df["label"].isin(["YES", "NO"])].copy()

    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    # optional filter
    df = df[df["prev_close"] > 5].copy()

    # signal from label
    df["signal"] = df["label"].map({"YES": 1, "NO": -1})

    # one stock per day
    firmday = (
        df.groupby(["date", "ticker"])
        .agg(
            signal=("signal", "mean"),
            event_return=("event_return", "mean")
        )
        .reset_index()
    )

    # rank into percentiles within each date
    # pct=True gives percentile rank between 0 and 1
    firmday["rank_pct"] = (
        firmday.groupby("date")["signal"]
        .rank(method="first", pct=True)
    )

    # keep only tails
    long_side = firmday[firmday["rank_pct"] >= 1 - pct].copy()
    short_side = firmday[firmday["rank_pct"] <= pct].copy()

    # equal-weight returns
    long_ret = long_side.groupby("date")["event_return"].mean().rename("long")
    short_ret = short_side.groupby("date")["event_return"].mean().rename("short")

    daily = pd.concat([long_ret, short_ret], axis=1).fillna(0).sort_index()
    daily["ls_ret"] = 3* daily["long"] - 2 * daily["short"]

    return daily, (1 + daily["ls_ret"]).cumprod()






def plot_like_paper(cumret_long_short, cumret_not_small, cumret_price5, cumret_market):
    # Restrict sample through end of March 2024
    end_date = pd.Timestamp("2024-03-31")
    cumret_long_short = cumret_long_short.loc[cumret_long_short.index <= end_date]
    cumret_not_small = cumret_not_small.loc[cumret_not_small.index <= end_date]
    cumret_price5 = cumret_price5.loc[cumret_price5.index <= end_date]
    cumret_market = cumret_market.loc[cumret_market.index <= end_date]

    # Global style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot lines
    ax.plot(
        cumret_long_short.index,
        cumret_long_short.values,
        color="blue",
        linewidth=0.8,
        linestyle="-",
        label="Long–Short"
    )

    # ax.plot(
    #     cumret_not_small.index,
    #     cumret_not_small.values,
    #     color="black",
    #     linewidth=0.8,
    #     linestyle="--",
    #     label="Not Small"
    # )

    ax.plot(
        cumret_price5.index,
        cumret_price5.values,
        color="green",
        linewidth=0.8,
        linestyle=":",
        label="Price > 5"
    )

    ax.plot(
        cumret_market.index,
        cumret_market.values,
        color="orange",
        linewidth=0.8,
        linestyle="--",
        label="Market Value-Weighted"
    )

    # X-axis: show years only
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Y-axis: log scale
    ax.set_yscale("log")
    ax.set_ylim(0.7, 10.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yticklabels([str(x) for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]])

    # Labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value in $")

    # Grid
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.4, alpha=0.3)

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.4")

    # Legend
    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=1.2
    )

    fig.tight_layout()

    output_dir = Path(config("OUTPUT_DIR")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cumulative_returns_paper_style.png", dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    intraday_headlines, overnight_headlines = _divide_intraday_overnight()
    overnight_headlines_cleaned = _merge_overnight(overnight_headlines)

    cumret_long_short = long_short_strategy(overnight_headlines_cleaned)
    cumret_not_small = long_short_not_small(overnight_headlines_cleaned)
    cumret_price5 = long_short_greater_5(overnight_headlines_cleaned)
    cumret_market = value_weight_market_portfolio()


    plot_like_paper(cumret_long_short, cumret_not_small, cumret_price5, cumret_market)