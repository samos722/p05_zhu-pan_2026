from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("_data")

def _divide_intraday_overnight():
    """Split intraday portfolio returns into intraday and overnight components."""
    gpt_headlines = pd.read_parquet(DATA_DIR / "gpt_labels.parquet")
    ravenpack = pd.read_parquet(DATA_DIR / "ravenpack_intraday_story.parquet")

    gpt_headlines = gpt_headlines.merge(
    ravenpack,
    on=['rp_story_id', 'ticker'],
    how='left')

    intraday_headlines = gpt_headlines[gpt_headlines['is_intraday'] == True]
    overnight_headlines = gpt_headlines[gpt_headlines['is_intraday'] == False]

    return intraday_headlines, overnight_headlines


def _merge_overnight(overnight_headlines):
    """Merge intraday and overnight headlines with stock prices"""
    daily_stock = pd.read_parquet(DATA_DIR / "clean" / "crsp_daily.parquet")

    daily_stock['date'] = pd.to_datetime(daily_stock['date'])
    overnight_headlines['date'] = pd.to_datetime(overnight_headlines['date'])

    # Remove duplicates 
    daily_stock_unique = (
    daily_stock
    .sort_values(['ticker','date'])
    .drop_duplicates(['ticker','date'], keep='first'))

    # compute next open, next close, and prior close
    daily_stock_unique['next_open'] = (
        daily_stock_unique
        .groupby('ticker')['openprc']
        .shift(-1)
    )

    daily_stock_unique['next_close'] = (
        daily_stock_unique
        .groupby('ticker')['closeprc']
        .shift(-1)
    )

    daily_stock_unique['prev_close'] = (
        daily_stock_unique
        .groupby('ticker')['closeprc']
        .shift(1)
    )

    # merge with overnight_headlines
    cols_to_keep = ['exchcd', 'ticker', 'date', 'ret', 'openprc', 'closeprc', 'vol', 'next_open', 'next_close', 'prev_close', 'mktcap']
    overnight_headlines = overnight_headlines.merge(
    daily_stock_unique[cols_to_keep],
    on=['ticker', 'date'],
    how='left')

    # clean up the merged dataframe
    cols = ['exchcd', 'ticker', 'date', 'headline', 'label', 'openprc', 'closeprc', 'timestamp_et', 'next_open', 'next_close', 'prev_close', 'mktcap']
    overnight_headlines_cleaned = overnight_headlines[cols]
    overnight_headlines_cleaned = overnight_headlines_cleaned.copy().dropna()


    # classify: before open (09:00), after close (16:00)
    overnight_headlines_cleaned["timestamp_et"] = pd.to_datetime(overnight_headlines_cleaned["timestamp_et"])
    
    overnight_headlines_cleaned["before_open"] = (
        overnight_headlines_cleaned["timestamp_et"].dt.time < pd.to_datetime("09:00").time()
    )

    overnight_headlines_cleaned["after_close"] = (
        overnight_headlines_cleaned["timestamp_et"].dt.time > pd.to_datetime("16:00").time()
    )


    # compute returns
    overnight_headlines_cleaned['same_day_return'] = (overnight_headlines_cleaned['closeprc'] - overnight_headlines_cleaned['openprc']) / overnight_headlines_cleaned['openprc']
    overnight_headlines_cleaned['next_day_return'] = (overnight_headlines_cleaned['next_close'] - overnight_headlines_cleaned['next_open']) / overnight_headlines_cleaned['next_open']

    return overnight_headlines_cleaned


def long_short_strategy(df):
    df = df.copy()

    # keep tradable timing only
    df = df[df["before_open"] | df["after_close"]].copy()

    # keep only YES / NO
    df = df[df["label"].isin(["YES", "NO"])].copy()

    # choose event return
    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    # collapse to one firm-day observation
    # here: average multiple headlines for same stock-day-label bucket
    df["date"] = pd.to_datetime(df["date"])
    firmday = (
        df.groupby(["date", "ticker", "label"], as_index=False)["event_return"]
        .mean()
    )

    # split long and short legs
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

    daily = pd.concat([long_df, short_df], axis=1)

    # if one side missing on a day, set to 0 or drop; paper choice matters
    daily = daily.fillna(0)

    daily["ls_ret"] = daily["long_ret"] - daily["short_ret"]
    cumulative = (1 + daily["ls_ret"]).cumprod()

    return cumulative

def long_short_not_small(df):

    df = df.copy()

    # tradable event timing
    df = df[df["before_open"] | df["after_close"]]

    # only tradable labels
    df = df[df["label"].isin(["YES","NO"])]

    # choose correct return
    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    # compute NYSE 20th percentile breakpoint each day
    nyse_bp = (
        df[df["exchcd"] == 1]
        .groupby("date")["mktcap"]
        .quantile(0.20)
        .rename("nyse_bp")
    )

    df = df.merge(nyse_bp, on="date", how="left")

    # keep only not-small stocks
    df = df[df["mktcap"] > df["nyse_bp"]]

    # signal
    df["signal"] = df["label"].map({"YES":1,"NO":-1})

    # firm-day aggregation
    firmday = (
        df.groupby(["date","ticker"])
        .agg(
            signal=("signal","mean"),
            event_return=("event_return","mean")
        )
        .reset_index()
    )

    # long and short legs
    long_ret = firmday[firmday["signal"]>0].groupby("date")["event_return"].mean()
    short_ret = firmday[firmday["signal"]<0].groupby("date")["event_return"].mean()

    daily = pd.concat([long_ret,short_ret],axis=1).fillna(0)
    daily.columns = ["long","short"]

    daily["ls_ret"] = daily["long"] - daily["short"]

    return (1 + daily["ls_ret"]).cumprod()


def long_short_greater_5(df):

    df = df.copy()

    df = df[df["before_open"] | df["after_close"]]
    df = df[df["label"].isin(["YES","NO"])]

    df["event_return"] = np.where(
        df["before_open"],
        df["same_day_return"],
        df["next_day_return"]
    )

    # price filter
    df = df[df["prev_close"] > 5]

    df["signal"] = df["label"].map({"YES":1,"NO":-1})

    firmday = (
        df.groupby(["date","ticker"])
        .agg(
            signal=("signal","mean"),
            event_return=("event_return","mean")
        )
        .reset_index()
    )

    long_ret = firmday[firmday["signal"]>0].groupby("date")["event_return"].mean()
    short_ret = firmday[firmday["signal"]<0].groupby("date")["event_return"].mean()

    daily = pd.concat([long_ret,short_ret],axis=1).fillna(0)
    daily.columns = ["long","short"]

    daily["ls_ret"] = daily["long"] - daily["short"]

    return (1 + daily["ls_ret"]).cumprod() 



if __name__ == "__main__":
    intraday_headlines, overnight_headlines = _divide_intraday_overnight()
    overnight_headlines_cleaned = _merge_overnight(overnight_headlines)

    cumret_long_short = long_short_strategy(overnight_headlines_cleaned)
    cumret_not_small = long_short_not_small(overnight_headlines_cleaned)
    cumret_price5 = long_short_greater_5(overnight_headlines_cleaned)

    plt.figure(figsize=(12,6))

    plt.plot(cumret_long_short, label='Long-Short', color='blue')
    plt.plot(cumret_not_small, label='Not Small', color='green')
    plt.plot(cumret_price5, label='Price > 5', color='grey')

    plt.title("Cumulative Returns of News-Based Strategies")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()

    plt.show()

    # save plot to output directory
    output_dir = Path("_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "cumulative_returns.png")
    
    

