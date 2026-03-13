from pathlib import Path

from nbconvert import export
import numpy as np
import pandas as pd
import webbrowser

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
SAMPLE_START = "2024-05-31"
SAMPLE_END = "2025-12-31"


def _get_taq_trading_dates() -> pd.Series:
    """Get unique trading dates from TAQ (market-open days in sample)."""
    taq = pd.read_parquet(DATA_DIR / "clean" / "taq_nbbo_minute.parquet")
    taq["date"] = pd.to_datetime(taq["date"])
    dates = taq["date"].dt.date.unique()
    dates = pd.to_datetime(
        [d for d in dates if SAMPLE_START <= str(d) <= SAMPLE_END]
    )
    return pd.Series(dates)


def _divide_intraday_overnight():
    """Split intraday portfolio returns into intraday and overnight components."""
    gpt_headlines = pd.read_parquet(DATA_DIR / "interim" / "gpt_labels.parquet")
    ravenpack = pd.read_parquet(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet")

    taq_dates = _get_taq_trading_dates()
    ravenpack["date"] = pd.to_datetime(ravenpack["date"])
    ravenpack = ravenpack[ravenpack["date"].isin(taq_dates)].copy()
    ravenpack = ravenpack.drop_duplicates(
        subset=["rp_story_id", "ticker", "date"], keep="first"
    )

    gpt_headlines = gpt_headlines.merge(
        ravenpack,
        on=["rp_story_id", "ticker"],
        how="inner",
    )

    intraday_headlines = gpt_headlines[gpt_headlines["is_intraday"] == True]
    overnight_headlines = gpt_headlines[gpt_headlines["is_intraday"] == False]

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

    # build prev_close
    daily_stock_unique['prev_close'] = (
    daily_stock_unique
    .groupby('ticker')['closeprc']
    .shift(1))

    # merge with overnight_headlines
    cols_to_keep = ['ticker', 'date', 'openprc', 'closeprc', 'prev_close']
    overnight_headlines = overnight_headlines.merge(
    daily_stock_unique[cols_to_keep],
    on=['ticker', 'date'],
    how='left')

    # build initial reaction = (open price - previous close price) / previous close price
    overnight_headlines['initial_reaction'] = (overnight_headlines["openprc"] - overnight_headlines["prev_close"]) / overnight_headlines["prev_close"]

    # drift = (close price - open price) / open price
    overnight_headlines["drift"] = (
        overnight_headlines["closeprc"] - overnight_headlines["openprc"]
    ) / overnight_headlines["openprc"]

    for col in ["initial_reaction", "drift"]:
        lo = overnight_headlines[col].quantile(0.01)
        hi = overnight_headlines[col].quantile(0.99)
        overnight_headlines[col] = overnight_headlines[col].clip(lo, hi)

    # clean up columns
    cols = ['ticker', 'headline', 'label', 'date', 'initial_reaction', 'drift']
    overnight_headlines_cleaned = overnight_headlines[cols]

    overnight_headlines_cleaned = overnight_headlines_cleaned.copy().dropna()

    return overnight_headlines_cleaned


def _merge_intraday(intraday_headlines):
    """Merge intraday headlines with stock prices"""
    daily_stock = pd.read_parquet(DATA_DIR / "clean" / "crsp_daily.parquet")
    taq = pd.read_parquet(DATA_DIR / "clean" / "taq_nbbo_minute.parquet")  # for intraday prices

    daily_stock['date'] = pd.to_datetime(daily_stock['date'])
    intraday_headlines['date'] = pd.to_datetime(intraday_headlines['date'])
    intraday_headlines["minute_ts"] = (pd.to_datetime(intraday_headlines["t15"]).dt.tz_localize(None)) # convert to naive datetime for merging with taq
    taq['date'] = pd.to_datetime(taq['date'])

    # Remove duplicates
    daily_stock_unique = (
    daily_stock
    .sort_values(['ticker','date'])
    .drop_duplicates(['ticker','date'], keep='first'))

    # build next_close
    daily_stock_unique['next_close'] = (
    daily_stock_unique
    .groupby('ticker')['closeprc']
    .shift(-1))

    # build prev_close
    daily_stock_unique['prev_close'] = (
    daily_stock_unique
    .groupby('ticker')['closeprc']
    .shift(1))

    # merge with daily stock
    cols_to_keep = ['ticker', 'date', 'openprc', 'closeprc', 'next_close', 'prev_close']
    intraday_headlines_merged = intraday_headlines.merge(
    daily_stock_unique[cols_to_keep],
    on=['ticker', 'date'],
    how='left')

    # merge with taq for intraday prices
    intraday_headlines_merged = intraday_headlines_merged.merge(taq,
                               on=['date', 'ticker', 'minute_ts'],
                               how='left')

    # build initial reaction = (mid price at headline time + 15 mins - previous close price) / previous close price
    intraday_headlines_merged['initial_reaction'] = (intraday_headlines_merged['mid'] - intraday_headlines_merged['prev_close']) / intraday_headlines_merged['prev_close']

    # build drift = (next close price - close price) / close price
    intraday_headlines_merged["drift"] = (
        intraday_headlines_merged["next_close"]
        - intraday_headlines_merged["closeprc"]
    ) / intraday_headlines_merged["closeprc"]

    for col in ["initial_reaction", "drift"]:
        lo = intraday_headlines_merged[col].quantile(0.01)
        hi = intraday_headlines_merged[col].quantile(0.99)
        intraday_headlines_merged[col] = intraday_headlines_merged[col].clip(lo, hi)

    cols = ["ticker", "headline", "label", "date", "closeprc", "initial_reaction", "drift"]
    intraday_headlines_cleaned = intraday_headlines_merged[cols]
    intraday_headlines_cleaned = intraday_headlines_cleaned.copy().dropna()

    return intraday_headlines_cleaned


def calculate_portfolio_metrics(
    df: pd.DataFrame,
    portfolio: str = "long_short",   # "long_short" | "long_only" | "short_only"
    min_long: int = 2,
    min_short: int = 2,
    annualization: int = 252,
) -> dict:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["signal"] = d["label"].map({"YES": 1, "NO": -1, "UNKNOWN": 0})

    # drop neutral observations
    d = d[d["signal"] != 0].copy()

    if portfolio == "long_short":
        # require enough longs and shorts on each day
        long_counts = d[d["signal"] > 0].groupby("date").size()
        short_counts = d[d["signal"] < 0].groupby("date").size()

        valid_days = long_counts[long_counts >= min_long].index.intersection(
            short_counts[short_counts >= min_short].index
        )
        d = d[d["date"].isin(valid_days)].copy()

        # equal-weight long leg and short leg separately
        long_initial = (
            d[d["signal"] > 0]
            .groupby("date")["initial_reaction"]
            .mean()
            .rename("long_initial")
        )
        short_initial = (
            d[d["signal"] < 0]
            .groupby("date")["initial_reaction"]
            .mean()
            .rename("short_initial")
        )
        long_drift = (
            d[d["signal"] > 0]
            .groupby("date")["drift"]
            .mean()
            .rename("long_drift")
        )
        short_drift = (
            d[d["signal"] < 0]
            .groupby("date")["drift"]
            .mean()
            .rename("short_drift")
        )

        daily = pd.concat(
            [long_initial, short_initial, long_drift, short_drift],
            axis=1
        ).dropna()

        daily["port_initial"] = daily["long_initial"] - daily["short_initial"]
        daily["port_drift"] = daily["long_drift"] - daily["short_drift"]

    elif portfolio == "long_only":
        d = d[d["signal"] > 0].copy()

        counts = d.groupby("date").size()
        valid_days = counts[counts >= min_long].index
        d = d[d["date"].isin(valid_days)].copy()

        daily = (
            d.groupby("date")[["initial_reaction", "drift"]]
            .mean()
            .rename(columns={
                "initial_reaction": "port_initial",
                "drift": "port_drift"
            })
        )

    elif portfolio == "short_only":
        d = d[d["signal"] < 0].copy()

        counts = d.groupby("date").size()
        valid_days = counts[counts >= min_short].index
        d = d[d["date"].isin(valid_days)].copy()

        daily = (
            d.groupby("date")[["initial_reaction", "drift"]]
            .mean()
            .rename(columns={
                "initial_reaction": "port_initial",
                "drift": "port_drift"
            })
        )

        # short return = negative stock return
        daily["port_initial"] = -daily["port_initial"]
        daily["port_drift"] = -daily["port_drift"]

    else:
        raise ValueError("portfolio must be 'long_short', 'long_only', or 'short_only'")

    hit_initial = (daily["port_initial"] > 0).mean()
    hit_drift = (daily["port_drift"] > 0).mean()

    mean_initial = daily["port_initial"].mean()
    mean_drift = daily["port_drift"].mean()

    vol_drift = daily["port_drift"].std()
    sharpe = np.nan if vol_drift == 0 or np.isnan(vol_drift) else (
        np.sqrt(annualization) * mean_drift / vol_drift
    )

    return {
        "portfolio": portfolio,
        "firm_day_observations": int(d.drop_duplicates(["ticker", "date"]).shape[0]),
        "trading_days": int(daily.shape[0]),
        "hit_rate_initial": float(hit_initial) * 100,
        "hit_rate_drift": float(hit_drift) * 100,
        "mean_return_initial": float(mean_initial) * 100,
        "mean_return_drift": float(mean_drift) * 100,
        "sharpe_ratio_drift": float(sharpe) if pd.notna(sharpe) else np.nan,
    }



def _compute_metrics():

    intraday_headlines, overnight_headlines = _divide_intraday_overnight()

    overnight_headlines_cleaned = _merge_overnight(overnight_headlines)
    intraday_headlines_cleaned = _merge_intraday(intraday_headlines)

    overnight_long_short = calculate_portfolio_metrics(overnight_headlines_cleaned, portfolio="long_short", min_long=2, min_short=2)
    overnight_long_only = calculate_portfolio_metrics(overnight_headlines_cleaned, portfolio="long_only", min_long=2, min_short=2)
    overnight_short_only = calculate_portfolio_metrics(overnight_headlines_cleaned, portfolio="short_only", min_long=2, min_short=2)

    intraday_long_short = calculate_portfolio_metrics(intraday_headlines_cleaned, portfolio="long_short", min_long=2, min_short=2)
    intraday_long_only = calculate_portfolio_metrics(intraday_headlines_cleaned, portfolio="long_only", min_long=2, min_short=2)
    intraday_short_only = calculate_portfolio_metrics(intraday_headlines_cleaned, portfolio="short_only", min_long=2, min_short=2)

    return {
        "overnight": {
            "long_short": overnight_long_short,
            "long_only": overnight_long_only,
            "short_only": overnight_short_only,
        },
        "intraday": {
            "long_short": intraday_long_short,
            "long_only": intraday_long_only,
            "short_only": intraday_short_only,
        },
    }



def build_performance_table(results, overnight_obs=None, intraday_obs=None):
    """Build Table 1 in the paper"""
    row_order = [
        ("Long-Short Portfolio", "hit_rate_initial", "Hit Rate (%)"),
        ("Long-Short Portfolio", "mean_return_initial", "Mean Return (%)"),
        ("Long-Short Portfolio", "sharpe_ratio_drift", "Sharpe Ratio"),
        ("Long-Only Portfolio", "hit_rate_initial", "Hit Rate (%)"),
        ("Long-Only Portfolio", "mean_return_initial", "Mean Return (%)"),
        ("Long-Only Portfolio", "sharpe_ratio_drift", "Sharpe Ratio"),
        ("Short-Only Portfolio", "hit_rate_initial", "Hit Rate (%)"),
        ("Short-Only Portfolio", "mean_return_initial", "Mean Return (%)"),
        ("Short-Only Portfolio", "sharpe_ratio_drift", "Sharpe Ratio"),
    ]

    portfolio_map = {
        "Long-Short Portfolio": "long_short",
        "Long-Only Portfolio": "long_only",
        "Short-Only Portfolio": "short_only",
    }

    metric_map = {
        "hit_rate_initial": "hit_rate_initial",
        "mean_return_initial": "mean_return_initial",
        "sharpe_ratio_drift": "sharpe_ratio_drift",
    }

    records = []

    for group_label, metric_key, metric_label in row_order:
        p = portfolio_map[group_label]

        overnight_metrics = results["overnight"][p]
        intraday_metrics = results["intraday"][p]

        if metric_key == "hit_rate_initial":
            row = [
                group_label,
                metric_label,
                overnight_metrics["hit_rate_initial"],
                overnight_metrics["hit_rate_drift"],
                intraday_metrics["hit_rate_initial"],
                intraday_metrics["hit_rate_drift"],
            ]
        elif metric_key == "mean_return_initial":
            row = [
                group_label,
                metric_label,
                overnight_metrics["mean_return_initial"],
                overnight_metrics["mean_return_drift"],
                intraday_metrics["mean_return_initial"],
                intraday_metrics["mean_return_drift"],
            ]
        elif metric_key == "sharpe_ratio_drift":
            row = [
                group_label,
                metric_label,
                np.nan,
                overnight_metrics["sharpe_ratio_drift"],
                np.nan,
                intraday_metrics["sharpe_ratio_drift"],
            ]

        records.append(row)

    df = pd.DataFrame(
        records,
        columns=[
            "Portfolio Group",
            "Metric",
            ("Overnight News", "Initial Reaction"),
            ("Overnight News", "Drift"),
            ("Intraday News", "Initial Reaction"),
            ("Intraday News", "Drift"),
        ],
    )

    # add summary rows
    if overnight_obs is None:
        overnight_obs = results["overnight"]["long_short"].get("firm_day_observations", np.nan)
    if intraday_obs is None:
        intraday_obs = results["intraday"]["long_short"].get("firm_day_observations", np.nan)

    overnight_days = results["overnight"]["long_short"]["trading_days"]
    intraday_days = results["intraday"]["long_short"]["trading_days"]

    summary_rows = pd.DataFrame(
        [
            [
                "",
                "Firm-Day Observations",
                overnight_obs,
                overnight_obs,
                intraday_obs,
                intraday_obs,
            ],
            [
                "",
                "Trading Days",
                overnight_days,
                overnight_days,
                intraday_days,
                intraday_days,
            ],
        ],
        columns=df.columns,
    )

    df = pd.concat([df, summary_rows], ignore_index=True)

    # make multiindex columns
    df.columns = pd.MultiIndex.from_tuples(
        [
            ("", "Portfolio Group"),
            ("", "Metric"),
            ("Overnight News", "Initial Reaction"),
            ("Overnight News", "Drift"),
            ("Intraday News", "Initial Reaction"),
            ("Intraday News", "Drift"),
        ]
    )

    return df


def style_performance_table(df):
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        if isinstance(x, (float, np.floating)):
            return f"{x:.2f}"
        return x

    return df.style.format(fmt)


if __name__ == "__main__":

    results = _compute_metrics()
    performance_table = build_performance_table(results)
    styled_table = style_performance_table(performance_table)

    output_dir = Path(config("OUTPUT_DIR")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "performance_table.html"
    tex_path = output_dir / "performance_table.tex"

    # Save HTML version for browser viewing
    styled_table.to_html(html_path)

    latex_table = performance_table.copy().fillna("--").reset_index(drop=True)

    latex_table.columns = [
        "Portfolio Group",
        "Metric",
        "Overnight Initial Reaction",
        "Overnight Drift",
        "Intraday Initial Reaction",
        "Intraday Drift",
    ]

    latex_table["Portfolio Group"] = latex_table["Portfolio Group"].replace({
        "Long-Short Portfolio": "Long-Short",
        "Long-Only Portfolio": "Long-Only",
        "Short-Only Portfolio": "Short-Only",
        "": "All Portfolios"
    })

    latex_table.to_latex(
        tex_path,
        index=False,
        escape=True,
        float_format="%.2f",
        na_rep="--",
        column_format="llrrrr"
    )

    # Open HTML table in browser
    webbrowser.open(html_path.resolve().as_uri())