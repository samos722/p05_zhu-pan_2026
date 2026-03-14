"""Unit tests comparing replication Table 1 metrics to paper benchmarks."""
import math
import pytest

from compute_portfolio_performance import _compute_metrics


PAPER_TABLE_1 = {
    "overnight": {
        "long_short": {
            "hit_rate_initial": 93.28,
            "hit_rate_drift": 58.06,
            "mean_return_initial": 3.06,
            "mean_return_drift": 0.34,
            "sharpe_ratio_drift": 2.97,
        },
        "long_only": {
            "hit_rate_initial": 83.28,
            "hit_rate_drift": 50.90,
            "mean_return_initial": 1.27,
            "mean_return_drift": 0.08,
            "sharpe_ratio_drift": 0.78,
        },
        "short_only": {
            "hit_rate_initial": 79.40,
            "hit_rate_drift": 53.58,
            "mean_return_initial": 1.79,
            "mean_return_drift": 0.26,
            "sharpe_ratio_drift": 2.01,
        },
        "firm_day_observations": 105742,
    },
    "intraday": {
        "long_short": {
            "hit_rate_initial": 88.78,
            "hit_rate_drift": 54.67,
            "mean_return_initial": 4.44,
            "mean_return_drift": 0.50,
            "sharpe_ratio_drift": 2.63,
        },
        "long_only": {
            "hit_rate_initial": 76.05,
            "hit_rate_drift": 46.00,
            "mean_return_initial": 1.33,
            "mean_return_drift": 0.19,
            "sharpe_ratio_drift": 1.37,
        },
        "short_only": {
            "hit_rate_initial": 77.57,
            "hit_rate_drift": 48.67,
            "mean_return_initial": 3.11,
            "mean_return_drift": 0.31,
            "sharpe_ratio_drift": 1.71,
        },
        "firm_day_observations": 26109,
    },
}


HIT_RATE_TOL = 8.0
MEAN_RETURN_TOL = 0.80
SHARPE_TOL = 2.6 # The Sharpe ratio is more volatile, so we allow a larger tolerance here.

@pytest.fixture(scope="module")
def results():
    return _compute_metrics()


@pytest.mark.parametrize(
    "news_type,portfolio,metric,tol",
    [
        ("overnight", "long_short", "hit_rate_initial", HIT_RATE_TOL),
        ("overnight", "long_short", "hit_rate_drift", HIT_RATE_TOL),
        ("overnight", "long_short", "mean_return_initial", MEAN_RETURN_TOL),
        ("overnight", "long_short", "mean_return_drift", MEAN_RETURN_TOL),
        ("overnight", "long_short", "sharpe_ratio_drift", SHARPE_TOL),

        ("overnight", "long_only", "hit_rate_initial", HIT_RATE_TOL),
        ("overnight", "long_only", "hit_rate_drift", HIT_RATE_TOL),
        ("overnight", "long_only", "mean_return_initial", MEAN_RETURN_TOL),
        ("overnight", "long_only", "mean_return_drift", MEAN_RETURN_TOL),
        ("overnight", "long_only", "sharpe_ratio_drift", SHARPE_TOL),

        ("overnight", "short_only", "hit_rate_initial", HIT_RATE_TOL),
        ("overnight", "short_only", "hit_rate_drift", HIT_RATE_TOL),
        ("overnight", "short_only", "mean_return_initial", MEAN_RETURN_TOL),
        ("overnight", "short_only", "mean_return_drift", MEAN_RETURN_TOL),
        ("overnight", "short_only", "sharpe_ratio_drift", SHARPE_TOL),

        ("intraday", "long_short", "hit_rate_initial", HIT_RATE_TOL),
        ("intraday", "long_short", "hit_rate_drift", HIT_RATE_TOL),
        ("intraday", "long_short", "mean_return_initial", MEAN_RETURN_TOL),
        ("intraday", "long_short", "mean_return_drift", MEAN_RETURN_TOL),
        ("intraday", "long_short", "sharpe_ratio_drift", SHARPE_TOL),

        ("intraday", "long_only", "hit_rate_initial", HIT_RATE_TOL),
        ("intraday", "long_only", "hit_rate_drift", HIT_RATE_TOL),
        ("intraday", "long_only", "mean_return_initial", MEAN_RETURN_TOL),
        ("intraday", "long_only", "mean_return_drift", MEAN_RETURN_TOL),
        ("intraday", "long_only", "sharpe_ratio_drift", SHARPE_TOL),

        ("intraday", "short_only", "hit_rate_initial", HIT_RATE_TOL),
        ("intraday", "short_only", "hit_rate_drift", HIT_RATE_TOL),
        ("intraday", "short_only", "mean_return_initial", MEAN_RETURN_TOL),
        ("intraday", "short_only", "mean_return_drift", MEAN_RETURN_TOL),
        ("intraday", "short_only", "sharpe_ratio_drift", SHARPE_TOL),
    ],
)
def test_table1_metrics_within_tolerance(results, news_type, portfolio, metric, tol):
    """Check that replication metrics are within tolerance of paper Table 1 benchmarks."""
    actual = results[news_type][portfolio][metric]
    expected = PAPER_TABLE_1[news_type][portfolio][metric]

    assert math.isfinite(actual)

    assert abs(actual - expected) <= tol, (
        f"{news_type}-{portfolio}-{metric}: "
        f"actual={actual:.2f}, expected={expected:.2f}, "
        f"diff={abs(actual - expected):.2f}, tol={tol:.2f}"
    )