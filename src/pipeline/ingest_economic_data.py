import pandas as pd
from src.utils.fetch_economic_data import (
    create_fred_client,
    fetch_sp500,               # still exists, but no longer used
    fetch_nasdaq,
    fetch_cpi,
    fetch_usd_to_eur,
    fetch_usd_to_jpy,
    fetch_usd_to_gbp,
    fetch_usd_to_cny,
    fetch_full_sp500_history,  # now used for S&P 500
    fetch_unemployment_rate,
    fetch_consumer_sentiment,
    fetch_fed_funds_rate,
    fetch_3m_treasury_yield,
    fetch_10y_treasury_yield
)
from src.utils.preprocess import save_series_to_csv


def compute_yield_spread(gs10: pd.Series, gs3m: pd.Series) -> pd.Series:
    return gs10 - gs3m


def fetch_economic_data_series():
    print("📡 Fetching economic data series...")
    fred = create_fred_client()

    data_sources = {
        "sp500.csv": fetch_full_sp500_history(),  # ← use Yahoo Finance full history
        "nasdaq.csv": fetch_nasdaq(fred),
        "cpi.csv": fetch_cpi(fred),
        "usd_eur.csv": fetch_usd_to_eur(fred),
        "usd_jpy.csv": fetch_usd_to_jpy(fred),
        "usd_gbp.csv": fetch_usd_to_gbp(fred),
        "usd_cny.csv": fetch_usd_to_cny(fred),
        "unrate.csv": fetch_unemployment_rate(fred),
        "umcsent.csv": fetch_consumer_sentiment(fred),
        "fed_funds.csv": fetch_fed_funds_rate(fred),
        "treasury_10y.csv": fetch_10y_treasury_yield(fred),
        "treasury_3m.csv": fetch_3m_treasury_yield(fred),
    }

    gs10 = fetch_10y_treasury_yield(fred)
    gs3m = fetch_3m_treasury_yield(fred)
    spread = compute_yield_spread(gs10, gs3m)
    save_series_to_csv(spread, destination_folder="raw", filename="yield_spread.csv")

    for filename, series in data_sources.items():
        save_series_to_csv(series, filename, destination_folder='raw')


if __name__ == "__main__":
    fetch_economic_data_series()
