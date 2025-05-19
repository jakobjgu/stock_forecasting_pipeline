import pandas as pd
import yfinance as yf
from fredapi import Fred
from settings import FRED_API_KEY


def create_fred_client():
    """Initialize and return a Fred API client."""
    return Fred(api_key=FRED_API_KEY)


def fetch_series(fred_client, series_id: str) -> pd.Series:
    """Fetch a time series from FRED by its series ID."""
    return fred_client.get_series(series_id=series_id)


def fetch_summary(series: pd.Series) -> pd.DataFrame:
    """Return basic summary statistics of a time series."""
    summary = {
        'start_date': series.index.min(),
        'end_date': series.index.max(),
        'num_points': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
    }
    return pd.DataFrame([summary])


def fetch_sp500(fred_client) -> pd.Series:
    """Fetch the S&P 500 Index (daily close)."""
    return fetch_series(fred_client, 'SP500')


def fetch_nasdaq(fred_client) -> pd.Series:
    """Fetch the NASDAQ Composite Index."""
    return fetch_series(fred_client, 'NASDAQCOM')


def fetch_cpi(fred_client) -> pd.Series:
    """Fetch the US Consumer Price Index (CPI for All Urban Consumers)."""
    return fetch_series(fred_client, 'CPIAUCSL')


def fetch_usd_to_eur(fred_client) -> pd.Series:
    """USD to Euro exchange rate (DEXUSEU)."""
    return fetch_series(fred_client, 'DEXUSEU')


def fetch_usd_to_jpy(fred_client) -> pd.Series:
    """USD to Japanese Yen exchange rate (DEXJPUS)."""
    return fetch_series(fred_client, 'DEXJPUS')


def fetch_usd_to_gbp(fred_client) -> pd.Series:
    """USD to British Pound exchange rate (DEXUSUK)."""
    return fetch_series(fred_client, 'DEXUSUK')


def fetch_usd_to_cny(fred_client) -> pd.Series:
    """USD to Chinese Yuan exchange rate (DEXCHUS)."""
    return fetch_series(fred_client, 'DEXCHUS')


def fetch_full_sp500_history() -> pd.Series:
    sp500 = yf.download("^GSPC", start="1950-01-01", interval="1d")
    # Flatten MultiIndex columns
    sp500.columns = sp500.columns.get_level_values(0)
    sp500_series = sp500["Close"]
    sp500_series.name = "value"
    return sp500_series


def fetch_unemployment_rate(fred: Fred) -> pd.Series:
    return fred.get_series("UNRATE")  # Monthly unemployment rate


def fetch_consumer_sentiment(fred: Fred) -> pd.Series:
    return fred.get_series("UMCSENT")  # University of Michigan consumer sentiment index


def fetch_fed_funds_rate(fred: Fred) -> pd.Series:
    return fred.get_series("FEDFUNDS")


def fetch_10y_treasury_yield(fred: Fred) -> pd.Series:
    return fred.get_series("GS10")


def fetch_3m_treasury_yield(fred: Fred) -> pd.Series:
    return fred.get_series("GS3M")


def compute_yield_spread(gs10: pd.Series, gs3m: pd.Series) -> pd.Series:
    return gs10 - gs3m