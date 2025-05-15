from src.utils.fetch_economic_data import (
    create_fred_client,
    fetch_sp500,
    fetch_nasdaq,
    fetch_cpi,
    fetch_usd_to_eur,
    fetch_usd_to_jpy,
    fetch_usd_to_gbp,
    fetch_usd_to_cny,
)
from src.utils.raw_to_csv import save_series_to_csv

def main():
    fred = create_fred_client()

    data_sources = {
        "sp500.csv": fetch_sp500(fred),
        "nasdaq.csv": fetch_nasdaq(fred),
        "cpi.csv": fetch_cpi(fred),
        "usd_eur.csv": fetch_usd_to_eur(fred),
        "usd_jpy.csv": fetch_usd_to_jpy(fred),
        "usd_gbp.csv": fetch_usd_to_gbp(fred),
        "usd_cny.csv": fetch_usd_to_cny(fred),
    }

    for filename, series in data_sources.items():
        save_series_to_csv(series, filename)

if __name__ == "__main__":
    main()
