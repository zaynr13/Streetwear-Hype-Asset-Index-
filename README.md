# Hype Asset Index

Created by Zayn Remtulla.

Live app: https://streetwear-hype-asset-index.streamlit.app/

A Streamlit dashboard for sneaker, streetwear, and luxury resale analytics.

## What it does

- Tracks individual item growth from each item's own release window
- Compares sneaker, streetwear, and luxury baskets against S&P 500, Nike, and gold
- Estimates future resale value with base / low / high scenarios
- Supports a unified Estimator with optional StockX link-assisted auto-fill
- Calculates premium, liquidity, risk, scarcity, grail score, and market segment

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py


## Important

The StockX link assist does not pull live sales automatically. It checks for a local dataset match first. If there is no match, it fills smart starting values that the user can confirm from StockX. True live verified updates would require official API access or another licensed data source.


## Benchmark fix

S&P 500 now uses fallback benchmark tickers and ignores broken partial data windows, so the line should cover the selected item's full comparison period instead of appearing as a tiny sliver.


## Basket maturity filter

Category and benchmark baskets now default to established items with at least six months of history. This keeps brand-new drops like Nike Mind 001 from creating a misleading sneaker-category spike.
