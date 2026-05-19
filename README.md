# Hype Asset Index

Created by Zayn Remtulla.

Live app: https://streetwear-hype-asset-index.streamlit.app/

A Streamlit dashboard for sneaker, streetwear, and luxury resale analytics.

## What it does

- Tracks individual item growth from each item's own release window
- Lets users estimate an item with a StockX link-assisted workflow
- Compares sneaker, streetwear, and luxury resale baskets against S&P 500, Nike, and gold
- Adds a correlation matrix to test diversification value
- Adds sensitivity analysis for the investability score
- Adds a simple chronological holdout check for the regression model
- Widens projection ranges for weaker data-quality rows
- Fixes projection-weight consistency across built-in and custom estimators
- Stabilizes liquidity scoring using an absolute sales-volume scale
- Adds event-study design templates for Yeezy/Adidas and Travis Scott without plotting simulated event evidence
- Labels projections as illustrative scenarios, not price predictions

## Data note

The project separates rows into verified snapshots, low-volume market snapshots, and seed estimates. Seed estimates are included for prototype coverage and should be replaced with verified sold-price histories before formal research claims. The app labels prototype-estimated histories clearly and defaults research-facing views to market-backed rows where possible.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```


## Patch note

Plotly event-line compatibility fix added for the Event Studies section.


## Professor-facing note

This build is intended to be shown as a prototype framework, not as a completed empirical paper. Event studies and regression findings require verified sold-price histories before they should be interpreted academically.


## Final consistency patch

Estimator scoring now matches dataset scoring for liquidity and cultural/grail boosts. The correlation heatmap also includes an in-chart prototype warning so screenshots do not overstate the evidence.
