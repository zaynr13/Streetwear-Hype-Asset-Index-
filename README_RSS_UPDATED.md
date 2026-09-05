# Hype Asset Index

Created by Zayn Remtulla.

Live app: https://streetwear-hype-asset-index.streamlit.app/

A Streamlit dashboard for sneaker, streetwear, and luxury resale analytics.

## What it does

- Tracks individual item growth from each item's own release window
- Lets users estimate an item with a StockX link-assisted workflow
- Compares sneaker, streetwear, and luxury resale baskets against S&P 500, Nike, and gold
- Measures liquidity, risk, premium, collector/grail effects, and Investability
- Uses a validated **Retail Scarcity Score (RSS)** based only on documented retail release structure
- Includes a Retail Scarcity research tab with the cleaned 70-item performance test
- Adds a correlation matrix to test diversification value
- Adds sensitivity analysis for the Investability Score
- Adds a chronological holdout check for the regression model
- Keeps projections labeled as illustrative scenarios rather than price predictions

## Retail Scarcity Score

Retail Scarcity Score measures retail-release scarcity rather than exact production quantity.

**RSS = (0.45D + 0.20A + 0.35R) / (0.45I_D + 0.20I_A + 0.35I_R)**

- **D — Distribution Restriction (45%)**: how narrowly the item was distributed at retail
- **A — Access Restriction (20%)**: ordinary checkout vs FCFS, raffle/draw/password, EA/invite, or similarly controlled access
- **R — Replenishment Scarcity (35%)**: whether equivalent authorized supply returned after launch

Observed components use a fixed 0/25/50/75/100 rubric. If replenishment is too new to judge, R is withheld and the remaining weights are renormalized instead of guessed.

Resale price, resale premium, CAGR, sales volume, S&P 500 performance, brand prestige, collaboration status, and legacy scarcity scores are excluded from RSS.

### Validation

- 116 / 116 products scored
- 76 high-confidence rows
- 40 medium-confidence rows
- 91 products with replenishment observed
- 25 products with replenishment withheld and weights renormalized
- 32 / 32 pre-specified anchor-order checks passed
- RSS ranking correlations remained approximately 0.993–0.996 under major component-weight shifts

Using the **final locked 116-item RSS values**, the cleaned 70-item benchmark sample gives:

- Pearson **r ≈ 0.449**, **p ≈ 0.000096**
- Spearman **ρ ≈ 0.454**, **p ≈ 0.000080**
- Top-10 excess-return average RSS: **81.96**
- Bottom-10 excess-return average RSS: **43.61**
- Difference: **38.36 points**

This is a moderate positive association. It does not prove that scarcity causes returns.

## Data note

The production dataset contains 116 products: 95 verified market snapshots and 21 low-volume market snapshots. The RSS evidence basis and confidence fields are stored separately from market-price fields.

Historical resale paths used by several dashboard views remain prototype-estimated until full verified sold-price time series are collected. Therefore, the broader regression, correlation, event-study, and projection layers should still be treated as prototype research tools rather than final empirical evidence.

## Investability

The current Investability Score is a proposed scoring framework:

- 40% liquidity
- 25% premium over retail
- 20% inverse risk
- 15% Retail Scarcity Score

RSS replaces the legacy scarcity variable throughout the production dataset and app.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Research note

The Retail Scarcity Score is the strongest currently validated methodological component of the project. The broader dashboard remains a research prototype until verified transaction histories replace simulated price paths.
