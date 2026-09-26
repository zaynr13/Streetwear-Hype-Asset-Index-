# Hype Asset Index

Created by Zayn Remtulla

Live app:
https://streetwear-hype-asset-index.streamlit.app/

The Hype Asset Index is a Streamlit app I made to look at sneakers, streetwear, and luxury items as investments.

## What it does

- Tracks how much different items have gone up or down since release
- Lets users enter an item and estimate its investment data
- Compares resale items to the S&P 500, Nike stock, and gold
- Looks at things like liquidity, risk, resale premium, scarcity, and overall investability
- Uses my Retail Scarcity Score to measure how hard an item was to get at retail
- Includes a 70-item test showing the relationship between scarcity and returns
- Tests whether the scoring system still works when the weights are changed
- Includes a basic holdout test for the regression model
- Keeps future prices as examples instead of calling them actual predictions

## Retail Scarcity Score

The Retail Scarcity Score, or RSS, measures how limited an item was when it originally released.

RSS uses 3 things:

- Distribution: how many places sold the item
- Access: how hard it was to actually buy
- Replenishment: whether more official stock came out after release

The formula is:

RSS = 45% Distribution + 20% Access + 35% Replenishment

Each part is scored from 0 to 100.

If there is not enough information about restocks yet, that part is left out instead of guessing.

The score does not use resale price, profits, sales volume, brand hype, collaborations, or stock market performance.

## RSS testing

I scored all 116 products in the dataset.

- 76 had high-confidence information
- 40 had medium-confidence information
- 91 had enough information to score restocks
- 25 did not, so the score was adjusted without guessing
- All 32 of my pre-made ranking checks passed
- Changing the weights did almost nothing to the overall rankings

I then tested RSS against returns using a cleaner group of 70 products.

The results were:

- Pearson correlation: about 0.45
- Spearman correlation: about 0.45
- Top 10 performing items had an average RSS of about 82
- Bottom 10 had an average RSS of about 44

Basically, items that were harder to get at retail usually had better returns in this sample.

That does not mean scarcity automatically causes higher returns.

## Data

The full dataset has 116 products.

- 95 have stronger market-price data
- 21 have lower-volume market data

The scarcity research is the strongest part of the project right now.

Some of the historical price charts and other parts of the dashboard still use estimated data because I do not yet have full verified sale histories for every product.

Because of that, things like the regression models, event studies, correlations, and projections should still be treated as experimental.

## Investability Score

I also created an Investability Score.

It currently uses:

- 40% liquidity
- 25% resale premium
- 20% risk
- 15% Retail Scarcity Score

The goal is to give a simple overall score for how investable an item may be.

## Running the app

Install the requirements:

python3 -m pip install -r requirements.txt

Then run:

python3 -m streamlit run app.py

## Final note

The Retail Scarcity Score is currently the most tested and reliable part of the project.

The rest of the dashboard is still a research prototype and would need better real sale-history data before I would treat the results as fully reliable.
