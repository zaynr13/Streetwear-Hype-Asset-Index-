
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Hype Asset Index", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
}
.creator-line {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-top: -0.25rem;
    margin-bottom: 0.15rem;
    line-height: 1.1;
}
.small-note {
    font-size: 0.92rem;
    opacity: 0.82;
}
.section-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


DATA_DIR = Path(__file__).resolve().parent / "data"
MARKET_PATH = DATA_DIR / "hype_asset_market_seed.csv"

# Transparent scenario assumptions used in both built-in and custom projections.
# These are not empirically optimized weights.
ITEM_TREND_WEIGHT = 0.42
CATEGORY_TREND_WEIGHT = 0.25
MARKET_TREND_WEIGHT = 0.18


@st.cache_data
def load_market() -> pd.DataFrame:
    df = pd.read_csv(MARKET_PATH)
    for col in ["retail_price","release_year","current_resale","price_low","price_high","volatility_pct","sales_volume","scarcity_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_metrics(df)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["premium_x"] = out["current_resale"] / out["retail_price"]
    out["return_from_retail"] = (out["current_resale"] - out["retail_price"]) / out["retail_price"]
    out["range_width_pct"] = (out["price_high"] - out["price_low"]) / out["current_resale"] * 100
    # Absolute liquidity scale instead of dataset-relative scaling.
    # This keeps scores stable when a very high-volume item is added later.
    out["liquidity_score"] = (100 * np.log1p(out["sales_volume"].fillna(0)) / np.log1p(1000)).clip(0, 100)

    vol = out["volatility_pct"].fillna(out["volatility_pct"].median())
    spread = out["range_width_pct"].fillna(out["range_width_pct"].median())
    out["risk_score"] = (
        0.45 * np.clip(vol / 45 * 100, 0, 100)
        + 0.25 * np.clip(spread / 220 * 100, 0, 100)
        + 0.30 * (100 - out["liquidity_score"])
    ).clip(0, 100)
    out["risk_label"] = pd.cut(out["risk_score"], [-1,35,65,101], labels=["Low","Medium","High"]).astype(str)

    out["investability_score"] = (
        0.40 * out["liquidity_score"]
        + 0.25 * np.clip(out["premium_x"] / 5 * 100, 0, 100)
        + 0.20 * (100 - out["risk_score"])
        + 0.15 * np.clip(out["scarcity_score"], 0, 100)
    ).clip(0, 100)

    out["data_quality"] = np.select(
        [
            out["data_status"].eq("verified_snapshot") & (out["sales_volume"] >= 100),
            out["data_status"].eq("verified_snapshot") | out["data_status"].eq("low_volume_snapshot") | (out["sales_volume"] >= 20),
        ],
        ["High", "Medium"],
        default="Seed"
    )

    # Grail score captures cultural/collector premium that ordinary resale variables miss.
    age_score = np.clip((2026 - out["release_year"]) / 12 * 100, 0, 100)
    premium_score = np.clip(out["premium_x"] / 10 * 100, 0, 100)
    scarcity = np.clip(out["scarcity_score"], 0, 100)
    low_volume_score = 100 - out["liquidity_score"]

    collab_boost = out["collaboration"].isin([
        "Off-White", "Yeezy", "Travis Scott", "Travis Scott x Fragment",
        "Kobe Bryant", "Ben & Jerry's", "JJJJound"
    ]).astype(int) * 12

    grail_keyword_boost = out["item"].str.contains(
        "Off-White|Yeezy 2|Red October|Solar Red|Chicago|Chunky Dunky|Travis Scott Jordan 1 High",
        case=False, regex=True, na=False
    ).astype(int) * 18

    # Use the larger cultural signal instead of adding both. This avoids double-counting
    # names like Travis Scott that can appear in both the collaboration field and item name.
    cultural_boost = np.maximum(collab_boost, grail_keyword_boost)

    out["grail_score"] = (
        0.30 * premium_score
        + 0.25 * scarcity
        + 0.20 * age_score
        + 0.15 * low_volume_score
        + cultural_boost
    ).clip(0, 100)

    out["market_segment"] = np.select(
        [
            out["grail_score"] >= 72,
            out["category"].eq("Luxury Streetwear"),
            out["category"].eq("Streetwear"),
            out["sales_volume"] >= 250,
        ],
        [
            "Grail / Collectible",
            "Luxury Streetwear",
            "Streetwear",
            "Liquid Sneaker",
        ],
        default="Specialty Sneaker"
    )
    return out


def estimated_release_start(row: pd.Series) -> pd.Timestamp:
    """
    Starts each item at its actual release window instead of forcing every item
    to begin on the same calendar date.
    """
    release_year = int(row["release_year"]) if pd.notna(row["release_year"]) else pd.Timestamp.today().year
    item_name = str(row["item"]).lower()

    if "nike mind 001" in item_name or "mind 001" in item_name:
        return pd.Timestamp(2026, 3, 1)

    # Birkin is a continuous luxury model, not a limited sneaker-style drop.
    # Start the prototype history from a modern tracking window instead of pretending
    # the app has resale history back to the original 1984 launch.
    if "birkin" in item_name or "hermes" in item_name or "hermès" in item_name:
        return pd.Timestamp(2020, 1, 1)

    return pd.Timestamp(release_year, 1, 1)


def estimated_release_resale(row: pd.Series) -> float:
    """
    Estimated starting resale value at/near release.
    This is still a prototype estimate until full historical sold-price data is uploaded.
    """
    retail = float(row["retail_price"])
    current = float(row["current_resale"])
    premium = current / max(retail, 1)

    if premium >= 10:
        start = max(retail * 2.2, current * 0.38)
    elif premium >= 5:
        start = max(retail * 1.8, current * 0.48)
    elif premium >= 2.5:
        start = max(retail * 1.35, current * 0.62)
    elif premium >= 1.25:
        start = max(retail * 1.05, current * 0.78)
    else:
        start = max(retail * 0.85, current * 0.92)

    return float(max(start, retail * 0.65))


def create_history(market: pd.DataFrame, months: int = 18, seed: int = 55) -> pd.DataFrame:
    """
    Release-based prototype history.

    Older versions used one shared 18-month window, so every item appeared to
    start around Dec 2024. This version begins each item at its release window
    and anchors the final point to current resale.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize().replace(day=1)
    rows = []

    for _, row in market.iterrows():
        release_start = estimated_release_start(row)
        dates = pd.date_range(start=release_start, end=end, freq="MS")

        if len(dates) == 0:
            dates = pd.DatetimeIndex([end])

        n_months = len(dates)
        current = float(row["current_resale"])
        start_price = estimated_release_resale(row)

        vol = float(row["volatility_pct"]) if pd.notna(row["volatility_pct"]) else 24.0
        vol_scale = min(max(vol / 100, 0.04), 0.45)

        # Smooth log path from release estimate to current value.
        base_path = np.linspace(np.log(start_price), np.log(current), n_months)

        # Damped noise: enough movement to look like a market, but anchored at both ends.
        noise = rng.normal(0, vol_scale / 7, n_months).cumsum()
        if n_months > 1:
            noise = noise - np.linspace(noise[0], noise[-1], n_months)
        else:
            noise = noise * 0

        # Hype-cycle shape: early bump for collabs/grails, then slower leveling.
        premium = current / max(float(row["retail_price"]), 1)
        age = np.linspace(0, 1, n_months)
        if premium >= 5:
            cycle = 0.14 * np.sin(np.pi * age) - 0.035 * age
        elif premium <= 1.2:
            cycle = -0.04 * np.sin(np.pi * age)
        else:
            cycle = 0.05 * np.sin(np.pi * age)

        series = np.exp(base_path + noise + cycle)

        # Anchor final point exactly to current resale.
        series[-1] = current

        # Prevent absurd negative/near-zero paths.
        floor = max(float(row["retail_price"]) * 0.45, 1)
        series = np.maximum(series, floor)
        series[-1] = current

        for month, price in zip(dates, series):
            rows.append({
                "month": month,
                "item": row["item"],
                "brand": row["brand"],
                "category": row["category"],
                "collaboration": row["collaboration"],
                "connection": row["connection"],
                "retail_price": row["retail_price"],
                "release_year": row["release_year"],
                "scarcity_score": row["scarcity_score"],
                "grail_score": row["grail_score"],
                "market_segment": row["market_segment"],
                "median_resale": price,
                "sale_count": max(1, int(row["sales_volume"] / 3)),
                "avg_size": 9.5 if row["category"] == "Sneaker" else 0,
                "data_status": row["data_status"],
            })

    panel = pd.DataFrame(rows)
    panel["premium_x"] = panel["median_resale"] / panel["retail_price"]
    panel["month_label"] = panel["month"].dt.strftime("%Y-%m")
    panel["months_since_release"] = panel.groupby("item").cumcount()
    return panel


def max_drawdown(series: pd.Series) -> float:
    vals = series.dropna().astype(float).values
    if len(vals) == 0:
        return 0.0
    peaks = np.maximum.accumulate(vals)
    return float(((vals - peaks) / peaks).min())


def annualized_volatility(returns: pd.Series) -> float:
    vals = returns.dropna()
    if len(vals) < 2:
        return 0.0
    return float(vals.std() * math.sqrt(12))


def make_index(panel: pd.DataFrame, category: str = "All", data_filter: str = "All items", min_history_months: int = 0) -> pd.DataFrame:
    work = panel.copy()
    if category != "All":
        work = work[work["category"] == category]

    if data_filter == "Market-backed":
        work = work[work["data_status"].isin(["verified_snapshot", "low_volume_snapshot"])]

    if min_history_months > 0 and not work.empty:
        history_counts = work.groupby("item")["month"].nunique()
        keep_items = history_counts[history_counts >= min_history_months].index
        work = work[work["item"].isin(keep_items)]

    if work["item"].nunique() < 2:
        return pd.DataFrame(columns=["month","index_value","monthly_return"])

    pivot = work.pivot_table(index="month", columns="item", values="median_resale", aggfunc="mean").sort_index()

    # Use fill_method=None so new items do not get silently forward-filled into months
    # where they did not exist.
    returns = pivot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    basket_return = returns.mean(axis=1, skipna=True).fillna(0)
    index_value = 100 * (1 + basket_return).cumprod()
    return pd.DataFrame({"month": index_value.index, "index_value": index_value.values, "monthly_return": basket_return.values})


def fit_regression(panel: pd.DataFrame):
    """
    Fixed-effect hedonic index model.

    includes item fixed effects + grail score + market segment.
    This lets the model learn that grail items have a permanent collector premium,
    while month effects still measure market movement.
    """
    work = panel.copy()
    work["log_price"] = np.log(work["median_resale"].clip(lower=1))

    numeric = ["retail_price", "scarcity_score", "grail_score", "sale_count", "avg_size"]
    categorical = ["item", "brand", "category", "collaboration", "connection", "market_segment", "month_label"]

    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    # Lower alpha reduces over-shrinking of legitimate grail prices.
    model = Pipeline([("transform", transform), ("ridge", Ridge(alpha=0.15))])

    X = work[numeric + categorical]
    y = work["log_price"]
    model.fit(X, y)
    pred = model.predict(X)

    work["predicted_resale"] = np.exp(pred)
    work["prediction_error"] = work["median_resale"] - work["predicted_resale"]

    mae = mean_absolute_error(work["median_resale"], work["predicted_resale"])
    r2 = r2_score(y, pred)

    encoder = model.named_steps["transform"].named_transformers_["cat"]
    features = numeric + list(encoder.get_feature_names_out(categorical))
    coefs = pd.DataFrame({"feature": features, "coefficient": model.named_steps["ridge"].coef_})
    coefs["abs_coef"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False)

    month_coefs = coefs[coefs["feature"].str.startswith("month_label_")].copy()
    if len(month_coefs) > 0:
        month_coefs["month_label"] = month_coefs["feature"].str.replace("month_label_", "", regex=False)
        month_coefs = month_coefs.sort_values("month_label")
        base = month_coefs["coefficient"].iloc[0]
        adjusted = pd.DataFrame({
            "month": pd.to_datetime(month_coefs["month_label"]),
            "adjusted_index": 100 * np.exp(month_coefs["coefficient"] - base)
        })
    else:
        adjusted = pd.DataFrame(columns=["month", "adjusted_index"])
    return work, coefs, adjusted, mae, r2



def regression_holdout_metrics(panel: pd.DataFrame, holdout_months: int = 3) -> dict:
    """
    Chronological holdout test.
    The last N months are held out so the model is evaluated out of sample.
    This still uses prototype-estimated histories, so it is a model check rather than an empirical proof.
    """
    work = panel.copy()
    months = sorted(work["month"].dropna().unique())
    if len(months) <= holdout_months + 3:
        return {"holdout_mae": np.nan, "holdout_r2": np.nan, "train_rows": 0, "test_rows": 0}

    cutoff_months = months[-holdout_months:]
    train = work[~work["month"].isin(cutoff_months)].copy()
    test = work[work["month"].isin(cutoff_months)].copy()

    if train.empty or test.empty:
        return {"holdout_mae": np.nan, "holdout_r2": np.nan, "train_rows": len(train), "test_rows": len(test)}

    train["log_price"] = np.log(train["median_resale"].clip(lower=1))
    test["log_price"] = np.log(test["median_resale"].clip(lower=1))

    numeric = ["retail_price", "scarcity_score", "grail_score", "sale_count", "avg_size"]
    categorical = ["item", "brand", "category", "collaboration", "connection", "market_segment", "month_label"]

    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([("transform", transform), ("ridge", Ridge(alpha=0.15))])
    model.fit(train[numeric + categorical], train["log_price"])
    pred_log = model.predict(test[numeric + categorical])
    pred_price = np.exp(pred_log)

    return {
        "holdout_mae": mean_absolute_error(test["median_resale"], pred_price),
        "holdout_r2": r2_score(test["log_price"], pred_log) if len(test) > 1 else np.nan,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def period_months(period: str) -> int:
    return {"6mo": 6, "1y": 12, "2y": 24}.get(period, 12)


def normalize_period_index(df: pd.DataFrame, value_col: str, period: str) -> pd.DataFrame:
    """
    Filter a series to the selected period and re-base it to 100.
    This keeps resale, S&P 500, Nike, and gold on the same visible timeline.
    """
    if df.empty:
        return df
    out = df.sort_values("month").copy()
    months = period_months(period)
    last_month = out["month"].max()
    cutoff = last_month - pd.DateOffset(months=months)
    out = out[out["month"] >= cutoff].copy()
    if out.empty:
        return out
    first = float(out[value_col].iloc[0])
    if first != 0:
        out[value_col] = 100 * out[value_col] / first
    return out


def smooth_monthly_index(df: pd.DataFrame, value_col: str, asset_name: str = "Resale Basket") -> pd.DataFrame:
    """
    Interpolate monthly index values to weekly points so the comparison chart does not look jagged.
    This is visual smoothing only; it does not create new real sales observations.
    """
    if df.empty:
        return pd.DataFrame(columns=["month", "index_value", "asset"])
    work = df[["month", value_col]].dropna().sort_values("month").copy()
    work = work.drop_duplicates("month")
    work = work.set_index("month")
    weekly_index = pd.date_range(work.index.min(), work.index.max(), freq="W-FRI")
    if len(weekly_index) == 0:
        weekly_index = work.index
    smooth = work.reindex(work.index.union(weekly_index)).sort_index().interpolate(method="time").reindex(weekly_index)
    smooth = smooth.reset_index().rename(columns={"index": "month", value_col: "index_value"})
    smooth["asset"] = asset_name
    return smooth


@st.cache_data(ttl=3600)
def fetch_benchmarks_for_range(start_month: pd.Timestamp, end_month: pd.Timestamp):
    """
    Fetch benchmark data across the same window as the selected item.
    Uses multiple ticker fallbacks for S&P 500 because SPY can occasionally
    return partial data in some yfinance sessions.
    """
    if yf is None:
        return pd.DataFrame()

    assets = {
        "S&P 500": ["SPY", "^GSPC"],
        "Nike": ["NKE"],
        "Gold ETF": ["GLD", "IAU"],
    }

    frames = []
    start = pd.to_datetime(start_month)
    end = pd.to_datetime(end_month) + pd.Timedelta(days=35)

    for label, tickers in assets.items():
        best_weekly = None
        best_coverage = -1

        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start - pd.Timedelta(days=10),
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if data is None or data.empty:
                    continue

                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                close = close[(close.index >= start) & (close.index <= end)]

                if len(close) < 2:
                    continue

                weekly = close.resample("W-FRI").last().dropna()
                coverage = len(weekly)

                if coverage > best_coverage:
                    best_weekly = weekly
                    best_coverage = coverage
            except Exception:
                continue

        if best_weekly is None or len(best_weekly) < 2:
            continue

        # Do not plot weird tiny slivers that cover almost none of the selected window.
        expected_weeks = max(2, int(((end - start).days) / 7))
        if len(best_weekly) < max(4, expected_weeks * 0.35):
            continue

        idx = 100 * best_weekly / best_weekly.iloc[0]
        frames.append(pd.DataFrame({
            "month": idx.index,
            "index_value": idx.values,
            "asset": label
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trailing_monthly_growth(values: pd.Series, months: int = 6) -> float:
    vals = values.dropna().astype(float)
    if len(vals) < 2:
        return 0.0
    start_idx = max(0, len(vals) - months - 1)
    start = vals.iloc[start_idx]
    end = vals.iloc[-1]
    periods = len(vals) - 1 - start_idx
    if start <= 0 or periods <= 0:
        return 0.0
    return float((end / start) ** (1 / periods) - 1)


def category_monthly_growth(panel: pd.DataFrame, category: str, months: int = 6) -> float:
    idx = make_index(panel, category)
    if idx.empty:
        return 0.0
    return trailing_monthly_growth(idx["index_value"], months)


def market_monthly_growth(panel: pd.DataFrame, months: int = 6) -> float:
    idx = make_index(panel, "All")
    if idx.empty:
        return 0.0
    return trailing_monthly_growth(idx["index_value"], months)


def estimate_expected_monthly_growth(item_row: pd.Series, item_history: pd.DataFrame, panel: pd.DataFrame) -> tuple[float, dict]:
    item_trend = trailing_monthly_growth(item_history["median_resale"], 6)
    cat_trend = category_monthly_growth(panel, item_row["category"], 6)
    mkt_trend = market_monthly_growth(panel, 6)

    investability_adj = ((float(item_row["investability_score"]) - 50) / 100) * 0.010
    liquidity_adj = ((float(item_row["liquidity_score"]) - 50) / 100) * 0.006
    risk_adj = -((float(item_row["risk_score"]) - 50) / 100) * 0.008

    segment = item_row["market_segment"]
    if segment == "Liquid Sneaker":
        segment_adj = 0.002
    elif segment == "Grail / Collectible":
        # Grails can keep value, but low sales makes growth less predictable.
        segment_adj = 0.001 if float(item_row["grail_score"]) >= 85 else -0.001
    elif segment == "Streetwear":
        segment_adj = -0.001
    elif segment == "Luxury Streetwear":
        segment_adj = -0.002
    else:
        segment_adj = 0.000

    # Mean reversion: huge premiums with thin sales should not receive crazy growth.
    premium = float(item_row["premium_x"])
    sales = float(item_row["sales_volume"])
    if premium > 12 and sales < 20:
        mean_reversion = -0.012
    elif premium > 8 and sales < 50:
        mean_reversion = -0.007
    elif premium < 1.2 and sales > 500:
        mean_reversion = 0.004
    else:
        mean_reversion = 0.0

    raw_growth = (
        ITEM_TREND_WEIGHT * item_trend
        + CATEGORY_TREND_WEIGHT * cat_trend
        + MARKET_TREND_WEIGHT * mkt_trend
        + investability_adj
        + liquidity_adj
        + risk_adj
        + segment_adj
        + mean_reversion
    )

    # Keep scenario forecast realistic for resale assets.
    expected = float(np.clip(raw_growth, -0.035, 0.045))

    drivers = {
        "Item trend": item_trend,
        "Category trend": cat_trend,
        "Market trend": mkt_trend,
        "Investability adj.": investability_adj,
        "Liquidity adj.": liquidity_adj,
        "Risk adj.": risk_adj,
        "Segment adj.": segment_adj,
        "Mean reversion": mean_reversion,
        "Expected monthly growth": expected,
    }
    return expected, drivers



def forecast_band_pct(annual_vol: float, segment: str, step: int, horizon: int) -> float:
    """
    Tight scenario range. The range grows with horizon but stays realistic.
    """
    segment_caps = {
        "Liquid Sneaker": 0.045,
        "Specialty Sneaker": 0.060,
        "Grail / Collectible": 0.080,
        "Streetwear": 0.065,
        "Luxury Streetwear": 0.075,
    }
    cap = segment_caps.get(segment, 0.060)
    scale = math.sqrt(step / max(horizon, 1))
    raw_band = annual_vol * math.sqrt(step / 12) * 0.18
    floor = 0.012 * scale
    max_allowed = cap * scale
    return float(np.clip(raw_band, floor, max_allowed))



def data_uncertainty_multiplier(data_status: str) -> float:
    """
    Widen illustrative projection bands when the underlying data is weaker.
    This makes low-volume and seed-estimate rows visually less certain.
    """
    status = str(data_status).lower()
    if "seed" in status:
        return 2.25
    if "low_volume" in status:
        return 1.65
    if "user_estimate" in status:
        return 1.85
    return 1.00



def leveled_growth_price(current_price: float, expected_monthly: float, premium_x: float, step: int, segment: str) -> float:
    """
    Forecast with leveling-off. Early months can follow the trend, but high-premium items slow down over time.
    This avoids pretending a resale item compounds forever like a savings account.
    """
    # Higher premium = stronger mean reversion / saturation.
    if premium_x >= 10:
        decay = 0.82
    elif premium_x >= 5:
        decay = 0.88
    elif premium_x >= 2.5:
        decay = 0.93
    else:
        decay = 0.97

    if segment == "Grail / Collectible":
        decay += 0.03  # grails can hold value better
    elif segment in ["Streetwear", "Luxury Streetwear"]:
        decay -= 0.02

    decay = min(max(decay, 0.78), 0.98)

    price = current_price
    monthly = expected_monthly
    for _ in range(step):
        price *= (1 + monthly)
        monthly *= decay
    return float(price)


def build_item_forecast(item_name: str, market: pd.DataFrame, panel: pd.DataFrame, months_forward: int = 12) -> tuple[pd.DataFrame, dict]:
    item_row = market[market["item"].eq(item_name)].iloc[0]
    item_history = panel[panel["item"].eq(item_name)].sort_values("month").copy()
    expected_monthly, drivers = estimate_expected_monthly_growth(item_row, item_history, panel)

    current_price = float(item_row["current_resale"])
    annual_vol = float(item_row["volatility_pct"]) / 100 if pd.notna(item_row["volatility_pct"]) else 0.25
    annual_vol = float(np.clip(annual_vol, 0.08, 0.65))
    premium_x = float(item_row["premium_x"])
    segment = str(item_row["market_segment"])

    start_month = item_history["month"].max()
    future_months = pd.date_range(start=start_month + pd.offsets.MonthBegin(1), periods=months_forward, freq="MS")

    rows = []
    for step, month in enumerate(future_months, start=1):
        base_price = leveled_growth_price(current_price, expected_monthly, premium_x, step, segment)
        band = forecast_band_pct(annual_vol, segment, step, months_forward)

        band = min(0.65, band * data_uncertainty_multiplier(item_row.get("data_status", "seed_estimate")))
        low = base_price * (1 - band)
        high = base_price * (1 + band)
        rows.append({
            "month": month,
            "item": item_name,
            "forecast_price": base_price,
            "low_estimate": low,
            "high_estimate": high,
            "expected_monthly_growth": expected_monthly,
            "expected_annual_growth": (base_price / current_price) ** (12 / step) - 1 if step > 0 else 0,
        })

    forecast = pd.DataFrame(rows)
    return forecast, drivers


def forecast_summary_table(selected_items: list[str], market: pd.DataFrame, panel: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    rows = []
    for item in selected_items:
        if item not in set(market["item"]):
            continue
        forecast, drivers = build_item_forecast(item, market, panel, horizon)
        row = market[market["item"].eq(item)].iloc[0]
        final = forecast.iloc[-1]
        current = float(row["current_resale"])
        rows.append({
            "Item": item,
            "Segment": row["market_segment"],
            "Current resale": current,
            f"{horizon}M base estimate": float(final["forecast_price"]),
            f"{horizon}M low": float(final["low_estimate"]),
            f"{horizon}M high": float(final["high_estimate"]),
            "Illustrative annual growth": float(final["expected_annual_growth"]),
            "Risk": float(row["risk_score"]),
            "Liquidity": float(row["liquidity_score"]),
            "Grail": float(row["grail_score"]),
            "Data quality": row["data_quality"],
        })
    return pd.DataFrame(rows)


def infer_custom_segment(category: str, premium_x: float, sales_volume: float, grail_score: float) -> str:
    if grail_score >= 72:
        return "Grail / Collectible"
    if category == "Luxury Streetwear":
        return "Luxury Streetwear"
    if category == "Streetwear":
        return "Streetwear"
    if category == "Sneaker" and sales_volume >= 250:
        return "Liquid Sneaker"
    return "Specialty Sneaker"


def custom_scores(
    market: pd.DataFrame,
    item_name: str,
    category: str,
    collaboration_yes: str,
    collaboration_name: str,
    demand_driver: str,
    retail_price: float,
    current_resale: float,
    price_low: float,
    price_high: float,
    sales_volume: float,
    release_year: int,
    manual_volatility: float | None = None,
    manual_scarcity: float | None = None,
) -> dict:
    retail_price = max(float(retail_price), 1.0)
    current_resale = max(float(current_resale), 1.0)
    price_low = max(float(price_low), 1.0)
    price_high = max(float(price_high), price_low)
    sales_volume = max(float(sales_volume), 0.0)

    premium_x = current_resale / retail_price
    # Match add_metrics(): absolute liquidity scale anchored at 1000 sales.
    # This keeps Estimator scores comparable to dataset scores.
    liquidity_score = 100 * math.log1p(sales_volume) / math.log1p(1000)
    liquidity_score = float(np.clip(liquidity_score, 0, 100))

    range_width_pct = (price_high - price_low) / current_resale * 100

    # Auto-estimate volatility from recent low/high range.
    # This is easier for users because they can read low/current/high from StockX.
    auto_volatility = min(max(range_width_pct * 0.32, 6), 45)
    volatility_pct = float(manual_volatility) if manual_volatility is not None else auto_volatility

    # Auto-estimate scarcity from sales volume, premium, age, and collab.
    age_score = min(max((2026 - int(release_year)) / 12 * 100, 0), 100)
    premium_score = min(max(premium_x / 8 * 100, 0), 100)
    low_volume_score = 100 - liquidity_score
    collab_bonus = 12 if collaboration_yes == "Yes" else 0
    auto_scarcity = (
        0.35 * low_volume_score
        + 0.30 * premium_score
        + 0.20 * age_score
        + 0.15 * collab_bonus * 8
    )
    auto_scarcity = min(max(auto_scarcity, 0), 100)
    scarcity_score = float(manual_scarcity) if manual_scarcity is not None else auto_scarcity

    risk_score = (
        0.45 * min(max(volatility_pct / 45 * 100, 0), 100)
        + 0.25 * min(max(range_width_pct / 220 * 100, 0), 100)
        + 0.30 * (100 - liquidity_score)
    )
    risk_score = min(max(risk_score, 0), 100)

    collab_text = f"{item_name} {collaboration_name}"
    collab_words = ["Off-White", "Yeezy", "Travis", "Kobe", "Ben & Jerry", "JJJJound", "Chrome Hearts", "Supreme", "Wales Bonner"]
    collab_boost = 12 if any(w.lower() in collab_text.lower() for w in collab_words) else (6 if collaboration_yes == "Yes" else 0)

    grail_words = ["Off-White", "Yeezy 2", "Red October", "Solar Red", "Chicago", "Travis Scott Jordan 1 High", "Chrome Hearts"]
    grail_keyword_boost = 18 if any(w.lower() in item_name.lower() for w in grail_words) else 0

    # Match add_metrics(): use the larger cultural signal instead of stacking both.
    cultural_boost = max(collab_boost, grail_keyword_boost)

    grail_score = (
        0.30 * premium_score
        + 0.25 * scarcity_score
        + 0.20 * age_score
        + 0.15 * low_volume_score
        + cultural_boost
    )
    grail_score = min(max(grail_score, 0), 100)

    collaboration = collaboration_name.strip() if collaboration_yes == "Yes" and collaboration_name.strip() else "Collaboration" if collaboration_yes == "Yes" else "None"
    segment = infer_custom_segment(category, premium_x, sales_volume, grail_score)

    investability_score = (
        0.40 * liquidity_score
        + 0.25 * min(max(premium_x / 5 * 100, 0), 100)
        + 0.20 * (100 - risk_score)
        + 0.15 * min(max(scarcity_score, 0), 100)
    )
    investability_score = min(max(investability_score, 0), 100)

    return {
        "item": item_name,
        "category": category,
        "collaboration": collaboration,
        "connection": demand_driver,
        "retail_price": retail_price,
        "current_resale": current_resale,
        "price_low": price_low,
        "price_high": price_high,
        "volatility_pct": volatility_pct,
        "auto_volatility_pct": auto_volatility,
        "sales_volume": sales_volume,
        "scarcity_score": scarcity_score,
        "auto_scarcity_score": auto_scarcity,
        "release_year": release_year,
        "premium_x": premium_x,
        "liquidity_score": liquidity_score,
        "risk_score": risk_score,
        "grail_score": grail_score,
        "investability_score": investability_score,
        "market_segment": segment,
    }



def stockx_slug_to_name(url: str) -> str:
    """
    Link-assisted import only. This does not scrape StockX.
    It cleans the product slug from a StockX link into a readable item name.
    """
    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url.strip())
        slug = parsed.path.strip("/").split("/")[0]
        slug = unquote(slug)
    except Exception:
        slug = url.strip().split("/")[-1]
    slug = slug.split("?")[0].strip()
    if not slug:
        return ""
    words = slug.replace("-", " ").replace("_", " ").split()
    small_words = {"og", "sp", "sb", "v2", "fw", "ss", "xx", "x"}
    cleaned = []
    for word in words:
        upper_tokens = {"jordan", "nike", "sb", "og", "sp", "yeezy", "bape", "cpfm", "kobe", "dunk", "air"}
        if word.lower() in {"og", "sp", "sb", "v2", "fw", "ss", "xx"}:
            cleaned.append(word.upper())
        elif word.lower() == "x":
            cleaned.append("x")
        else:
            cleaned.append(word.capitalize())
    return " ".join(cleaned)


def infer_from_stockx_name(name: str) -> dict:
    lower = name.lower()
    category = "Sneaker"
    brand = ""
    collab_yes = "No"
    collab_name = ""
    demand_driver = "None"

    brand_rules = [
        ("hermes", "Hermès", "Luxury Streetwear"),
        ("hermès", "Hermès", "Luxury Streetwear"),
        ("birkin", "Hermès", "Luxury Streetwear"),
        ("chrome hearts", "Chrome Hearts", "Luxury Streetwear"),
        ("supreme", "Supreme", "Streetwear"),
        ("denim tears", "Denim Tears", "Streetwear"),
        ("bape", "BAPE", "Streetwear"),
        ("stussy", "Stussy", "Streetwear"),
        ("cactus plant flea market", "CPFM", "Streetwear"),
        ("cpfm", "CPFM", "Streetwear"),
        ("gallery dept", "Gallery Dept.", "Luxury Streetwear"),
        ("hellstar", "Hellstar", "Streetwear"),
        ("sp5der", "Sp5der", "Streetwear"),
        ("corteiz", "Corteiz", "Streetwear"),
        ("fear of god", "Fear of God", "Streetwear"),
        ("palace", "Palace", "Streetwear"),
        ("kith", "Kith", "Streetwear"),
        ("aime leon dore", "Aime Leon Dore", "Streetwear"),
        ("jordan", "Jordan", "Sneaker"),
        ("nike mind", "Nike", "Sneaker"),
        ("nike", "Nike", "Sneaker"),
        ("adidas", "Adidas", "Sneaker"),
        ("yeezy", "Adidas Yeezy", "Sneaker"),
        ("new balance", "New Balance", "Sneaker"),
        ("asics", "ASICS", "Sneaker"),
    ]
    for key, inferred_brand, inferred_category in brand_rules:
        if key in lower:
            brand = inferred_brand
            category = inferred_category
            break

    collab_rules = [
        ("off white", "Off-White", "Designer"),
        ("off-white", "Off-White", "Designer"),
        ("travis scott", "Travis Scott", "Celebrity"),
        ("fragment", "Fragment", "Designer"),
        ("kobe", "Kobe Bryant", "Athlete"),
        ("jjjjound", "JJJJound", "Designer"),
        ("jarritos", "Jarritos", "Brand Collab"),
        ("born x raised", "Born x Raised", "Streetwear"),
        ("ben jerry", "Ben & Jerry's", "Brand Collab"),
        ("supreme", "Supreme", "Streetwear"),
        ("burberry", "Burberry", "Luxury"),
        ("louis vuitton", "Louis Vuitton", "Luxury"),
        ("wales bonner", "Wales Bonner", "Designer"),
        ("a ma maniere", "A Ma Maniere", "Designer"),
        ("mm6", "MM6 Maison Margiela", "Designer"),
        ("levis", "Levi's", "Brand Collab"),
        ("levi", "Levi's", "Brand Collab"),
    ]
    for key, inferred_collab, driver in collab_rules:
        if key in lower:
            collab_yes = "Yes"
            collab_name = inferred_collab
            demand_driver = driver
            break

    if demand_driver == "None":
        if "nike mind" in lower:
            demand_driver = "Brand Collab" if "fragment" in lower else "None"
        elif category == "Sneaker" and any(x in lower for x in ["jordan", "kobe"]):
            demand_driver = "Athlete"
        elif category == "Luxury Streetwear":
            demand_driver = "Luxury"
        elif category == "Streetwear":
            demand_driver = "Streetwear"

    if not brand:
        brand = "Unknown"

    return {
        "item_name": name,
        "brand": brand,
        "category": category,
        "collaboration_yes": collab_yes,
        "collaboration_name": collab_name,
        "demand_driver": demand_driver,
    }


def default_category_index(category: str) -> int:
    options = ["Sneaker", "Streetwear", "Luxury Streetwear"]
    return options.index(category) if category in options else 0


def default_driver_index(driver: str) -> int:
    options = ["Athlete", "Celebrity", "Designer", "Streetwear", "Luxury", "Brand Collab", "None"]
    return options.index(driver) if driver in options else 6


def simple_similarity(a: str, b: str) -> float:
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    except Exception:
        return 0.0


def find_market_match(item_name: str, market: pd.DataFrame) -> tuple[pd.Series | None, float]:
    if not item_name:
        return None, 0.0
    best_score = 0.0
    best_row = None
    target = item_name.lower()

    for _, row in market.iterrows():
        candidate = str(row["item"]).lower()
        score = simple_similarity(target, candidate)

        target_tokens = set(target.replace("-", " ").split())
        candidate_tokens = set(candidate.replace("-", " ").split())
        if target_tokens and candidate_tokens:
            overlap = len(target_tokens & candidate_tokens) / max(len(target_tokens | candidate_tokens), 1)
            score = max(score, overlap)

        if target in candidate or candidate in target:
            score = max(score, 0.92)

        if score > best_score:
            best_score = score
            best_row = row

    if best_score >= 0.58:
        return best_row, best_score
    return None, best_score


def sales_band_to_volume(band: str) -> float:
    mapping = {
        "Very high, 1000+ recent sales": 1200.0,
        "High, 300-999 recent sales": 500.0,
        "Medium, 75-299 recent sales": 150.0,
        "Low, 15-74 recent sales": 40.0,
        "Very low, under 15 recent sales": 8.0,
    }
    return mapping.get(band, 150.0)


def volume_to_sales_band(volume: float) -> str:
    if volume >= 1000:
        return "Very high, 1000+ recent sales"
    if volume >= 300:
        return "High, 300-999 recent sales"
    if volume >= 75:
        return "Medium, 75-299 recent sales"
    if volume >= 15:
        return "Low, 15-74 recent sales"
    return "Very low, under 15 recent sales"


def retail_guess_from_name(name: str, category: str) -> float:
    lower = name.lower()
    if category != "Sneaker":
        if "birkin" in lower or "hermes" in lower or "hermès" in lower:
            return 11000.0
        if "chrome hearts" in lower or "louis vuitton" in lower:
            return 850.0
        if "hoodie" in lower:
            return 180.0
        if "tee" in lower or "t-shirt" in lower:
            return 54.0
        if "jean" in lower or "denim" in lower:
            return 295.0
        return 200.0

    if "nike mind 001" in lower or "mind 001" in lower:
        return 90.0 if "fragment" in lower else 95.0
    if "yeezy slide" in lower or "slide" in lower:
        return 70.0
    if "foam runner" in lower:
        return 90.0
    if "yeezy 350" in lower:
        return 220.0
    if "yeezy 700" in lower:
        return 300.0
    if "air yeezy 2" in lower:
        return 245.0
    if "jordan 1" in lower and ("off white" in lower or "off-white" in lower):
        return 190.0
    if "jordan 1" in lower:
        return 180.0
    if "jordan 4" in lower:
        return 225.0
    if "jordan 3" in lower:
        return 210.0
    if "jordan 11" in lower:
        return 230.0
    if "sb dunk" in lower:
        return 130.0
    if "dunk" in lower:
        return 120.0
    if "kobe" in lower:
        return 180.0
    if "new balance" in lower:
        return 180.0
    if "samba" in lower:
        return 120.0
    return 150.0


def stockx_import_defaults(url: str, market: pd.DataFrame) -> dict:
    name = stockx_slug_to_name(url)
    inferred = infer_from_stockx_name(name) if name else {
        "item_name": "",
        "brand": "Unknown",
        "category": "Sneaker",
        "collaboration_yes": "No",
        "collaboration_name": "",
        "demand_driver": "None",
    }

    match_row, match_score = find_market_match(name, market)
    if match_row is not None and match_score >= 0.72:
        current = float(match_row["current_resale"])
        low = float(match_row["price_low"])
        high = float(match_row["price_high"])
        retail = float(match_row["retail_price"])
        volume = float(match_row["sales_volume"])
        return {
            **inferred,
            "retail_price": retail,
            "current_resale": current,
            "price_low": low,
            "price_high": high,
            "sales_band": volume_to_sales_band(volume),
            "sales_volume": volume,
            "release_year": int(match_row["release_year"]),
            "match_status": f"Matched local dataset: {match_row['item']} ({match_score:.0%})",
            "matched_item": str(match_row["item"]),
            "match_score": match_score,
        }

    # Heuristic defaults for items not in the local dataset.
    retail = retail_guess_from_name(name, inferred["category"])
    lower = name.lower()

    if "birkin" in lower or "hermes" in lower or "hermès" in lower:
        multiplier = 2.1
        sales_band = "Very low, under 15 recent sales"
    elif "nike mind 001" in lower and "fragment" in lower:
        multiplier = 9.7
        sales_band = "Low, 15-74 recent sales"
    elif "nike mind 001" in lower or "mind 001" in lower:
        multiplier = 2.0
        sales_band = "Medium, 75-299 recent sales"
    elif "yeezy slide" in lower or "slide" in lower:
        multiplier = 1.6
        sales_band = "Very high, 1000+ recent sales"
    elif "jordan 1" in lower and ("off white" in lower or "off-white" in lower):
        multiplier = 9.0
        sales_band = "Low, 15-74 recent sales"
    elif "travis scott" in lower:
        multiplier = 6.0
        sales_band = "Medium, 75-299 recent sales"
    elif "chrome hearts" in lower:
        multiplier = 2.2
        sales_band = "Low, 15-74 recent sales"
    elif inferred["category"] == "Streetwear":
        multiplier = 1.5
        sales_band = "Medium, 75-299 recent sales"
    elif inferred["category"] == "Luxury Streetwear":
        multiplier = 1.8
        sales_band = "Low, 15-74 recent sales"
    else:
        multiplier = 1.45
        sales_band = "Medium, 75-299 recent sales"

    current = retail * multiplier
    low = current * 0.82
    high = current * 1.22

    return {
        **inferred,
        "retail_price": retail,
        "current_resale": current,
        "price_low": low,
        "price_high": high,
        "sales_band": sales_band,
        "sales_volume": sales_band_to_volume(sales_band),
        "release_year": 2023,
        "match_status": "No local data match. Using smart defaults; user should confirm visible StockX numbers.",
        "matched_item": "",
        "match_score": match_score,
    }


def build_custom_forecast(
    custom: dict,
    panel: pd.DataFrame,
    horizon: int,
    price_6m_ago: float = 0,
    price_12m_ago: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = float(custom["current_resale"])
    category = str(custom["category"])
    segment = str(custom["market_segment"])

    cat_trend = category_monthly_growth(panel, category, 6)
    mkt_trend = market_monthly_growth(panel, 6)

    if price_6m_ago and price_6m_ago > 0:
        item_trend = (current / price_6m_ago) ** (1 / 6) - 1
        trend_source = "6M user input"
    elif price_12m_ago and price_12m_ago > 0:
        item_trend = (current / price_12m_ago) ** (1 / 12) - 1
        trend_source = "12M user input"
    else:
        item_trend = 0.50 * cat_trend + 0.50 * mkt_trend
        trend_source = "category/market proxy"

    investability_adj = ((float(custom["investability_score"]) - 50) / 100) * 0.010
    liquidity_adj = ((float(custom["liquidity_score"]) - 50) / 100) * 0.006
    risk_adj = -((float(custom["risk_score"]) - 50) / 100) * 0.008

    if segment == "Liquid Sneaker":
        segment_adj = 0.002
    elif segment == "Grail / Collectible":
        segment_adj = 0.001 if float(custom["grail_score"]) >= 85 else -0.001
    elif segment == "Streetwear":
        segment_adj = -0.001
    elif segment == "Luxury Streetwear":
        segment_adj = -0.002
    else:
        segment_adj = 0.000

    premium = float(custom["premium_x"])
    sales = float(custom["sales_volume"])
    if premium > 12 and sales < 20:
        mean_reversion = -0.012
    elif premium > 8 and sales < 50:
        mean_reversion = -0.007
    elif premium < 1.2 and sales > 500:
        mean_reversion = 0.004
    else:
        mean_reversion = 0.0

    expected_monthly = (
        ITEM_TREND_WEIGHT * item_trend
        + CATEGORY_TREND_WEIGHT * cat_trend
        + MARKET_TREND_WEIGHT * mkt_trend
        + investability_adj
        + liquidity_adj
        + risk_adj
        + segment_adj
        + mean_reversion
    )
    expected_monthly = float(np.clip(expected_monthly, -0.035, 0.045))

    annual_vol = min(max(float(custom["volatility_pct"]) / 100, 0.08), 0.65)
    start = pd.Timestamp.today().normalize().replace(day=1)
    dates = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    rows = []
    premium_x = float(custom["premium_x"])
    for step, month in enumerate(dates, start=1):
        base_price = leveled_growth_price(current, expected_monthly, premium_x, step, segment)
        band = forecast_band_pct(annual_vol, segment, step, horizon)

        if trend_source == "category/market proxy":

            band = min(0.65, band * 1.60)
        rows.append({
            "month": month,
            "forecast_price": base_price,
            "low_estimate": base_price * (1 - band),
            "high_estimate": base_price * (1 + band),
            "expected_monthly_growth": expected_monthly,
            "expected_annual_growth": (base_price / current) ** (12 / step) - 1 if step > 0 else 0,
        })

    forecast = pd.DataFrame(rows)
    drivers = pd.DataFrame([
        {"Driver": "Item trend", "Value": item_trend, "Notes": trend_source},
        {"Driver": "Category trend", "Value": cat_trend, "Notes": category},
        {"Driver": "Market trend", "Value": mkt_trend, "Notes": "broad resale basket"},
        {"Driver": "Investability adj.", "Value": investability_adj, "Notes": ""},
        {"Driver": "Liquidity adj.", "Value": liquidity_adj, "Notes": ""},
        {"Driver": "Risk adj.", "Value": risk_adj, "Notes": ""},
        {"Driver": "Segment adj.", "Value": segment_adj, "Notes": segment},
        {"Driver": "Mean reversion", "Value": mean_reversion, "Notes": ""},
        {"Driver": "Expected monthly growth", "Value": expected_monthly, "Notes": "final model estimate"},
    ])
    return forecast, drivers


market = load_market()
panel = create_history(market)

# Academic-facing model views exclude seed estimates by default.
model_panel = panel[panel["data_status"].isin(["verified_snapshot", "low_volume_snapshot"])].copy()
if model_panel["item"].nunique() < 5:
    model_panel = panel.copy()

fitted, coefs, adjusted_index, mae, r2 = fit_regression(model_panel)
holdout_report = regression_holdout_metrics(model_panel)


def aligned_benchmark_wide(resale_idx: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align resale basket and traditional benchmark returns on monthly periods.
    Correlations are based on monthly percentage returns, not price levels.
    """
    if resale_idx.empty:
        return pd.DataFrame()

    resale = resale_idx[["month", "index_value"]].copy()
    resale["period"] = pd.to_datetime(resale["month"]).dt.to_period("M").dt.to_timestamp()
    resale = resale.groupby("period")["index_value"].last().rename("Resale Basket")

    pieces = [resale]
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench["period"] = pd.to_datetime(bench["month"]).dt.to_period("M").dt.to_timestamp()
        for asset, group in bench.groupby("asset"):
            pieces.append(group.groupby("period")["index_value"].last().rename(asset))

    wide = pd.concat(pieces, axis=1).sort_index()
    return wide


def monthly_return_correlation(wide_index: pd.DataFrame) -> pd.DataFrame:
    if wide_index.empty or wide_index.shape[1] < 2:
        return pd.DataFrame()
    returns = wide_index.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[0] < 3 or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr()


# ---------- Presentation helpers ----------

FRIENDLY_COLUMNS = {
    "item": "Item",
    "brand": "Brand",
    "category": "Category",
    "market_segment": "Market Segment",
    "current_resale": "Current Resale",
    "retail_price": "Retail Price",
    "premium_x": "Premium (× retail)",
    "volatility_pct": "Volatility %",
    "sales_volume": "Sales Volume",
    "grail_score": "Grail Score",
    "risk_score": "Risk Score",
    "investability_score": "Investability Score",
    "data_quality": "Data Quality",
    "data_status": "Data Source",
    "source": "Source Note",
    "source_url": "Source URL",
    "release_year": "Release Year",
    "liquidity_score": "Liquidity Score",
    "price_low": "Recent Low",
    "price_high": "Recent High",
    "collaboration": "Collaboration",
    "connection": "Demand Driver",
    "feature": "Model Feature",
    "coefficient": "Coefficient",
    "prediction_error": "Prediction Error",
    "predicted_resale": "Model Estimate",
    "median_resale": "Observed / Estimated Resale",
}

DATA_STATUS_LABELS = {
    "verified_snapshot": "Verified snapshot",
    "low_volume_snapshot": "Low-volume market snapshot",
    "seed_estimate": "Seed estimate",
    "user_verified_stockx": "User-verified StockX",
    "user_estimate": "User estimate",
}

def humanize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "data_status":
            out[col] = out[col].map(DATA_STATUS_LABELS).fillna(out[col])
    out = out.rename(columns=FRIENDLY_COLUMNS)
    return out


def format_money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return value


def format_pct(value):
    try:
        return f"{float(value):+.1%}"
    except Exception:
        return value


def add_prototype_annotation(fig: go.Figure, text: str = "Prototype-estimated resale history"):
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=1,
        y=1.08,
        showarrow=False,
        align="right",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.10)",
        bordercolor="rgba(128,128,128,0.45)",
        borderwidth=1,
    )
    return fig


def add_projection_annotation(fig: go.Figure):
    fig.add_annotation(
        text="Illustrative projection, not a price prediction",
        xref="paper",
        yref="paper",
        x=1,
        y=1.08,
        showarrow=False,
        align="right",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.10)",
        bordercolor="rgba(128,128,128,0.45)",
        borderwidth=1,
    )
    return fig


def chart_caption(text: str):
    st.caption(text)


def category_label(name: str) -> str:
    return {"All": "All Resale", "Sneaker": "Sneakers", "Streetwear": "Streetwear", "Luxury Streetwear": "Luxury Streetwear"}.get(name, name)


def get_default_correlation_summary() -> str:
    try:
        idx = make_index(panel, "All", "Market-backed", min_history_months=6)
        idx = normalize_period_index(idx, "index_value", "1y")
        if idx.empty:
            return "Not enough overlapping market-backed data to summarize correlation yet."
        bench = fetch_benchmarks_for_range(idx["month"].min(), idx["month"].max())
        wide = aligned_benchmark_wide(idx, bench)
        corr = monthly_return_correlation(wide)
        if corr.empty or "Resale Basket" not in corr.columns:
            return "Not enough overlapping monthly benchmark returns to summarize correlation yet."
        vals = corr["Resale Basket"].drop("Resale Basket", errors="ignore").dropna()
        if len(vals) == 0:
            return "Not enough benchmark overlap to summarize correlation yet."
        lowest = vals.sort_values().iloc[0]
        highest = vals.sort_values().iloc[-1]
        return f"Preliminary 1-year market-backed resale correlations range from {lowest:.2f} to {highest:.2f} versus traditional benchmarks."
    except Exception:
        return "Correlation summary unavailable in the current local run."


def investability_scores_for_weights(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    premium_component = np.clip(df["premium_x"] / 5 * 100, 0, 100)
    components = {
        "Liquidity": df["liquidity_score"].fillna(0),
        "Premium": premium_component.fillna(0),
        "Inverse Risk": (100 - df["risk_score"]).fillna(0),
        "Scarcity": df["scarcity_score"].fillna(0),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {"Liquidity": 0.40, "Premium": 0.25, "Inverse Risk": 0.20, "Scarcity": 0.15}
        total_weight = 1.0
    score = sum((weights[k] / total_weight) * components[k] for k in weights)
    return score


def sensitivity_analysis(df: pd.DataFrame, top_n: int = 10, delta: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights = {"Liquidity": 0.40, "Premium": 0.25, "Inverse Risk": 0.20, "Scarcity": 0.15}
    work = df.copy()
    work["Base Score"] = investability_scores_for_weights(work, base_weights)
    base_top = list(work.sort_values("Base Score", ascending=False)["item"].head(top_n))

    scenario_rows = []
    ranking_rows = []

    scenarios = {"Base": base_weights}
    for key in base_weights:
        up = base_weights.copy()
        up[key] = up[key] * (1 + delta)
        scenarios[f"{key} +10%"] = up

        down = base_weights.copy()
        down[key] = down[key] * (1 - delta)
        scenarios[f"{key} -10%"] = down

    for scenario, weights in scenarios.items():
        scores = investability_scores_for_weights(work, weights)
        temp = work[["item", "category", "market_segment", "data_status"]].copy()
        temp["Score"] = scores
        top_items = list(temp.sort_values("Score", ascending=False)["item"].head(top_n))
        overlap = len(set(base_top) & set(top_items))
        scenario_rows.append({
            "Scenario": scenario,
            f"Top {top_n} overlap with base": overlap,
            "Overlap %": overlap / top_n,
        })

        for rank, item in enumerate(top_items[:5], start=1):
            ranking_rows.append({"Scenario": scenario, "Rank": rank, "Item": item})

    return pd.DataFrame(scenario_rows), pd.DataFrame(ranking_rows)


def event_group_index(keyword: str) -> pd.DataFrame:
    group = panel[panel["item"].str.contains(keyword, case=False, na=False)].copy()
    if group.empty:
        return pd.DataFrame()
    pivot = group.pivot_table(index="month", columns="item", values="median_resale", aggfunc="mean").sort_index()
    indexed = pivot.apply(lambda s: 100 * s / s.dropna().iloc[0] if s.dropna().shape[0] else s, axis=0)
    avg = indexed.mean(axis=1, skipna=True).dropna()
    return pd.DataFrame({"month": avg.index, "index_value": avg.values})


def display_item_table(df: pd.DataFrame, cols: list[str]):
    display = df[cols].copy()
    money_cols = [c for c in ["current_resale", "retail_price", "price_low", "price_high", "predicted_resale", "median_resale", "prediction_error"] if c in display.columns]
    for c in money_cols:
        display[c] = display[c].map(lambda x: "" if pd.isna(x) else format_money(x))
    if "premium_x" in display.columns:
        display["premium_x"] = display["premium_x"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}×")
    if "volatility_pct" in display.columns:
        display["volatility_pct"] = display["volatility_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}%")
    for c in ["grail_score", "risk_score", "investability_score", "liquidity_score"]:
        if c in display.columns:
            display[c] = display[c].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    if "data_status" in display.columns:
        display["data_status"] = display["data_status"].map(DATA_STATUS_LABELS).fillna(display["data_status"])
    st.dataframe(humanize_table(display), use_container_width=True, hide_index=True)


# ---------- App UI ----------

st.title("Hype Asset Index™")
st.markdown("<div class='creator-line'>Created by Zayn Remtulla</div>", unsafe_allow_html=True)
st.caption("Sneaker, streetwear, and luxury resale analytics · Alternative asset research prototype")

tabs = st.tabs(["Home", "Estimator", "Explore Items", "Market Comparison", "Research", "Data"])

with tabs[0]:
    st.subheader("What this app does")
    st.write(
        "Hype Asset Index tracks limited-edition sneakers, streetwear, and luxury resale goods as alternative assets. "
        "It compares resale baskets against S&P 500, Nike, and gold, while also estimating risk, liquidity, premium, and collector or grail effects."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Try an item**")
        st.write("Use the Estimator to paste a StockX link or manually enter visible market numbers.")
    with c2:
        st.markdown("**2. Explore growth**")
        st.write("Compare resale items from their own release windows instead of forcing every item onto one start date.")
    with c3:
        st.markdown("**3. Read the research layer**")
        st.write("Use Research for findings, correlation, sensitivity analysis, regression, event studies, and methodology.")

    verified = market[market["data_status"].eq("verified_snapshot")]
    low_volume = market[market["data_status"].eq("low_volume_snapshot")]
    seed = market[market["data_status"].str.contains("seed", case=False, na=False)]
    verified_sales = int(verified["sales_volume"].fillna(0).sum())
    market_backed_sales = int(market[market["data_status"].isin(["verified_snapshot", "low_volume_snapshot"])]["sales_volume"].fillna(0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Items tracked", f"{len(market)}")
    m2.metric("Market-backed rows", f"{len(verified) + len(low_volume)}")
    m3.metric("Seed-estimate rows", f"{len(seed)}")
    m4.metric("Market-backed sales proxy", f"{market_backed_sales:,}")

    with st.expander("Data status note"):
        st.write(
            f"The current dataset includes {len(verified)} verified snapshot rows, {len(low_volume)} low-volume market snapshot rows, "
            f"and {len(seed)} seed-estimate rows. Verified rows contain about {verified_sales:,} reported or snapshot sales. "
            "Seed estimates are prototype placeholders for testing the model structure and should be replaced with verified sold-price histories before making formal research claims. Research-facing model sections default to market-backed rows where possible."
        )

    st.subheader("Condensed research framing")
    st.write(
        "The research question is whether limited-edition resale goods show alternative-asset behavior through premium, growth, liquidity, volatility, drawdown, and low correlation with traditional assets. "
        "The current app should be read as a prototype framework, not a final empirical paper. The strongest next step is replacing seed estimates with verified item-level sold-price histories."
    )

    st.info(get_default_correlation_summary())

    st.subheader("Current market segments")
    seg = market["market_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Items"]
    fig_seg = px.bar(seg, x="Segment", y="Items", title="Tracked items by market segment")
    fig_seg.update_layout(height=340, xaxis_tickangle=-20, margin=dict(b=90))
    st.plotly_chart(fig_seg, use_container_width=True)
    chart_caption("Bar chart showing how many items fall into each market segment. This chart uses the item list, not simulated price history.")


with tabs[1]:
    st.subheader("Estimator")
    st.write(
        "Paste a StockX link if you have one, then click **Auto-fill from StockX link**. "
        "The app fills matched local data or smart starting values into the fields below. Every number can still be edited before estimating."
    )

    estimator_defaults = {
        "est_item_name": "Nike Dunk Low Panda Black White",
        "est_brand": "Nike",
        "est_category": "Sneaker",
        "est_release_year": 2021,
        "est_collab_yes": "No",
        "est_collab_name": "",
        "est_demand_driver": "None",
        "est_retail_price": 100.0,
        "est_current_resale": 130.0,
        "est_price_low": 105.0,
        "est_price_high": 170.0,
        "est_sales_band": "Very high, 1000+ recent sales",
        "est_exact_sales": 1200.0,
        "est_horizon": 12,
        "est_price_6m_ago": 0.0,
        "est_price_12m_ago": 0.0,
    }
    for key, value in estimator_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    stockx_url = st.text_input(
        "StockX product link, optional",
        value=st.session_state.get("est_stockx_url", "https://stockx.com/adidas-yeezy-slide-black-onyx"),
        key="est_stockx_url",
        help="Paste a StockX product URL. The app does not scrape live sales; it uses the link to identify the item and fill starting fields."
    )

    fill_col, status_col = st.columns([0.7, 2.3])
    with fill_col:
        fill_from_link = st.button("Auto-fill from StockX link", type="primary", use_container_width=True)
    with status_col:
        st.caption("Link assist checks for a local dataset match first. If there is no match, it fills smart starting values that the user can confirm from StockX.")

    if fill_from_link:
        defaults = stockx_import_defaults(stockx_url, market)
        st.session_state["est_item_name"] = defaults["item_name"]
        st.session_state["est_brand"] = defaults["brand"]
        st.session_state["est_category"] = defaults["category"]
        st.session_state["est_release_year"] = int(defaults["release_year"])
        st.session_state["est_collab_yes"] = defaults["collaboration_yes"]
        st.session_state["est_collab_name"] = defaults["collaboration_name"]
        st.session_state["est_demand_driver"] = defaults["demand_driver"]
        st.session_state["est_retail_price"] = float(defaults["retail_price"])
        st.session_state["est_current_resale"] = float(round(defaults["current_resale"], 0))
        st.session_state["est_price_low"] = float(round(defaults["price_low"], 0))
        st.session_state["est_price_high"] = float(round(defaults["price_high"], 0))
        st.session_state["est_sales_band"] = defaults["sales_band"]
        st.session_state["est_exact_sales"] = float(defaults["sales_volume"])
        st.session_state["est_import_status"] = defaults["match_status"]
        st.session_state["est_matched_item"] = defaults["matched_item"]
        st.rerun()

    if st.session_state.get("est_import_status"):
        if st.session_state.get("est_matched_item"):
            st.success(st.session_state["est_import_status"])
        else:
            st.info(st.session_state["est_import_status"])

    with st.form("unified_estimator_form"):
        st.markdown("### 1. Item info")
        c1, c2 = st.columns(2)
        with c1:
            item_name = st.text_input("Item name", key="est_item_name", help="Name of the sneaker, streetwear item, or luxury item being estimated.")
            brand = st.text_input("Brand", key="est_brand", help="Main brand, such as Nike, Jordan, Supreme, Chrome Hearts, or Hermès.")
            category = st.selectbox(
                "Category",
                ["Sneaker", "Streetwear", "Luxury Streetwear"],
                key="est_category",
                help="Choose the closest resale market category for the item."
            )
            release_year = st.number_input("Release year", min_value=1985, max_value=2026, step=1, key="est_release_year", help="Use the original release/drop year. If unsure, use the year shown on StockX or GOAT.")
        with c2:
            collaboration_yes = st.radio("Collaboration?", ["No", "Yes"], horizontal=True, key="est_collab_yes", help="Choose Yes if the item is a formal collaboration, such as Travis Scott, Off-White, Fragment, or Supreme x Burberry.")
            collaboration_name = st.text_input("Collaboration name, if yes", key="est_collab_name", help="Enter the collaborator name. Leave blank if there is no collaboration.")
            demand_driver = st.selectbox(
                "Main demand driver",
                ["Athlete", "Celebrity", "Designer", "Streetwear", "Luxury", "Brand Collab", "None"],
                key="est_demand_driver",
                help="Why people care about the item: athlete, celebrity, designer, streetwear brand, luxury status, brand collab, or none."
            )
            data_label = st.selectbox("Data label", ["User-verified from StockX page", "User estimate"], index=0, help="Use user-verified only if you personally checked the visible market numbers on the product page.")

        st.markdown("### 2. Confirm market numbers")
        st.caption("Use visible numbers from StockX, GOAT, eBay sold listings, or your own sale records.")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            retail_price = st.number_input("Retail price", min_value=1.0, step=10.0, key="est_retail_price", help="Original retail price at release, before resale markup.")
        with p2:
            current_resale = st.number_input("Current resale / market price", min_value=1.0, step=10.0, key="est_current_resale", help="Current market resale value. Use recent sale average or current market price.")
        with p3:
            price_low = st.number_input("Recent low sale", min_value=1.0, step=10.0, key="est_price_low", help="Lowest recent sale in the visible sales window. This helps estimate volatility.")
        with p4:
            price_high = st.number_input("Recent high sale", min_value=1.0, step=10.0, key="est_price_high", help="Highest recent sale in the visible sales window. This helps estimate volatility.")

        s1, s2, s3 = st.columns(3)
        sales_options = [
            "Very high, 1000+ recent sales",
            "High, 300-999 recent sales",
            "Medium, 75-299 recent sales",
            "Low, 15-74 recent sales",
            "Very low, under 15 recent sales",
        ]
        with s1:
            sales_band = st.selectbox("Recent sales volume band", sales_options, key="est_sales_band", help="Approximate recent sales activity. Use the exact count only if you know it.")
        with s2:
            use_exact_sales = st.checkbox("Use exact sales count", help="Turn this on only if you have an actual count from StockX, your records, or another source.")
            if use_exact_sales:
                exact_sales = st.number_input("Exact recent sales", min_value=0.0, step=10.0, key="est_exact_sales", help="Exact number of recent sales in the visible market window.")
            else:
                exact_sales = sales_band_to_volume(sales_band)
        with s3:
            horizon = st.slider("Projection horizon, months", min_value=3, max_value=24, step=3, key="est_horizon", help="How far out the illustrative projection should run.")

        h1, h2 = st.columns(2)
        with h1:
            price_6m_ago = st.number_input("Price 6 months ago, optional", min_value=0.0, step=10.0, key="est_price_6m_ago", help="Optional past resale price. This improves item-specific trend estimation.")
        with h2:
            price_12m_ago = st.number_input("Price 12 months ago, optional", min_value=0.0, step=10.0, key="est_price_12m_ago", help="Optional past resale price. Use this if 6-month history is unavailable.")

        with st.expander("Advanced override, optional"):
            use_manual_vol = st.checkbox("Manually override volatility", help="Advanced users can override auto-estimated volatility from the low/high sale range.")
            manual_volatility = st.slider("Manual volatility %", 5, 60, 18, help="Manual volatility estimate as a percent.") if use_manual_vol else None
            use_manual_scarcity = st.checkbox("Manually override scarcity", help="Advanced users can override auto-estimated scarcity.")
            manual_scarcity = st.slider("Manual scarcity score", 0, 100, 70, help="Manual scarcity score from 0 to 100.") if use_manual_scarcity else None

        submitted = st.form_submit_button("Estimate illustrative value")

    if submitted:
        sales_volume = exact_sales if use_exact_sales else sales_band_to_volume(sales_band)

        custom = custom_scores(
            market=market,
            item_name=item_name,
            category=category,
            collaboration_yes=collaboration_yes,
            collaboration_name=collaboration_name,
            demand_driver=demand_driver,
            retail_price=retail_price,
            current_resale=current_resale,
            price_low=price_low,
            price_high=price_high,
            sales_volume=sales_volume,
            release_year=int(release_year),
            manual_volatility=manual_volatility,
            manual_scarcity=manual_scarcity,
        )
        forecast, drivers = build_custom_forecast(custom, panel, horizon, price_6m_ago, price_12m_ago)
        final = forecast.iloc[-1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Premium", f"{custom['premium_x']:.1f}× retail")
        m2.metric("Segment", custom["market_segment"])
        m3.metric(f"{horizon}M illustrative base", f"${final['forecast_price']:,.0f}", f"{final['expected_annual_growth']:+.1%} annualized")
        m4.metric(f"{horizon}M illustrative range", f"${final['low_estimate']:,.0f} – ${final['high_estimate']:,.0f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Auto volatility", f"{custom['volatility_pct']:.0f}%")
        m6.metric("Auto scarcity", f"{custom['scarcity_score']:.0f}/100")
        m7.metric("Risk score", f"{custom['risk_score']:.0f}/100")
        m8.metric("Liquidity", f"{custom['liquidity_score']:.0f}/100")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[pd.Timestamp.today().normalize().replace(day=1)], y=[current_resale], mode="markers", name="Current resale", marker=dict(size=11)))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["forecast_price"], mode="lines+markers", name="Illustrative base projection", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["high_estimate"], mode="lines", line=dict(width=0), showlegend=False, name="High estimate"))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["low_estimate"], mode="lines", fill="tonexty", line=dict(width=0), name="Illustrative range"))
        fig.update_layout(height=520, yaxis_title="Estimated resale value", hovermode="x unified", legend=dict(orientation="h", y=-0.22), margin=dict(b=125))
        add_projection_annotation(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Projection chart showing current resale value and an illustrative low/base/high scenario. This is not a prediction.")

        export_row = pd.DataFrame([{
            "stockx_url": stockx_url,
            "item": item_name,
            "brand": brand,
            "category": category,
            "collaboration": custom["collaboration"],
            "connection": demand_driver,
            "retail_price": retail_price,
            "release_year": int(release_year),
            "current_resale": current_resale,
            "price_low": price_low,
            "price_high": price_high,
            "volatility_pct": custom["volatility_pct"],
            "sales_volume": sales_volume,
            "sales_volume_band": sales_band,
            "scarcity_score": custom["scarcity_score"],
            "data_status": "user_verified_stockx" if data_label == "User-verified from StockX page" else "user_estimate",
            "source": data_label,
            "source_url": stockx_url,
        }])
        st.download_button("Download this item row as CSV", export_row.to_csv(index=False).encode("utf-8"), file_name="estimator_item_row.csv", mime="text/csv")

        dshow = drivers.copy()
        dshow["Value"] = dshow["Value"].map(lambda x: f"{x:+.2%}")
        st.dataframe(humanize_table(dshow), use_container_width=True, hide_index=True)

        if price_6m_ago == 0 and price_12m_ago == 0:
            st.info("No past price was entered, so the illustrative projection uses category/market trend as a proxy. Add a 6M or 12M past price for a stronger scenario.")
        st.warning("Illustrative projection only. Confirm visible market numbers before interpreting the result.")


with tabs[2]:
    explore_tabs = st.tabs(["Item Growth", "Illustrative Projection"])

    with explore_tabs[0]:
        st.subheader("Item Growth")
        st.write("Compare resale items from their own release windows. This avoids forcing old and new items onto the same calendar start date.")

        options = sorted(panel["item"].unique())
        default_items = [
            "Air Jordan 1 High OG Lost and Found",
            "Nike Mind 001 Slide Fragment Black",
            "Travis Scott Jordan 1 Low Black Phantom",
            "Kobe 6 Protro Grinch",
        ]
        default_items = [item for item in default_items if item in options]
        selected = st.multiselect("Choose resale items to compare", options, default=default_items, help="Select two or more items to compare growth from each item's release window.")

        fig = go.Figure()
        for item in selected:
            hist = panel[panel["item"].eq(item)].sort_values("month").copy()
            if hist.empty:
                continue
            hist["growth_index"] = 100 * hist["median_resale"] / hist["median_resale"].iloc[0]
            fig.add_trace(go.Scatter(x=hist["months_since_release"], y=hist["growth_index"], mode="lines", name=item, line_shape="spline"))
        fig.update_layout(height=540, yaxis_title="Growth index, base = 100 at release", xaxis_title="Months since release", hovermode="x unified", legend=dict(orientation="h", y=-0.30, x=0), margin=dict(b=165))
        add_prototype_annotation(fig, "Prototype-estimated resale paths")
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Line chart comparing selected resale items by months since release. Resale paths are prototype-estimated until full sold-price histories are collected.")

        st.markdown("### One item vs S&P 500, Nike, and gold since release")
        primary_item = st.selectbox("Select item for benchmark comparison", selected if selected else options, index=0, help="Benchmarks start from the selected item's release window.")
        primary_hist = panel[panel["item"].eq(primary_item)].sort_values("month").copy()
        primary_hist["growth_index"] = 100 * primary_hist["median_resale"] / primary_hist["median_resale"].iloc[0]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=primary_hist["month"], y=primary_hist["growth_index"], mode="lines", name=primary_item, line=dict(width=3), line_shape="spline"))
        bench = fetch_benchmarks_for_range(primary_hist["month"].min(), primary_hist["month"].max())
        if not bench.empty:
            for asset, group in bench.groupby("asset"):
                fig2.add_trace(go.Scatter(x=group["month"], y=group["index_value"], mode="lines", name=asset, line=dict(dash="dash"), line_shape="spline"))
        else:
            st.info("Benchmark data did not load. yfinance or local internet may be blocked.")
        fig2.update_layout(height=540, yaxis_title="Growth index, base = 100 at item release", xaxis_title="", hovermode="x unified", legend=dict(orientation="h", y=-0.30, x=0), margin=dict(b=165))
        add_prototype_annotation(fig2, "Resale path is prototype-estimated; benchmarks use market data")
        st.plotly_chart(fig2, use_container_width=True)
        chart_caption("Line chart comparing one resale item against traditional benchmarks from the item's release window.")

        latest = market[market["item"].isin(selected)].copy()
        if len(latest):
            display_item_table(latest, ["item","category","market_segment","current_resale","retail_price","premium_x","volatility_pct","sales_volume","grail_score","risk_score","data_quality","data_status"])

    with explore_tabs[1]:
        st.subheader("Illustrative Projection")
        st.write("This is an illustrative projection, not a price prediction. It uses recent item movement, category trend, broad market trend, liquidity, risk, grail score, and segment profile.")

        forecast_item = st.selectbox("Select item to project", sorted(market["item"].unique()), index=0, help="Choose an item from the dataset for an illustrative low/base/high scenario.")
        horizon = st.slider("Projection horizon, months", 3, 24, 12, step=3, help="How far forward the scenario should extend.")

        forecast, drivers = build_item_forecast(forecast_item, market, panel, horizon)
        hist = panel[panel["item"].eq(forecast_item)].sort_values("month")
        row = market[market["item"].eq(forecast_item)].iloc[0]
        final = forecast.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current resale", f"${row['current_resale']:,.0f}")
        c2.metric(f"{horizon}M illustrative base", f"${final['forecast_price']:,.0f}", f"{final['expected_annual_growth']:+.1%} annualized")
        c3.metric(f"{horizon}M illustrative range", f"${final['low_estimate']:,.0f} – ${final['high_estimate']:,.0f}")
        c4.metric("Risk / liquidity", f"{row['risk_score']:.0f} / {row['liquidity_score']:.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["median_resale"], mode="lines+markers", name="Prototype history"))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["forecast_price"], mode="lines+markers", name="Illustrative base projection", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["high_estimate"], mode="lines", name="High estimate", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["low_estimate"], mode="lines", name="Illustrative range", fill="tonexty", line=dict(width=0)))
        fig.update_layout(height=560, yaxis_title="Resale value", xaxis_title="", hovermode="x unified", legend=dict(orientation="h", y=-0.22), margin=dict(b=125))
        add_projection_annotation(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Chart showing prototype history and an illustrative future scenario. It is not a guaranteed prediction.")

        drivers_df = pd.DataFrame([{"Driver": k, "Monthly Impact": v, "Annualized Equivalent": (1 + v) ** 12 - 1} for k, v in drivers.items()])
        show_drivers = drivers_df.copy()
        show_drivers["Monthly Impact"] = show_drivers["Monthly Impact"].map(lambda x: f"{x:+.2%}")
        show_drivers["Annualized Equivalent"] = show_drivers["Annualized Equivalent"].map(lambda x: f"{x:+.1%}")
        st.dataframe(show_drivers, use_container_width=True, hide_index=True)


with tabs[3]:
    market_tabs = st.tabs(["Category Baskets", "Benchmarks", "Correlation"])

    with market_tabs[0]:
        st.subheader("Category Baskets")
        st.write("Compare sneaker, streetwear, and luxury resale baskets. The maturity filter prevents brand-new drops from distorting the whole category.")

        c1, c2 = st.columns(2)
        with c1:
            data_filter = st.selectbox("Data basis", ["Market-backed", "All items"], index=0, help="Market-backed keeps verified and low-volume market snapshots. All items also includes seed estimates.")
        with c2:
            maturity_filter = st.selectbox("Maturity filter", ["Established items only, 6+ months", "All release windows"], index=0, help="Established items only is the cleaner default for baskets.")
        min_history_months = 6 if maturity_filter.startswith("Established") else 0

        categories = ["Sneaker", "Streetwear", "Luxury Streetwear"]
        fig = go.Figure()
        summary_rows = []
        for cat in categories:
            idx = make_index(panel, cat, data_filter, min_history_months=min_history_months)
            filtered_items = panel[panel["category"].eq(cat)].copy()
            if data_filter == "Market-backed":
                filtered_items = filtered_items[filtered_items["data_status"].isin(["verified_snapshot", "low_volume_snapshot"])]
            if min_history_months > 0:
                counts = filtered_items.groupby("item")["month"].nunique()
                filtered_items = filtered_items[filtered_items["item"].isin(counts[counts >= min_history_months].index)]
            n_items = filtered_items["item"].nunique()
            if idx.empty or n_items < 2:
                summary_rows.append({"Category": category_label(cat), "Items": n_items, "Status": "Not enough items"})
                continue
            total_return = idx["index_value"].iloc[-1] / idx["index_value"].iloc[0] - 1
            vol = annualized_volatility(idx["monthly_return"])
            dd = max_drawdown(idx["index_value"])
            summary_rows.append({"Category": category_label(cat), "Items": n_items, "Return": total_return, "Volatility": vol, "Max Drawdown": dd, "Status": "OK"})
            fig.add_trace(go.Scatter(x=idx["month"], y=idx["index_value"], mode="lines", name=category_label(cat), line_shape="spline"))

        fig.update_layout(height=540, yaxis_title="Category basket index, base = 100", xaxis_title="", hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=130))
        add_prototype_annotation(fig, "Basket uses prototype-estimated resale histories")
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Line chart comparing category basket indexes. Resale histories are prototype-estimated until full transaction data is collected.")

        summary = pd.DataFrame(summary_rows)
        for col in ["Return","Volatility","Max Drawdown"]:
            if col in summary.columns:
                summary[col] = summary[col].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}" if col=="Return" else f"{x:.1%}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with market_tabs[1]:
        st.subheader("Benchmark Comparison")
        st.write("Compare a resale basket against S&P 500, Nike, and gold over the same visible period.")

        c1, c2, c3, c4 = st.columns([0.8, 1.1, 1.1, 1.2])
        with c1:
            period = st.selectbox("Benchmark period", ["6mo","1y","2y"], index=1, help="Time window for the comparison.")
        with c2:
            resale_choice = st.selectbox("Resale basket", ["All Resale", "Sneakers", "Streetwear", "Luxury Streetwear"], index=0, help="Choose which resale basket to compare.")
        with c3:
            benchmark_data_filter = st.selectbox("Data basis", ["Market-backed", "All items"], index=0, help="Market-backed excludes seed estimates where possible.")
        with c4:
            benchmark_maturity_filter = st.selectbox("Maturity", ["Established 6M+", "All drops"], index=0, help="Established 6M+ prevents brand-new drops from distorting the basket.")
        show_assets = st.multiselect("Traditional assets", ["S&P 500", "Nike", "Gold ETF"], default=["S&P 500", "Nike", "Gold ETF"], help="Choose which traditional benchmarks to show.")

        category_map = {"All Resale": "All", "Sneakers": "Sneaker", "Streetwear": "Streetwear", "Luxury Streetwear": "Luxury Streetwear"}
        category = category_map[resale_choice]
        benchmark_min_history = 6 if benchmark_maturity_filter.startswith("Established") else 0
        resale_raw = make_index(panel, category, benchmark_data_filter, min_history_months=benchmark_min_history)
        resale_period = normalize_period_index(resale_raw, "index_value", period)
        resale_smooth = smooth_monthly_index(resale_period, "index_value", f"{resale_choice} Basket")

        fig = go.Figure()
        if not resale_smooth.empty:
            fig.add_trace(go.Scatter(x=resale_smooth["month"], y=resale_smooth["index_value"], mode="lines", name=f"{resale_choice} Basket", line_shape="spline", line=dict(width=3)))
        else:
            st.info(f"Not enough data for {resale_choice} basket.")

        bench = fetch_benchmarks_for_range(resale_period["month"].min(), resale_period["month"].max()) if not resale_period.empty else pd.DataFrame()
        if not bench.empty and show_assets:
            bench = bench[bench["asset"].isin(show_assets)]
            for asset, group in bench.groupby("asset"):
                fig.add_trace(go.Scatter(x=group["month"], y=group["index_value"], mode="lines", name=asset, line=dict(dash="dash"), line_shape="spline"))
        elif show_assets:
            st.info("Benchmark data did not load. yfinance or local internet may be blocked.")

        fig.update_layout(height=560, yaxis_title="Index value, base = 100", xaxis_title="", hovermode="x unified", legend=dict(orientation="h", y=-0.28, x=0), margin=dict(b=155, r=40, l=60, t=40))
        add_prototype_annotation(fig, "Resale basket uses prototype-estimated histories; benchmarks use market data")
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Benchmark chart comparing resale baskets with traditional assets. Resale basket history is prototype-estimated.")

        if not resale_period.empty:
            basket_return = resale_period["index_value"].iloc[-1] / resale_period["index_value"].iloc[0] - 1
            basket_vol = annualized_volatility(resale_period["monthly_return"]) if "monthly_return" in resale_period.columns else 0
            basket_dd = max_drawdown(resale_period["index_value"])
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{resale_choice} return", f"{basket_return:+.1%}")
            m2.metric("Volatility", f"{basket_vol:.1%}")
            m3.metric("Max drawdown", f"{basket_dd:.1%}")

    with market_tabs[2]:
        st.subheader("Correlation Matrix")
        st.warning(
            "Prototype correlation only: the resale basket is based on prototype-estimated histories, while S&P 500, Nike, and gold use real market data. "
            "Use this as a structure for future analysis, not as current evidence."
        )
        st.write("Monthly return correlations test whether resale baskets move differently from traditional assets once verified histories are available.")

        c1, c2, c3 = st.columns(3)
        with c1:
            corr_period = st.selectbox("Correlation period", ["6mo", "1y", "2y"], index=1, help="Time window for overlapping monthly returns.")
        with c2:
            corr_basket = st.selectbox("Resale basket", ["All Resale", "Sneakers", "Streetwear", "Luxury Streetwear"], index=0, key="corr_basket", help="Choose which resale basket to test.")
        with c3:
            corr_basis = st.selectbox("Data basis", ["Market-backed", "All items"], index=0, key="corr_basis", help="Market-backed excludes seed estimates where possible.")

        corr_category = {"All Resale": "All", "Sneakers": "Sneaker", "Streetwear": "Streetwear", "Luxury Streetwear": "Luxury Streetwear"}[corr_basket]
        corr_resale_raw = make_index(panel, corr_category, corr_basis, min_history_months=6)
        corr_resale_period = normalize_period_index(corr_resale_raw, "index_value", corr_period)

        if corr_resale_period.empty:
            st.info("Not enough resale data for this basket and period.")
        else:
            corr_bench = fetch_benchmarks_for_range(corr_resale_period["month"].min(), corr_resale_period["month"].max())
            wide_index = aligned_benchmark_wide(corr_resale_period, corr_bench)
            corr = monthly_return_correlation(wide_index)

            if corr.empty:
                st.info("Not enough overlapping monthly returns to calculate a reliable correlation matrix.")
            else:
                fig_corr = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto", title="Prototype monthly return correlation matrix")
                fig_corr.add_annotation(
                    text="Prototype resale index vs real market benchmarks",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1.12,
                    showarrow=False,
                    align="right",
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.10)",
                    bordercolor="rgba(128,128,128,0.45)",
                    borderwidth=1,
                )
                fig_corr.update_layout(height=520, margin=dict(b=80, t=95))
                st.plotly_chart(fig_corr, use_container_width=True)
                chart_caption("Heatmap of monthly return correlations. Values near zero suggest lower co-movement; values near one suggest stronger co-movement.")

                st.dataframe(corr.round(3), use_container_width=True)
                if "Resale Basket" in corr.columns:
                    comparison = corr["Resale Basket"].drop("Resale Basket", errors="ignore").sort_values()
                    if len(comparison):
                        st.write("Resale basket correlation with traditional assets:")
                        st.dataframe(comparison.rename("Correlation").round(3), use_container_width=True)

        st.caption("Correlation uses monthly percentage returns over the selected overlapping period. Recalculate after replacing seed estimates with verified histories.")


with tabs[4]:
    research_tabs = st.tabs(["Findings", "Sensitivity", "Event Studies", "Regression", "Risk", "Methodology"])

    with research_tabs[0]:
        st.subheader("Findings and Current Claim")
        st.write(
            "Current claim: the app supports a prototype framework for testing resale goods as alternative assets, but it does not yet prove the claim empirically. "
            "The strongest evidence to watch is low correlation with traditional assets, stable segment-level behavior, and robustness of item rankings under different scoring weights."
        )
        st.markdown("""
**What the current prototype supports**
- Resale goods should not be treated as one single market. Liquid sneakers, grails, streetwear, and luxury items behave differently.
- Fixed item effects and market segmentation are necessary because grail items have persistent collector premiums.
- Correlation with equities, Nike, and gold is the cleanest test for alternative-asset diversification value.

**What it does not yet prove**
- It does not yet prove long-run resale returns because many histories are prototype-estimated.
- It does not yet prove investability rankings because the score weights are proposed assumptions.
- It does not yet prove live market accuracy because StockX link assist is not an official API feed.

**Most important limitation**
Many price histories are prototype-estimated. The framework is research-ready, but the empirical claim is not complete until verified sold-price time series replace the simulated paths.

**Next research step**
Collect 20-30 real item-level sold-price time series, document transaction counts behind each item, then rerun index, correlation, sensitivity, and holdout validation.

**How to frame this to a professor**
This is a prototype research framework and dashboard. The contribution is the structure for index construction, segmentation, scoring, sensitivity analysis, and data collection. It is not yet an empirical paper because verified transaction histories are still needed.
""")

    with research_tabs[1]:
        st.subheader("Sensitivity Analysis")
        st.write("This tests whether top investability rankings stay stable when the scoring weights change by ±10%.")
        data_basis = st.selectbox("Sensitivity data basis", ["Market-backed", "All items"], index=0, help="Market-backed excludes seed estimates where possible.")
        sens_df = market.copy()
        if data_basis == "Market-backed":
            sens_df = sens_df[sens_df["data_status"].isin(["verified_snapshot", "low_volume_snapshot"])]
        scenario_table, ranking_table = sensitivity_analysis(sens_df, top_n=10, delta=0.10)

        show = scenario_table.copy()
        show["Overlap %"] = show["Overlap %"].map(lambda x: f"{x:.0%}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.write("Top-five items by scenario:")
        st.dataframe(humanize_table(ranking_table), use_container_width=True, hide_index=True)
        st.caption("If top-item overlap remains high, the scoring framework is more stable. If rankings flip heavily, that becomes a research finding rather than a weakness.")

    with research_tabs[2]:
        st.subheader("Event Study Design")
        st.warning(
            "Event-study charts are intentionally not plotted from prototype histories. "
            "A real event study should only be shown after verified sold-price histories are collected."
        )
        st.write(
            "This section defines how the event study would be run once real transaction histories are available. "
            "It avoids presenting simulated paths as evidence."
        )

        event_plan = pd.DataFrame([
            {
                "Event": "Yeezy / Adidas split",
                "Event Date": "2022-10",
                "Treatment Group": "Yeezy models",
                "Comparison Group": "Non-Yeezy Adidas/Nike resale items",
                "Outcome": "Monthly resale return and abnormal return",
                "Required Data": "Item-level sold-price history before and after event"
            },
            {
                "Event": "Travis Scott demand shock",
                "Event Date": "2021-11",
                "Treatment Group": "Travis Scott collaboration items",
                "Comparison Group": "Comparable Jordan/Nike collaborations",
                "Outcome": "Monthly resale return and liquidity change",
                "Required Data": "Verified sold-price and sales-count history"
            },
        ])
        st.dataframe(event_plan, use_container_width=True, hide_index=True)

        st.markdown("""
**Planned method**
1. Collect verified monthly resale prices and transaction counts for treatment and comparison items.
2. Index each item to 100 before the event window.
3. Compare post-event abnormal returns against the comparison group.
4. Report confidence intervals and transaction counts.
5. Treat the result as descriptive unless the comparison group and event window are strong enough for causal claims.
""")


    with research_tabs[3]:
        st.subheader("Regression Model")
        st.write(
            "Regression is used to control for item mix and separate persistent item premiums from broader market movement. "
            "This section defaults to market-backed rows, but the histories are still prototype-estimated until full transaction time series are collected."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("In-sample MAE", f"${mae:,.0f}")
        c2.metric("In-sample log R²", f"{r2:.2f}")
        c3.metric("Prototype holdout MAE", "n/a" if pd.isna(holdout_report["holdout_mae"]) else f"${holdout_report['holdout_mae']:,.0f}")
        c4.metric("Prototype holdout rows", f"{holdout_report['test_rows']:,}")

        st.caption(
            "The prototype holdout uses the final three months of the generated panel as a model-discipline check, not as true out-of-sample validation. "
            "This improves model discipline, but it is still not empirical validation because the underlying histories are prototype-estimated."
        )

        sample = fitted.sample(min(800, len(fitted)), random_state=5)
        fig = px.scatter(
            sample,
            x="median_resale",
            y="predicted_resale",
            color="market_segment",
            hover_name="item",
            hover_data=["month_label","brand","grail_score","premium_x","sale_count","data_status"],
            title="Observed/estimated resale vs model estimate"
        )
        fig.add_trace(go.Scatter(
            x=[sample["median_resale"].min(), sample["median_resale"].max()],
            y=[sample["median_resale"].min(), sample["median_resale"].max()],
            mode="lines",
            name="Perfect fit"
        ))
        fig.update_layout(height=560, xaxis_title="Observed / estimated resale", yaxis_title="Model estimate")
        add_prototype_annotation(fig, "Regression uses market-backed rows but prototype-estimated histories")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Largest coefficient magnitudes:")
        coeffs = coefs.head(25)[["feature","coefficient"]].copy()
        coeffs["coefficient"] = coeffs["coefficient"].map(lambda x: f"{x:+.3f}")
        st.dataframe(humanize_table(coeffs), use_container_width=True, hide_index=True)


    with research_tabs[4]:
        st.subheader("Risk Framework")
        st.write("Risk combines volatility, drawdown, and liquidity. Low-volume grails may have high premiums but still be risky because the market is thin.")

        fig = px.scatter(
            market,
            x="liquidity_score",
            y="volatility_pct",
            size="current_resale",
            color="market_segment",
            hover_name="item",
            hover_data=["current_resale","grail_score","risk_score","investability_score","data_status"],
            title="Liquidity vs volatility"
        )
        fig.update_layout(height=560, xaxis_title="Liquidity score", yaxis_title="Volatility %")
        st.plotly_chart(fig, use_container_width=True)
        chart_caption("Scatter plot comparing liquidity and volatility. Marker size reflects current resale value.")

        risk_table = market[["item","category","market_segment","current_resale","premium_x","volatility_pct","sales_volume","grail_score","risk_score","investability_score","data_quality","data_status"]].sort_values("risk_score")
        display_item_table(risk_table, ["item","category","market_segment","current_resale","premium_x","volatility_pct","sales_volume","grail_score","risk_score","investability_score","data_quality","data_status"])

    with research_tabs[5]:
        st.subheader("Methodology")
        st.markdown("""
### Research purpose
Hype Asset Index is a prototype framework for testing whether sneakers, streetwear, and luxury resale goods behave like alternative assets. The core variables are resale premium, price growth, liquidity, volatility, drawdown, risk, and correlation with traditional assets.

### Data provenance
The dataset separates observations into verified snapshots, low-volume market snapshots, and seed estimates. Seed estimates are placeholders used to test the dashboard and model structure. They should not be interpreted as verified transaction data. Prototype histories are generated with a fixed random seed for reproducibility, and monthly sale counts are currently proxies derived from available sales-volume fields.

### Index construction
Category and market baskets are built from monthly percentage changes in resale values. Baskets default to established items with at least six months of history so brand-new drops do not distort the entire category.

### Regression model
The regression uses item fixed effects, category, brand, collaboration, market segment, grail score, scarcity, liquidity, and month effects. Item fixed effects are used because grail items have persistent collector premiums that should not be forced onto the same curve as ordinary high-volume resale items. The research view reports both in-sample fit and a simple chronological holdout check, but this should still be treated as a model-discipline check rather than empirical validation until real histories are used. Because prototype histories are partly shaped by score inputs, regression coefficients should not be interpreted as findings yet.

### Scoring framework
The investability score is a proposed framework, not an empirically optimized score. Current weights are 40% liquidity, 25% premium over retail, 20% inverse risk, and 15% scarcity. Liquidity is scored on an absolute 1000-sale anchor so scores remain stable as more items are added. The Sensitivity tab tests how stable rankings are when these assumptions change.

### Projection framework
The projection tool is illustrative only. It combines recent item trend, category trend, market trend, liquidity, risk, grail score, segment, and mean reversion. It should be read as a scenario model, not a price prediction. Projection ranges widen when the underlying row is lower-quality or seed-estimated. The projection weights are transparent assumptions, not optimized parameters.

### Limitations
The main limitations are incomplete historical transaction data, seed estimates, low-volume luxury markets, and no official live API access. The most important limitation is the closed prototype loop: simulated histories can make regression, correlation, and event-study outputs look more empirical than they are. The strongest next step is replacing seed rows and prototype histories with verified sold-price histories and documenting the number of transactions behind each item.
""")


with tabs[5]:
    st.subheader("Data")
    st.write("Data provenance summary:")
    status_summary = market.groupby("data_status", dropna=False).agg(
        Items=("item", "count"),
        Sales_Volume=("sales_volume", "sum"),
        Avg_Current_Resale=("current_resale", "mean")
    ).reset_index().rename(columns={"data_status": "Data Source"})
    status_summary["Data Source"] = status_summary["Data Source"].map(DATA_STATUS_LABELS).fillna(status_summary["Data Source"])
    status_summary["Avg_Current_Resale"] = status_summary["Avg_Current_Resale"].map(lambda x: f"${x:,.0f}")
    status_summary = status_summary.rename(columns={"Sales_Volume": "Sales Volume", "Avg_Current_Resale": "Average Current Resale"})
    st.dataframe(status_summary, use_container_width=True, hide_index=True)

    st.write("Current market seed:")
    display_seed = market.copy()
    display_item_table(display_seed, ["item","brand","category","collaboration","connection","release_year","retail_price","current_resale","price_low","price_high","volatility_pct","sales_volume","scarcity_score","data_status","source"])

    st.download_button("Download market seed", market.to_csv(index=False).encode("utf-8"), file_name="hype_asset_market_seed.csv", mime="text/csv")
    st.download_button("Download history panel", panel.to_csv(index=False).encode("utf-8"), file_name="hype_asset_history_panel.csv", mime="text/csv")
