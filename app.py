
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
    padding-top: 1.4rem;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #16a34a, #22c55e) !important;
    border: 1px solid #22c55e !important;
    color: #06140a !important;
    font-weight: 800 !important;
    box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.25), 0 8px 20px rgba(34, 197, 94, 0.18);
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #15803d, #16a34a) !important;
    border-color: #86efac !important;
    color: white !important;
}
div[data-testid="stMetricValue"] {
    letter-spacing: -0.03em;
}

.creator-name {
    font-size: 2.15rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin-top: -0.35rem;
    margin-bottom: 0.15rem;
    line-height: 1.05;
}
@media (max-width: 900px) {
    .creator-name {
        font-size: 1.65rem;
    }
}

</style>
""", unsafe_allow_html=True)


DATA_DIR = Path(__file__).resolve().parent / "data"
MARKET_PATH = DATA_DIR / "hype_asset_market_seed.csv"


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
    max_sales = max(out["sales_volume"].fillna(0).max(), 1)
    out["liquidity_score"] = 100 * np.log1p(out["sales_volume"].fillna(0)) / np.log1p(max_sales)

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

    grail_keywords = out["item"].str.contains(
        "Off-White|Yeezy 2|Red October|Solar Red|Chicago|Chunky Dunky|Travis Scott Jordan 1 High",
        case=False, regex=True, na=False
    ).astype(int) * 18

    out["grail_score"] = (
        0.30 * premium_score
        + 0.25 * scarcity
        + 0.20 * age_score
        + 0.15 * low_volume_score
        + collab_boost
        + grail_keywords
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
        0.42 * item_trend
        + 0.28 * cat_trend
        + 0.18 * mkt_trend
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
            "Est. annual growth": float(final["expected_annual_growth"]),
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
    max_sales = max(float(market["sales_volume"].max()), 1.0)
    liquidity_score = 100 * math.log1p(sales_volume) / math.log1p(max_sales)

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
    grail_boost = 18 if any(w.lower() in item_name.lower() for w in grail_words) else 0

    grail_score = (
        0.30 * premium_score
        + 0.25 * scarcity_score
        + 0.20 * age_score
        + 0.15 * low_volume_score
        + collab_boost
        + grail_boost
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
        0.42 * item_trend
        + 0.25 * cat_trend
        + 0.18 * mkt_trend
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
fitted, coefs, adjusted_index, mae, r2 = fit_regression(panel)

st.title("Hype Asset Index™")
st.markdown("<div class='creator-name'>Created by Zayn Remtulla</div>", unsafe_allow_html=True)
st.caption("Sneaker, streetwear, and luxury resale analytics")
tabs = st.tabs(["Overview", "Item Growth", "Growth Forecast", "Estimator", "Category Baskets", "Market Basket", "Regression", "Risk", "Benchmarks", "Methodology", "Data"])

with tabs[0]:
    verified = market[market["data_status"].eq("verified_snapshot")]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Items tracked", f"{len(market)}")
    c2.metric("Verified snapshot rows", f"{len(verified)}")
    c3.metric("Avg current premium", f"{market['premium_x'].mean():.1f}x")
    c4.metric("3M sales in verified rows", f"{int(verified['sales_volume'].sum()):,}")

    st.subheader("Best current resale assets by investability score")
    top = market.sort_values("investability_score", ascending=False).head(12)
    fig = px.bar(
        top, x="investability_score", y="item", color="data_quality", orientation="h",
        hover_data=["current_resale","premium_x","sales_volume","volatility_pct","risk_score","grail_score","market_segment","data_status"],
        labels={"investability_score":"Investability score", "item":""},
    )
    fig.update_layout(height=520, yaxis={"categoryorder":"total ascending"}, legend=dict(orientation="h", y=-0.18), margin=dict(b=90))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Market segments")
    seg = market["market_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Items"]
    fig_seg = px.bar(seg, x="Segment", y="Items", title="Tracked items by market segment")
    fig_seg.update_layout(height=330, xaxis_tickangle=-20, margin=dict(b=90))
    st.plotly_chart(fig_seg, use_container_width=True)

    st.write("The cleanest research question is whether individual items and category baskets show asset-like behavior: premium, growth, liquidity, volatility, and drawdown.")

with tabs[1]:
    st.subheader("Individual Item Growth")
    st.write(
        "Each item now starts from its own release window instead of every line starting on the same Dec 2024 date. "
        "The first chart compares items by months since release. The second chart compares one selected item to stocks/gold since that item's release."
    )

    options = sorted(panel["item"].unique())
    default_items = [
        "Air Jordan 1 High OG Lost and Found",
        "Nike Mind 001 Slide Fragment Black",
        "Travis Scott Jordan 1 Low Black Phantom",
        "Kobe 6 Protro Grinch",
    ]
    default_items = [item for item in default_items if item in options]
    selected = st.multiselect("Choose resale items to compare", options, default=default_items)

    st.markdown("### Resale items since their own release")
    fig = go.Figure()
    for item in selected:
        hist = panel[panel["item"].eq(item)].sort_values("month").copy()
        if hist.empty:
            continue
        hist["growth_index"] = 100 * hist["median_resale"] / hist["median_resale"].iloc[0]
        fig.add_trace(go.Scatter(
            x=hist["months_since_release"],
            y=hist["growth_index"],
            mode="lines",
            name=item,
            hovertemplate="Months since release=%{x}<br>Growth index=%{y:.1f}<extra></extra>",
            line_shape="spline",
        ))

    fig.update_layout(
        height=540,
        yaxis_title="Growth index, base = 100 at release",
        xaxis_title="Months since release",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.30, x=0),
        margin=dict(b=165),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### One item vs S&P 500, Nike, and gold since release")
    if selected:
        primary_item = st.selectbox("Select item for stock/gold benchmark comparison", selected, index=0)
    else:
        primary_item = st.selectbox("Select item for stock/gold benchmark comparison", options, index=0)

    primary_hist = panel[panel["item"].eq(primary_item)].sort_values("month").copy()
    primary_hist["growth_index"] = 100 * primary_hist["median_resale"] / primary_hist["median_resale"].iloc[0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=primary_hist["month"],
        y=primary_hist["growth_index"],
        mode="lines",
        name=f"{primary_item}",
        line=dict(width=3),
        line_shape="spline",
    ))

    bench = fetch_benchmarks_for_range(primary_hist["month"].min(), primary_hist["month"].max())
    if not bench.empty:
        for asset, group in bench.groupby("asset"):
            fig2.add_trace(go.Scatter(
                x=group["month"],
                y=group["index_value"],
                mode="lines",
                name=asset,
                line=dict(dash="dash"),
                line_shape="spline",
            ))
    else:
        st.warning("Benchmark data did not load. yfinance or local internet may be blocked.")

    fig2.update_layout(
        height=540,
        yaxis_title="Growth index, base = 100 at item release",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.30, x=0),
        margin=dict(b=165),
    )
    st.plotly_chart(fig2, use_container_width=True)

    latest = market[market["item"].isin(selected)].copy()
    if len(latest):
        display = latest[["item","category","market_segment","current_resale","retail_price","premium_x","volatility_pct","sales_volume","grail_score","risk_score","data_quality","data_status"]].copy()
        display["current_resale"] = display["current_resale"].map(lambda x: f"${x:,.0f}")
        display["retail_price"] = display["retail_price"].map(lambda x: f"${x:,.0f}")
        display["premium_x"] = display["premium_x"].map(lambda x: f"{x:.1f}x")
        display["volatility_pct"] = display["volatility_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}%")
        display["grail_score"] = display["grail_score"].map(lambda x: f"{x:.0f}")
        display["risk_score"] = display["risk_score"].map(lambda x: f"{x:.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Historical resale paths are prototype-estimated until full sold-price history is collected. Each item begins at its own release window, and stock/gold benchmarks start from the selected item's release date. If a benchmark source returns only a tiny partial sliver, the app now uses a fallback ticker or hides that bad partial line.")


with tabs[2]:
    st.subheader("Growth Forecast")
    st.write(
        "This estimates future resale value using recent item movement, category trend, broad resale trend, liquidity, risk, grail score, and market segment. "
        "Treat it as a scenario forecast, not a guaranteed price prediction."
    )

    forecast_item = st.selectbox("Select item to forecast", sorted(market["item"].unique()), index=sorted(market["item"].unique()).index("Air Jordan 1 High OG Lost and Found") if "Air Jordan 1 High OG Lost and Found" in sorted(market["item"].unique()) else 0)
    horizon = st.slider("Forecast horizon, months", 3, 24, 12, step=3)

    forecast, drivers = build_item_forecast(forecast_item, market, panel, horizon)
    hist = panel[panel["item"].eq(forecast_item)].sort_values("month")
    row = market[market["item"].eq(forecast_item)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    final = forecast.iloc[-1]
    c1.metric("Current resale", f"${row['current_resale']:,.0f}")
    c2.metric(f"{horizon}M base estimate", f"${final['forecast_price']:,.0f}", f"{final['expected_annual_growth']:+.1%} annualized")
    c3.metric(f"{horizon}M low/high", f"${final['low_estimate']:,.0f} – ${final['high_estimate']:,.0f}")
    c4.metric("Risk / liquidity", f"{row['risk_score']:.0f} / {row['liquidity_score']:.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["month"], y=hist["median_resale"],
        mode="lines+markers", name="Historical/prototype price"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["month"], y=forecast["forecast_price"],
        mode="lines+markers", name="Base forecast", line=dict(dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=forecast["month"], y=forecast["high_estimate"],
        mode="lines", name="High estimate", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast["month"], y=forecast["low_estimate"],
        mode="lines", name="Forecast range",
        fill="tonexty", line=dict(width=0)
    ))
    fig.update_layout(
        height=560,
        yaxis_title="Resale value",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(b=125),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast drivers")
    drivers_df = pd.DataFrame([
        {"Driver": k, "Monthly impact": v, "Annualized equivalent": (1 + v) ** 12 - 1}
        for k, v in drivers.items()
    ])
    show_drivers = drivers_df.copy()
    show_drivers["Monthly impact"] = show_drivers["Monthly impact"].map(lambda x: f"{x:+.2%}")
    show_drivers["Annualized equivalent"] = show_drivers["Annualized equivalent"].map(lambda x: f"{x:+.1%}")
    st.dataframe(show_drivers, use_container_width=True, hide_index=True)

    st.subheader("Forecast watchlist")
    default_watch = [
        "Air Jordan 1 High OG Lost and Found",
        "Travis Scott Jordan 1 Low Black Phantom",
        "Off-White Jordan 1 Chicago",
        "Kobe 6 Protro Grinch",
        "Jordan 4 SB Pine Green",
        "Yeezy 350 V2 Zebra",
    ]
    default_watch = [x for x in default_watch if x in set(market["item"])]
    watch_items = st.multiselect("Choose forecast watchlist", sorted(market["item"].unique()), default=default_watch)
    summary = forecast_summary_table(watch_items, market, panel, horizon)
    if len(summary):
        display = summary.copy()
        money_cols = ["Current resale", f"{horizon}M base estimate", f"{horizon}M low", f"{horizon}M high"]
        for col in money_cols:
            display[col] = display[col].map(lambda x: f"${x:,.0f}")
        display["Est. annual growth"] = display["Est. annual growth"].map(lambda x: f"{x:+.1%}")
        for col in ["Risk", "Liquidity", "Grail"]:
            display[col] = display[col].map(lambda x: f"{x:.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)




with tabs[3]:
    st.subheader("Estimator")
    st.info("Paste a StockX link to auto-fill matched or smart default fields, then confirm the visible market numbers before estimating future resale value.")
    st.write(
        "Use this as one combined estimator. Paste a StockX link if you have one, then click **Auto-fill from StockX link**. "
        "The app will move matched local data or smarter defaults into the fields below. If no link is available, just type the item manually."
    )

    # Initialize estimator field defaults one time.
    estimator_defaults = {
        "est_item_name": "Jordan 1 Retro High Off-White University Blue",
        "est_brand": "Jordan",
        "est_category": "Sneaker",
        "est_release_year": 2023,
        "est_collab_yes": "Yes",
        "est_collab_name": "Off-White",
        "est_demand_driver": "Designer",
        "est_retail_price": 190.0,
        "est_current_resale": 450.0,
        "est_price_low": 420.0,
        "est_price_high": 500.0,
        "est_sales_band": "Medium, 75-299 recent sales",
        "est_exact_sales": 150.0,
        "est_horizon": 12,
        "est_price_6m_ago": 0.0,
        "est_price_12m_ago": 0.0,
    }
    for key, value in estimator_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown("### Optional StockX link assist")
    stockx_url = st.text_input(
        "StockX product link, optional",
        value=st.session_state.get("est_stockx_url", "https://stockx.com/adidas-yeezy-slide-black-onyx"),
        key="est_stockx_url"
    )

    fill_col, status_col = st.columns([0.6, 2.4])
    with fill_col:
        fill_from_link = st.button("Auto-fill from StockX link", type="primary", use_container_width=True)
    with status_col:
        st.caption("The link assist checks for a local dataset match first. If there is no match, it fills smart starting values that the user can confirm from StockX.")

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
            st.warning(st.session_state["est_import_status"])

    with st.form("unified_estimator_form"):
        st.markdown("### 1. Item info")
        c1, c2 = st.columns(2)
        with c1:
            item_name = st.text_input("Item name", key="est_item_name")
            brand = st.text_input("Brand", key="est_brand")
            category = st.selectbox(
                "Category",
                ["Sneaker", "Streetwear", "Luxury Streetwear"],
                key="est_category"
            )
            release_year = st.number_input("Release year", min_value=1985, max_value=2026, step=1, key="est_release_year")
        with c2:
            collaboration_yes = st.radio("Collaboration?", ["No", "Yes"], horizontal=True, key="est_collab_yes")
            collaboration_name = st.text_input("Collaboration name, if yes", key="est_collab_name")
            demand_driver = st.selectbox(
                "Main demand driver",
                ["Athlete", "Celebrity", "Designer", "Streetwear", "Luxury", "Brand Collab", "None"],
                key="est_demand_driver",
                help="Why people care about the item: athlete, celebrity, designer, streetwear, luxury, brand collab, or none."
            )
            data_label = st.selectbox("Data label", ["User-verified from StockX page", "User estimate"], index=0)

        st.markdown("### 2. Confirm market numbers")
        st.caption("If the link found a local match, these fields are filled from the app's dataset. If not, they are smart guesses that the user should confirm from StockX/GOAT.")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            retail_price = st.number_input("Retail price", min_value=1.0, step=10.0, key="est_retail_price")
        with p2:
            current_resale = st.number_input("Current resale / market price", min_value=1.0, step=10.0, key="est_current_resale")
        with p3:
            price_low = st.number_input("Recent low sale", min_value=1.0, step=10.0, key="est_price_low")
        with p4:
            price_high = st.number_input("Recent high sale", min_value=1.0, step=10.0, key="est_price_high")

        s1, s2, s3 = st.columns(3)
        sales_options = [
            "Very high, 1000+ recent sales",
            "High, 300-999 recent sales",
            "Medium, 75-299 recent sales",
            "Low, 15-74 recent sales",
            "Very low, under 15 recent sales",
        ]
        with s1:
            sales_band = st.selectbox("Recent sales volume band", sales_options, key="est_sales_band")
        with s2:
            use_exact_sales = st.checkbox("Use exact sales count")
            if use_exact_sales:
                exact_sales = st.number_input("Exact recent sales", min_value=0.0, step=10.0, key="est_exact_sales")
            else:
                exact_sales = sales_band_to_volume(sales_band)
        with s3:
            horizon = st.slider("Forecast horizon, months", min_value=3, max_value=24, step=3, key="est_horizon")

        st.caption("Optional but recommended: add one past price if StockX shows enough history or if you know it from your own records.")
        h1, h2 = st.columns(2)
        with h1:
            price_6m_ago = st.number_input("Price 6 months ago, optional", min_value=0.0, step=10.0, key="est_price_6m_ago")
        with h2:
            price_12m_ago = st.number_input("Price 12 months ago, optional", min_value=0.0, step=10.0, key="est_price_12m_ago")

        with st.expander("Advanced override, optional"):
            st.caption("Most users should leave these blank. The app estimates volatility and scarcity automatically.")
            use_manual_vol = st.checkbox("Manually override volatility")
            manual_volatility = st.slider("Manual volatility %", 5, 60, 18) if use_manual_vol else None
            use_manual_scarcity = st.checkbox("Manually override scarcity")
            manual_scarcity = st.slider("Manual scarcity score", 0, 100, 70) if use_manual_scarcity else None

        submitted = st.form_submit_button("Estimate future value")

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
        m1.metric("Premium", f"{custom['premium_x']:.1f}x retail")
        m2.metric("Segment", custom["market_segment"])
        m3.metric(f"{horizon}M base estimate", f"${final['forecast_price']:,.0f}", f"{final['expected_annual_growth']:+.1%} annualized")
        m4.metric(f"{horizon}M low/high", f"${final['low_estimate']:,.0f} – ${final['high_estimate']:,.0f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Auto volatility", f"{custom['volatility_pct']:.0f}%")
        m6.metric("Auto scarcity", f"{custom['scarcity_score']:.0f}/100")
        m7.metric("Risk score", f"{custom['risk_score']:.0f}/100")
        m8.metric("Liquidity", f"{custom['liquidity_score']:.0f}/100")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp.today().normalize().replace(day=1)],
            y=[current_resale],
            mode="markers",
            name="Current resale",
            marker=dict(size=11),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["month"],
            y=forecast["forecast_price"],
            mode="lines+markers",
            name="Base forecast",
            line=dict(dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["month"],
            y=forecast["high_estimate"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="High estimate",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["month"],
            y=forecast["low_estimate"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Forecast range",
        ))
        fig.update_layout(
            height=520,
            yaxis_title="Estimated resale value",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.22),
            margin=dict(b=125),
        )
        st.plotly_chart(fig, use_container_width=True)

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

        st.download_button(
            "Download this item row as CSV",
            export_row.to_csv(index=False).encode("utf-8"),
            file_name="estimator_item_row.csv",
            mime="text/csv",
        )

        dshow = drivers.copy()
        dshow["Value"] = dshow["Value"].map(lambda x: f"{x:+.2%}")
        st.dataframe(dshow, use_container_width=True, hide_index=True)

        if price_6m_ago == 0 and price_12m_ago == 0:
            st.warning("No past price was entered, so the forecast uses category/market trend as a proxy. It is better if you add a 6M or 12M past price.")
        st.warning("This is not a live StockX update. The link can fill matched or smart default fields, but the user should confirm visible market numbers unless official API access is added.")


with tabs[4]:
    st.subheader("Category Baskets")
    st.write("A category basket is like a mini-index. The maturity filter prevents very new drops, like Nike Mind 001, from causing a huge short-term spike in the whole sneaker basket.")
    data_filter = st.selectbox(
        "Data filter",
        ["All items", "Market-backed"],
        index=1,
        help="Market-backed uses verified snapshots plus low-volume luxury snapshots. All items also includes seed estimates."
    )
    maturity_filter = st.selectbox(
        "Basket maturity filter",
        ["Established items only, 6+ months", "All release windows"],
        index=0,
        help="Established items only is the cleaner default. All release windows includes new drops, but a hot new release can distort the category basket."
    )
    min_history_months = 6 if maturity_filter.startswith("Established") else 0
    categories = ["Sneaker","Streetwear","Luxury Streetwear"]
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
            summary_rows.append({"Category": cat, "Items": n_items, "Status": "Not enough items"})
            continue
        total_return = idx["index_value"].iloc[-1] / idx["index_value"].iloc[0] - 1
        vol = annualized_volatility(idx["monthly_return"])
        dd = max_drawdown(idx["index_value"])
        summary_rows.append({"Category": cat, "Items": n_items, "Return": total_return, "Volatility": vol, "Max drawdown": dd, "Status": "OK"})
        fig.add_trace(go.Scatter(x=idx["month"], y=idx["index_value"], mode="lines", name=cat))

    fig.update_layout(height=540, yaxis_title="Category basket index, base = 100", xaxis_title="", hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=130))
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(summary_rows)
    for col in ["Return","Volatility","Max drawdown"]:
        if col in summary.columns:
            summary[col] = summary[col].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}" if col=="Return" else f"{x:.1%}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Market Basket Index")
    st.warning("This is not saying one person owns every item. It is a broad market basket, like a simple sneaker/streetwear version of an index.")
    basket_filter = st.selectbox(
        "Basket maturity filter",
        ["Established items only, 6+ months", "All release windows"],
        index=0,
        help="All release windows includes very new drops, which can create short-term jumps."
    )
    min_history_months = 6 if basket_filter.startswith("Established") else 0
    raw = make_index(panel, "All", "All items", min_history_months=min_history_months)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw["month"], y=raw["index_value"], mode="lines+markers", name="Raw market basket"))
    fig.add_trace(go.Scatter(x=adjusted_index["month"], y=adjusted_index["adjusted_index"], mode="lines+markers", name="Regression-adjusted basket"))
    fig.update_layout(height=540, yaxis_title="Index value, base = 100", hovermode="x unified", legend=dict(orientation="h", y=-0.22), margin=dict(b=120))
    st.plotly_chart(fig, use_container_width=True)

with tabs[6]:
    st.subheader("Regression")
    st.write("Regression now uses item fixed effects, grail score, and market segment. That lets the model recognize legitimate grail premiums instead of forcing OW Jordan 1s or Nike Yeezy 2s back toward normal sneaker prices.")
    c1,c2 = st.columns(2)
    c1.metric("Fixed-effect MAE", f"${mae:,.0f}")
    c2.metric("Log-price R²", f"{r2:.2f}")
    fig = px.scatter(
        fitted, x="predicted_resale", y="median_resale", color="category", hover_name="item",
        hover_data=["month_label","brand","market_segment","grail_score","premium_x","sale_count","data_status"],
        labels={"predicted_resale":"Predicted resale","median_resale":"Actual/estimated resale"},
    )
    maxv = max(fitted["predicted_resale"].max(), fitted["median_resale"].max()) * 1.05
    fig.add_trace(go.Scatter(x=[0,maxv], y=[0,maxv], mode="lines", name="Perfect fit"))
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2), margin=dict(b=100))
    st.plotly_chart(fig, use_container_width=True)
    st.write("Top non-month coefficients:")
    show = coefs[~coefs["feature"].str.startswith("month_label_")].head(20).copy()
    show["coefficient"] = show["coefficient"].map(lambda x: round(float(x),3))
    st.dataframe(show[["feature","coefficient"]], use_container_width=True, hide_index=True)

with tabs[7]:
    st.subheader("Risk")
    fig = px.scatter(
        market, x="volatility_pct", y="premium_x", size="sales_volume", color="data_quality",
        hover_name="item", hover_data=["current_resale","market_segment","grail_score","risk_score","investability_score","data_status"],
        title="Current premium vs volatility, sized by sales volume",
    )
    fig.update_layout(height=530, legend=dict(orientation="h", y=-0.22), margin=dict(b=130))
    st.plotly_chart(fig, use_container_width=True)
    risk_table = market[["item","category","market_segment","current_resale","premium_x","volatility_pct","sales_volume","grail_score","risk_score","investability_score","data_quality","data_status"]].sort_values("risk_score")
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tabs[8]:
    st.subheader("Benchmarks")

    c1, c2, c3, c4 = st.columns([0.8, 1.1, 1.1, 1.2])
    with c1:
        period = st.selectbox("Benchmark period", ["6mo","1y","2y"], index=1)
    with c2:
        resale_choice = st.selectbox(
            "Resale basket to compare",
            ["All Resale", "Sneakers", "Streetwear", "Luxury Streetwear"],
            index=0
        )
    with c3:
        benchmark_data_filter = st.selectbox(
            "Resale data basis",
            ["All items", "Market-backed"],
            index=1,
            help="Market-backed keeps verified snapshots plus low-volume luxury snapshots. All items also includes seed estimates."
        )
    with c4:
        benchmark_maturity_filter = st.selectbox(
            "Maturity",
            ["Established 6M+", "All drops"],
            index=0,
            help="Established 6M+ prevents brand-new drops from distorting the basket."
        )
    show_assets = st.multiselect(
        "Traditional assets",
        ["S&P 500", "Nike", "Gold ETF"],
        default=["S&P 500", "Nike", "Gold ETF"]
    )

    category_map = {
        "All Resale": "All",
        "Sneakers": "Sneaker",
        "Streetwear": "Streetwear",
        "Luxury Streetwear": "Luxury Streetwear",
    }
    category = category_map[resale_choice]
    resale_label = f"{resale_choice} Basket"

    benchmark_min_history = 6 if benchmark_maturity_filter.startswith("Established") else 0
    resale_raw = make_index(panel, category, benchmark_data_filter, min_history_months=benchmark_min_history)
    resale_period = normalize_period_index(resale_raw, "index_value", period)
    resale_smooth = smooth_monthly_index(resale_period, "index_value", resale_label)

    fig = go.Figure()
    if not resale_smooth.empty:
        fig.add_trace(go.Scatter(
            x=resale_smooth["month"],
            y=resale_smooth["index_value"],
            mode="lines",
            name=resale_label,
            line_shape="spline",
            line=dict(width=3),
        ))
    else:
        st.warning(f"Not enough data for {resale_choice} basket.")

    if not resale_period.empty:
        bench = fetch_benchmarks_for_range(resale_period["month"].min(), resale_period["month"].max())
    else:
        bench = pd.DataFrame()

    if not bench.empty and show_assets:
        bench = bench[bench["asset"].isin(show_assets)]
        for asset, group in bench.groupby("asset"):
            fig.add_trace(go.Scatter(
                x=group["month"],
                y=group["index_value"],
                mode="lines",
                name=asset,
                line=dict(dash="dash"),
                line_shape="spline",
            ))
    elif show_assets:
        st.warning("Benchmark data did not load. yfinance or local internet may be blocked.")

    fig.update_layout(
        height=560,
        yaxis_title="Index value, base = 100",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.28, x=0),
        margin=dict(b=155, r=40, l=60, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not resale_period.empty:
        basket_return = resale_period["index_value"].iloc[-1] / resale_period["index_value"].iloc[0] - 1
        basket_vol = annualized_volatility(resale_period["monthly_return"]) if "monthly_return" in resale_period.columns else 0
        basket_dd = max_drawdown(resale_period["index_value"])
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{resale_choice} return", f"{basket_return:+.1%}")
        m2.metric("Volatility", f"{basket_vol:.1%}")
        m3.metric("Max drawdown", f"{basket_dd:.1%}")

    st.caption("Choose All Resale, Sneakers, Streetwear, or Luxury Streetwear to compare a dedicated resale basket against S&P 500, Nike, and Gold. Smoothing only interpolates between existing resale index months; it does not create new real resale observations.")

with tabs[9]:
    st.subheader("Methodology")
    st.markdown("""
### What fixes
fixes the grail problem. Items like Off-White Jordan 1 Chicago and Nike Yeezy 2 are not normal resale sneakers, so the model should not force them onto a normal sneaker curve.

### Fixed-effect regression
The regression now includes item fixed effects. That means each item gets its own baseline value. This is the right setup for an index because the model controls for the item itself and uses the month effects to estimate market movement.

### Grail score
Grail score captures cultural and collector premium using premium over retail, scarcity, age, low liquidity, and known grail/collaboration signals. It helps identify items that trade like collectibles instead of liquid resale inventory.

### Market segments
The app separates items into Liquid Sneaker, Specialty Sneaker, Grail / Collectible, Streetwear, and Luxury Streetwear. This prevents rare collectibles from being compared too directly to high-volume shoes.

### Raw basket vs regression-adjusted basket
The raw basket averages monthly item movement. The regression-adjusted basket controls for item, brand, category, collaboration, grail score, segment, sales volume, and month so the result is less distorted by item mix.

### Release-aware histories
Items no longer appear before their release year. Item Growth now starts each item at its own release window and compares S&P 500, Nike, and gold from the selected item's release date. Nike Mind 001 Slide rows start in 2026, so their graphs do not backfill fake 2024/2025 history.

### Growth forecast
The growth forecast is scenario-based. It combines recent item trend, category trend, broad resale trend, liquidity, risk, grail score, and market segment. also adds leveling-off, meaning the forecast slows over time instead of assuming resale prices compound forever.

### Estimator
The Estimator combines the old Custom Estimator and StockX Import into one page. If a StockX link is pasted, clicking Fill fields from link moves matched local data or smarter defaults into the confirmation fields. The user can still manually edit every number. It is link-assisted, not live scraping.

### User inputs
The estimator asks for things a user can reasonably find: retail price, current resale, recent low/high, sales volume band, release year, and whether it is a collaboration. Volatility and scarcity are auto-estimated, with optional advanced overrides.

### Basket maturity filter
The category and benchmark baskets default to established items with at least six months of history. This prevents very new releases, especially Nike Mind 001 rows, from creating a misleading category-wide spike. You can still choose All release windows to include every new drop.

### Benchmark smoothing
The benchmark chart now filters resale and traditional assets to the same selected period and interpolates the resale basket between monthly points so the comparison does not look chopped off or jagged. lets you switch the resale comparison between All Resale, Sneakers, Streetwear, and Luxury Streetwear, and choose either All items or Market-backed. Market-backed includes low-volume luxury snapshots so luxury does not disappear. This is visual smoothing only.

### Current premium and volatility
Current premium is current resale divided by retail. Volatility is how unstable the resale price is. High premium plus low volatility and high liquidity is the cleanest asset-like profile.
""")

with tabs[10]:
    st.subheader("Data")
    st.write("Current market seed:")
    st.dataframe(market, use_container_width=True, hide_index=True)
    st.download_button("Download market seed", market.to_csv(index=False).encode("utf-8"), "hype_asset_market_seed.csv", "text/csv")
    st.write("Prototype history panel:")
    st.dataframe(panel, use_container_width=True, hide_index=True)
    st.download_button("Download prototype history panel", panel.to_csv(index=False).encode("utf-8"), "hype_asset_history_panel.csv", "text/csv")
