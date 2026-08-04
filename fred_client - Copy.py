from __future__ import annotations

import os
import io
from datetime import date

import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xls"

# Three primary indicators the app is built around, each representing a
# distinct transmission channel into the supply chain:
#   diesel_price          -> operational/freight cost
#   import_price_index    -> cost of goods sourced internationally (COGS)
#   cpi                    -> consumer purchasing power / demand elasticity
SERIES = {
    "diesel_price": "GASDESW",
    "import_price_index": "IR",
    "cpi": "CPIAUCSL",
    # Secondary/contextual indicators, shown but not driving the model directly
    "wti_crude": "DCOILWTICO",
    "freight_ppi": "PCU484121484121",
    "fed_funds_rate": "FEDFUNDS",
}


def _fallback_series(series_key: str) -> pd.DataFrame:
    """
    Small representative fallback series (monthly, 2023-2025) so the app
    is fully functional out-of-the-box without a live API key. Values are
    approximate historical levels, NOT live data -- replace with a real
    FRED_API_KEY for production use.
    """
    dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    baselines = {
        "diesel_price": 4.20,          # $/gallon, drifting with mild seasonality
        "import_price_index": 142.0,   # index level (2000=100)
        "cpi": 305.0,                   # index level
        "wti_crude": 78.0,              # $/barrel
        "freight_ppi": 245.0,           # index level
        "fed_funds_rate": 5.1,          # %
    }
    import numpy as np
    base = baselines[series_key]
    trend = np.linspace(-0.03, 0.03, len(dates))
    seasonal = 0.02 * np.sin(np.linspace(0, 6 * 3.14159, len(dates)))
    noise = np.random.default_rng(42).normal(0, 0.01, len(dates))
    values = base * (1 + trend + seasonal + noise)
    return pd.DataFrame({"date": dates, "value": values})


def fetch_fred_series(series_key: str, api_key: str | None = None,
                       start_date: str = "2023-01-01") -> pd.DataFrame:
    """
    Fetch one FRED series. Falls back to synthetic representative data if
    no API key is set or the request fails (e.g. no network access).
    Returns a DataFrame with columns ['date', 'value'].
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    series_id = SERIES.get(series_key, series_key)

    if not api_key:
        df = _fallback_series(series_key)
        df.attrs["source"] = "fallback"
        return df

    try:
        resp = requests.get(
            FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
            },
            timeout=10,
        )
        resp.raise_for_status()
        obs = resp.json()["observations"]
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df.attrs["source"] = "fred_live"
        return df
    except Exception as exc:  # network unavailable, bad key, rate limit, etc.
        df = _fallback_series(series_key)
        df.attrs["source"] = f"fallback (FRED fetch failed: {exc})"
        return df


def fetch_gscpi() -> pd.DataFrame:
    """
    Fetch NY Fed Global Supply Chain Pressure Index. Falls back to a
    synthetic representative series if unreachable.
    """
    try:
        resp = requests.get(GSCPI_URL, timeout=10)
        resp.raise_for_status()
        raw = pd.read_excel(io.BytesIO(resp.content), sheet_name="GSCPI Monthly Data", skiprows=3)
        raw.columns = ["date", "gscpi"]
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna()
        raw.attrs["source"] = "nyfed_live"
        return raw.rename(columns={"gscpi": "value"})
    except Exception as exc:
        dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
        import numpy as np
        values = 0.3 * np.sin(np.linspace(0, 4 * 3.14159, len(dates))) + np.random.default_rng(7).normal(0, 0.1, len(dates))
        df = pd.DataFrame({"date": dates, "value": values})
        df.attrs["source"] = f"fallback (GSCPI fetch failed: {exc})"
        return df


def fetch_all_macro(api_key: str | None = None) -> dict[str, pd.DataFrame]:
    data = {key: fetch_fred_series(key, api_key=api_key) for key in SERIES}
    data["gscpi"] = fetch_gscpi()
    return data


def latest_value(df: pd.DataFrame) -> float:
    return float(df.sort_values("date")["value"].iloc[-1])
