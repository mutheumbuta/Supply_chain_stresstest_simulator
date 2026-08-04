"""
Core analytics: lane-level lead-time variance and dynamic safety stock.

Runs entirely in pandas against the cleaned CSV export
(data/dataco_clean.csv) -- no database required.

Operates on the cleaned dataset (see notebooks/cleaning.ipynb) -- snake_case
column names throughout (e.g. order_region, not "Order Region").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_SCORES = {
    "90%": 1.28,
    "95%": 1.65,
    "97.5%": 1.96,
    "99%": 2.33,
}


def build_lanes(df: pd.DataFrame) -> pd.DataFrame:
    lanes = (
        df[["market", "order_region", "shipping_mode"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    lanes["lane_id"] = lanes.index + 1
    return lanes


def lane_lead_time_stats(df: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
    """Avg + std lead time (days_for_shipping_real) per lane."""
    merged = df.merge(lanes, on=["market", "order_region", "shipping_mode"], how="left")
    stats = (
        merged.groupby("lane_id")["days_for_shipping_real"]
        .agg(avg_lead_time="mean", std_lead_time="std")
        .fillna(0)
        .reset_index()
    )
    return stats


def product_demand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Avg + std of daily demand (units) per product."""
    daily = (
        df.groupby(["product_card_id", "order_date"])["order_item_quantity"]
        .sum()
        .reset_index()
    )
    stats = (
        daily.groupby("product_card_id")["order_item_quantity"]
        .agg(avg_daily_demand="mean", std_daily_demand="std")
        .fillna(0)
        .reset_index()
    )
    return stats


def apply_cpi_demand_elasticity(
    demand_stats: pd.DataFrame,
    pct_cpi_change: float,
    elasticity: float = -0.35,
) -> pd.DataFrame:
    """
    Adjust forecasted demand for a CPI (inflation) shock.

    elasticity = -0.35 means a 10% CPI increase shrinks demand by 3.5%
    (a standard-goods elasticity assumption; discretionary categories in
    this dataset would realistically be more elastic, staples less so --
    a single elasticity is a simplification, documented here rather than
    hidden). Both mean and std of daily demand are scaled together so
    volatility shrinks/grows proportionally with volume.
    """
    adjusted = demand_stats.copy()
    demand_multiplier = 1 + elasticity * pct_cpi_change
    demand_multiplier = max(demand_multiplier, 0.05)  # floor: demand can't go negative
    adjusted["avg_daily_demand"] = adjusted["avg_daily_demand"] * demand_multiplier
    adjusted["std_daily_demand"] = adjusted["std_daily_demand"] * demand_multiplier
    adjusted.attrs["demand_multiplier"] = demand_multiplier
    return adjusted


def dynamic_safety_stock(
    df: pd.DataFrame,
    lanes: pd.DataFrame,
    lead_time_stats: pd.DataFrame,
    demand_stats: pd.DataFrame,
    service_level: str = "95%",
    lead_time_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    King's formula:
        SS = Z * sqrt( LT_avg * sigma_D^2 + D_avg^2 * sigma_LT^2 )

    lead_time_multiplier lets a macro scenario (e.g. port congestion from
    a diesel shock) stretch out avg/std lead time before recomputing SS,
    which is the "dynamic" part of dynamic safety stock.
    """
    z = Z_SCORES[service_level]

    pairs = (
        df.merge(lanes, on=["market", "order_region", "shipping_mode"], how="left")
        [["product_card_id", "lane_id"]]
        .drop_duplicates()
    )

    merged = (
        pairs.merge(demand_stats, on="product_card_id", how="left")
        .merge(lead_time_stats, on="lane_id", how="left")
        .merge(lanes, on="lane_id", how="left")
    )

    lt_avg = merged["avg_lead_time"] * lead_time_multiplier
    lt_std = merged["std_lead_time"] * lead_time_multiplier

    merged["safety_stock"] = z * np.sqrt(
        lt_avg * merged["std_daily_demand"] ** 2
        + merged["avg_daily_demand"] ** 2 * lt_std ** 2
    )
    merged["reorder_point"] = merged["safety_stock"] + lt_avg * merged["avg_daily_demand"]

    return merged[
        [
            "product_card_id", "lane_id", "market", "order_region", "shipping_mode",
            "avg_lead_time", "std_lead_time", "avg_daily_demand", "std_daily_demand",
            "safety_stock", "reorder_point",
        ]
    ]