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

#z scores for various service levels, used in safety stock calculations and were not just assumption but obtained from the standard distributiion table for the normal distribution. The z score is the number of standard deviations a data point is from the mean. In this case, it is used to determine the safety stock level based on the desired service level.
#Z-score is how many standard deviations of buffer you need above average demand to hit your target service level — it comes straight from the normal distribution's math, not something I chose. 1.65 for 95% means only 5% of the bell curve lies beyond that point, so you'd stock out 5% of the time at that level.
# the percentages are not calculated from my data but  are standard inventory management convention tears, bench marks any real supplier uses to ser risk tollerance
# 90% = a loose service level, often used for low priority items, 95% = a standard service level-good enough for most products, 97.5% = a high service level- a step up in strictness , reusing universally known constant, 99% = a very high service level - hard to replace items
#The spread from 90% to 99% also lets the tool reflect a real business tradeoff, how much stockout risk you're willing to accept varies by how critical or expensive a product is.
#


Z_SCORES = {
    "90%": 1.28,
    "95%": 1.65,
    "97.5%": 1.96,
    "99%": 2.33,
}

# looks at every order in the dataset and finds every unique combination of market, order_region, and shipping_mode — then assigns each unique combination an ID number (lane_id).
# lane is the actual unit that the project works on about for making shipping decisioons e.g Europe - western Europe - standard class is one lane, Europe - western Europe - first class is another lane, etc. The lane_id is just a unique identifier for each of these combinations.
# returns one row per lane, with the lane_id as a new column. This is useful for joining with other dataframes later on, as it allows you to easily reference a specific lane by its ID rather than having to use the combination of market, order_region, and shipping_mode each time.

def build_lanes(df: pd.DataFrame) -> pd.DataFrame:
    lanes = (
        df[["market", "order_region", "shipping_mode"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    lanes["lane_id"] = lanes.index + 1
    return lanes

#for each lane, calculates the average and the standard deviation of days_for_shipping_real — i.e., how long deliveries actually took on that specific lane, historically.
# this produces the average lead time and lead time variability for each lane, which is important for calculating safety stock and reorder points later on. The average lead time tells you how long it typically takes for an order to be delivered on that lane, while the standard deviation tells you how much variability there is in that lead time — a higher standard deviation means more uncertainty and risk of stockouts.
# returns one row per lane, average lead time and how much that lead time varies


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

#we cant just average order item quantity across rows we have to sum up the daily totals first as one day could have multiple orders
# produces average and standard deviation of daily demand (in units) for each product, which is important for calculating safety stock and reorder points later on. The average daily demand tells you how many units of a product are typically sold per day, while the standard deviation tells you how much variability there is in that demand — a higher standard deviation means more uncertainty and risk of stockouts.
# returns one row  per product , average daily deman and how much that demand varies



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

# takes demand stats table and scales it up or down based on a cpi shock
# elasticity is a measure of chane in demand relative to change in price, in this case the price change is represented by the cpi shock. A negative elasticity means that as prices go up, demand goes down, which is typical for most goods. The function adjusts both the average and standard deviation of daily demand proportionally to reflect the expected change in demand due to the cpi shock.
#
#

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

#this is the payoff fuction everything exist to feed this function

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