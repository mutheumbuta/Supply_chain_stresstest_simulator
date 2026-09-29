
from __future__ import annotations

import pandas as pd
import pulp

# The base rates are assumptions I made because the data doesn't contain real costs but they're not random assumptions, they follow the actual pricing pattern real carriers use.
# they are no vfreight columns anywhere in the data so I had to make assumptions about what the base cost per order is for each shipping mode. These are not random numbers, they are based on real-world pricing patterns that carriers use, where standard class is the cheapest and same-day is the most expensive.
# Each tier costs roughly 1.5–1.8x the one below it, mirroring how express shipping is real-world priced at a steep premium over ground shipping, not just "a little more
# Same Day sits at exactly 5x Standard Class — a large gap on purpose, since same-day delivery genuinely is the most disproportionately expensive tier in real logistics (it requires dedicated capacity, not shared truck routes)
BASE_RATE = {
    "Standard Class": 8.0,
    "Second Class": 14.0,
    "First Class": 22.0,
    "Same Day": 40.0,
}

# this were also assumptions but represents what fraction of this shipping modes cost is tied on fuel prices versus labour and handling costs
# more diesel-exposed; premium air/same-day is comparatively insulated.
# basic reason is standard class is mostly ground shipping, which is heavily dependent on diesel fuel prices, while same-day and first-class are more likely to be air shipping or involve more labor and handling costs, which are less sensitive to diesel price changes. So, the cost of standard class will fluctuate more with diesel price changes than the cost of same-day or first-class shipping.
# inverse relationship is the entie mechanism that makes a diesel shock change which mode is optimal, if exposure were flat across all four modes a diesel change would change by the same percentage across all four modes and the optimal mode would never change.

DIESEL_EXPOSURE = {
    "Standard Class": 0.55,
    "Second Class": 0.45,
    "First Class": 0.30,
    "Same Day": 0.20,
}

# function is a safety net to make sure that the lead time for each shipping mode never goes below a certain threshold, regardless of any macroeconomic conditions or shocks. This is important because even if there are changes in fuel prices or other factors, there is a physical limit to how fast a delivery can be made. For example, even if diesel prices drop significantly, a same-day delivery can't be completed in less than half a day due to the time it takes to process and transport the order. These floors ensure that the model doesn't produce unrealistic lead times that could lead to poor decision-making in the supply chain.
MODE_AVG_LEAD_DAYS_FLOOR = {
    # Physical floor on lead time per mode regardless of macro conditions
    "Same Day": 0.5,
    "First Class": 1.5,
    "Second Class": 3.0,
    "Standard Class": 4.5,
}

# produces one cost per estimate per lane per shipping mode, which is important for calculating the total cost of fulfilling orders on that lane. The cost per order tells you how much it costs to ship an order on that lane using a specific shipping mode, which is crucial for making decisions about which shipping mode to use for each lane in order to minimize costs while still meeting service level requirements.
#The avg_order_value itself is real — it's the actual average order value for that specific lane, pulled directly from historical order data. But the 500 denominator is an assumed normalizing constant, chosen so that a lane with a $500 average order (a reasonable mid-range value in this dataset) gets exactly a 2x cost multiplier. This is a genuine hybrid: real data run through an assumed scaling rule, because there's no real shipment-weight data to calculate size/cost from directly.
#

def build_lane_cost_table(
    df: pd.DataFrame,
    lane_lead_time_stats: pd.DataFrame,
    lanes: pd.DataFrame,
    pct_diesel_change: float = 0.0,
    congestion_lead_time_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Build a (market, order_region, shipping_mode) -> {cost, lead_time}
    table under a given macro scenario.
    """
    lane_value = (
        df.groupby(["market", "order_region"])["order_item_total"]
        .mean()
        .rename("avg_order_value")
        .reset_index()
    )

    rows = []
    for _, lane_row in lanes.iterrows():
        market, region, mode = lane_row["market"], lane_row["order_region"], lane_row["shipping_mode"]
        base = BASE_RATE[mode]
        exposure = DIESEL_EXPOSURE[mode]
        elasticity = 1 + exposure * pct_diesel_change

        avg_val_row = lane_value[
            (lane_value["market"] == market) & (lane_value["order_region"] == region)
        ]
        volume_factor = 1 + (avg_val_row["avg_order_value"].iloc[0] / 500 if len(avg_val_row) else 0)

        cost = base * volume_factor * elasticity

        lt_row = lane_lead_time_stats[lane_lead_time_stats["lane_id"] == lane_row["lane_id"]]
        observed_lt = lt_row["avg_lead_time"].iloc[0] if len(lt_row) else MODE_AVG_LEAD_DAYS_FLOOR[mode]
        lead_time = max(observed_lt, MODE_AVG_LEAD_DAYS_FLOOR[mode]) * congestion_lead_time_multiplier

        rows.append(
            {
                "market": market,
                "order_region": region,
                "shipping_mode": mode,
                "cost_per_order": round(cost, 2),
                "lead_time_days": round(lead_time, 2),
            }
        )
    return pd.DataFrame(rows)


#
#
#
def optimize_lane_mode_assignment(
    cost_table: pd.DataFrame,
    max_lead_time_days: float = 5.0,
) -> pd.DataFrame:
    """
    LP: for each lane (market x order_region), pick exactly one
    shipping_mode minimizing cost, subject to lead_time_days <=
    max_lead_time_days. If no mode satisfies the constraint for a lane,
    relax and pick the fastest available mode for that lane (flagged in
    the output).
    """
    lanes = cost_table[["market", "order_region"]].drop_duplicates().values.tolist()
    results = []

    for market, region in lanes:
        subset = cost_table[
            (cost_table["market"] == market) & (cost_table["order_region"] == region)
        ].reset_index(drop=True)

        feasible = subset[subset["lead_time_days"] <= max_lead_time_days]
        relaxed = False
        if feasible.empty:
            feasible = subset
            relaxed = True

        prob = pulp.LpProblem(f"lane_{market}_{region}", pulp.LpMinimize)
        choice_vars = {
            i: pulp.LpVariable(f"choose_{market}_{region}_{i}", cat="Binary")
            for i in feasible.index
        }
        prob += pulp.lpSum(choice_vars[i] * feasible.loc[i, "cost_per_order"] for i in feasible.index)
        prob += pulp.lpSum(choice_vars.values()) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen_idx = [i for i in feasible.index if choice_vars[i].value() == 1][0]
        chosen = feasible.loc[chosen_idx]

        results.append(
            {
                "Market": market,
                "Order Region": region,
                "Recommended Shipping Mode": chosen["shipping_mode"],
                "Cost per Order ($)": chosen["cost_per_order"],
                "Expected Lead Time (days)": chosen["lead_time_days"],
                "Service Level Constraint Relaxed": relaxed,
            }
        )

    return pd.DataFrame(results).sort_values("Cost per Order ($)", ascending=False)