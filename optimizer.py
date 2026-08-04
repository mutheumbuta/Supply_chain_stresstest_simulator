
from __future__ import annotations

import pandas as pd
import pulp

BASE_RATE = {
    "Standard Class": 8.0,
    "Second Class": 14.0,
    "First Class": 22.0,
    "Same Day": 40.0,
}

# Fraction of each mode's cost that scales with diesel price (rest is fixed
# handling/labor cost, insulated from fuel swings). Ground-heavy modes are
# more diesel-exposed; premium air/same-day is comparatively insulated.
DIESEL_EXPOSURE = {
    "Standard Class": 0.55,
    "Second Class": 0.45,
    "First Class": 0.30,
    "Same Day": 0.20,
}

MODE_AVG_LEAD_DAYS_FLOOR = {
    # Physical floor on lead time per mode regardless of macro conditions
    "Same Day": 0.5,
    "First Class": 1.5,
    "Second Class": 3.0,
    "Standard Class": 4.5,
}


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