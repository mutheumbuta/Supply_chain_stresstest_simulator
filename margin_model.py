

from __future__ import annotations

import pandas as pd

DEFAULT_IMPORT_EXPOSURE = 0.6


def apply_import_price_shock(
    df: pd.DataFrame,
    pct_import_price_change: float,
    import_exposure: float | None = None,
    exposure_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns a per-order DataFrame with implied COGS, shocked COGS, and
    shocked profit under an import price index scenario.

    Pass either a single `import_exposure` (applies to every order --
    the v1 behavior) OR an `exposure_table` with columns
    ['Department Name', 'import_exposure_prior'] (from
    elasticity_model.estimate_department_elasticities) to use a
    department-specific exposure instead of one global number.
    """
    out = df[[
        "order_id", "market", "order_region", "category_name", "department_name",
        "order_item_total", "order_profit_per_order",
    ]].copy()

    out["implied_cogs"] = out["order_item_total"] - out["order_profit_per_order"]
    out["implied_cogs"] = out["implied_cogs"].clip(lower=0)

    if exposure_table is not None:
        exp_map = exposure_table.set_index("Department Name")["import_exposure_prior"]
        out["exposure"] = out["department_name"].map(exp_map).fillna(DEFAULT_IMPORT_EXPOSURE)
    else:
        out["exposure"] = import_exposure if import_exposure is not None else DEFAULT_IMPORT_EXPOSURE

    cost_multiplier = 1 + out["exposure"] * pct_import_price_change
    out["shocked_cogs"] = out["implied_cogs"] * cost_multiplier
    out["shocked_profit"] = out["order_item_total"] - out["shocked_cogs"]
    out["margin_impact"] = out["shocked_profit"] - out["order_profit_per_order"]

    return out


def margin_impact_summary(shocked_df: pd.DataFrame, group_col: str = "category_name") -> pd.DataFrame:
    """Aggregate margin impact by a grouping column (category_name, market, order_region)."""
    summary = (
        shocked_df.groupby(group_col)
        .agg(
            orders=("order_id", "count"),
            baseline_profit=("order_profit_per_order", "sum"),
            shocked_profit=("shocked_profit", "sum"),
            margin_impact=("margin_impact", "sum"),
        )
        .reset_index()
    )
    summary["margin_impact_pct"] = (
        summary["margin_impact"] / summary["baseline_profit"].replace(0, pd.NA)
    ) * 100
    return summary.sort_values("margin_impact")
