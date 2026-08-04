
from __future__ import annotations

import numpy as np
import pandas as pd


DEPARTMENT_PRIORS = {
    "Technology":         {"demand_elasticity": -0.60, "import_exposure": 0.85},
    "Golf":                {"demand_elasticity": -0.60, "import_exposure": 0.55},
    "Apparel":             {"demand_elasticity": -0.45, "import_exposure": 0.75},
    "Footwear":            {"demand_elasticity": -0.45, "import_exposure": 0.75},
    "Fan Shop":            {"demand_elasticity": -0.50, "import_exposure": 0.55},
    "Outdoors":            {"demand_elasticity": -0.45, "import_exposure": 0.55},
    "Fitness":             {"demand_elasticity": -0.55, "import_exposure": 0.60},
    "Discs Shop":          {"demand_elasticity": -0.30, "import_exposure": 0.30},
    "Pet Shop":            {"demand_elasticity": -0.20, "import_exposure": 0.40},
    "Book Shop":           {"demand_elasticity": -0.15, "import_exposure": 0.30},
    "Health and Beauty":   {"demand_elasticity": -0.25, "import_exposure": 0.45},
}
DEFAULT_PRIOR = {"demand_elasticity": -0.35, "import_exposure": 0.60}


def historical_cpi_series() -> pd.DataFrame:
    """
    REAL BLS CPI-U (NSA, 1982-84=100) monthly index values, Jan 2015 -
    Jan 2018 -- matching the dataset's actual order date range exactly.
    Source: BLS "Historical Consumer Price Index for All Urban
    Consumers" table (bls.gov/cpi/tables/supplemental-files).
    """
    months = pd.period_range("2015-01", "2018-01", freq="M")
    cpi_values = [
        233.707, 234.722, 236.119, 236.599, 237.805, 238.638,
        238.654, 238.316, 237.945, 237.838, 237.336, 236.525,
        236.916, 237.111, 238.132, 239.261, 240.229, 241.018,
        240.628, 240.849, 241.428, 241.729, 241.353, 241.432,
        242.839, 243.603, 243.801, 244.524, 244.733, 244.955,
        244.786, 245.519, 246.819, 246.663, 246.669, 246.524,
        247.867,
    ]
    assert len(months) == len(cpi_values)
    return pd.DataFrame({"month": months, "cpi": cpi_values})


def estimate_department_elasticities(
    df: pd.DataFrame,
    k_shrinkage: float = 12.0,
    min_t_stat: float = 2.0,
) -> pd.DataFrame:
    """
    Returns one row per department: prior, data-estimated slope (where
    computable and statistically distinguishable from noise), sample
    size, shrinkage weight, and the blended final demand_elasticity
    used by the app.

    Iterates over every department present in the data (not just ones
    with usable regression history) so departments with zero months in
    the regression window still get their own documented prior instead
    of silently disappearing or falling back to the generic default.
    """
    d = df.copy()
    d["month"] = d["order_date"].dt.to_period("M")

    all_departments = sorted(d["department_name"].dropna().unique())

    monthly = (
        d.groupby(["department_name", "month"])["order_item_quantity"]
        .sum()
        .reset_index()
    )

    cpi = historical_cpi_series()
    # Exclude the truncated tail (Oct 2017 onward) from the regression
    # window -- order volume roughly halves dataset-wide from that point,
    # a known export artifact, not real demand collapse.
    regression_window_end = pd.Period("2017-09", freq="M")
    monthly = monthly[monthly["month"] <= regression_window_end]
    cpi = cpi[cpi["month"] <= regression_window_end]

    monthly = monthly.merge(cpi, on="month", how="inner")
    monthly = monthly[monthly["order_item_quantity"] > 0]
    monthly["log_qty"] = np.log(monthly["order_item_quantity"])
    monthly["log_cpi"] = np.log(monthly["cpi"])

    monthly_by_dept = {dept: group for dept, group in monthly.sort_values("month").groupby("department_name")}

    rows = []
    for dept in all_departments:
        group = monthly_by_dept.get(dept)
        prior = DEPARTMENT_PRIORS.get(dept, DEFAULT_PRIOR)
        n_months = len(group) if group is not None else 0

        estimated, t_stat = None, None
        if group is not None and n_months >= 8 and group["log_cpi"].std() > 1e-6:
            time_idx = np.arange(n_months)
            X = np.column_stack([np.ones(n_months), time_idx, group["log_cpi"].values])
            y = group["log_qty"].values
            beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = max(n_months - X.shape[1], 1)
            sigma2 = (resid @ resid) / dof
            xtx_inv = np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(sigma2 * xtx_inv))
            beta_cpi, se_cpi = beta[2], se[2]
            t_stat = beta_cpi / se_cpi if se_cpi > 1e-9 else 0.0
            if abs(t_stat) >= min_t_stat:
                estimated = float(np.clip(beta_cpi, -3.0, 0.5))

        weight = (n_months / (n_months + k_shrinkage)) if estimated is not None else 0.0
        blended = weight * estimated + (1 - weight) * prior["demand_elasticity"] if estimated is not None else prior["demand_elasticity"]

        rows.append(
            {
                "Department Name": dept,
                "n_months": n_months,
                "prior_elasticity": prior["demand_elasticity"],
                "estimated_elasticity": estimated,
                "t_stat": round(t_stat, 2) if t_stat is not None else None,
                "data_weight": round(weight, 2),
                "blended_elasticity": round(blended, 3),
                "import_exposure_prior": prior["import_exposure"],
            }
        )

    result = pd.DataFrame(rows).sort_values("n_months", ascending=False).reset_index(drop=True)
    return result


def apply_department_cpi_elasticity(
    df: pd.DataFrame,
    demand_stats: pd.DataFrame,
    elasticity_table: pd.DataFrame,
    pct_cpi_change: float,
) -> pd.DataFrame:
    """
    Same output shape as analytics.apply_cpi_demand_elasticity, but the
    elasticity used varies per product based on its department_name.
    """
    product_dept = df[["product_card_id", "department_name"]].drop_duplicates()
    merged = demand_stats.merge(product_dept, on="product_card_id", how="left")
    merged = merged.merge(
        elasticity_table[["Department Name", "blended_elasticity"]].rename(
            columns={"Department Name": "department_name"}
        ),
        on="department_name", how="left",
    )
    merged["blended_elasticity"] = merged["blended_elasticity"].fillna(DEFAULT_PRIOR["demand_elasticity"])

    multiplier = (1 + merged["blended_elasticity"] * pct_cpi_change).clip(lower=0.05)
    merged["avg_daily_demand"] = merged["avg_daily_demand"] * multiplier
    merged["std_daily_demand"] = merged["std_daily_demand"] * multiplier

    return merged[["product_card_id", "avg_daily_demand", "std_daily_demand"]]
