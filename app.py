
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from analytics import (
    build_lanes, lane_lead_time_stats, product_demand_stats,
    apply_cpi_demand_elasticity, dynamic_safety_stock, Z_SCORES,
)
from optimizer import build_lane_cost_table, optimize_lane_mode_assignment
from margin_model import apply_import_price_shock, margin_impact_summary, DEFAULT_IMPORT_EXPOSURE
from elasticity_model import estimate_department_elasticities, apply_department_cpi_elasticity
from fred_client import fetch_all_macro, latest_value

st.set_page_config(page_title="Supply Chain Stress Test Simulator", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DataCoSupplyChainDataset_clean.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # CSV doesn't preserve dtypes -- dates come back as plain strings and
    # must be re-parsed, unlike parquet which round-trips them natively.
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["shipping_date"] = pd.to_datetime(df["shipping_date"])
    return df


@st.cache_data
def load_macro(api_key: str | None):
    return fetch_all_macro(api_key=api_key)


@st.cache_data
def precompute_base_stats(df: pd.DataFrame):
    lanes = build_lanes(df)
    lt_stats = lane_lead_time_stats(df, lanes)
    demand_stats = product_demand_stats(df)
    elasticity_table = estimate_department_elasticities(df)
    return lanes, lt_stats, demand_stats, elasticity_table


df = load_data()
lanes, lt_stats, demand_stats, elasticity_table = precompute_base_stats(df)

st.title("Supply Chain Stress Test Simulator")
st.caption(
    "Three macro shocks, three distinct channels into the supply chain: diesel price -> freight cost, "
    "import price index -> cost of goods, CPI -> demand. Built on the DataCo Supply Chain dataset (180K+ orders)."
)

# --------------------------------------------------------------------
# Sidebar: scenario controls, one slider per primary indicator
# --------------------------------------------------------------------
st.sidebar.header(" Macro Scenario Controls")

api_key = st.sidebar.text_input("FRED API key (optional)", type="password",
                                 help="Leave blank to use built-in representative data.")
macro = load_macro(api_key if api_key else None)

st.sidebar.subheader("1️⃣ Diesel Price (GASDESW)")
st.sidebar.caption(f"Source: {macro['diesel_price'].attrs.get('source', 'unknown')}")
pct_diesel_change = st.sidebar.slider(
    "Diesel price shock (%)", min_value=-30, max_value=100, value=0, step=5,
    help="Drives freight cost per shipment."
) / 100.0

st.sidebar.subheader("2️⃣ Import Price Index (IR)")
st.sidebar.caption(f"Source: {macro['import_price_index'].attrs.get('source', 'unknown')}")
pct_import_change = st.sidebar.slider(
    "Import price index shock (%)", min_value=-20, max_value=50, value=0, step=5,
    help="Drives cost of goods sold on imported/sourced inputs."
) / 100.0
use_dept_import_exposure = st.sidebar.checkbox(
    "Use per-department import exposure (recommended)", value=True,
    help="Off = one global exposure value for every order (v1 behavior)."
)
if use_dept_import_exposure:
    st.sidebar.caption("Exposure varies by department — see table below the margin section.")
else:
    import_exposure = st.sidebar.slider(
        "Share of COGS exposed to import prices (global)", min_value=0.0, max_value=1.0,
        value=DEFAULT_IMPORT_EXPOSURE, step=0.05,
    )

st.sidebar.subheader("3️⃣ CPI / Inflation (CPIAUCSL)")
st.sidebar.caption(f"Source: {macro['cpi'].attrs.get('source', 'unknown')}")
pct_cpi_change = st.sidebar.slider(
    "CPI (inflation) shock (%)", min_value=-5, max_value=20, value=0, step=1,
    help="Drives forecasted demand via price elasticity."
) / 100.0
use_dept_elasticity = st.sidebar.checkbox(
    "Use per-department demand elasticity (recommended)", value=True,
    help="Off = one global elasticity for every product (v1 behavior). "
         "On = department-specific, blending a documented prior with a "
         "data-estimated slope where the history actually supports one."
)
if not use_dept_elasticity:
    demand_elasticity = st.sidebar.slider(
        "Demand elasticity to CPI (global)", min_value=-1.0, max_value=-0.05, value=-0.35, step=0.05,
    )

st.sidebar.divider()
congestion_multiplier = st.sidebar.slider(
    "Port/lane congestion multiplier on lead time", min_value=1.0, max_value=2.5, value=1.0, step=0.1,
    help="Operational lever, independent of the 3 macro indicators above (e.g. port backlog)."
)
service_level = st.sidebar.selectbox("Target service level", list(Z_SCORES.keys()), index=1)
max_lead_time = st.sidebar.slider(
    "Max acceptable lead time for routing (days)", min_value=1.0, max_value=10.0, value=5.0, step=0.5
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Cost model note:** the source dataset has no freight cost or COGS field. "
    "Both cost models are documented, adjustable assumption sets (see `src/optimizer.py` and "
    "`src/margin_model.py` docstrings), not fitted to real carrier rates or supplier costs."
)

# --------------------------------------------------------------------
# Macro indicators row — the three primary indicators front and center
# --------------------------------------------------------------------
st.subheader("The Three Macro Indicators")
c1, c2, c3 = st.columns(3)
c1.metric("⛽ Diesel Price (GASDESW)", f"${latest_value(macro['diesel_price']):.2f}/gal",
          delta=f"{pct_diesel_change:+.0%} scenario shock")
c2.metric("📦 Import Price Index (IR)", f"{latest_value(macro['import_price_index']):.1f} idx",
          delta=f"{pct_import_change:+.0%} scenario shock")
c3.metric("🛒 CPI (CPIAUCSL)", f"{latest_value(macro['cpi']):.1f} idx",
          delta=f"{pct_cpi_change:+.0%} scenario shock")

with st.expander("Other tracked indicators (contextual, not driving the model directly)"):
    o1, o2, o3 = st.columns(3)
    o1.metric("WTI Crude", f"${latest_value(macro['wti_crude']):.2f}/bbl")
    o2.metric("Freight PPI", f"{latest_value(macro['freight_ppi']):.1f} idx")
    o3.metric("Fed Funds Rate", f"{latest_value(macro['fed_funds_rate']):.2f}%")

st.divider()

# --------------------------------------------------------------------
# CHANNEL 1: Diesel -> Freight cost & routing
# --------------------------------------------------------------------
st.subheader("1️⃣ Diesel Price → Cost-Optimized Shipping Routes")

cost_table = build_lane_cost_table(
    df, lt_stats, lanes,
    pct_diesel_change=pct_diesel_change,
    congestion_lead_time_multiplier=congestion_multiplier,
)
recommendations = optimize_lane_mode_assignment(cost_table, max_lead_time_days=max_lead_time)

left, right = st.columns([2, 1])
with left:
    fig = px.bar(
        recommendations, x="Cost per Order ($)", y="Order Region", color="Market",
        pattern_shape="Recommended Shipping Mode", orientation="h",
        title=f"Optimal Cost per Order by Lane (diesel shock: {pct_diesel_change:+.0%})",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    baseline_cost = build_lane_cost_table(df, lt_stats, lanes, 0.0, 1.0)
    baseline_rec = optimize_lane_mode_assignment(baseline_cost, max_lead_time_days=max_lead_time)
    delta = recommendations["Cost per Order ($)"].sum() - baseline_rec["Cost per Order ($)"].sum()
    st.metric("Total Freight Cost Across All Lanes", f"${recommendations['Cost per Order ($)'].sum():,.0f}",
              delta=f"{delta:+,.0f} vs. no-shock baseline")
    relaxed_count = recommendations["Service Level Constraint Relaxed"].sum()
    if relaxed_count:
        st.warning(f"{relaxed_count} lane(s) can't meet the {max_lead_time}-day target under this scenario.")
    st.dataframe(recommendations["Recommended Shipping Mode"].value_counts(), use_container_width=True)

with st.expander("Full lane-by-lane freight recommendation table"):
    st.dataframe(recommendations, use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------
# CHANNEL 2: Import Price Index -> COGS & Margin
# --------------------------------------------------------------------
st.subheader("2️⃣ Import Price Index → Cost of Goods & Margin Impact")
st.caption(
    "Implied COGS = Order Item Total − Order Profit Per Order. A rising import price index inflates the "
    "import-exposed share of that COGS, compressing margin. See `src/margin_model.py` for the assumption set."
)

shocked = apply_import_price_shock(
    df, pct_import_change,
    import_exposure=None if use_dept_import_exposure else import_exposure,
    exposure_table=elasticity_table if use_dept_import_exposure else None,
)
by_category = margin_impact_summary(shocked, group_col="category_name")

m1, m2, m3 = st.columns(3)
m1.metric("Baseline Total Profit", f"${shocked['order_profit_per_order'].sum():,.0f}")
m2.metric("Shocked Total Profit", f"${shocked['shocked_profit'].sum():,.0f}")
m3.metric("Total Margin Impact", f"${shocked['margin_impact'].sum():,.0f}",
          delta=f"{shocked['margin_impact'].sum():,.0f}")

exposure_label = "per-department exposure" if use_dept_import_exposure else f"{import_exposure:.0%} global exposure"
fig3 = px.bar(
    by_category.tail(15), x="margin_impact", y="category_name", orientation="h",
    title=f"Margin Impact by Product Category (import price shock: {pct_import_change:+.0%}, {exposure_label})",
    color="margin_impact", color_continuous_scale="RdYlGn", height=500,
    labels={"category_name": "Category Name"},
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("Full category margin impact table"):
    st.dataframe(by_category, use_container_width=True, hide_index=True)

with st.expander("📐 Department-level elasticity & import exposure (methodology + edit)"):
    st.markdown(
        "Prior = documented economic assumption. Estimated = a real regression slope from this "
        "dataset's own order history (log demand vs. log CPI, controlling for a linear time trend), "
        "**only shown where the t-stat clears a significance bar** — most departments here don't have "
        "enough independent variation to say anything, and correctly fall back to the prior instead of "
        "a fake-precise number. Blended = shrinkage-weighted combination, used by the app. Edit any cell "
        "to override — changes apply immediately below."
    )
    edited_table = st.data_editor(
        elasticity_table, use_container_width=True, hide_index=True,
        column_config={
            "blended_elasticity": st.column_config.NumberColumn("Elasticity used by app", step=0.05),
            "import_exposure_prior": st.column_config.NumberColumn("Import exposure used by app", step=0.05, min_value=0.0, max_value=1.0),
        },
        disabled=["Department Name", "n_months", "prior_elasticity", "estimated_elasticity", "t_stat", "data_weight"],
        key="elasticity_editor",
    )
    elasticity_table = edited_table  # any edits flow into the CPI section below

st.divider()

# --------------------------------------------------------------------
# CHANNEL 3: CPI -> Demand elasticity -> Dynamic safety stock
# --------------------------------------------------------------------
st.subheader("3️⃣ CPI / Inflation → Demand Shift → Dynamic Safety Stock")
st.caption(
    "Safety stock = Z × √(LT_avg·σ_D² + D_avg²·σ_LT²). CPI shock adjusts D_avg and σ_D (demand) via the "
    "elasticity assumption; the congestion multiplier (sidebar) independently adjusts LT_avg and σ_LT."
)

if use_dept_elasticity:
    adjusted_demand = apply_department_cpi_elasticity(df, demand_stats, elasticity_table, pct_cpi_change)
    st.info(f"CPI shock of {pct_cpi_change:+.0%} applied with **per-department elasticities** "
            f"(edit the table above to override). Range in use: "
            f"{elasticity_table['blended_elasticity'].min():.2f} to {elasticity_table['blended_elasticity'].max():.2f}.")
else:
    adjusted_demand = apply_cpi_demand_elasticity(demand_stats, pct_cpi_change, elasticity=demand_elasticity)
    demand_mult = adjusted_demand.attrs.get("demand_multiplier", 1.0)
    st.info(f"CPI shock of {pct_cpi_change:+.0%} at global elasticity {demand_elasticity:.2f} → "
            f"forecasted demand multiplier: **{demand_mult:.2f}x** (same for every product).")

ss = dynamic_safety_stock(
    df, lanes, lt_stats, adjusted_demand,
    service_level=service_level, lead_time_multiplier=congestion_multiplier,
)

product_names = df[["product_card_id", "product_name"]].drop_duplicates()
ss_named = ss.merge(product_names, on="product_card_id", how="left")

top_products = (
    ss_named.groupby(["product_card_id", "product_name"])["safety_stock"]
    .mean().reset_index().sort_values("safety_stock", ascending=False).head(15)
)

c1, c2 = st.columns([2, 1])
with c1:
    fig2 = px.bar(
        top_products, x="safety_stock", y="product_name", orientation="h",
        title=f"Top 15 Products by Recommended Safety Stock (service level: {service_level})",
        height=550, labels={"product_name": "Product Name"},
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    st.metric("Avg Safety Stock (units)", f"{ss['safety_stock'].mean():,.1f}")
    st.metric("Avg Reorder Point (units)", f"{ss['reorder_point'].mean():,.1f}")
    st.write("Push CPI positive to see demand-driven safety stock fall (less to stock against smaller "
             "expected demand) — a real tension against the freight/COGS shocks above, which usually rise together.")

with st.expander("Full product-lane safety stock table"):
    st.dataframe(
        ss_named[["product_name", "market", "order_region", "shipping_mode",
                  "avg_lead_time", "std_lead_time", "avg_daily_demand",
                  "safety_stock", "reorder_point"]]
        .rename(columns={
            "product_name": "Product Name", "market": "Market", "order_region": "Order Region",
            "shipping_mode": "Shipping Mode", "avg_lead_time": "Avg Lead Time", "std_lead_time": "Std Lead Time",
            "avg_daily_demand": "Avg Daily Demand", "safety_stock": "Safety Stock", "reorder_point": "Reorder Point",
        })
        .sort_values("Safety Stock", ascending=False),
        use_container_width=True, hide_index=True,
    )

st.divider()
st.caption(
    "Data: DataCo Supply Chain Dataset. Macro data: FRED (GASDESW, IR, CPIAUCSL) with offline fallback "
    "data when no API key/network is available. Freight cost and COGS models are documented assumption "
    "sets, not fitted to real carrier rates or supplier costs."
)

