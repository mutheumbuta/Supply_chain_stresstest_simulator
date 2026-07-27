# Supply_chain_stresstest_simulator
## Overview

This project is a decision-support tool that models how three macroeconomic shocks — diesel price spikes, import price inflation, and consumer inflation (CPI) — propagate through a real e-commerce supply chain (180K+ orders, DataCo dataset) and change three concrete business decisions:

- **Shipping mode selection** — solved via linear programming
- **Profit margin** — modeled via a cost-of-goods shock
- **Inventory safety stock levels** — computed via a demand-elasticity-adjusted version of the standard safety stock formula

### Methodology
The dataset doesn't include real freight cost data, so instead of faking precision, the project uses clear, documented assumptions wherever that data is missing.

Where possible, those assumptions are replaced with real estimates — built from historical BLS/EIA data and tested for statistical significance. Most department-level effects don't pass that test, which shows the historical data doesn't cover enough time to estimate them reliably. That's treated as a real, useful finding, not a flaw to hide.

### Deliverables

- Interactive Streamlit dashboard
- Jupyter notebook walkthrough
- Documented Postgres schema for the production path
