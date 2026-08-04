# Supply Chain Stress Test Simulator

A decision-support tool that models how three macroeconomic shocks —
diesel price spikes, import price inflation, and consumer inflation
(CPI) — propagate through a real e-commerce supply chain (180K+ orders,
DataCo dataset) and change three concrete business decisions: shipping
mode selection, profit margin, and inventory safety stock levels.

See `GLOBAL_TRADE_INTRODUCTION.md` for the global trade context this
project responds to, and how the project's own data sits inside it.

---

## Project Files (Google Drive)

**https://drive.google.com/drive/folders/194_fSkCtJigzi8pu_f4qRewF7_NljmXJ?usp=drive_link**


## Project Goals

This project exists to answer a concrete question: **when a macroeconomic
shock hits — a diesel price spike, an import cost surge, or a burst of
inflation — what should a supply chain actually *do* differently?**

Specifically, the goals are to:

1. **Model three distinct, real macroeconomic transmission channels**
   into a real supply chain, rather than treating "macro risk" as one
   vague concept:
   - Diesel price → freight cost → which shipping mode to use
   - Import price index → cost of goods sold → profit margin
   - CPI (inflation) → consumer demand → how much inventory to hold
2. **Be honest about the limits of the data.** The source dataset has no
   freight cost, no COGS, and only a ~3-year window of history. Rather
   than fabricate false precision, the project uses clearly documented
   assumptions where the data can't support a claim, and replaces those
   assumptions with tested statistical estimates only where the data
   actually earns it.
3. **Produce something usable, not just a model.** The end product is an
   interactive tool a non-technical stakeholder could actually use to
   ask "what if diesel spikes 30%?" and get a concrete answer — not just
   a static analysis notebook.
4. **Demonstrate the full pipeline**, from raw, messy source data through
   cleaning, analysis, and optimization, not just a polished final chart.

### What This Is *Not*

To be equally clear about scope: this is not a production forecasting
system, not fitted to real carrier rate cards, and not validated against
live business outcomes. It's a decision-support prototype built to reason
correctly under real data limitations — see "Methodology" below for how
that shows up in the actual numbers.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

No database, no parquet, no setup beyond the above — `app.py` reads
`data/DataCoSupplyChainDataset_clean.csv` directly. Add a free
[FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) in the
sidebar for live macro data; without one, the app runs on labeled
fallback data.

To regenerate `data/dataco_clean.csv` from the raw dataset, run all cells
in `notebooks/cleaning.ipynb`.

---

## Data Lineage

### Where the data came from

The core dataset is the **DataCo Supply Chain Dataset**, a public dataset
of real (anonymized) e-commerce order data, originally published on
Mendeley Data: https://data.mendeley.com/datasets/8gx2fvg2k6/5

- **180,519 rows** — each row is one **order line item** (one product
  within one customer order; a single order can span multiple rows if it
  contains multiple products)
- **65,752 unique orders**, **20,652 unique customers**
- **Date range:** January 2015 – January 2018
- **53 original columns**, reduced to **38** after cleaning (see below)

Macroeconomic data (diesel prices, import price index, CPI) comes from
two public U.S. government sources:
- **FRED** (Federal Reserve Economic Data, Federal Reserve Bank of St.
  Louis) — `GASDESW` (diesel), `IR` (import price index), `CPIAUCSL` (CPI)
- **BLS** (Bureau of Labor Statistics) — real historical CPI-U index
  values used specifically for the department-level elasticity backtest,
  matched to the dataset's own 2015–2018 order date range

### What the critical fields represent

| Field (cleaned name) | Original CSV header | What it means |
|---|---|---|
| `order_id` | Order Id | Identifier for the parent order (one order → 1+ line items) |
| `order_item_id` | (generated) | Identifier for this specific line item |
| `order_date` | order date (DateOrders) | When the customer placed the order |
| `shipping_date` | shipping date (DateOrders) | When the order actually shipped |
| `days_for_shipping_real` | Days for shipping (real) | Actual delivery time, in days — the "ground truth" lead time |
| `days_for_shipment_scheduled` | Days for shipment (scheduled) | Promised/planned delivery time — the SLA target |
| `late_delivery_risk` | Late_delivery_risk | Binary flag: 1 if delivery was late |
| `delivery_status` | Delivery Status | Advance shipping / Late delivery / Shipping on time / Shipping canceled |
| `shipping_mode` | Shipping Mode | Standard Class / Second Class / First Class / Same Day — the four service tiers this project optimizes over |
| `market` | Market | Top-level geography: LATAM, Europe, Pacific Asia, USCA, Africa |
| `order_region` | Order Region | Finer-grained shipping destination region within a Market |
| `department_name` | Department Name | Product department (Apparel, Technology, Golf, etc.) — the unit the elasticity model is estimated at |
| `category_name` | Category Name | Finer-grained product category within a department |
| `product_card_id` | Product Card Id | Unique product identifier |
| `product_price` | Product Price | List price of the product |
| `order_item_quantity` | Order Item Quantity | Units of this product in this line item |
| `sales` | Sales | Pre-discount revenue for this line item |
| `order_item_total` | Order Item Total | Post-discount revenue actually charged — the "real" revenue figure used throughout this project |
| `order_profit_per_order` | Order Profit Per Order | Realized profit ($) — used together with `order_item_total` to back into an **implied cost of goods sold**, since no COGS field exists directly (see `margin_model.py`) |

**Fields that exist in the raw data but were deliberately dropped** (with
reasons) are documented in "Cleaning Steps" below.

### Known data quality characteristics (found, not assumed)

- **No freight cost, weight, or shipping distance field exists anywhere**
  in the dataset — this is why `optimizer.py` builds a documented cost
  *assumption* model instead of fitting one.
- **No COGS field exists** — `margin_model.py` backs one out from
  `order_item_total − order_profit_per_order`.
- **Order volume roughly halves dataset-wide from October 2017 onward** —
  a known truncation artifact of this specific export, not a real demand
  collapse. `elasticity_model.py` explicitly excludes this window from
  its regression to avoid being misled by it.
- **5 of 11 departments** (Technology, Book Shop, Pet Shop, Discs Shop,
  Health and Beauty) only ever have orders *inside* that truncated tail
  window, meaning they have zero months of usable regression history —
  handled explicitly rather than silently dropped or defaulted.

---

## Cleaning Steps — What Was Done, and Why

Cleaning is performed in `notebooks/cleaning.ipynb`, which produces
`data/dataco_clean.csv` from the raw CSV. Every step below was decided by
first checking the actual data, not assumed.

### 1. Dropped 15 columns, in four categories

**Exact duplicates** (verified as 100% identical to another column before
dropping — not assumed):
- `Order Customer Id` — identical to `Customer Id`
- `Order Item Cardprod Id` — identical to `Product Card Id`
- `Product Category Id` — identical to `Category Id`
- `Order Item Product Price` — identical to `Product Price`

**Zero information:**
- `Product Description` — 100% null across every row
- `Product Status` — a single constant value (0) for every row; zero
  variance means zero analytical value

**Mostly missing:**
- `Order Zipcode` — 86.2% null, too sparse to be usable

**Fabricated PII / genuinely unused:**
- `Customer Email`, `Customer Password`, `Customer Fname`, `Customer
  Lname`, `Customer Street` — fabricated placeholder data in this public
  dataset (not real people), with no legitimate reason to carry
  PII-shaped fields through a pipeline even when fake
- `Product Image` (a URL string) and `Latitude`/`Longitude` (customer-level
  coordinates) — not used anywhere in this project's analysis

### 2. Stripped whitespace from text/category columns

**Why this mattered, concretely:** `Order Region` had **17,925 rows** and
`Department Name` had **362 rows** with trailing or extra whitespace
(e.g. `"Health and Beauty "` vs. `"Health and Beauty"`). Left uncleaned,
this silently fragments any `groupby()` into duplicate categories — no
error is raised, the aggregation just quietly produces a wrong answer.
This was caught by direct inspection: before this fix, the `Health and
Beauty` department's documented prior (elasticity, import exposure) was
*never actually matching* in `elasticity_model.py`, because the
whitespace mismatch meant the lookup silently fell through to the
generic default every time.

### 3. Converted date columns from text to real datetime

`order_date` and `shipping_date` were stored as plain text strings in the
raw CSV. Converted once, here, using `pd.to_datetime()`, so every
downstream lead-time and monthly-aggregation calculation doesn't have to
re-parse them on every use. (Note: since the cleaned file is CSV, not
parquet, `app.py` re-parses these two columns back to datetime on load —
CSV doesn't preserve dtypes the way parquet does.)

### 4. Renamed all columns to snake_case

The raw CSV headers mix spacing, casing, and parenthetical suffixes (e.g.
`order date (DateOrders)`, `Days for shipping (real)`). Renamed
mechanically to consistent `snake_case` (`order_date`,
`days_for_shipping_real`) so the entire codebase can reference columns
consistently without bracket-and-quote syntax everywhere.

### What was checked, but deliberately *not* changed

It's just as important to document what was verified clean as what was
fixed — cleaning steps should be evidence-based in both directions:

- **Zero exact duplicate rows** in the dataset
- **Zero cases** of `shipping_date` occurring before `order_date`
  (internal date logic is consistent)
- **Zero** negative or zero `order_item_quantity`, negative
  `product_price`, or invalid (>100%) `order_item_discount_rate` values
- **`late_delivery_risk` and `delivery_status` are perfectly consistent**
  with each other (every "Late delivery" row has risk=1, every other
  status has risk=0) — a redundant pair, but both were kept since they're
  used differently downstream
- **76 orders with profit below -$1,000** were investigated and left as
  real data (not treated as errors) — these look like genuine large-loss
  or heavily-discounted/fraud-flagged orders, not data corruption

---

## Methodology Summary

The dataset doesn't include real freight cost or COGS data, so instead of
faking precision, the project uses clear, documented assumptions
wherever that data is missing (see `optimizer.py` and `margin_model.py`
docstrings for the exact assumption sets).

Where possible, those assumptions are replaced with real estimates —
built from historical BLS CPI data and tested for statistical
significance (`elasticity_model.py`). Most department-level effects
don't pass that significance test, which shows the historical data
doesn't cover enough time or enough CPI movement to estimate them
reliably. That result is treated as a real, useful finding — evidence
the significance gate is working correctly — not a flaw to hide.

---

## Project Structure

```
├── app.py                       # Streamlit dashboard (entry point)
├── analytics.py                  # Lead-time variance + dynamic safety stock
├── optimizer.py                  # LP: cost-minimizing shipping mode per lane
├── margin_model.py                # Import price -> COGS / margin impact model
├── elasticity_model.py            # CPI demand elasticity, per-department + tested
├── fred_client.py                  # FRED + NY Fed GSCPI client, with offline fallback
├── data/
│   └── DatacosupplyChainDataset_clean.csv            # Cleaned dataset (38 cols, snake_case)
│    cleaning.ipynb               # Raw CSV -> cleaned CSV, narrated
│   
├── GLOBAL_TRADE_INTRODUCTION.md       # Global trade context + project relevan
└── requirements.txt
```

## Deliverables

- Interactive Streamlit dashboard (`app.py`)
- Jupyter notebook walkthrough (`notebooks/`)

## License

MIT

## Author

Nancy Mbuta
