# Supply Chain Stress Test Simulator

I built this as a decision-support tool that models how three
macroeconomic shocks — diesel price spikes, import price inflation, and
consumer inflation (CPI) — flow through a real e-commerce supply chain
(180K+ orders, the DataCo dataset) and change three concrete business
decisions: which shipping mode to use, how much profit margin survives,
and how much inventory to hold.

For the global trade context that got me thinking about this in the
first place, see `Global_trade_context_intro.md`.

## Project Files (Google Drive)

**https://drive.google.com/drive/folders/194_fSkCtJigzi8pu_f4qRewF7_NljmXJ?usp=drive_link**


## Project Goals
The question I actually wanted to answer: **when a macroeconomic shock
hits — a diesel price spike, an import cost surge, a burst of inflation —
what should a supply chain actually *do* differently?** Most of the
supply chain tools I'd seen treat "macro risk" as something to note in a
report, not something the system actually reacts to. I wanted to close
that gap.

What I set out to do:

1. **Model three distinct macroeconomic channels**, not just gesture at
   "macro risk" as one vague thing:
   - Diesel price → freight cost → which shipping mode to use
   - Import price index → cost of goods sold → profit margin
   - CPI (inflation) → consumer demand → how much inventory to hold
2. **Be honest about what the data can and can't support.** The dataset
   I'm working with has no freight cost field, no COGS field, and only
   about three years of history. Rather than fake precision I didn't
   have, I built documented assumptions where the data couldn't support
   a real claim — and where it *could*, I tested the assumption
   statistically instead of just asserting it.
3. **Make something usable, not just a model.** I wanted a non-technical
   person to be able to open the app, move a slider, and get a real
   answer to "what if diesel spikes 30%?" — not just read a static
   write-up.
4. **Show the full pipeline**, from messy raw data through cleaning,
   analysis, and optimization — not just a polished final chart with
   the messy parts hidden.

### What this isn't

To be clear about scope: this isn't a production forecasting system.
It's not fitted to real carrier rate cards, and I haven't validated it
against live business outcomes. It's a decision-support prototype, built
to reason correctly under real data limitations — see "Methodology"
below for how that shows up in the actual numbers I got.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

No database, no parquet, nothing else to set up — `app.py` reads
`data/dataco_clean.csv` directly. I added a free
[FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) field
in the sidebar for live macro data; without one, it runs on fallback
data that's clearly labeled as such.

If I want to regenerate `data/dataco_clean.csv` from the raw dataset, I
just run all cells in `notebooks/cleaning.ipynb`.

---

## Where the Data Came From

The core dataset is the **DataCo Supply Chain Dataset**, a public
dataset of real (anonymized) e-commerce order data, originally published
on Mendeley Data: https://data.mendeley.com/datasets/8gx2fvg2k6/5

- **180,519 rows** — each row is one order line item (one product within
  one order; a single order can span multiple rows if it has multiple
  products)
- **65,752 unique orders**, **20,652 unique customers**
- **Date range:** January 2015 – January 2018
- **53 original columns**, down to **38** after I cleaned it

For the macro data (diesel prices, import price index, CPI), I'm pulling
from two public U.S. government sources:
- **FRED** (Federal Reserve Economic Data) — `GASDESW` for diesel, `IR`
  for the import price index, `CPIAUCSL` for CPI
- **BLS** (Bureau of Labor Statistics) — real historical CPI-U index
  values, which I used specifically for the department-level elasticity
  backtest, matched to the dataset's own 2015–2018 order dates

### What the fields I actually use represent

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
| `shipping_mode` | Shipping Mode | Standard Class / Second Class / First Class / Same Day — the four service tiers I optimize over |
| `market` | Market | Top-level geography: LATAM, Europe, Pacific Asia, USCA, Africa |
| `order_region` | Order Region | Finer-grained shipping destination within a Market |
| `department_name` | Department Name | Product department (Apparel, Technology, Golf, etc.) — the unit I estimate the elasticity model at |
| `category_name` | Category Name | Finer-grained product category within a department |
| `product_card_id` | Product Card Id | Unique product identifier |
| `product_price` | Product Price | List price |
| `order_item_quantity` | Order Item Quantity | Units of this product in this line item |
| `sales` | Sales | Pre-discount revenue for this line item |
| `order_item_total` | Order Item Total | Post-discount revenue actually charged — the "real" revenue figure I use throughout |
| `order_profit_per_order` | Order Profit Per Order | Realized profit ($) — I use this together with `order_item_total` to back into an implied cost of goods sold, since there's no COGS field directly (see `margin_model.py`) |

The columns I deliberately dropped, and why, are in "Cleaning Steps"
below.

### Things I found about the data quality (not assumed going in)

- **There's no freight cost, weight, or shipping distance field anywhere**
  in this dataset — that's why `optimizer.py` builds a documented cost
  *assumption* model instead of trying to fit one to data that doesn't
  exist.
- **No COGS field exists either** — `margin_model.py` backs one out from
  `order_item_total − order_profit_per_order`.
- **Order volume roughly halves dataset-wide from October 2017 onward.**
  I dug into this and it looks like a truncation artifact in this
  specific export, not a real demand collapse — `elasticity_model.py`
  excludes that window from its regression so it doesn't get misled by
  it.
- **5 of 11 departments** (Technology, Book Shop, Pet Shop, Discs Shop,
  Health and Beauty) only ever have orders *inside* that truncated tail
  window — meaning they have zero months of usable regression history. I
  handle that explicitly rather than letting them silently disappear or
  fall back to a generic default.

---

## Cleaning Steps — What I Did, and Why

I did the cleaning in `notebooks/cleaning.ipynb`, which turns the raw CSV
into `data/dataco_clean.csv`. Every step below is something I checked
against the actual data first — I didn't want to clean things that
didn't need cleaning, or miss things that did.

### 1. I dropped 15 columns, in four categories

**Exact duplicates** (I verified these were 100% identical to another
column before dropping them, not just assumed):
- `Order Customer Id` — identical to `Customer Id`
- `Order Item Cardprod Id` — identical to `Product Card Id`
- `Product Category Id` — identical to `Category Id`
- `Order Item Product Price` — identical to `Product Price`

**Zero information:**
- `Product Description` — 100% null across every row
- `Product Status` — the same single value (0) for every row; zero
  variance means zero analytical value

**Mostly missing:**
- `Order Zipcode` — 86.2% null, too sparse to use

**Fabricated PII / stuff I don't need:**
- `Customer Email`, `Customer Password`, `Customer Fname`, `Customer
  Lname`, `Customer Street` — this is fabricated placeholder data in a
  public dataset, not real people, but I didn't see a reason to carry
  PII-shaped fields through a pipeline even when they're fake
- `Product Image` (a URL string) and `Latitude`/`Longitude` — not used
  anywhere in my analysis

### 2. I stripped whitespace from the text/category columns

This one actually mattered, and I only caught it by looking closely:
`Order Region` had **17,925 rows** and `Department Name` had **362 rows**
with trailing or extra whitespace (e.g. `"Health and Beauty "` vs.
`"Health and Beauty"`). Left alone, that silently fragments any
`groupby()` into duplicate categories — no error gets raised, the
aggregation just quietly produces a wrong answer. I actually caught this
because it was breaking a lookup in `elasticity_model.py` — the "Health
and Beauty" department's prior was never matching because of the
whitespace mismatch, and it was silently falling back to a generic
default every time.

### 3. I converted the date columns from text to real datetime

`order_date` and `shipping_date` came in as plain text strings. I convert
them once here with `pd.to_datetime()` so I'm not re-parsing them on
every downstream calculation. One thing worth noting: since I ended up
using CSV instead of parquet for the cleaned output, `app.py` has to
re-parse these two columns back to datetime every time it loads the
file — CSV doesn't remember column types the way parquet does. Small
tradeoff I accepted for simplicity.

### 4. I renamed everything to snake_case

The raw CSV headers are a mix of spacing, casing, and parenthetical
suffixes (`order date (DateOrders)`, `Days for shipping (real)`). I
renamed everything to consistent `snake_case` so I'm not writing
bracket-and-quote column references everywhere in the code.

### What I checked but decided *not* to change

I think it's just as important to say what I verified was already clean
as what I fixed:

- **Zero exact duplicate rows**
- **Zero cases** of `shipping_date` before `order_date` — the date logic
  checks out
- **Zero** negative or zero `order_item_quantity`, negative
  `product_price`, or invalid (>100%) `order_item_discount_rate`
- **`late_delivery_risk` and `delivery_status` are perfectly consistent**
  with each other — I kept both anyway since I use them differently
  downstream
- **76 orders with profit below -$1,000** — I looked into these and left
  them as real data rather than treating them as errors; they look like
  genuine large-loss or heavily-discounted/fraud-flagged orders

---

## Methodology, in Short

Since the dataset doesn't have real freight cost or COGS data, I built
clear, documented assumptions wherever that data was missing (see the
`optimizer.py` and `margin_model.py` docstrings for exactly what I
assumed and why).

Where I could, I replaced those assumptions with real estimates — built
from historical BLS CPI data and tested for statistical significance
(`elasticity_model.py`). Most department-level effects didn't pass that
test, which tells me the historical window I have just doesn't cover
enough time or enough CPI movement to estimate them reliably. I'm
treating that as a real, useful finding rather than something to hide —
it means the significance gate I built is actually doing its job, not
just rubber-stamping every guess as "data-driven."


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
