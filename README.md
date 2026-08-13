# investing-backtester

Historical **safe-withdrawal-rate** and **lump-sum-vs-dollar-cost-averaging** backtests on
Robert Shiller's long-run US market data (monthly, Jan 1871 – Sep 2023). Everything runs in
**real (CPI-deflated)** dollars against actual overlapping historical cohorts — no Monte
Carlo, no assumed return distributions.

All code lives in [`swr-team/quant/`](swr-team/quant/).

## What's inside

**Retirement / safe-withdrawal engine**
- A cohort simulation engine with pluggable spending policies — fixed-real (Bengen),
  CAPE-scaled initial withdrawal, and Guyton-Klinger guardrails — plus a SAFEMAX solver
  (the largest withdrawal rate every historical cohort survived).

**Windfall deployment analysis** — how to put a lump sum to work:
- **Time to recover:** the longest a lump sum has ever taken to get back to real breakeven,
  by allocation (100% equity / 80-20 / 68-17-15), with drawdown depths.
- **Lump vs. DCA vs. half-and-half:** the full risk/return frontier — average outcome,
  downside floor, and how often each beats the others — with the cash bucket earning a
  configurable real money-market yield.
- **Dip-accelerated DCA:** scaling the monthly buy up into drawdowns, and a check of how
  often that rule simply reduces to plain DCA when no crash occurs.

## Quick start

```bash
cd swr-team/quant
python3 -m venv venv
./venv/bin/pip install pandas numpy xlrd openpyxl

# 1. build the clean monthly real-return series (already checked in as shiller_monthly.csv)
./venv/bin/python load_data.py       # ie_data.xls -> shiller_monthly.csv (+ validation)

# 2. safe-withdrawal-rate tables
./venv/bin/python run_analysis.py    # headline tables (fixed-real, glidepath, CAPE, GK)
./venv/bin/python supplement.py      # deeper diagnostics

# 3. windfall deployment
./venv/bin/python recovery.py        # time-to-real-breakeven by allocation
./venv/bin/python deploy.py 0.015    # lump vs DCA vs half; arg = cash real yield (default 1.5%)
./venv/bin/python dip_dca.py         # dip-accelerated DCA vs lump / DCA / half
./venv/bin/python dip_calm.py        # how often dip-DCA == plain DCA
```

## Repository layout

| File | What it does |
|---|---|
| `swr-team/quant/ie_data.xls` | Raw Shiller "Irrational Exuberance" data, 1871–2023 (from econ.yale.edu) |
| `swr-team/quant/load_data.py` | Parses the xls into monthly **real total-return** series for stocks & bonds |
| `swr-team/quant/shiller_monthly.csv` | Clean monthly series (P, D, CPI, GS10, CAPE, real stock/bond returns) |
| `swr-team/quant/backtest.py` | Simulation engine + spending policies + SAFEMAX solver |
| `swr-team/quant/run_analysis.py` | Headline safe-withdrawal-rate tables |
| `swr-team/quant/supplement.py` | Diagnostics behind the headline numbers |
| `swr-team/quant/recovery.py` | Lump-sum time-to-recover and drawdown depths |
| `swr-team/quant/deploy.py` | Lump vs. DCA vs. half-and-half deployment |
| `swr-team/quant/dip_dca.py` | Dip-accelerated dollar-cost averaging |
| `swr-team/quant/dip_calm.py` | Dip-DCA vs. plain DCA in calm markets |

## Method notes

- **Real dollars throughout;** portfolios normalized to 1.0. Monthly compounding, annual
  rebalancing to target weights.
- **Stocks** = S&P Composite real total return (dividends reinvested). **Bonds** = a 10-year
  Treasury constant-maturity real total-return approximation built from the GS10 yield
  (coupon accrual + reprice on yield change), validated against Shiller's own bond
  total-return column (monthly-return correlation 1.000, matching 2.44% annualized real).
- **Retirement runs:** annual withdrawal at the start of each year (Bengen convention);
  an H-year retirement funds H withdrawals; success = the portfolio never goes negative.
- **Caveats:** US large-cap only (a survivorship-optimistic sample — no small-cap or
  international); no taxes, fees, or credit spread; overlapping cohorts are not independent.

## Data source

Robert J. Shiller, *Irrational Exuberance* online data —
<http://www.econ.yale.edu/~shiller/data.htm>. To refresh:

```bash
curl -L -A "Mozilla/5.0" http://www.econ.yale.edu/~shiller/data/ie_data.xls -o swr-team/quant/ie_data.xls
```

## Disclaimer

Historical-data analysis for educational purposes only — **not financial advice**. Past
returns do not predict future results; the US historical record is a single, unusually
fortunate sample.
