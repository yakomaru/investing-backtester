# SWR backtests — quant workbench

Historical safe-withdrawal-rate backtests on Robert Shiller's long-run US data.

## Rerun

```bash
cd swr-team/quant
python3 -m venv venv
./venv/bin/pip install pandas numpy xlrd openpyxl
# ie_data.xls is already here; to refresh:  curl -L -A "Mozilla/5.0" \
#   http://www.econ.yale.edu/~shiller/data/ie_data.xls -o ie_data.xls
./venv/bin/python load_data.py      # -> shiller_monthly.csv (+ validation prints)
./venv/bin/python run_analysis.py   # headline tables (fixed-real, glidepath, CAPE, GK)
./venv/bin/python supplement.py     # GK 1966 trace, CAPE failure diagnosis, safety-matched search
```

## Files

| file | what |
|---|---|
| `ie_data.xls` | raw Shiller "Irrational Exuberance" data, Jan 1871–Sep 2023 (from econ.yale.edu) |
| `load_data.py` | parses the xls, builds monthly REAL total-return series for stocks & bonds |
| `shiller_monthly.csv` | clean monthly series (P, D, CPI, GS10, CAPE, real stock/bond returns) |
| `backtest.py` | simulation engine + spending policies (FixedReal, CapeInitial, GuytonKlinger) |
| `run_analysis.py` | all headline tables |
| `supplement.py` | deeper diagnostics behind the headline numbers |

## Method notes

- Real (CPI-deflated) dollars throughout. Portfolio normalized to 1.0.
- Annual real withdrawals at the START of each year; monthly compounding within the year;
  rebalance to target weights annually. H-year retirement = H withdrawals; success = all
  funded (portfolio never negative).
- Stocks = S&P Composite real total return (divs reinvested). Bonds = 10y Treasury
  constant-maturity real total return **approximation** from the GS10 yield (coupon +
  reprice on yield change). Validated against Shiller's own bond TR column: monthly-return
  correlation 1.000, identical 2.44% annualized real return.
- Caveats: US large-cap only (no small/international — the survivorship-optimistic market);
  no taxes, fees, or credit spread; overlapping cohorts are not independent.
