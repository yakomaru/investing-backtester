"""
recovery.py -- "invested at the worst possible time, how long to get back to even?"

A one-time LUMP SUM invested and left alone (no withdrawals), dividends reinvested,
measured in REAL (CPI-deflated) terms. For each start month we find how many months
until the real value first returns to the amount invested (real breakeven), for a few
fixed-weight, monthly-rebalanced allocations relevant to the plan:

  100/0/0   pure equity (worst case; what a windfall dumped straight into stocks looks like)
  80/20/0   the plan's invested sleeve (where 85% of a windfall lands)
  68/17/15  whole-portfolio deployment incl. the capped cash bucket (cash = 0% real)

Right-censoring: start months that had not recovered by the end of the data (Sep 2023)
are reported as ">= X yrs (not yet)".
"""
import numpy as np
import pandas as pd

df = pd.read_csv("shiller_monthly.csv")
df = df.dropna(subset=["stock_ret_real", "bond_ret_real"]).reset_index(drop=True)
rs = df["stock_ret_real"].values
rb = df["bond_ret_real"].values
rc = np.zeros(len(df))          # cash: 0% real (conservative; matches plan's cash_real=0)
ym = [(int(y), int(m)) for y, m in zip(df["year"], df["month"])]
n = len(df)


def label(i):
    y, m = ym[i]
    return f"{y}-{m:02d}"


def wealth_index(ws, wb, wc):
    """Fixed-weight, monthly-rebalanced real wealth index (starts at 1.0)."""
    r = ws * rs + wb * rb + wc * rc
    w = np.empty(n + 1)
    w[0] = 1.0
    w[1:] = np.cumprod(1.0 + r)
    return w  # w[t] = real value at start of month t (w[0..n])


def recovery_months(w):
    """For each start t, months until real value first >= value at t. -1 if censored."""
    out = np.full(n, -1, dtype=int)
    for t in range(n):
        base = w[t]
        # first u > t with w[u] >= base
        future = w[t + 1:]
        hit = np.argmax(future >= base) if np.any(future >= base) else -1
        out[t] = (hit + 1) if hit != -1 or (len(future) and future[0] >= base) else -1
        if not np.any(future >= base):
            out[t] = -1
    return out


def max_drawdown_from(w, t):
    """Deepest real drawdown experienced starting from month t (over remaining data)."""
    path = w[t:]
    peak = np.maximum.accumulate(path)
    dd = path / peak - 1.0
    return dd.min()


ALLOCS = [
    ("100/0/0  pure equity", 1.00, 0.00, 0.00),
    ("80/20/0  plan sleeve", 0.80, 0.20, 0.00),
    ("68/17/15 whole portfolio", 0.68, 0.17, 0.15),
]

print("=" * 72)
print("LONGEST TIME TO REAL BREAKEVEN for a lump sum (1871-2023, dividends reinvested)")
print("=" * 72)
for name, ws, wb, wc in ALLOCS:
    w = wealth_index(ws, wb, wc)
    rec = recovery_months(w)
    # worst *completed* recovery
    completed = [(t, rec[t]) for t in range(n) if rec[t] > 0]
    worst_t, worst_m = max(completed, key=lambda x: x[1])
    # censored starts (never recovered in-sample)
    censored = [t for t in range(n) if rec[t] == -1]
    print(f"\n{name}")
    print(f"  worst completed recovery : {worst_m/12:5.1f} yrs  (invested {label(worst_t)})")
    if censored:
        first_c = min(censored)
        span = (n - first_c)
        print(f"  still under water at end : starts {label(first_c)} .. onward "
              f"({len(censored)} months); earliest has been down {span/12:.1f} yrs and counting")


print("\n" + "=" * 72)
print("THE CLASSIC WORST ENTRY POINTS -- years to real breakeven")
print("=" * 72)
anchors = ["1929-8", "1929-9", "1906-1", "1966-1", "1968-11", "1972-12", "2000-3", "2000-8", "2007-10"]
idx = {label(i).replace("-0", "-").lstrip(): i for i in range(n)}
# build a simple lookup by "YYYY-M"
lut = {}
for i in range(n):
    y, m = ym[i]
    lut[f"{y}-{m}"] = i
for name, ws, wb, wc in ALLOCS:
    w = wealth_index(ws, wb, wc)
    rec = recovery_months(w)
    print(f"\n{name}")
    for a in anchors:
        if a not in lut:
            continue
        t = lut[a]
        r = rec[t]
        dd = max_drawdown_from(w, t)
        rt = f"{r/12:4.1f} yrs" if r > 0 else ">= not yet"
        print(f"  invest {a:>8}  ->  breakeven {rt:>10}   (worst real drawdown {dd*100:5.0f}%)")


print("\n" + "=" * 72)
print("HISTOGRAM: how long is a lump sum under water? (share of all start months)")
print("=" * 72)
buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 99)]
for name, ws, wb, wc in ALLOCS:
    w = wealth_index(ws, wb, wc)
    rec = recovery_months(w)
    yrs = np.array([rec[t] / 12 if rec[t] > 0 else 99.0 for t in range(n)])
    print(f"\n{name}   (median {np.median(yrs[yrs<99]):.1f} yrs to breakeven)")
    for lo, hi in buckets:
        share = np.mean((yrs >= lo) & (yrs < hi)) * 100
        tag = f"{lo}-{hi} yrs" if hi < 99 else f"{lo}+ yrs / never"
        bar = "#" * int(round(share / 2))
        print(f"  {tag:>14}: {share:5.1f}%  {bar}")
