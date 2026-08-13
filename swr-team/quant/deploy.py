"""
deploy.py -- how to put a windfall into the 80/20 sleeve: lump vs DCA vs half-and-half.

For every historical start month we deploy $1 into the plan's 80/20 stock/bond sleeve on
different schedules; undeployed money waits in cash (0% real, matching the plan engine).
We then look at REAL wealth a fixed horizon later (5 and 10 yrs from the start), across all
start months, to see the classic tradeoff: lump = highest average but worst floor; DCA =
lower average, higher floor; half-and-half sits in between.

Strategies:
  LUMP      : 100% deployed at month 0
  HALF-12   : 50% at month 0, remaining 50% spread over months 1..12
  DCA-12    : 1/12 per month over months 0..11
  DCA-24    : 1/24 per month over months 0..23
"""
import numpy as np
import pandas as pd

df = pd.read_csv("shiller_monthly.csv")
df = df.dropna(subset=["stock_ret_real", "bond_ret_real"]).reset_index(drop=True)
rs, rb = df["stock_ret_real"].values, df["bond_ret_real"].values
sleeve_r = 0.80 * rs + 0.20 * rb          # monthly REAL return of the 80/20 sleeve
ym = [(int(y), int(m)) for y, m in zip(df["year"], df["month"])]
n = len(df)

# High-yield money market REAL return while cash waits to be deployed.
# 0% = cash exactly tracks inflation; +1.5% ~ today's HY money market (nom ~4.3% - CPI ~2.8%).
import sys
CASH_REAL_ANNUAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
CASH_R = (1 + CASH_REAL_ANNUAL) ** (1 / 12) - 1
print(f"[assumption] undeployed cash earns {CASH_REAL_ANNUAL*100:+.1f}% REAL / yr "
      f"({CASH_R*100:+.3f}%/mo)\n")


def schedule(kind):
    """dict month_offset -> fraction of the $1 deployed that month."""
    if kind == "LUMP":
        return {0: 1.0}
    if kind == "HALF-12":
        s = {0: 0.5}
        for k in range(1, 13):
            s[k] = 0.5 / 12
        return s
    if kind == "DCA-12":
        return {k: 1 / 12 for k in range(12)}
    if kind == "DCA-24":
        return {k: 1 / 24 for k in range(24)}
    raise ValueError(kind)


def terminal_wealth(start, months, sched):
    """Real value at `start+months` of $1 deployed on `sched` into the sleeve."""
    sleeve = 0.0
    cash = 1.0
    for k in range(months):
        # deploy this month's tranche first, then grow both buckets this month
        move = sched.get(k, 0.0)  # fraction of original $1
        sleeve += move
        cash -= move
        sleeve *= (1.0 + sleeve_r[start + k])
        cash *= (1.0 + CASH_R)     # money market earns its real yield while waiting
    return sleeve + cash


def label(i):
    y, m = ym[i]
    return f"{y}-{m:02d}"


for H in (5, 10):
    months = H * 12
    print("=" * 74)
    print(f"REAL wealth {H} years after deploying $1  (across all {ym[0][0]}-{ym[-1][0]} starts)")
    print("=" * 74)
    print(f"{'strategy':<10}{'mean':>8}{'median':>8}{'5th pct':>9}{'worst':>8}"
          f"{'  worst start':>16}{'  P(beat lump)':>15}")
    lump_vals = None
    starts = [t for t in range(n) if t + months < n]
    for kind in ("LUMP", "HALF-12", "DCA-12", "DCA-24"):
        sched = schedule(kind)
        vals = np.array([terminal_wealth(t, months, sched) for t in starts])
        if kind == "LUMP":
            lump_vals = vals
            pbeat = ""
        else:
            pbeat = f"{np.mean(vals >= lump_vals) * 100:5.1f}%"
        worst_i = starts[int(np.argmin(vals))]
        print(f"{kind:<10}{vals.mean():>8.2f}{np.median(vals):>8.2f}"
              f"{np.percentile(vals, 5):>9.2f}{vals.min():>8.2f}"
              f"{label(worst_i):>16}{pbeat:>15}")
    print()

# Focused view: what half-and-half saves you at the historically worst entries.
print("=" * 74)
print("WORST-CASE ENTRIES: real wealth 10 yrs later, $100k deployed")
print("=" * 74)
anchors = {"1929-9": "1929 crash", "1973-1": "1973 stagflation",
           "2000-8": "2000 dot-com", "2007-10": "2007 GFC"}
lut = {f"{y}-{m}": i for i, (y, m) in enumerate(ym)}
months = 120
print(f"{'entry':<20}{'LUMP':>10}{'HALF-12':>10}{'DCA-12':>10}{'DCA-24':>10}")
for key, name in anchors.items():
    t = lut.get(key)
    if t is None or t + months >= n:
        continue
    row = []
    for kind in ("LUMP", "HALF-12", "DCA-12", "DCA-24"):
        v = terminal_wealth(t, months, schedule(kind)) * 100_000
        row.append(f"${v/1000:6.0f}k")
    print(f"{name+' ('+key+')':<20}" + "".join(f"{x:>10}" for x in row))
