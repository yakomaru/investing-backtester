"""
dip_dca.py -- "12-month DCA, but double the monthly buy if the market is down 5%,
triple if down 10%, and so on."

Multiplier m = 1 + floor(drawdown / 5%), applied to the base 1/12 monthly tranche, but
never more than the cash you have left (so a deep crash just deploys everything at once).
'Drawdown' = decline in the S&P price from its running peak since deployment began
(the everyday "market is down X%" meaning). Undeployed cash earns 0% real (user's case).
Deploys into the 80/20 sleeve. We also test a stricter trigger = down X% from the START
price, and compare to plain DCA-12, HALF-12, and LUMP.

Reported at 5- and 10-yr horizons across all 1871-2023 starts, plus the classic crashes.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("shiller_monthly.csv")
df = df.dropna(subset=["stock_ret_real", "bond_ret_real"]).reset_index(drop=True)
rs, rb = df["stock_ret_real"].values, df["bond_ret_real"].values
sleeve_r = 0.80 * rs + 0.20 * rb
P = df["P"].values                      # S&P price, for the "market is down X%" trigger
ym = [(int(y), int(m)) for y, m in zip(df["year"], df["month"])]
n = len(df)
CASH_R = 0.0                            # 0% real cash (money market ~= inflation)
WINDOW = 12
STEP = 0.05                             # every 5% down => +1x


def label(i):
    y, m = ym[i]
    return f"{y}-{m:02d}"


def wealth_static(start, months, sched):
    """Fixed schedule (LUMP / HALF-12 / DCA-12): dict k->fraction."""
    sleeve, cash = 0.0, 1.0
    for k in range(months):
        move = sched.get(k, 0.0)
        sleeve += move
        cash -= move
        sleeve *= (1 + sleeve_r[start + k])
        cash *= (1 + CASH_R)
    return sleeve + cash


def wealth_dip(start, months, ref="peak"):
    """Dip-accelerated DCA over WINDOW months, then hold to `months`."""
    sleeve, cash = 0.0, 1.0
    base = 1.0 / WINDOW
    peak = P[start]
    p0 = P[start]
    for k in range(months):
        if k < WINDOW and cash > 1e-12:
            px = P[start + k]
            peak = max(peak, px)
            anchor = peak if ref == "peak" else p0
            dd = max(0.0, 1.0 - px / anchor)
            m = 1 + int(dd / STEP)          # 5% down ->2x, 10% ->3x, ...
            buy = min(m * base, cash)
            sleeve += buy
            cash -= buy
        sleeve *= (1 + sleeve_r[start + k])
        cash *= (1 + CASH_R)
    return sleeve + cash


SCHED = {
    "LUMP":    {0: 1.0},
    "HALF-12": {**{0: 0.5}, **{k: 0.5 / 12 for k in range(1, 13)}},
    "DCA-12":  {k: 1 / 12 for k in range(12)},
}

for H in (5, 10):
    months = H * 12
    starts = [t for t in range(n) if t + months < n]
    print("=" * 82)
    print(f"REAL wealth {H} yrs after deploying $1  (all {ym[0][0]}-{ym[-1][0]} starts, 0% real cash)")
    print("=" * 82)
    print(f"{'strategy':<22}{'mean':>7}{'median':>8}{'5th pct':>9}{'worst':>7}"
          f"{'  P>lump':>9}{'  P>DCA12':>10}")
    res = {}
    for name, s in SCHED.items():
        res[name] = np.array([wealth_static(t, months, s) for t in starts])
    res["DIP-DCA (peak ref)"] = np.array([wealth_dip(t, months, "peak") for t in starts])
    res["DIP-DCA (start ref)"] = np.array([wealth_dip(t, months, "start") for t in starts])
    lump, dca = res["LUMP"], res["DCA-12"]
    for name in ("LUMP", "HALF-12", "DCA-12", "DIP-DCA (peak ref)", "DIP-DCA (start ref)"):
        v = res[name]
        pl = "" if name == "LUMP" else f"{np.mean(v >= lump)*100:5.1f}%"
        pd_ = "" if name == "DCA-12" else f"{np.mean(v >= dca)*100:5.1f}%"
        print(f"{name:<22}{v.mean():>7.2f}{np.median(v):>8.2f}"
              f"{np.percentile(v,5):>9.2f}{v.min():>7.2f}{pl:>9}{pd_:>10}")
    print()

print("=" * 82)
print("CLASSIC CRASH ENTRIES: real value 10 yrs later, $100k deployed")
print("=" * 82)
lut = {f"{y}-{m}": i for i, (y, m) in enumerate(ym)}
anchors = {"1929-9": "1929 crash", "1973-1": "1973 stagflation",
           "2000-8": "2000 dot-com", "2007-10": "2007 GFC", "1987-8": "1987 crash"}
months = 120
cols = ["LUMP", "DCA-12", "DIP-DCA (peak ref)", "DIP-DCA (start ref)"]
print(f"{'entry':<24}" + "".join(f"{c.split(' ')[0][:9]:>11}"
      if 'DIP' not in c else f"{'DIP-'+c.split('(')[1][:4]:>11}" for c in cols))
for key, nm in anchors.items():
    t = lut.get(key)
    if t is None or t + months >= n:
        continue
    row = []
    for c in cols:
        if c.startswith("DIP"):
            ref = "peak" if "peak" in c else "start"
            v = wealth_dip(t, months, ref) * 100_000
        else:
            v = wealth_static(t, months, SCHED[c]) * 100_000
        row.append(f"${v/1000:6.0f}k")
    print(f"{nm+' ('+key+')':<24}" + "".join(f"{x:>11}" for x in row))
