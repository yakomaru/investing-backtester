"""
dip_calm.py -- when there's no crash, does dip-DCA collapse to plain DCA?
For every 12-month deployment window, find the deepest drawdown (from running peak) the
S&P touched. If it never fell >=5%, the multiplier never fired -> deployment is byte-for-byte
plain DCA. We report how often that happens, and how tiny the divergence is in calm windows.
Monthly-average prices understate intra-month dips, so real-world triggers a bit more often.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("shiller_monthly.csv")
df = df.dropna(subset=["stock_ret_real", "bond_ret_real"]).reset_index(drop=True)
sleeve_r = 0.80 * df["stock_ret_real"].values + 0.20 * df["bond_ret_real"].values
P = df["P"].values
n = len(df)
WINDOW, STEP = 12, 0.05


def max_dd_window(start):
    peak = P[start]
    worst = 0.0
    for k in range(WINDOW):
        peak = max(peak, P[start + k])
        worst = max(worst, 1.0 - P[start + k] / peak)
    return worst


def wealth(start, months, dip):
    sleeve, cash, base, peak = 0.0, 1.0, 1.0 / WINDOW, P[start]
    for k in range(months):
        if k < WINDOW and cash > 1e-12:
            peak = max(peak, P[start + k])
            m = 1 + int(max(0.0, 1 - P[start + k] / peak) / STEP) if dip else 1
            buy = min(m * base, cash)
            sleeve += buy
            cash -= buy
        sleeve *= (1 + sleeve_r[start + k])
    return sleeve + cash


starts = [t for t in range(n) if t + 120 < n]
dds = np.array([max_dd_window(t) for t in starts])

print("Deepest drawdown the S&P touched during the 12-month deploy window:")
for lo, hi, tag in [(0, .05, "never fell 5%  -> IDENTICAL to plain DCA"),
                    (.05, .10, "5-10% dip     -> some 2x buys"),
                    (.10, .20, "10-20% drop   -> 2x/3x buys"),
                    (.20, 1.0, "20%+ crash    -> heavy front-loading")]:
    share = np.mean((dds >= lo) & (dds < hi)) * 100
    print(f"  {tag:<44}{share:5.1f}% of windows")

# divergence in calm windows (no >=10% drop) at 10-yr horizon
calm = [t for t in starts if max_dd_window(t) < 0.10]
d = np.array([wealth(t, 120, True) for t in calm])
p = np.array([wealth(t, 120, False) for t in calm])
diff = (d - p) / p * 100
print(f"\nAmong the {len(calm)/len(starts)*100:.0f}% of windows with no >=10% drop, "
      f"dip-DCA vs plain DCA (10-yr real wealth):")
print(f"  identical (no 5% dip at all): {np.mean(dds[np.array([max_dd_window(t)<0.10 for t in starts])] < 0.05)*100:.0f}% of these")
print(f"  mean difference : {diff.mean():+.2f}%")
print(f"  max difference  : {diff.max():+.2f}% / {diff.min():+.2f}%")
