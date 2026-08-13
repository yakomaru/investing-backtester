"""
run_analysis.py -- runs every SWR experiment and prints results for the panel.
Run: ./venv/bin/python run_analysis.py
"""
import numpy as np
import backtest as bt

def pct(x): return f"{x*100:.2f}%"

ALLOCS = {"50/50": 0.50, "60/40": 0.60, "75/25": 0.75, "100/0": 1.00}

print("=" * 78)
print("SHILLER US DATA  Jan 1871 - Sep 2023  | real total returns")
print("Stocks = S&P Composite (divs reinvested) | Bonds = 10y Treasury approx")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. FIXED-REAL: 30-year, classic allocations
# ---------------------------------------------------------------------------
H = 30
sy30 = bt.cohort_start_years(H)
print(f"\n### FIXED-REAL WITHDRAWALS, {H}-YR HORIZON  "
      f"(start years {sy30[0]}-{sy30[-1]}, n={len(sy30)} cohorts)\n")
print(f"{'alloc':7} {'succ@4%':>8} {'SAFEMAX':>8} {'worst yr':>9}   worst-5 (yr:max sustainable WR %)")
res30 = {}
for name, w in ALLOCS.items():
    r = bt.safemax_and_worst(H, bt.static_w(w), sy30)
    res30[name] = r
    print(f"{name:7} {pct(r['succ4']):>8} {pct(r['safemax']):>8} {r['worst_year']:>9}   {r['worst5']}")

# ---------------------------------------------------------------------------
# 2. FIXED-REAL: 40-year horizon (early retirement)
# ---------------------------------------------------------------------------
H = 40
sy40 = bt.cohort_start_years(H)
print(f"\n### FIXED-REAL WITHDRAWALS, {H}-YR HORIZON  "
      f"(start years {sy40[0]}-{sy40[-1]}, n={len(sy40)} cohorts)\n")
print(f"{'alloc':7} {'succ@4%':>8} {'SAFEMAX':>8} {'worst yr':>9}   worst-5")
res40 = {}
for name, w in ALLOCS.items():
    r = bt.safemax_and_worst(H, bt.static_w(w), sy40)
    res40[name] = r
    print(f"{name:7} {pct(r['succ4']):>8} {pct(r['safemax']):>8} {r['worst_year']:>9}   {r['worst5']}")

# ---------------------------------------------------------------------------
# 3. EQUITY GLIDEPATH vs STATIC (30-yr)
# ---------------------------------------------------------------------------
H = 30
print(f"\n### RISING EQUITY GLIDEPATH vs STATIC, {H}-YR\n")
print(f"{'strategy':22} {'succ@4%':>8} {'SAFEMAX':>8} {'worst yr':>9}")
glides = {
    "static 60/40":            bt.static_w(0.60),
    "static 75/25":            bt.static_w(0.75),
    "glide 60->100 / 15yr":    bt.glidepath(0.60, 1.00, 15),
    "glide 30->70 / 15yr":     bt.glidepath(0.30, 0.70, 15),  # Pfau-Kitces style low->moderate
}
for name, wf in glides.items():
    r = bt.safemax_and_worst(H, wf, sy30)
    print(f"{name:22} {pct(r['succ4']):>8} {pct(r['safemax']):>8} {r['worst_year']:>9}")

# ---------------------------------------------------------------------------
# 4. CAPE-BASED INITIAL WITHDRAWAL RATE (30-yr)
#    wr0 = a + b/CAPE, then fixed-real. Measures whether variable *initial*
#    spending beats a flat 4% on BOTH success and average lifetime spending.
# ---------------------------------------------------------------------------
H = 30
print(f"\n### CAPE-BASED INITIAL WR  (wr0 = a + b/CAPE, then fixed-real), {H}-YR, 60/40\n")
wf = bt.static_w(0.60)
cape_variants = [
    ("a=1.5%,b=0.5",        0.015, 0.5, None),
    ("a=1.5%,b=0.5 cap6%",  0.015, 0.5, 0.06),
    ("a=1.0%,b=0.5",        0.010, 0.5, None),
    ("a=2.0%,b=0.4",        0.020, 0.4, None),
]
print(f"{'variant':20} {'succ':>6} {'avg wr0':>8} {'min wr0':>8} {'max wr0':>8} "
      f"{'avg spend':>10} {'#fail':>6}")
for label, a, b, cap in cape_variants:
    wr0s, succ, spends, fails = [], 0, [], []
    for y in sy30:
        pol = bt.CapeInitial(a, b, cap=cap)
        r = bt.simulate(bt.jan_index(y), H, wf, pol)
        wr0s.append(pol.initial_wr)
        succ += r["success"]
        # average real spending over the retirement (per $1 initial)
        spends.append(r["spend_path"].mean())
        if not r["success"]:
            fails.append(y)
    print(f"{label:20} {succ}/{len(sy30):<3} {pct(np.mean(wr0s)):>8} "
          f"{pct(np.min(wr0s)):>8} {pct(np.max(wr0s)):>8} {pct(np.mean(spends)):>10} "
          f"{len(fails):>6}  fails={fails}")

# baseline: flat 4% and flat SAFEMAX average spend for comparison
for wr in [0.04]:
    spends, succ = [], 0
    for y in sy30:
        r = bt.simulate(bt.jan_index(y), H, wf, bt.FixedReal(wr))
        spends.append(r["spend_path"].mean()); succ += r["success"]
    print(f"{'flat 4.0% (ref)':20} {succ}/{len(sy30):<3} {'4.00%':>8} {'4.00%':>8} "
          f"{'4.00%':>8} {pct(np.mean(spends)):>10} {0:>6}")

# what did CAPE allow in the cheapest vs richest markets?
print("\n  Initial WR the a=1.5%,b=0.5 rule produced at CAPE extremes:")
capevals = [(y, bt.CAPE[bt.jan_index(y)]) for y in sy30]
capevals = [(y, c) for y, c in capevals if not np.isnan(c)]
cheap = sorted(capevals, key=lambda t: t[1])[:3]
rich = sorted(capevals, key=lambda t: t[1], reverse=True)[:3]
for y, c in cheap:
    print(f"    {y}: CAPE={c:5.1f}  -> wr0={pct(0.015+0.5/c)}")
for y, c in rich:
    print(f"    {y}: CAPE={c:5.1f}  -> wr0={pct(0.015+0.5/c)}")

# ---------------------------------------------------------------------------
# 5. GUYTON-KLINGER GUARDRAILS (30-yr, 60/40) starting at 5%
# ---------------------------------------------------------------------------
H = 30
print(f"\n### GUYTON-KLINGER GUARDRAILS, {H}-YR, 60/40, start 5.0%\n")
wf = bt.static_w(0.60)
FLAT4 = 0.04  # the 4%-rule real spending level (per $1 initial)
for start_wr in [0.05, 0.055]:
    succ = 0
    min_spend_frac = []     # worst real spending / initial real spending, per cohort
    below4_years = []       # fraction of years spending < 4%-rule level, per cohort
    ever_below4 = 0
    end_spend = []          # final-year spending / initial spending
    for y in sy30:
        pol = bt.GuytonKlinger(start_wr)
        r = bt.simulate(bt.jan_index(y), H, wf, pol)
        succ += r["success"]
        sp = r["spend_path"]
        s0 = sp[0]
        min_spend_frac.append(sp.min() / s0)
        below4 = np.mean(sp < FLAT4)
        below4_years.append(below4)
        if (sp < FLAT4).any():
            ever_below4 += 1
        end_spend.append(sp[-1] / s0)
    min_spend_frac = np.array(min_spend_frac)
    below4_years = np.array(below4_years)
    print(f"start {pct(start_wr)}: success={succ}/{len(sy30)}  "
          f"({pct(succ/len(sy30))})")
    print(f"   worst-cohort min real spend = {pct(min_spend_frac.min())} of its own start "
          f"(i.e., cut to {min_spend_frac.min()*start_wr*100:.2f}% of initial portfolio)")
    print(f"   avg-cohort min real spend    = {pct(min_spend_frac.mean())} of its own start")
    print(f"   cohorts whose spending EVER fell below the flat-4% level: "
          f"{ever_below4}/{len(sy30)} ({pct(ever_below4/len(sy30))})")
    print(f"   of ALL cohort-years, fraction spent below flat-4% level: "
          f"{pct(below4_years.mean())}")
    print(f"   avg final-year real spend = {pct(np.mean(end_spend))} of initial spend "
          f"(median {pct(np.median(end_spend))})")
    # a couple of the worst cohorts by min spend
    order = np.argsort(min_spend_frac)[:4]
    worst = [(sy30[i], pct(min_spend_frac[i])) for i in order]
    print(f"   deepest-cut cohorts (start yr : min spend vs own start): {worst}")

print("\nDONE")
