"""
supplement.py -- deeper looks that the headline tables raise:
 (A) trace the GK 1966 cohort spending path (concrete worst-case experience)
 (B) confirm exactly which cohorts flat-4% fails, and whether CAPE-initial fixes them
 (C) safety-matched CAPE search: best avg spending with failures <= flat-4%'s failures
 (D) downside-only CAPE rule (never exceed 4% start) -- does asymmetric help?
"""
import numpy as np
import backtest as bt

def pct(x): return f"{x*100:.2f}%"
H = 30
sy = bt.cohort_start_years(H)
wf = bt.static_w(0.60)

# ---- (A) GK 1966 spending trace ------------------------------------------
print("### (A) Guyton-Klinger 60/40 start-5% -- 1966 cohort spending path (real $ per $1 init)")
pol = bt.GuytonKlinger(0.05)
r = bt.simulate(bt.jan_index(1966), H, wf, pol)
sp = r["spend_path"]
for yr in range(H):
    tag = ""
    if yr > 0:
        if sp[yr] < sp[yr-1] - 1e-9: tag = "  <- CUT 10%"
        elif sp[yr] > sp[yr-1] + 1e-9: tag = "  <- RAISE 10%"
    print(f"  yr {1966+yr} (age+{yr:2d}): spend={sp[yr]*100:5.2f}%  bal={r['balance_path'][yr]*100:6.1f}%{tag}")
ncuts = int(np.sum(np.diff(sp) < -1e-9))
nraise = int(np.sum(np.diff(sp) > 1e-9))
below4 = int(np.sum(sp < 0.04))
print(f"  => cuts={ncuts}, raises={nraise}, min spend={pct(sp.min())} "
      f"(={pct(sp.min()/sp[0])} of start), years below flat-4% level={below4}/{H}, "
      f"final spend={pct(sp[-1])}")

# ---- (B) flat-4% failures and whether CAPE-initial rescues them ------------
print("\n### (B) 60/40 flat-4.0% failing cohorts, and the CAPE(1.5%,0.5) verdict on each")
for y in sy:
    rf = bt.simulate(bt.jan_index(y), H, wf, bt.FixedReal(0.04))
    if not rf["success"]:
        pol = bt.CapeInitial(0.015, 0.5)
        rc = bt.simulate(bt.jan_index(y), H, wf, pol)
        c0 = bt.CAPE[bt.jan_index(y)]
        print(f"  {y}: CAPE={c0:4.1f}  flat4%=FAIL  CAPE-wr0={pct(pol.initial_wr)} "
              f"CAPE-outcome={'OK ' if rc['success'] else 'FAIL'}")

print("\n### (B2) cohorts the CAPE(1.5%,0.5) rule FAILS -- what CAPE/rate triggered it")
fails = []
for y in sy:
    pol = bt.CapeInitial(0.015, 0.5)
    rc = bt.simulate(bt.jan_index(y), H, wf, pol)
    if not rc["success"]:
        c0 = bt.CAPE[bt.jan_index(y)]
        fails.append((y, c0, pol.initial_wr))
for y, c, w in fails:
    print(f"  {y}: CAPE={c:4.1f}  CAPE-wr0={pct(w)}  (rule granted a high rate in a 'cheap' market)")

# ---- (C) safety-matched CAPE grid: failures <= flat-4% failures ------------
n_flat_fail = sum(not bt.simulate(bt.jan_index(y), H, wf, bt.FixedReal(0.04))["success"] for y in sy)
flat_avg_spend = np.mean([bt.simulate(bt.jan_index(y), H, wf, bt.FixedReal(0.04))["spend_path"].mean() for y in sy])
print(f"\n### (C) safety-matched CAPE search (allow <= {n_flat_fail} failures, = flat-4%)")
print(f"    flat-4% baseline: {n_flat_fail} failures, avg lifetime spend {pct(flat_avg_spend)}")
best = None
for a in [0.005, 0.01, 0.015, 0.02]:
    for b in [0.3, 0.4, 0.5]:
        for cap in [0.045, 0.05, 0.055, 0.06, None]:
            nf, spends, wr0s = 0, [], []
            for y in sy:
                pol = bt.CapeInitial(a, b, cap=cap)
                rc = bt.simulate(bt.jan_index(y), H, wf, pol)
                nf += (not rc["success"])
                spends.append(rc["spend_path"].mean()); wr0s.append(pol.initial_wr)
            if nf <= n_flat_fail:
                avg = np.mean(spends)
                if best is None or avg > best[0]:
                    best = (avg, a, b, cap, nf, np.mean(wr0s))
if best:
    avg, a, b, cap, nf, awr = best
    print(f"    BEST safety-matched: a={a}, b={b}, cap={cap}: failures={nf}, "
          f"avg wr0={pct(awr)}, avg lifetime spend={pct(avg)} "
          f"(vs flat {pct(flat_avg_spend)}  -> +{(avg-flat_avg_spend)*100:.2f} pp)")
else:
    print("    No CAPE variant in the grid matched flat-4% safety.")

# ---- (D) downside-only CAPE (never start above 4%, cut when rich) ----------
print("\n### (D) downside-only CAPE: wr0 = min(4%, 1.5% + 0.5/CAPE)")
nf, spends, wr0s = 0, [], []
for y in sy:
    pol = bt.CapeInitial(0.015, 0.5, cap=0.04)
    rc = bt.simulate(bt.jan_index(y), H, wf, pol)
    nf += (not rc["success"]); spends.append(rc["spend_path"].mean()); wr0s.append(pol.initial_wr)
print(f"    failures={nf}/{len(sy)}, avg wr0={pct(np.mean(wr0s))}, "
      f"avg lifetime spend={pct(np.mean(spends))} (vs flat {pct(flat_avg_spend)})")
print("    (this spends LESS than flat-4% on average -> buys safety, does not 'beat' 4% on income)")
