"""
backtest.py  -- historical safe-withdrawal-rate backtests on Shiller monthly real data.

Framework (unless noted):
  * Real dollars throughout (CPI-deflated). Portfolio normalized to 1.0 at retirement.
  * Annual withdrawals taken at the START of each year (Bengen convention).
  * Monthly compounding of real returns within the year; rebalance to target weights
    once per year right after the withdrawal.
  * H-year retirement = H annual withdrawals. SUCCESS = all H withdrawals funded in full
    (portfolio never goes negative). FAILURE = a withdrawal cannot be met.
  * Cohorts start in JANUARY of each start year (Shiller data is contiguous monthly).

Return series come from shiller_monthly.csv (built by load_data.py):
  stock = S&P Composite real total return (divs reinvested)
  bond  = 10y Treasury constant-maturity real total return approximation (matches
          Shiller's own bond TR series at corr 1.000).
"""
import numpy as np
import pandas as pd

df = pd.read_csv("shiller_monthly.csv")
RS = df["stock_ret_real"].values   # monthly real stock return
RB = df["bond_ret_real"].values    # monthly real bond return
CAPE = df["CAPE"].values
YEAR = df["year"].values
MONTH = df["month"].values
N = len(df)
BASE_YEAR = int(YEAR[0])            # 1871

def jan_index(y):
    """Month index of January of year y (data is contiguous from Jan 1871)."""
    return (y - BASE_YEAR) * 12

def cohort_start_years(H, last_data_idx=N - 1):
    """Start years (Jan) with a full H-year window available."""
    years = []
    y = BASE_YEAR
    while jan_index(y) + H * 12 - 1 <= last_data_idx:
        years.append(y)
        y += 1
    return years


# ----------------------------------------------------------------------------
# equity-weight schedules
# ----------------------------------------------------------------------------
def static_w(w):
    return lambda yr: w

def glidepath(start_w, end_w, years):
    def f(yr):
        return start_w + (end_w - start_w) * min(yr, years) / years
    return f


# ----------------------------------------------------------------------------
# core single-cohort simulation
# ----------------------------------------------------------------------------
def simulate(start_idx, H, weight_fn, policy):
    """
    policy: object exposing initial_wr and next_spend(yr, port_before_wd, prev_spend, cape0).
    Returns dict with success, terminal (real), spend_path (list), balance_path (list).
    """
    port = 1.0
    cape0 = CAPE[start_idx]
    prev_spend = None
    spend_path = []
    bal_path = []
    success = True
    for yr in range(H):
        spend = policy.next_spend(yr, port, prev_spend, cape0)
        if spend > port:                 # cannot fund full withdrawal -> failure
            spend_path.append(port)      # spend whatever remains
            bal_path.append(0.0)
            port = 0.0
            success = False
            # fill remaining years with zero for path-length consistency
            for _ in range(yr + 1, H):
                spend_path.append(0.0)
                bal_path.append(0.0)
            break
        port -= spend
        prev_spend = spend
        spend_path.append(spend)
        w = weight_fn(yr)
        stock = port * w
        bond = port * (1 - w)
        base = start_idx + yr * 12
        for m in range(12):
            stock *= (1.0 + RS[base + m])
            bond *= (1.0 + RB[base + m])
        port = stock + bond
        bal_path.append(port)
    return {"success": success, "terminal": port,
            "spend_path": np.array(spend_path), "balance_path": np.array(bal_path)}


# ----------------------------------------------------------------------------
# spending policies
# ----------------------------------------------------------------------------
class FixedReal:
    """Withdraw wr * initial (=wr, since port0=1) in real terms every year."""
    def __init__(self, wr):
        self.initial_wr = wr
        self.amt = wr
    def next_spend(self, yr, port, prev, cape0):
        return self.amt

class CapeInitial:
    """Set initial rate wr0 = a + b/CAPE at retirement, then fixed-real forever.
    Falls back to `fallback` if CAPE missing at start."""
    def __init__(self, a, b, cap=None, fallback=0.04):
        self.a = a; self.b = b; self.cap = cap; self.fallback = fallback
        self.initial_wr = None; self.amt = None
    def next_spend(self, yr, port, prev, cape0):
        if yr == 0:
            if np.isnan(cape0) or cape0 <= 0:
                wr0 = self.fallback
            else:
                wr0 = self.a + self.b / cape0
            if self.cap is not None:
                wr0 = min(wr0, self.cap)
            self.initial_wr = wr0
            self.amt = wr0
        return self.amt

class GuytonKlinger:
    """Simplified guardrails. Start at initial_wr (of initial portfolio).
    Each subsequent year, let cur_wr = prev_spend / current_port:
      cur_wr > 1.2*initial_wr  -> cut spending 10% (capital-preservation guardrail)
      cur_wr < 0.8*initial_wr  -> raise spending 10% (prosperity guardrail)
      else                     -> hold spending flat (real)
    All in real dollars."""
    def __init__(self, initial_wr, upper=1.2, lower=0.8, cut=0.10, raise_=0.10):
        self.initial_wr = initial_wr
        self.upper = upper; self.lower = lower
        self.cut = cut; self.raise_ = raise_
    def next_spend(self, yr, port, prev, cape0):
        if yr == 0:
            return self.initial_wr
        cur_wr = prev / port if port > 0 else np.inf
        if cur_wr > self.upper * self.initial_wr:
            return prev * (1 - self.cut)
        if cur_wr < self.lower * self.initial_wr:
            return prev * (1 + self.raise_)
        return prev


# ----------------------------------------------------------------------------
# metrics helpers
# ----------------------------------------------------------------------------
def success_rate(H, weight_fn, policy_factory, start_years):
    """policy_factory() returns a fresh policy per cohort."""
    ok = 0
    for y in start_years:
        r = simulate(jan_index(y), H, weight_fn, policy_factory())
        ok += r["success"]
    return ok / len(start_years)

def max_sustainable_wr(start_idx, H, weight_fn, lo=0.0, hi=0.20, tol=1e-5):
    """Highest fixed-real WR this cohort can sustain for H years (binary search)."""
    for _ in range(40):
        mid = (lo + hi) / 2
        if simulate(start_idx, H, weight_fn, FixedReal(mid))["success"]:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo

def safemax_and_worst(H, weight_fn, start_years):
    """SAFEMAX = min over cohorts of max-sustainable WR; worst start year = argmin.
    Also returns success rate at 4%."""
    msw = {y: max_sustainable_wr(jan_index(y), H, weight_fn) for y in start_years}
    worst_year = min(msw, key=msw.get)
    safemax = msw[worst_year]
    succ4 = np.mean([msw[y] >= 0.04 for y in start_years])
    # a couple of runner-up worst years for context
    order = sorted(msw, key=msw.get)[:5]
    return {"safemax": safemax, "worst_year": worst_year, "succ4": succ4,
            "worst5": [(y, round(msw[y] * 100, 2)) for y in order], "msw": msw}
