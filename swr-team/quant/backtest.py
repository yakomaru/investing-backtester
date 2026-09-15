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
from pathlib import Path
from typing import Callable, NamedTuple, Protocol, runtime_checkable

DEFAULT_CSV = Path(__file__).with_name("shiller_monthly.csv")


class Series(NamedTuple):
    """One monthly real-return world: the arrays every backtest reads.

    Bundled behind `load_series` so the data can be loaded from an explicit
    path instead of only from the process's current working directory, which
    is what previously made this module importable only from inside
    swr-team/quant/.
    """
    df: pd.DataFrame
    RS: np.ndarray        # monthly real stock return
    RB: np.ndarray        # monthly real bond return
    CAPE: np.ndarray
    YEAR: np.ndarray
    MONTH: np.ndarray
    N: int
    BASE_YEAR: int        # first year in the file (1871 for the Shiller series)


def load_series(path=DEFAULT_CSV):
    """Load the monthly series from `path`, defaulting to the CSV beside this file."""
    df = pd.read_csv(path)
    year = df["year"].values
    return Series(
        df=df,
        RS=df["stock_ret_real"].values,
        RB=df["bond_ret_real"].values,
        CAPE=df["CAPE"].values,
        YEAR=year,
        MONTH=df["month"].values,
        N=len(df),
        BASE_YEAR=int(year[0]),
    )


# The default series, loaded at import as before. The existing run_*.py scripts
# read these bare module names (bt.RS, bt.CAPE, bt.N, ...), so they are kept
# exactly as they were; the only behavioural change is that the CSV is now
# located relative to this file rather than the current working directory.
SERIES = load_series()
df = SERIES.df
RS = SERIES.RS
RB = SERIES.RB
CAPE = SERIES.CAPE
YEAR = SERIES.YEAR
MONTH = SERIES.MONTH
N = SERIES.N
BASE_YEAR = SERIES.BASE_YEAR

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
# the two plug-in points of a backtest, named
# ----------------------------------------------------------------------------
WeightFn = Callable[[int], float]
"""Target equity weight for retirement year `yr`.

Takes `yr` and nothing else on purpose: a weight function is structurally
incapable of seeing a return, so allocation cannot react to the future. That
property is enforced by this signature, not by a test.
"""


@runtime_checkable
class SpendingPolicy(Protocol):
    """What `simulate` requires of a spending rule.

    This protocol was always implicit -- FixedReal, CapeInitial and
    GuytonKlinger have satisfied it since they were written. Naming it gives
    `Strategy` (see strategies.py) something to bind a declared allocation to,
    and gives the invariants a single contract to check against.

    `initial_wr` is the rate the policy started from, which is what the
    guardrail logic and the reporting scripts read back afterwards. Note that
    CapeInitial cannot know it until the cohort's CAPE is seen, so it is None
    until `next_spend` has been called with yr == 0.

    `next_spend` receives (yr, port, prev, cape0) and nothing else -- like
    WeightFn, it has no channel through which future returns could reach it.

    runtime_checkable so the harness can assert conformance with isinstance.
    That only verifies the members exist, not their signatures, which is the
    right strength here: it catches a policy that forgot `initial_wr`, and it
    does not pretend to catch one whose arithmetic is wrong.
    """

    initial_wr: float

    def next_spend(self, yr: int, port: float, prev, cape0: float) -> float:
        ...


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

class SearchSaturated(RuntimeError):
    """max_sustainable_wr was asked for a rate above the ceiling it searches under.

    Raised instead of returning `hi`, which would be a bound wearing the costume
    of an answer. See the guard in max_sustainable_wr for why this matters.
    """


def max_sustainable_wr(start_idx, H, weight_fn, lo=0.0, hi=0.20, tol=1e-5):
    """Highest fixed-real WR this cohort can sustain for H years (binary search).

    Raises SearchSaturated if the cohort can already sustain `hi`, because the
    search cannot then distinguish "the answer is 20%" from "the answer is at
    least 20%" -- it converges lo up to hi and returns the ceiling silently.
    That is not hypothetical: at H<=3 every cohort in the Shiller series
    saturates (the true answer at H=1 is ~100%), and at H=5 116 of 148 do. The
    headline H=30/40 runs peak at 11.0%, so nothing published goes near it.

    The check is an explicit simulate at `hi` rather than an after-the-fact
    `lo >= hi - tol` test, so the precondition is verified before the search
    runs rather than inferred from where it landed.
    """
    if simulate(start_idx, H, weight_fn, FixedReal(hi))["success"]:
        raise SearchSaturated(
            f"cohort at index {start_idx} sustains the search ceiling hi={hi:.4f} "
            f"for H={H}, so the true max is above it; raise hi to get a real answer"
        )
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
