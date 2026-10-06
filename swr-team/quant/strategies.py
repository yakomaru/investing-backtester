"""
strategies.py -- a strategy is a *declared* allocation envelope bound to the
weight function and spending policy that are supposed to implement it.

Why this module exists
----------------------
`backtest.simulate(start_idx, H, weight_fn, policy)` takes allocation and
spending as independent arguments and has no idea what either one is meant to
be. A strategy's stated allocation lives only in the label a human typed into a
print statement. Nothing connects "68/17/15 with a cash buffer" to the weights
the backtest actually runs, so the two can drift apart and the run still
returns plausible numbers. That is the seam the original bug in this codebase
slipped through, and it is not a seam a test can close from outside: there is
nothing to compare the realized weights *to*.

A Strategy supplies the missing half. `equity_min`/`equity_max` is intent,
stated separately from `weight_fn`, which is mechanism. The invariants in
tests/test_allocation.py assert the two agree -- and, just as importantly, that
the envelope is tight, so a strategy cannot buy conformance by declaring
(0.0, 1.0) and conforming to nothing.

The catalog below mirrors every strategy run_analysis.py reports. It is a
second copy of those definitions, which is a real cost; tests/test_golden.py
pays for it by asserting the catalog reproduces the published headline numbers,
so the copy cannot drift from the script without a test going red.
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import backtest as bt


@dataclass(frozen=True)
class Strategy:
    """A declared allocation envelope bound to the machinery meant to realize it.

    equity_min / equity_max are the fraction of the portfolio in equities that
    this strategy claims it will hold, as a closed interval over its horizon.
    A static 60/40 declares (0.60, 0.60); a 60->100 glidepath declares
    (0.60, 1.00) and is expected to actually reach both ends.
    """

    name: str
    equity_min: float
    equity_max: float
    weight_fn: bt.WeightFn
    policy_factory: Callable[[], bt.SpendingPolicy]
    horizons: Tuple[int, ...] = (30,)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def declared_weights(self, H=None):
        """The weight schedule this strategy *says* it will run, year by year.

        Named `declared`, not `realized`: this asks the weight function and
        nothing else. It is a statement of intent, and checking it against the
        envelope only proves the catalog is internally consistent. Whether the
        engine honours this schedule is a separate question, answered by
        tests/test_allocation.py::test_engine_blends_at_the_declared_weights,
        which reads what simulate() actually did.
        """
        return [self.weight_fn(yr) for yr in range(H or self.max_horizon)]

    def run(self, start_idx, H):
        """Simulate one cohort. Builds a fresh policy, which is the contract
        success_rate() assumes and tests/test_policy_reuse.py pins."""
        return bt.simulate(start_idx, H, self.weight_fn, self.policy_factory())


# ---------------------------------------------------------------------------
# catalog: every strategy run_analysis.py reports
# ---------------------------------------------------------------------------
CATALOG = (
    # Section 1 & 2: fixed-real withdrawals at the four classic allocations.
    # succ@4% is the reported metric, so the bound policy is a flat 4%.
    Strategy("static 50/50", 0.50, 0.50, bt.static_w(0.50),
             lambda: bt.FixedReal(0.04), horizons=(30, 40)),
    Strategy("static 60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.FixedReal(0.04), horizons=(30, 40)),
    Strategy("static 75/25", 0.75, 0.75, bt.static_w(0.75),
             lambda: bt.FixedReal(0.04), horizons=(30, 40)),
    Strategy("static 100/0", 1.00, 1.00, bt.static_w(1.00),
             lambda: bt.FixedReal(0.04), horizons=(30, 40)),

    # Section 3: rising-equity glidepaths. These are the entries that make the
    # envelope worth stating as an interval rather than a point.
    Strategy("glide 60->100 / 15yr", 0.60, 1.00, bt.glidepath(0.60, 1.00, 15),
             lambda: bt.FixedReal(0.04), horizons=(30,)),
    Strategy("glide 30->70 / 15yr", 0.30, 0.70, bt.glidepath(0.30, 0.70, 15),
             lambda: bt.FixedReal(0.04), horizons=(30,)),

    # Section 4: CAPE-based initial withdrawal rate, held at 60/40 throughout.
    # The spending rule varies; the allocation claim does not.
    Strategy("cape a=1.5%,b=0.5 @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.CapeInitial(0.015, 0.5), horizons=(30,)),
    Strategy("cape a=1.5%,b=0.5 cap6% @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.CapeInitial(0.015, 0.5, cap=0.06), horizons=(30,)),
    Strategy("cape a=1.0%,b=0.5 @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.CapeInitial(0.010, 0.5), horizons=(30,)),
    Strategy("cape a=2.0%,b=0.4 @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.CapeInitial(0.020, 0.4), horizons=(30,)),

    # Section 5: Guyton-Klinger guardrails, also 60/40 throughout.
    Strategy("GK start 5.0% @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.GuytonKlinger(0.05), horizons=(30,)),
    Strategy("GK start 5.5% @60/40", 0.60, 0.60, bt.static_w(0.60),
             lambda: bt.GuytonKlinger(0.055), horizons=(30,)),
)

BY_NAME = {s.name: s for s in CATALOG}
