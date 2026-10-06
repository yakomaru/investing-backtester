"""Invariants M and H -- the search that produces SAFEMAX, and its preconditions.

SAFEMAX is the headline number this repo exists to produce, and it is not
measured, it is *searched for*. max_sustainable_wr binary-searches on a boolean
success flag. That is only a valid thing to do if success is monotone in the
withdrawal rate; if it were not, bisection would converge to an arbitrary point
and report it with the same confidence. Nothing stated that precondition
anywhere before this file.

Invariant M: success is monotone in the withdrawal rate (the precondition).
Invariant H: SAFEMAX is non-increasing as the horizon grows (the consequence).

Both are run against the real series over every cohort rather than sampled,
because the input space here is small enough to enumerate exhaustively -- which
is strictly stronger than random search and, unlike it, gives the same answer
every run.
"""
import numpy as np
import pytest

import backtest as bt

WF = bt.static_w(0.60)
SEARCH_TOL = 1e-5  # max_sustainable_wr's own convergence tolerance


# ---------------------------------------------------------------------------
# the saturation guard
# ---------------------------------------------------------------------------
def test_search_refuses_to_return_its_own_ceiling():
    """A cohort that can sustain `hi` raises instead of returning `hi`.

    Without the guard the function returns 0.20 -- the search bound -- with no
    signal that it is a bound rather than an answer. At H<=3 every cohort in
    the series saturates (the true answer at H=1 is about 100%), and at H=5,
    116 of 148 do. Any invariant comparing two saturated values would then pass
    by comparing the ceiling to itself.
    """
    with pytest.raises(bt.SearchSaturated):
        bt.max_sustainable_wr(bt.jan_index(1950), 5, WF)

    # And it does not fire where the headline runs live.
    for horizon in (30, 40):
        for year in bt.cohort_start_years(horizon):
            got = bt.max_sustainable_wr(bt.jan_index(year), horizon, WF)
            assert got < 0.20, f"{year} at H={horizon} reached the search ceiling"


def test_search_refuses_to_return_an_unsustainable_floor():
    """A bracket whose `lo` already fails raises instead of returning `lo`.

    The mirror image of the ceiling guard, and the sharper of the two: a
    saturated ceiling returns a rate that is merely too low to be the answer,
    while an unsustainable floor returns a rate that *does not survive the
    horizon* and presents it as the safe withdrawal rate. Verified before
    fixing -- max_sustainable_wr for 1966 at H=30 with lo=0.05 returned exactly
    0.050000, and simulating at that rate runs the portfolio out.

    Cannot fire on the default bracket, where lo=0.0 and withdrawing nothing
    always succeeds. It exists for the caller who narrows the bracket to save
    iterations, which is the only way to reach it and gives no other signal.
    """
    i = bt.jan_index(1966)
    with pytest.raises(bt.SearchBracketInvalid):
        bt.max_sustainable_wr(i, 30, WF, lo=0.05)

    # The default bracket is unaffected, and its answer really is sustainable.
    got = bt.max_sustainable_wr(i, 30, WF)
    assert got < 0.05
    assert bt.simulate(i, 30, WF, bt.FixedReal(got))["success"]


# ---------------------------------------------------------------------------
# invariant M -- monotonicity in the withdrawal rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [30, 40])
def test_search_result_actually_brackets_the_failure_point(horizon):
    """The returned rate succeeds and a hair above it fails, for every cohort.

    The tightest available statement of "the search found the right answer",
    checked on all of them rather than argued from the algorithm.
    """
    for year in bt.cohort_start_years(horizon):
        i = bt.jan_index(year)
        wr = bt.max_sustainable_wr(i, horizon, WF)
        assert bt.simulate(i, horizon, WF, bt.FixedReal(wr))["success"], (
            f"{year} at H={horizon}: search returned {wr:.6f} but that rate fails"
        )
        above = wr + 2 * SEARCH_TOL
        assert not bt.simulate(i, horizon, WF, bt.FixedReal(above))["success"], (
            f"{year} at H={horizon}: search returned {wr:.6f} but {above:.6f} "
            f"also succeeds, so the answer was not maximal"
        )


@pytest.mark.parametrize("horizon", [30, 40])
@pytest.mark.parametrize("w", [0.50, 0.60, 0.75, 1.00])
def test_success_is_monotone_in_withdrawal_rate(w, horizon):
    """Once a cohort fails at some rate, it fails at every higher rate.

    The soundness precondition of the bisection in max_sustainable_wr. Spending
    strictly more every year leaves a portfolio strictly smaller at every point,
    so failure cannot un-happen -- but that argument is about FixedReal, and
    the search hard-codes FixedReal, so this pins the pairing rather than the
    policy alone.

    Swept over every published allocation at both published horizons, not just
    the 60/40 30-year case. The property has no reason to be allocation- or
    horizon-specific, and SAFEMAX is reported for all eight combinations, so
    testing one of them left the other seven resting on an argument rather
    than a measurement. Exhaustive over cohorts at 0.25% steps; a finer 0.05%
    grid was also checked by hand and finds no violation either, so the step
    is chosen for runtime, not because it is near a boundary.
    """
    wf = bt.static_w(w)
    grid = np.arange(0.0, 0.2001, 0.0025)
    offenders = []
    for year in bt.cohort_start_years(horizon):
        i = bt.jan_index(year)
        failed_at = None
        for wr in grid:
            ok = bt.simulate(i, horizon, wf, bt.FixedReal(float(wr)))["success"]
            if not ok and failed_at is None:
                failed_at = wr
            elif ok and failed_at is not None:
                offenders.append((year, float(failed_at), float(wr)))
                break
    assert not offenders, (
        f"w={w} H={horizon}: success came back after failing, so bisection on "
        f"it is unsound: {offenders[:5]}"
    )


# ---------------------------------------------------------------------------
# invariant H -- monotonicity in the horizon
# ---------------------------------------------------------------------------
HORIZONS = [10, 15, 20, 25, 30, 35, 40]


@pytest.mark.parametrize("short,long", list(zip(HORIZONS, HORIZONS[1:])))
def test_safemax_does_not_rise_as_the_horizon_grows(short, long):
    """A longer retirement cannot support a higher safe rate.

    Two things this has to get right to mean anything:

    1. Only horizons of 10 years and up. Below that the search saturates at its
       ceiling and the comparison is 0.20 against 0.20 -- at H=1 vs H=2 that is
       151 of 151 cohorts tied, a green test measuring nothing. The saturation
       guard now raises there instead, but the floor is kept explicit because
       the vacuousness is the reason for it, not the exception.

    2. Only cohorts present at *both* horizons. The aggregate SAFEMAX is a min
       over cohorts, and a longer horizon has fewer start years available -- the
       1990s cohorts drop out first. A min over a smaller set can legitimately
       be higher, so comparing the two published aggregates directly would find
       violations that are an artifact of the changing cohort set rather than a
       bug.
    """
    common = sorted(set(bt.cohort_start_years(short)) & set(bt.cohort_start_years(long)))
    assert common, f"no cohorts shared between H={short} and H={long}"

    tol = 1e-4  # two independent bisections, each converged to 1e-5
    short_msw = {y: bt.max_sustainable_wr(bt.jan_index(y), short, WF) for y in common}
    long_msw = {y: bt.max_sustainable_wr(bt.jan_index(y), long, WF) for y in common}

    risen = [
        (y, short_msw[y], long_msw[y])
        for y in common if long_msw[y] > short_msw[y] + tol
    ]
    assert not risen, (
        f"cohorts sustain a higher rate over {long} years than over {short}: {risen[:5]}"
    )

    assert min(long_msw.values()) <= min(short_msw.values()) + tol, (
        f"SAFEMAX over the shared cohort set rose from {min(short_msw.values()):.5f} "
        f"at H={short} to {min(long_msw.values()):.5f} at H={long}"
    )


# ---------------------------------------------------------------------------
# cross-check: two routes to success@4%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [30, 40])
@pytest.mark.parametrize("w", [0.50, 0.60, 0.75, 1.00])
def test_succ4_agrees_with_simulating_at_four_percent(horizon, w):
    """safemax_and_worst derives succ@4% from the search; check it against reality.

    succ4 is computed as `msw[y] >= 0.04` -- the fraction of cohorts whose
    *searched* maximum clears 4% -- not by running a 4% withdrawal. Those are
    different computations that must agree, and the search's answer is a lower
    bound converged to 1e-5, so a cohort sitting within a hair of 4% is exactly
    where they could disagree. A free cross-check of the search against the
    thing it approximates.
    """
    wf = bt.static_w(w)
    years = bt.cohort_start_years(horizon)
    derived = bt.safemax_and_worst(horizon, wf, years)["succ4"]
    direct = np.mean([
        bt.simulate(bt.jan_index(y), horizon, wf, bt.FixedReal(0.04))["success"]
        for y in years
    ])
    assert derived == pytest.approx(direct, abs=1e-12), (
        f"H={horizon} w={w}: succ4 from the search is {derived}, simulating at "
        f"4% gives {direct}"
    )
