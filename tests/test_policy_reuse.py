"""Invariant P -- a policy instance gives the same answers however it is reused.

Catches: a spending policy that carries state from one cohort into the next.

An honest note about what this invariant is for
-----------------------------------------------
This was proposed on the theory that reuse was *already* broken -- that
CapeInitial, which writes self.initial_wr and self.amt during a run, would
corrupt the next cohort, and that success_rate() therefore takes a
policy_factory() out of necessity.

That is not what the code does. CapeInitial's `yr == 0` branch assigns both
fields unconditionally, with no `if self.amt is None` guard, so every new
cohort overwrites them before they are read. Reusing one instance across all
123 thirty-year cohorts gives 106/123 -- identical to fresh instances, and
identical to the published table. FixedReal and GuytonKlinger never mutate
during a run at all. policy_factory() is defensive, not load-bearing.

So this invariant does not find a bug; it pins a property that currently holds
and that one plausible-looking edit would silently break. Adding the `is None`
guard to CapeInitial reads like an obvious optimization -- don't recompute the
rate every year -- and would convert every cohort after the first into a
backtest of the *first* cohort's CAPE, at full plausibility and with no error.
That is the same shape as the original bug, which is the argument for keeping
it even though it is green for a different reason than proposed.
"""
import numpy as np
import pytest

import backtest as bt
import strategies as st

WF = bt.static_w(0.60)
H = 30

FACTORIES = {
    "FixedReal": lambda: bt.FixedReal(0.04),
    "CapeInitial": lambda: bt.CapeInitial(0.015, 0.5),
    "CapeInitial capped": lambda: bt.CapeInitial(0.015, 0.5, cap=0.06),
    "GuytonKlinger": lambda: bt.GuytonKlinger(0.05),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_one_shared_instance_matches_fresh_instances_per_cohort(name):
    """Running every cohort through one policy object gives the published results."""
    factory = FACTORIES[name]
    years = bt.cohort_start_years(H)

    shared_policy = factory()
    shared = [bt.simulate(bt.jan_index(y), H, WF, shared_policy) for y in years]
    fresh = [bt.simulate(bt.jan_index(y), H, WF, factory()) for y in years]

    for year, a, b in zip(years, shared, fresh):
        assert a["success"] == b["success"], (
            f"{name}: cohort {year} succeeds only when the policy is fresh; "
            f"state is leaking between cohorts"
        )
        assert a["terminal"] == pytest.approx(b["terminal"], rel=1e-15), (
            f"{name}: cohort {year} ends at {a['terminal']} reusing the policy "
            f"and {b['terminal']} with a fresh one"
        )
        assert np.array_equal(a["spend_path"], b["spend_path"]), (
            f"{name}: cohort {year} spends differently when the policy is reused"
        )


@pytest.mark.parametrize("name", list(FACTORIES))
def test_replaying_one_cohort_twice_is_deterministic(name):
    """The same instance on the same cohort twice gives the same run.

    Weaker than the reuse check above and kept separate because it fails for a
    different reason: this one going red means a policy accumulated state
    within a single cohort, not across cohorts.
    """
    policy = FACTORIES[name]()
    i = bt.jan_index(1966)
    first = bt.simulate(i, H, WF, policy)
    second = bt.simulate(i, H, WF, policy)
    assert first["success"] == second["success"]
    assert first["terminal"] == pytest.approx(second["terminal"], rel=1e-15)
    assert np.array_equal(first["spend_path"], second["spend_path"])


def test_cape_policy_reports_the_rate_of_the_cohort_just_run():
    """After a run, initial_wr belongs to that cohort, not an earlier one.

    run_analysis.py reads `pol.initial_wr` back off the policy after each
    simulate() to build the avg/min/max wr0 column. If a stale value survived a
    run, that column would silently describe the wrong cohort while every
    success count stayed correct -- the exact failure mode this harness exists
    to make impossible.
    """
    policy = bt.CapeInitial(0.015, 0.5)
    for year in [1900, 1921, 1929, 1966, 1982]:
        i = bt.jan_index(year)
        bt.simulate(i, H, WF, policy)
        expected = 0.015 + 0.5 / bt.CAPE[i]
        assert policy.initial_wr == pytest.approx(expected, rel=1e-12), (
            f"after running cohort {year}, initial_wr is {policy.initial_wr} "
            f"but that cohort's CAPE implies {expected}"
        )


def test_catalog_strategies_hand_out_independent_policies():
    """Strategy.policy_factory returns a new object each call.

    Strategy.run() builds a policy per cohort. A factory that returned a
    singleton -- easy to write as `policy=some_instance` instead of a lambda --
    would put every cohort through one object and quietly depend on the
    property above continuing to hold.
    """
    for strat in st.CATALOG:
        a, b = strat.policy_factory(), strat.policy_factory()
        assert a is not b, f"{strat.name} hands out the same policy object twice"
