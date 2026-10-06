"""Invariant A -- allocation conformance.

Catches: a strategy whose realized allocation is not the one it claims. This is
the original bug in this codebase, stated directly. A correctly implemented rule
converted a cash-buffered strategy into an all-equity one, and every backtest
after that kept returning plausible numbers, because nothing in the engine knew
what the strategy was supposed to hold.

Why the envelope is an interval, not a point
--------------------------------------------
A glidepath legitimately holds a different weight every year, so a point claim
would make the rising-equity strategies unstateable. The interval says "this
strategy stays between these bounds", which is exactly the claim a reader of the
results table is relying on.

What these checks can and cannot see
------------------------------------
The envelope and tightness checks below ask the weight function what it
intends and compare that to what the strategy declared. That is worth having
-- it is what stops the catalog quietly disagreeing with itself -- but on its
own it is a declaration checked against a declaration, and it cannot see the
engine at all. Replacing `w = weight_fn(yr)` with `w = 1.0` inside simulate(),
which is the original bug's exact shape, leaves every one of them green.

test_engine_blends_at_the_declared_weights closes that gap by reading what
simulate() actually did. Both halves are needed: the first says the strategy
means what it says, the second says the engine does what the strategy means.

Why tightness is asserted too
-----------------------------
An interval invariant alone is trivially satisfiable: declare (0.0, 1.0) and
conform to nothing. So conformance is paired with tightness -- the strategy must
actually *reach* both ends of the envelope it declares. Together they pin the
realized weight path from both sides. This is the difference between a test that
catches the original bug and one that a future author can silently neuter by
widening a bound. Widening the envelope now breaks tightness, which is a loud
failure rather than a quiet loss of coverage.
"""
import numpy as np
import pytest

import backtest as bt
import strategies as st

TOL = 1e-12

ALL = pytest.mark.parametrize("strat", st.CATALOG, ids=lambda s: s.name)


@ALL
def test_declared_schedule_stays_inside_declared_envelope(strat):
    """Every year's equity weight is inside the strategy's declared envelope."""
    for H in strat.horizons:
        for yr, w in enumerate(strat.declared_weights(H)):
            assert strat.equity_min - TOL <= w <= strat.equity_max + TOL, (
                f"{strat.name} declares equity in "
                f"[{strat.equity_min}, {strat.equity_max}] but holds {w} in year {yr} "
                f"of a {H}-year retirement"
            )


@ALL
def test_declared_envelope_is_tight(strat):
    """The strategy reaches both ends of its envelope, so the claim is not vacuous.

    Checked over the strategy's longest horizon: a 60->100 glidepath over 15
    years only attains 1.00 if the horizon is long enough to get there, which
    every horizon in the catalog is.
    """
    w = strat.declared_weights(strat.max_horizon)
    assert min(w) == pytest.approx(strat.equity_min, abs=1e-9), (
        f"{strat.name} declares a floor of {strat.equity_min} but never goes "
        f"below {min(w)}; the declared envelope is looser than the strategy"
    )
    assert max(w) == pytest.approx(strat.equity_max, abs=1e-9), (
        f"{strat.name} declares a ceiling of {strat.equity_max} but never "
        f"exceeds {max(w)}; the declared envelope is looser than the strategy"
    )


@ALL
def test_weights_are_a_long_only_unlevered_portfolio(strat):
    """No weight escapes [0, 1].

    Separate from the envelope check because this holds for *any* strategy this
    engine can express, declared envelope or not: simulate() splits the
    portfolio into `port * w` and `port * (1 - w)`, so a weight outside [0, 1]
    is silently a leveraged or short position that the engine will happily
    compound and that no table would mention.
    """
    for w in strat.declared_weights(strat.max_horizon):
        assert 0.0 <= w <= 1.0, f"{strat.name} holds equity weight {w}"


@ALL
def test_weight_fn_depends_only_on_year(strat):
    """Asking for the same year twice gives the same weight.

    The allocation schedule must be a pure function of the retirement year.
    A weight_fn that accumulated state across calls would make the realized
    path depend on how many times it had been evaluated -- including by this
    harness, which would then be measuring something the backtest never ran.
    """
    first = strat.declared_weights(strat.max_horizon)
    second = strat.declared_weights(strat.max_horizon)
    third = [strat.weight_fn(yr) for yr in reversed(range(strat.max_horizon))]
    assert first == second
    assert first == list(reversed(third)), (
        f"{strat.name}'s weight function gives different answers depending on "
        f"the order years are requested in"
    )


@ALL
def test_policy_satisfies_the_spending_protocol(strat):
    """The bound policy actually implements SpendingPolicy.

    isinstance against a runtime_checkable Protocol verifies the members exist,
    not that they are correct. That is the intended strength: it catches a
    policy missing `initial_wr` (which the reporting scripts read back), and it
    makes no claim about the arithmetic.
    """
    policy = strat.policy_factory()
    assert isinstance(policy, bt.SpendingPolicy), (
        f"{strat.name}'s policy {type(policy).__name__} does not satisfy "
        f"SpendingPolicy"
    )
    assert hasattr(policy, "next_spend")


@pytest.mark.parametrize("year", [1871, 1929, 1966, 1982, 1993])
@ALL
def test_engine_blends_at_the_declared_weights(strat, year):
    """simulate() actually allocates at the weights the strategy declares.

    Catches: the engine ignoring, overriding, or mis-indexing the weight
    schedule -- the original bug. Every other check in this file asks the
    weight function what it intends; this one reads what the run did.

    How, and why this much mirroring is the right amount. The engine's own
    spend_path is taken as given and only the allocation blend is re-derived:
    each year, the portfolio after that year's withdrawal is grown by
    w*Gs + (1-w)*Gb, where Gs and Gb are the 12-month compounded stock and
    bond growth for that year, and the result is compared to the balance the
    engine reported.

    This file's sibling (tests/test_simulate_contract.py) rejects a
    conservation identity precisely because it would re-implement simulate()
    inside the test, and a mirror is wrong in the same way as the thing it
    mirrors. The distinction that makes this one worth having: consuming the
    engine's spend_path means the withdrawal sequence, the spending policy,
    the failure branch and the zero-padding are all still the engine's -- only
    the blend is restated, which is the single thing being asserted. It is as
    much mirror as the property requires and no more.

    Stated as a forward recomputation rather than solving for w out of the
    balance path, which would work algebraically but divides by (Gs - Gb) and
    so needs a tolerance that scales with how close the two assets' returns
    came that year. Forward has no division, needs no conditioning guard, and
    separates clean from broken by about twelve orders of magnitude.

    The absolute tolerance is 1e-12 rather than machine epsilon, and that is
    not slack for its own sake. The engine compounds twelve monthly returns on
    the already-split stock and bond amounts; this check compounds each series
    and then blends. Those are equal in exact arithmetic but not bitwise, so
    the two paths diverge in the last few bits. On a cohort that runs out, the
    engine books the remaining balance as that year's spend and hard-zeros the
    portfolio, while the recomputation is left holding that divergence -- about
    1e-15, which then rides along through the zero-padding. Clean runs sit near
    1e-15 and a broken allocation shows up around 0.5, so 1e-12 keeps roughly
    twelve orders of separation and still tolerates the float path.
    """
    for H in strat.horizons:
        if year not in bt.cohort_start_years(H):
            continue  # no full H-year window from this start; not this test's business
        i = bt.jan_index(year)
        r = strat.run(i, H)
        spend, bal = r["spend_path"], r["balance_path"]

        port = 1.0
        for yr in range(H):
            port -= spend[yr]
            w = strat.weight_fn(yr)
            sl = slice(i + yr * 12, i + yr * 12 + 12)
            growth = (w * np.prod(1.0 + bt.RS[sl])
                      + (1.0 - w) * np.prod(1.0 + bt.RB[sl]))
            port = port * growth
            assert port == pytest.approx(bal[yr], rel=1e-9, abs=1e-12), (
                f"{strat.name} cohort {year} (H={H}): the engine ended year "
                f"{yr} at {bal[yr]!r}, but blending at the declared weight {w} "
                f"gives {port!r} -- the engine is not running the declared "
                f"allocation"
            )


def test_catalog_covers_every_distinct_allocation_run_analysis_reports():
    """The catalog has not quietly lost a strategy the headline tables report."""
    envelopes = {(s.equity_min, s.equity_max) for s in st.CATALOG}
    for expected in [(0.50, 0.50), (0.60, 0.60), (0.75, 0.75), (1.00, 1.00),
                     (0.60, 1.00), (0.30, 0.70)]:
        assert expected in envelopes, f"no catalog strategy declares {expected}"
