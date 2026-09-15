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
import pytest

import backtest as bt
import strategies as st

TOL = 1e-12

ALL = pytest.mark.parametrize("strat", st.CATALOG, ids=lambda s: s.name)


@ALL
def test_realized_weights_stay_inside_declared_envelope(strat):
    """Every year's equity weight is inside the strategy's declared envelope."""
    for H in strat.horizons:
        for yr, w in enumerate(strat.realized_weights(H)):
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
    w = strat.realized_weights(strat.max_horizon)
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
    for w in strat.realized_weights(strat.max_horizon):
        assert 0.0 <= w <= 1.0, f"{strat.name} holds equity weight {w}"


@ALL
def test_weight_fn_depends_only_on_year(strat):
    """Asking for the same year twice gives the same weight.

    The allocation schedule must be a pure function of the retirement year.
    A weight_fn that accumulated state across calls would make the realized
    path depend on how many times it had been evaluated -- including by this
    harness, which would then be measuring something the backtest never ran.
    """
    first = strat.realized_weights(strat.max_horizon)
    second = strat.realized_weights(strat.max_horizon)
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


def test_catalog_covers_every_distinct_allocation_run_analysis_reports():
    """The catalog has not quietly lost a strategy the headline tables report."""
    envelopes = {(s.equity_min, s.equity_max) for s in st.CATALOG}
    for expected in [(0.50, 0.50), (0.60, 0.60), (0.75, 0.75), (1.00, 1.00),
                     (0.60, 1.00), (0.30, 0.70)]:
        assert expected in envelopes, f"no catalog strategy declares {expected}"
