"""Invariant A, applied to the engine the original bug actually lived in.

The bug this harness exists to prevent -- a cash-buffered strategy silently
becoming an all-equity one -- did not happen in backtest.py. backtest.py has no
cash bucket at all; it splits a portfolio between stocks and bonds via a single
weight and nothing else. The three-bucket (stock / bond / cash) decumulation
engine is va_decum2.py, which imports backtest only for the return series and
defines its own simulate(). An allocation invariant over backtest.py alone
would guard a different engine than the one that broke.

So this file applies the same declared-vs-realized check there. The seam is the
pair (kind, sleeve_sf): `sleeve_sf` is the stock share *within* the invested
sleeve, and a strategy labelled "68/17/15" that is handed sleeve_sf=1.0 runs as
85/15 stocks/cash with no bonds at all, reports plausible numbers, and says
nothing. That is the original failure, reproduced exactly.

va_decum2.py and its dependencies hold personal financial figures and are
deliberately untracked, so this module skips when they are absent, and nothing
here hardcodes a dollar amount, an age, or a Social Security figure: the spend
is derived from the engine's own constant, and the claim age is pushed out of
range so no benefit is ever drawn.

One invariant deliberately NOT asserted
---------------------------------------
"Realized cash never exceeds the 15% cap" is false, and looks true enough to be
worth warning the next reader about. CASH_CAP governs how much the harvest leg
may move *into* cash, not the realized share of the portfolio. When the sleeve
crashes the denominator shrinks and the cash fraction rises above the cap on its
own -- measured as high as 15.9% -- without anything having gone wrong. The cap
is therefore checked only where the engine actually enforces it: at a STATIC
rebalance, where cash is assigned as total * CASH_CAP outright.
"""
import numpy as np
import pytest

va2 = pytest.importorskip(
    "va_decum2",
    reason="three-bucket engine is untracked (holds personal financial figures)",
)

import backtest as bt  # noqa: E402

# label -> (kind, stock share of the invested sleeve, holds a cash buffer)
# Transcribed from the labels va_decum2.STRATS reports under. This is the
# declaration half of the invariant: it must be written down separately from
# the parameters the engine runs with, or there is nothing to compare against.
DECLARED = {
    "A  literal VA (eq)":     ("VA",       1.00, True),
    "B  harvest-only (eq)":   ("HARVEST",  1.00, True),
    "C  85/15 rebal (eq)":    ("STATIC",   1.00, True),
    "A2 literal VA 80/20":    ("VA",       0.80, True),
    "B2 harvest-only 80/20":  ("HARVEST",  0.80, True),
    "C2 static 68/17/15":     ("STATIC",   0.80, True),
    "D  75/25 baseline":      ("BASE7525", 0.75, False),
}

COHORTS = [1906, 1929, 1966, 1973]
REBAL = 12
NEVER_CLAIMS = va2.START_AGE + 200   # keeps every Social Security figure out of this file
SPEND = va2.P0 * 0.04                # derived from the engine, not written down here


def run(year, kind, sleeve_sf):
    """One traced cohort. Returns per-month (stock, bond, cash) as an array."""
    _, cash_empty, _, tr = va2.simulate(
        bt.jan_index(year), SPEND, NEVER_CLAIMS, 0.0, kind, sleeve_sf, trace=True,
    )
    return np.array([(s, b, c) for _, s, b, c, _ in tr], dtype=float), cash_empty


def test_registry_parameters_match_the_labels_they_are_reported_under():
    """Every STRATS entry runs with the allocation its label claims.

    The cheapest possible statement of the original bug: the label says
    "68/17/15" and the parameter says sleeve_sf=0.8, and if someone edits one
    without the other, every downstream table silently describes a different
    portfolio than the one that ran.
    """
    assert set(va2.STRATS) == set(DECLARED), (
        "the strategy registry gained or lost an entry; update DECLARED "
        "deliberately after checking what the new label claims"
    )
    for label, (kind, sleeve_sf) in va2.STRATS.items():
        want_kind, want_sf, _ = DECLARED[label]
        assert kind == want_kind, f"{label!r} is labelled {want_kind} but runs as {kind}"
        assert sleeve_sf == pytest.approx(want_sf), (
            f"{label!r} claims a {want_sf:.0%} stock sleeve but runs with {sleeve_sf:.0%}"
        )


@pytest.mark.parametrize("label", list(DECLARED))
@pytest.mark.parametrize("year", COHORTS)
def test_strategy_declaring_bonds_actually_holds_bonds(label, year):
    """A sleeve declared with a bond component never runs all-equity.

    This is the original bug stated in the most direct form available: not
    "the weights drifted" but "the thing that was supposed to be there is
    simply gone". Robust to intra-year drift, which is why it is worth having
    alongside the exact ratio check below rather than being subsumed by it.
    """
    kind, sleeve_sf, _ = DECLARED[label]
    if sleeve_sf >= 1.0:
        pytest.skip("declares an all-equity sleeve, so holding no bonds is correct")

    buckets, _ = run(year, kind, sleeve_sf)
    stock, bond = buckets[:, 0], buckets[:, 1]
    invested = stock + bond
    live = invested > 1e-9
    assert live.any(), "cohort never held an invested sleeve at all"
    assert np.all(bond[live] > 0.0), (
        f"{label!r} declares a {1 - sleeve_sf:.0%} bond sleeve but held zero bonds "
        f"in {int((~(bond > 0))[live].sum())} of {int(live.sum())} live months "
        f"of the {year} cohort"
    )


@pytest.mark.parametrize("label", list(DECLARED))
@pytest.mark.parametrize("year", COHORTS)
def test_sleeve_is_rebalanced_to_its_declared_split(label, year):
    """At every rebalance, the sleeve holds exactly its declared stock share.

    Checked at rebalance boundaries rather than every month because that is
    where the engine actually enforces the ratio. Between annual rebalances a
    STATIC sleeve drifts with relative performance -- measured between 73.7%
    and 84.5% for a declared 80% -- and asserting the exact ratio there would
    be asserting that markets do not move. VA and HARVEST re-impose the ratio
    every month, so for those this is simply the tighter check.
    """
    kind, sleeve_sf, _ = DECLARED[label]
    buckets, _ = run(year, kind, sleeve_sf)
    stock, bond = buckets[:, 0], buckets[:, 1]
    invested = stock + bond

    months = np.arange(len(buckets))
    at_rebalance = (months % REBAL == REBAL - 1) & (invested > 1e-9)
    assert at_rebalance.any(), "cohort never reached a rebalance with money left"

    realized = stock[at_rebalance] / invested[at_rebalance]
    assert np.allclose(realized, sleeve_sf, atol=1e-9), (
        f"{label!r} declares a {sleeve_sf:.0%} stock sleeve but rebalances to "
        f"between {realized.min():.4f} and {realized.max():.4f} in the {year} cohort"
    )


@pytest.mark.parametrize("label", list(DECLARED))
@pytest.mark.parametrize("year", COHORTS)
def test_cash_buffer_exists_exactly_when_it_is_declared(label, year):
    """A strategy declaring a buffer starts with one; one that does not never holds cash.

    The 'silently converted to all-equity' failure in its cash dimension. A
    strategy whose buffer was never funded would run as a pure invested sleeve
    while every table still called it cash-buffered.
    """
    kind, sleeve_sf, holds_cash = DECLARED[label]
    buckets, _ = run(year, kind, sleeve_sf)
    cash = buckets[:, 2]

    if not holds_cash:
        assert np.all(cash == 0.0), (
            f"{label!r} declares no cash buffer but held cash in "
            f"{int((cash > 0).sum())} months"
        )
        return

    assert cash[0] > 0.0, f"{label!r} declares a cash buffer but started with none"

    if kind == "STATIC":
        # The one place the cap is enforced outright: cash is assigned as
        # total * CASH_CAP at each rebalance.
        total = buckets.sum(axis=1)
        months = np.arange(len(buckets))
        at_rebalance = (months % REBAL == REBAL - 1) & (total > 1e-9)
        share = cash[at_rebalance] / total[at_rebalance]
        assert np.allclose(share, va2.CASH_CAP, atol=1e-9), (
            f"{label!r} rebalances cash to between {share.min():.4f} and "
            f"{share.max():.4f} of the portfolio, not the declared {va2.CASH_CAP}"
        )


@pytest.mark.parametrize("sleeve_sf", [1.00, 0.80])
@pytest.mark.parametrize("year", COHORTS)
def test_buy_leg_drains_the_buffer_faster_than_harvesting_alone(year, sleeve_sf):
    """VA empties the buffer no later than HARVEST, which empties no later than STATIC.

    Pins the panel's conclusion that the value-averaging *buy* leg is harmful in
    decumulation: it spends the cash buffer back into a falling sleeve, so the
    buffer that was supposed to carry the retiree through the drawdown is gone
    within a year of it starting. Measured here at 5-25 months for VA against
    44-62 for harvest-only and 203+ (or never) for a static rebalance.

    Stated as an ordering rather than a month count on purpose: the counts move
    with the portfolio and spending constants, which are personal and change,
    while the ordering is the finding and has held on every cohort tested.
    """
    never = 10 ** 9
    emptied = {}
    for kind in ("VA", "HARVEST", "STATIC"):
        _, cash_empty = run(year, kind, sleeve_sf)
        emptied[kind] = never if cash_empty is None else cash_empty

    assert emptied["VA"] <= emptied["HARVEST"], (
        f"{year} sleeve_sf={sleeve_sf}: the buy leg preserved the buffer longer "
        f"than harvest-only ({emptied['VA']} vs {emptied['HARVEST']} months)"
    )
    assert emptied["HARVEST"] <= emptied["STATIC"], (
        f"{year} sleeve_sf={sleeve_sf}: harvest-only preserved the buffer longer "
        f"than a static rebalance ({emptied['HARVEST']} vs {emptied['STATIC']} months)"
    )
