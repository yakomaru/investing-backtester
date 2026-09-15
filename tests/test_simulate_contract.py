"""Invariant F -- the shape of what simulate() returns, especially on failure.

Catches: a portfolio that fails but reports paths as though it did not, and any
drift in the one branch of simulate() that contains real bookkeeping.

Why the failure branch specifically
-----------------------------------
The success path of simulate() is a loop with nothing to get wrong twice. The
failure path is where the bookkeeping lives: it appends the *remaining*
portfolio as that year's spend rather than the requested amount, zeroes the
portfolio, then pads both paths with zeros so every cohort's arrays are length
H regardless of when it died. Downstream code relies on that padding without
checking it -- run_analysis.py takes `spend_path.mean()` and `sp[-1] / s[0]`
across cohorts, so a failing cohort whose array came back short or unpadded
would silently skew an average rather than raise.

Why this is not the "conservation" invariant originally proposed
----------------------------------------------------------------
A full conservation identity -- spend plus terminal reconciles against the
return path -- can only be written by re-implementing simulate() inside the
test. A mirror catches transcription slips in a refactor but cannot catch a
wrong model, because it is wrong in exactly the same way. What is asserted here
instead are relations that hold *between* the returned arrays, which are
genuinely redundant information: the withdrawal that broke the cohort must
equal the balance the previous year ended with. That is a real cross-check, and
it is where the padding logic can actually go wrong.
"""
import numpy as np
import pytest

import backtest as bt

WF = bt.static_w(0.60)
H = 30
# 1966 is the worst start year in the Shiller series at every allocation, so a
# 10% fixed-real withdrawal is a reliable failure and 3% a reliable success.
FAIL_YEAR, FAIL_WR = 1966, 0.10
OK_YEAR, OK_WR = 1966, 0.03


def test_failing_cohort_returns_full_length_padded_paths():
    """A cohort that runs out still returns H entries in both paths."""
    r = bt.simulate(bt.jan_index(FAIL_YEAR), H, WF, bt.FixedReal(FAIL_WR))
    assert r["success"] is False
    assert len(r["spend_path"]) == H, "spend_path was not padded to the horizon"
    assert len(r["balance_path"]) == H, "balance_path was not padded to the horizon"
    assert r["terminal"] == 0.0


def test_failing_cohort_spends_exactly_what_was_left():
    """The withdrawal that broke the cohort equals the prior year's ending balance.

    This is the cross-check the padding can break: spend_path[k] is not the
    requested FAIL_WR, it is whatever the portfolio still held, and the only
    other record of that amount is balance_path[k-1].
    """
    r = bt.simulate(bt.jan_index(FAIL_YEAR), H, WF, bt.FixedReal(FAIL_WR))
    spend, bal = r["spend_path"], r["balance_path"]

    full = np.flatnonzero(spend < FAIL_WR - 1e-12)
    assert len(full) > 0, "expected a partial final withdrawal"
    k = int(full[0])
    assert k > 0, "cohort failed on its very first withdrawal; pick a lower rate"

    assert np.allclose(spend[:k], FAIL_WR), "years before failure were not funded in full"
    assert 0.0 <= spend[k] < FAIL_WR, "the partial withdrawal is not a partial withdrawal"
    assert spend[k] == pytest.approx(bal[k - 1], abs=1e-12), (
        f"failed in year {k} spending {spend[k]!r} but ended year {k - 1} "
        f"holding {bal[k - 1]!r}"
    )
    assert np.all(spend[k + 1:] == 0.0), "post-failure years were not zero-padded"
    assert np.all(bal[k:] == 0.0), "post-failure balances were not zero-padded"


def test_surviving_cohort_reconciles_terminal_with_its_last_balance():
    """terminal and balance_path[-1] are computed separately and must agree."""
    r = bt.simulate(bt.jan_index(OK_YEAR), H, WF, bt.FixedReal(OK_WR))
    assert r["success"] is True
    assert len(r["spend_path"]) == H
    assert r["terminal"] == pytest.approx(r["balance_path"][-1], rel=1e-15)
    assert r["terminal"] > 0.0
    assert np.allclose(r["spend_path"], OK_WR), "a funded cohort took a partial withdrawal"
    assert np.all(r["balance_path"] > 0.0)


@pytest.mark.parametrize("year", [1871, 1929, 1966, 1982, 1993])
def test_zero_spend_cohort_is_pure_compounded_growth(year):
    """With nothing withdrawn, terminal equals the compounded return of the weights.

    A calibration anchor rather than a mirror: the expected value is built by
    a different route (per-year products over the monthly slices) than
    simulate()'s running loop, and it pins the engine's core claim -- annual
    rebalancing to the target weight, monthly compounding within the year -- to
    an independently computed number.
    """
    i = bt.jan_index(year)
    got = bt.simulate(i, H, WF, bt.FixedReal(0.0))

    port = 1.0
    for yr in range(H):
        w = WF(yr)
        sl = slice(i + yr * 12, i + yr * 12 + 12)
        port = (port * w * np.prod(1.0 + bt.RS[sl])
                + port * (1.0 - w) * np.prod(1.0 + bt.RB[sl]))

    assert got["success"] is True
    assert got["terminal"] == pytest.approx(port, rel=1e-12)
    assert np.all(got["spend_path"] == 0.0)


def test_cohort_windows_stay_inside_the_data():
    """cohort_start_years only offers starts with a full H years of returns behind them.

    Off-by-one here would read past the end of the series or silently drop the
    last usable cohort, either of which moves SAFEMAX without any other signal.
    """
    for horizon in (30, 40):
        years = bt.cohort_start_years(horizon)
        last = bt.jan_index(years[-1]) + horizon * 12 - 1
        assert last <= bt.N - 1, "last cohort reads past the end of the series"
        beyond = bt.jan_index(years[-1] + 1) + horizon * 12 - 1
        assert beyond > bt.N - 1, "a usable cohort was dropped from the end"
        assert years[0] == bt.BASE_YEAR
