"""Invariant G -- golden snapshots.

Catches: any change that silently moves a published number.

This is the bluntest invariant in the harness and the one that would most
directly have caught the original bug. It does not know what a strategy is or
what it ought to do; it only knows that run_analysis.py printed these exact
characters before, so if a refactor changes them, someone has to look. Every
other invariant in this suite states a property and can only catch violations
of that property. This one catches everything, and explains nothing -- which is
why it is worth having alongside them rather than instead of them.

Stated as a byte-for-byte match of stdout rather than a tolerance on parsed
numbers because the whole point is to be maximally sensitive. A reformatting of
the table is a real change to the artifact this repo produces, and having to
re-bless the snapshot deliberately is the correct amount of friction.
"""
import subprocess
import sys

import numpy as np
from pathlib import Path

import pytest

QUANT = Path(__file__).resolve().parents[1] / "swr-team" / "quant"
GOLDEN = Path(__file__).parent / "golden" / "run_analysis.txt"

import strategies as st  # noqa: E402  (conftest puts QUANT on sys.path)


def test_run_analysis_output_is_unchanged():
    """The headline tables are byte-identical to the blessed snapshot."""
    proc = subprocess.run(
        [sys.executable, "run_analysis.py"],
        cwd=QUANT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"run_analysis.py failed:\n{proc.stderr}"
    expected = GOLDEN.read_text()
    if proc.stdout != expected:
        exp, got = expected.splitlines(), proc.stdout.splitlines()
        diff = [
            f"  line {i + 1}:\n    expected: {e!r}\n    got:      {g!r}"
            for i, (e, g) in enumerate(zip(exp, got)) if e != g
        ]
        if len(exp) != len(got):
            diff.append(f"  line count: expected {len(exp)}, got {len(got)}")
        pytest.fail(
            "run_analysis.py output drifted from tests/golden/run_analysis.txt.\n"
            "If the change is intentional, re-bless the snapshot deliberately.\n"
            + "\n".join(diff[:20])
        )


# Headline SAFEMAX / succ@4% / worst start year, read off the blessed snapshot.
# Duplicated here on purpose: this ties the strategies.py catalog to the numbers
# run_analysis.py publishes, so the catalog cannot drift away from the script it
# mirrors without going red.
HEADLINE = {
    # (strategy name, horizon): (succ@4% pct, SAFEMAX pct, worst start year)
    ("static 50/50", 30): (95.12, 3.69, 1966),
    ("static 60/40", 30): (96.75, 3.74, 1966),
    ("static 75/25", 30): (97.56, 3.78, 1966),
    ("static 100/0", 30): (97.56, 3.79, 1929),
    ("static 50/50", 40): (85.84, 3.41, 1966),
    ("static 60/40", 40): (92.04, 3.47, 1966),
    ("static 75/25", 40): (92.92, 3.54, 1966),
    ("static 100/0", 40): (93.81, 3.59, 1966),
    ("glide 60->100 / 15yr", 30): (98.37, 3.85, 1966),
    ("glide 30->70 / 15yr", 30): (96.75, 3.76, 1966),
}


@pytest.mark.parametrize("key", list(HEADLINE))
def test_catalog_reproduces_headline_numbers(key):
    """Running the catalog strategy reproduces the published table row.

    strategies.py restates definitions that also live in run_analysis.py. That
    duplication is only safe if something checks the two agree; this is it.
    """
    import backtest as bt

    name, H = key
    succ4, safemax, worst = HEADLINE[key]
    strat = st.BY_NAME[name]
    assert H in strat.horizons, f"{name} does not declare horizon {H}"

    got = bt.safemax_and_worst(H, strat.weight_fn, bt.cohort_start_years(H))
    assert round(got["safemax"] * 100, 2) == pytest.approx(safemax, abs=0.005)
    assert round(got["succ4"] * 100, 2) == pytest.approx(succ4, abs=0.005)
    assert got["worst_year"] == worst

# The policy-bearing rows of run_analysis.py sections 4 and 5, read off the
# blessed snapshot. Without these, the four CAPE and two Guyton-Klinger catalog
# entries have no numeric tie to the script at all: their declared allocation is
# a flat 60/40, so the envelope checks pass whatever spending rule is bound to
# them, and swapping CapeInitial(0.010, 0.5) for CapeInitial(0.099, 0.5) or
# GuytonKlinger(0.05) for GuytonKlinger(0.12) was verified to leave the whole
# suite green. These rows close that.
CAPE_HEADLINE = {
    # name: (successes, avg wr0 %, min wr0 %, max wr0 %, avg spend %, failures)
    "cape a=1.5%,b=0.5 @60/40":       (106, 5.19, 3.35, 11.26, 4.99, 17),
    "cape a=1.5%,b=0.5 cap6% @60/40": (113, 4.89, 3.35,  6.00, 4.79, 10),
    "cape a=1.0%,b=0.5 @60/40":       (113, 4.73, 2.85, 10.76, 4.66, 10),
    "cape a=2.0%,b=0.4 @60/40":       (111, 4.95, 3.48,  9.81, 4.88, 12),
}

GK_HEADLINE = {
    # name: (successes, worst min spend %, avg min spend %, cohorts ever below
    #        the flat-4% level, share of all cohort-years below it %,
    #        avg final-year spend %, median final-year spend %)
    "GK start 5.0% @60/40": (123, 34.87, 74.00, 67, 23.17, 111.04, 104.61),
    "GK start 5.5% @60/40": (123, 31.38, 69.79, 66, 22.82,  95.44,  86.45),
}

FLAT4 = 0.04  # the 4%-rule real spending level, as run_analysis.py defines it


@pytest.mark.parametrize("name", list(CAPE_HEADLINE))
def test_catalog_reproduces_cape_headline_row(name):
    """The bound CAPE policy reproduces its published row.

    Ties the spending rule to the table. The allocation checks cannot do this:
    every CAPE entry declares a flat 60/40, so they stay green no matter what
    policy is attached.
    """
    import backtest as bt

    succ_e, avg_e, min_e, max_e, spend_e, fails_e = CAPE_HEADLINE[name]
    strat = st.BY_NAME[name]
    years = bt.cohort_start_years(30)

    wr0s, spends, succ = [], [], 0
    for y in years:
        policy = strat.policy_factory()
        r = bt.simulate(bt.jan_index(y), 30, strat.weight_fn, policy)
        wr0s.append(policy.initial_wr)
        spends.append(r["spend_path"].mean())
        succ += r["success"]

    assert succ == succ_e, f"{name}: {succ}/{len(years)} funded, table says {succ_e}"
    assert len(years) - succ == fails_e
    assert round(float(np.mean(wr0s)) * 100, 2) == pytest.approx(avg_e, abs=0.005)
    assert round(float(np.min(wr0s)) * 100, 2) == pytest.approx(min_e, abs=0.005)
    assert round(float(np.max(wr0s)) * 100, 2) == pytest.approx(max_e, abs=0.005)
    assert round(float(np.mean(spends)) * 100, 2) == pytest.approx(spend_e, abs=0.005)


@pytest.mark.parametrize("name", list(GK_HEADLINE))
def test_catalog_reproduces_guardrail_headline_row(name):
    """The bound Guyton-Klinger policy reproduces its published row.

    The guardrail rows are the ones whose headline claim is about *spending
    cuts* rather than success, so the numbers pinned here are the depth and
    frequency of those cuts -- which is the whole point of the section and
    invisible to every other invariant in the suite.
    """
    import backtest as bt

    (succ_e, worst_e, avg_min_e, ever_e, share_e,
     final_avg_e, final_med_e) = GK_HEADLINE[name]
    strat = st.BY_NAME[name]
    years = bt.cohort_start_years(30)

    succ, min_fracs, below4_shares, end_fracs, ever_below = 0, [], [], [], 0
    for y in years:
        r = bt.simulate(bt.jan_index(y), 30, strat.weight_fn, strat.policy_factory())
        succ += r["success"]
        sp = r["spend_path"]
        min_fracs.append(sp.min() / sp[0])
        below4_shares.append(float(np.mean(sp < FLAT4)))
        end_fracs.append(sp[-1] / sp[0])
        ever_below += bool((sp < FLAT4).any())

    assert succ == succ_e
    assert ever_below == ever_e
    assert round(float(np.min(min_fracs)) * 100, 2) == pytest.approx(worst_e, abs=0.005)
    assert round(float(np.mean(min_fracs)) * 100, 2) == pytest.approx(avg_min_e, abs=0.005)
    assert round(float(np.mean(below4_shares)) * 100, 2) == pytest.approx(share_e, abs=0.005)
    assert round(float(np.mean(end_fracs)) * 100, 2) == pytest.approx(final_avg_e, abs=0.005)
    assert round(float(np.median(end_fracs)) * 100, 2) == pytest.approx(final_med_e, abs=0.005)


def test_interval_envelope_strategies_are_pinned_to_a_safemax():
    """Any strategy with a non-degenerate envelope must appear in HEADLINE.

    Tightness only pins the two ends of an envelope, not the shape of the path
    between them: glidepath(0.60, 1.00, 1) conforms to (0.60, 1.00) and is
    tight while holding 100% equity for 29 of its 30 years. What actually
    catches that is the published SAFEMAX, which moves when the glide duration
    does -- so the real requirement is that every interval-envelope strategy
    has a SAFEMAX pinned. Stated structurally rather than per-entry so a
    glidepath added later cannot slip in unpinned.
    """
    pinned = {n for n, _ in HEADLINE}
    loose = [
        s.name for s in st.CATALOG
        if s.equity_min != s.equity_max and s.name not in pinned
    ]
    assert not loose, (
        f"strategies with an interval envelope and no pinned SAFEMAX: {loose}. "
        f"Tightness alone cannot see the shape of their weight path; add a "
        f"HEADLINE row."
    )


def test_every_catalog_strategy_is_tied_to_a_published_number():
    """No catalog entry escapes all three headline tables.

    The catalog is a second copy of run_analysis.py's definitions, and the only
    thing keeping a copy honest is that something checks it. An entry pinned by
    nothing is free to drift, so adding one has to mean adding its row too.
    """
    pinned = ({n for n, _ in HEADLINE} | set(CAPE_HEADLINE) | set(GK_HEADLINE))
    missing = [s.name for s in st.CATALOG if s.name not in pinned]
    assert not missing, (
        f"catalog entries tied to no published number: {missing}. Add them to "
        f"HEADLINE, CAPE_HEADLINE or GK_HEADLINE."
    )
