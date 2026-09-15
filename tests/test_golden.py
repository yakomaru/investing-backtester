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
