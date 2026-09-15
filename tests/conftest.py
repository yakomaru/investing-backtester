"""Put swr-team/quant on the import path so the invariants run from the repo root.

backtest.py resolves its CSV relative to its own file (see load_series), so the
only thing the tests need is for the directory to be importable.
"""
import sys
from pathlib import Path

QUANT = Path(__file__).resolve().parents[1] / "swr-team" / "quant"
if str(QUANT) not in sys.path:
    sys.path.insert(0, str(QUANT))
