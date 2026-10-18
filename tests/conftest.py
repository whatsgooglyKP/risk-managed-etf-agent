import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.config import load_config  # noqa: E402
from trading_agent.risk.engine import RiskEngine  # noqa: E402

UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "IWM"]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def risk(cfg, tmp_path):
    return RiskEngine(cfg["risk"], UNIVERSE, tmp_path / "risk_state.json", tmp_path / "HALT")
