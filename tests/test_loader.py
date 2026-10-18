import numpy as np
import pandas as pd

from trading_agent.data.loader import _align, _cache_path


def test_late_listing_is_not_backfilled():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"OLD": [1, 2, 3, 4, np.nan, 6], "NEW": [np.nan, np.nan, 10, 11, 12, 13]},
                         index=idx, dtype=float)
    opens = close.copy()
    opens.loc[idx[3], "NEW"] = np.nan
    p = _align(opens, close)
    assert p.close["NEW"].iloc[:2].isna().all()          # nothing invented before listing
    assert p.close.loc[idx[4], "OLD"] == 4               # gap forward-filled, never back-filled
    assert p.open.loc[idx[3], "NEW"] == 11               # missing open -> that day's close
    assert len(p.close) == 6


def test_cache_path_stays_short(tmp_path):
    syms = [f"SYM{i}" for i in range(50)]
    assert len(_cache_path(tmp_path, syms, "2005-01-01").name) < 100
    assert _cache_path(tmp_path, ["SPY", "QQQ"], "2005-01-01").name == "prices_QQQ-SPY_2005-01-01.csv"
