"""Wide-universe selection: ranking, top-N, hysteresis, correlation filter."""
import numpy as np
import pandas as pd

from trading_agent.strategy import long_signals, rank_scores, select_portfolio

SCFG = {"enter_prob": 0.55, "exit_prob": 0.50, "min_agreement": 2, "trend_filter": False}


def fc(p: dict, agreement=3, mu=None):
    df = pd.DataFrame({"p_up": p}).astype(float)
    df["agreement"] = agreement
    df["mu"] = mu if mu is not None else (df["p_up"] - 0.5) / 100
    df["confidence"] = (2 * (df["p_up"] - 0.5).abs())
    return df


P = {"A": 0.70, "B": 0.66, "C": 0.62, "D": 0.60, "E": 0.58, "F": 0.40}


def test_top_n_null_is_the_original_rule():
    f = fc(P)
    assert select_portfolio(f, {}, SCFG, {"top_n": None}) == long_signals(f, {}, SCFG)


def test_picks_the_strongest_n():
    out = select_portfolio(fc(P), {}, SCFG, {"top_n": 3, "hold_buffer": 0})
    assert [s for s, v in out.items() if v] == ["A", "B", "C"]


def test_never_picks_ineligible_or_missing_forecasts():
    p = dict(P, G=np.nan)
    out = select_portfolio(fc(p), {}, SCFG, {"top_n": 10})
    assert not out["F"] and not out["G"]          # below enter_prob / no data yet
    assert sum(out.values()) == 5


def test_hysteresis_keeps_a_holding_that_slipped_slightly():
    held = {"E": True}                            # E ranks 5th of 5 eligible
    out = select_portfolio(fc(P), held, SCFG, {"top_n": 3, "hold_buffer": 2})
    assert out["E"] and sum(out.values()) == 3    # E kept, still only 3 names
    out = select_portfolio(fc(P), held, SCFG, {"top_n": 3, "hold_buffer": 0})
    assert not out["E"]


def test_correlation_filter_skips_near_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    rets = pd.DataFrame({s: rng.normal(0, 0.01, 200) for s in P})
    rets["B"] = base
    rets["A"] = base + rng.normal(0, 0.001, 200)  # A and B ~ identical
    out = select_portfolio(fc(P), {}, SCFG, {"top_n": 3, "hold_buffer": 0, "max_corr": 0.85}, returns=rets)
    assert [s for s, v in out.items() if v] == ["A", "C", "D"]


def test_rank_by_mu_over_vol():
    f = fc({"X": 0.60, "Y": 0.60}, mu=pd.Series({"X": 0.001, "Y": 0.001}))
    s = rank_scores(f, "mu_over_vol", {"X": 0.10, "Y": 0.40})
    assert s["X"] > s["Y"]                         # same mean, lower vol -> better
