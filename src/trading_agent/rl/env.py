"""Gymnasium environment: the agent picks an exposure scale each day.

Action  : Discrete(5) -> scale in {0, .25, .5, .75, 1.0} (mapped into [min_scale, max_scale]).
          The scale multiplies the RiskEngine's already-capped targets, so RL can only
          REDUCE risk; it cannot choose symbols, add leverage, or bypass any risk rule.
Reward  : 100 * (r - 0.5 * risk_aversion * r**2) - dd_penalty * 100 * (new drawdown beyond 5%),
          a mean-variance utility: full exposure pays in calm markets, while high-volatility
          regimes make the variance term dominate, so the policy learns when to de-risk.
Obs     : portfolio drawdown / momentum / volatility, SPY regime, forecast summary, exposure.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

SCALES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
OBS_DIM = 9


def build_obs(equity: np.ndarray, exposure: float, spy_vol: float, spy_mom20: float,
              mean_pup: float, frac_long: float) -> np.ndarray:
    """Shared by training, backtest evaluation and live runs."""
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    if len(eq) < 2:
        eq = np.array([1.0, 1.0])
    r = np.diff(np.log(eq))
    dd = 1 - eq[-1] / eq.max()
    ret5 = np.log(eq[-1] / eq[max(0, len(eq) - 6)])
    ret20 = np.log(eq[-1] / eq[max(0, len(eq) - 21)])
    vol20 = r[-20:].std() * np.sqrt(252) if len(r) >= 2 else 0.0
    obs = np.array([dd * 10, ret5 * 20, ret20 * 10, vol20 * 5, exposure,
                    (spy_vol if np.isfinite(spy_vol) else 0.15) * 5,
                    (spy_mom20 if np.isfinite(spy_mom20) else 0.0) * 10,
                    (mean_pup - 0.5) * 10, frac_long], dtype=np.float32)
    return np.clip(obs, -10, 10)


def scale_from_action(action: int, min_scale: float, max_scale: float) -> float:
    return float(min_scale + SCALES[int(action)] * (max_scale - min_scale))


def bt_obs(bt) -> np.ndarray:
    """Observation from a Backtester positioned just after open_and_close() of day k."""
    d = bt.dates[bt.k]
    eq = np.array([e for _, e in bt.eq[-260:]])
    expo = bt.expo[-1][1] if bt.expo else 0.0
    spy = bt.cfg["benchmark"]
    spy_vol = bt.vol.loc[d, spy]
    close = bt.prices.close[spy]
    i = close.index.get_loc(d)
    mom = np.log(close.iloc[i] / close.iloc[max(0, i - 20)])
    fc = bt.fc_by_date[d]
    frac_long = np.mean([bool(v) for v in bt.sig_long.values()])
    return build_obs(eq, expo, spy_vol, mom, float(fc["p_up"].mean()), frac_long)


class ExposureEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, backtester, episode_len: int = 252, risk_aversion: float = 10.0,
                 dd_penalty: float = 0.5,
                 min_scale: float = 0.0, max_scale: float = 1.0, seed: int | None = None):
        super().__init__()
        self.bt = backtester
        self.episode_len = episode_len
        self.dd_penalty = dd_penalty
        self.risk_aversion = risk_aversion
        self.min_scale, self.max_scale = min_scale, max_scale
        self.action_space = spaces.Discrete(len(SCALES))
        self.observation_space = spaces.Box(-10, 10, shape=(OBS_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def _dd(self) -> float:
        eq = np.array([e for _, e in self.bt.eq])
        return 1 - eq[-1] / eq.max()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bt.reset()
        n = len(self.bt.dates)
        hi = max(1, n - self.episode_len - 2)
        self.bt.k = int(self._rng.integers(0, hi))
        self.start_k = self.bt.k
        self.bt.open_and_close()
        return bt_obs(self.bt), {}

    def step(self, action):
        scale = scale_from_action(action, self.min_scale, self.max_scale)
        eq0, dd0 = self.bt.eq[-1][1], self._dd()
        self.bt.decide(scale)                       # advances k
        truncated = self.bt.done or (self.bt.k - self.start_k) >= self.episode_len
        if self.bt.done:
            return bt_obs_safe(self.bt), 0.0, False, True, {}
        eq1 = self.bt.open_and_close()
        dd1 = self._dd()
        r = np.log(max(eq1, 1e-9) / eq0)
        deep = max(0.0, dd1 - max(dd0, 0.05))          # only penalise drawdown beyond 5%
        reward = 100 * (r - 0.5 * self.risk_aversion * r * r) - self.dd_penalty * 100 * deep
        terminated = bool(self.bt.risk.state.halted)
        return bt_obs(self.bt), float(reward), terminated, bool(truncated), {"scale": scale}


def bt_obs_safe(bt):
    bt.k = min(bt.k, len(bt.dates) - 1)
    return bt_obs(bt)


class RLPolicy:
    """Callable policy for Backtester.run(policy=...) and live runs."""

    def __init__(self, model, min_scale: float = 0.0, max_scale: float = 1.0):
        self.model, self.min_scale, self.max_scale = model, min_scale, max_scale

    def scale(self, obs: np.ndarray) -> float:
        action, _ = self.model.predict(obs, deterministic=True)
        return scale_from_action(int(action), self.min_scale, self.max_scale)

    def __call__(self, bt) -> float:
        return self.scale(bt_obs(bt))
