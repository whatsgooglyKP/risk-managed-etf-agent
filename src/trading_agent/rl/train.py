"""Train the PPO exposure scaler on out-of-sample forecasts up to rl.train_end."""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("trading_agent.rl")


def train_ppo(prices, forecasts, cfg, model_path: str | Path, timesteps: int | None = None,
              seed: int = 7):
    import torch
    from stable_baselines3 import PPO

    from ..backtest.engine import Backtester
    from .env import ExposureEnv

    torch.set_num_threads(2)
    rl = cfg["rl"]
    bt = Backtester(prices, forecasts, cfg, cfg["backtest"]["start"], rl["train_end"])
    env = ExposureEnv(bt, episode_len=252, risk_aversion=rl.get("risk_aversion", 10.0),
                      dd_penalty=rl.get("dd_penalty", 0.5), min_scale=rl["min_scale"],
                      max_scale=rl["max_scale"], seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=1024, batch_size=128,
                learning_rate=3e-4, gamma=0.97, ent_coef=0.01,
                policy_kwargs={"net_arch": [32, 32]}, device="cpu")
    t = time.time()
    model.learn(total_timesteps=int(timesteps or rl["timesteps"]))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    log.info("PPO trained in %.0fs -> %s", time.time() - t, model_path)
    return model


def load_policy(cfg, model_path: str | Path):
    from stable_baselines3 import PPO

    from .env import RLPolicy

    model = PPO.load(str(model_path), device="cpu")
    return RLPolicy(model, cfg["rl"]["min_scale"], cfg["rl"]["max_scale"])
