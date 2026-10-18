"""Configuration loading (YAML) + secrets (environment / .env)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Leveraged / inverse ETFs are never tradable, even if someone adds them to the universe.
FORBIDDEN_SYMBOLS = {
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU", "SSO", "SDS", "QLD", "QID",
    "TNA", "TZA", "UVXY", "SVXY", "VXX", "TMF", "TMV", "SOXL", "SOXS", "LABU",
    "LABD", "FAS", "FAZ", "NUGT", "DUST", "JNUG", "JDST", "UDOW", "SDOW", "SH", "PSQ",
}


class Config(dict):
    """dict with attribute access for nested sections (cfg.risk.hard_floor)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - developer error
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    """Load default.yaml, optionally layered with a user YAML and dict overrides."""
    with open(DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    validate_config(cfg)
    return Config(cfg)


def validate_config(cfg: dict) -> None:
    bad = [s for s in cfg["universe"] if s.upper() in FORBIDDEN_SYMBOLS]
    if bad:
        raise ValueError(f"Leveraged/inverse ETFs are not allowed in the universe: {bad}")
    r = cfg["risk"]
    if not 0 < r["hard_floor"] < r["initial_capital"]:
        raise ValueError("risk.hard_floor must be between 0 and initial_capital")
    if r["max_gross_exposure"] > 1.0:
        raise ValueError("Cash-only: risk.max_gross_exposure must be <= 1.0 (no margin)")
    if not 0 < r["max_position"] <= 1.0:
        raise ValueError("risk.max_position must be in (0, 1]")
    if cfg["rl"]["max_scale"] > 1.0:
        raise ValueError("rl.max_scale must be <= 1.0: RL can only reduce risk-approved size")
    uni = [s.upper() for s in cfg["universe"]]
    if len(set(uni)) != len(uni):
        raise ValueError("universe contains duplicate symbols")
    if cfg["benchmark"].upper() not in uni:
        raise ValueError("benchmark must be part of the universe (it drives market features)")
    unknown = [s for s in (cfg.get("sectors") or {}) if s.upper() not in uni]
    if unknown:
        raise ValueError(f"sectors lists symbols that are not in the universe: {unknown}")
    pf = cfg.get("portfolio") or {}
    top_n = pf.get("top_n")
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError("portfolio.top_n must be a positive integer or null")
    if pf.get("rank_by", "p_up") not in ("p_up", "confidence", "mu_over_vol"):
        raise ValueError("portfolio.rank_by must be p_up, confidence or mu_over_vol")
    mc = pf.get("max_corr")
    if mc is not None and not 0 < mc <= 1:
        raise ValueError("portfolio.max_corr must be in (0, 1] or null")
    for k in ("max_sector", "portfolio_vol_cap"):
        v = r.get(k)
        if v is not None and v <= 0:
            raise ValueError(f"risk.{k} must be positive or null")
    mop = r.get("max_open_positions")
    if mop is not None and (not isinstance(mop, int) or mop < 1):
        raise ValueError("risk.max_open_positions must be a positive integer or null")
    mon = cfg.get("monitor") or {}
    if mon.get("on_daily_loss", "block_buys") not in ("block_buys", "flatten"):
        raise ValueError("monitor.on_daily_loss must be block_buys or flatten")
    if mon.get("on_drawdown", "flatten") not in ("flatten", "none"):
        raise ValueError("monitor.on_drawdown must be flatten or none")


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_secrets() -> tuple[str | None, str | None]:
    """Read Alpaca keys from the environment (a local .env file is loaded if present)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass
    return os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
