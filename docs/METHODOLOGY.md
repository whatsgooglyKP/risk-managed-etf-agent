# Methodology

How the agent forecasts, sizes, trades and protects the account, how it was tested, and
what the tests can and can't tell you. Numbers come from `trading-agent compare --sensitivity`
(tables in [`comparison.md`](comparison.md)) and `trading-agent backtest` (RL tables in
[`results.md`](results.md)).

## 1. Pipeline

```
daily bars -> ETS + ARIMA + gradient boosting -> ensemble P(up), mean, agreement
           -> signal rules (+ optional ranking / top-N) -> RiskEngine targets
           -> order planner -> GuardedBroker (RiskEngine.validate_order on every order)
           -> SimBroker (backtest) | Alpaca (paper / live)
intraday:  Alpaca IEX live prices -> monitor -> floor / drawdown / daily-loss actions
```

The backtester and the live runner share the forecasting, selection, risk and
`GuardedBroker` code. Only the broker adapter is different.

## 2. Data

- **Source:** yfinance daily bars adjusted for splits and dividends (total return), from
  2005-01-01. If yfinance fails, Alpaca's free IEX daily bars are the fallback. Symbols that
  yfinance drops from a large request are retried once. If a symbol still has no data, the
  load fails and nothing trades.
- **Staggered listings:** a session is kept if any symbol traded that day. A symbol that
  listed later (SLV, April 2006) stays NaN before its first bar. It is never back-filled
  from the future, and it gets forecasts only after `forecasting.min_history` (252) bars.
  After the first bar, isolated gaps are forward-filled and a missing open uses that day's close.
- **Freshness (live):** the agent refuses to trade if the last bar isn't the previous session.

## 3. Universes and survivorship bias

| | v1, default (`configs/default.yaml`) | v2, experiment (`configs/wide.yaml`) |
|---|---|---|
| Symbols | SPY, QQQ, TLT, GLD, IWM | 26 ETFs + 24 large-cap stocks |
| Selection | hold every symbol whose signal is on | rank daily, hold the best `top_n` = 5 |

**How the v2 list was built to limit survivorship bias:**

- **ETFs:** all launched before 2007. They were chosen by asset class: US index, international,
  Treasuries/TIPS, credit, metals, the nine Select Sector SPDRs and real estate. Later
  performance played no part. This includes ETFs that did badly over the period (EWZ, FXI, SLV,
  XLE).
- **Stocks:** the largest US companies by market cap at the **end of 2007**, i.e. a
  point-in-time list as of the backtest start, not today's winners. It includes later
  disappointments: C, AIG, GE, INTC, BAC, PFE, T, IBM.
- **Remaining bias:** members of that list that were later acquired or delisted can't be
  downloaded for free, so they're missing. For example, 2007 large caps like Wachovia or Merrill
  Lynch never get the chance to lose money in the backtest. This bias flatters v2, and v2
  still lost money.
- **v1 bias:** its five ETFs are hand-picked by asset class and all survived. A list that
  short, chosen with hindsight, is a mild form of the same bias.

## 4. Forecasting

- **ETS:** additive damped trend on log prices (statsmodels `ExponentialSmoothing`).
  - Refit every 21 days on a 504-day window.
  - Between refits the state is updated recursively, which is tested to match statsmodels exactly.
- **ARIMA:** the best of (1,0,0), (0,0,1), (1,0,1) by AIC, on log returns.
  - Same refit schedule as ETS.
  - Updated daily in between with `extend`, so parameters are not re-estimated.
- **Gradient boosting:** sklearn `HistGradientBoosting`, pooled across all symbols.
  - Features: lagged returns, momentum, volatility, MA distance, RSI, market regime and a symbol id.
  - Target: whether the 5-day forward return is positive.
  - Retrained every 63 days on an expanding window. Training rows whose 5-day target ends
    after the cutoff are dropped, so no labels leak.
- **Ensemble:** P(up) for each model is Φ(μ/σ). The three are blended 25/25/50 into `p_up`,
  `mu`, `agreement` (how many models have μ > 0) and `confidence` = 2·|p_up − 0.5|.
- **Leakage tests:** in `tests/test_forecasting.py`, changing prices after date t does not
  change any forecast at t.
- **Compute:**
  - ETS/ARIMA run one process per symbol, with one BLAS thread per process. Before this
    fix, N processes × M BLAS threads made a run of minutes take over half an hour.
  - Each symbol's walk-forward result is cached in `results/fc_cache/`, keyed by a hash of
    its exact price history and the model settings. Symbols shared between configs are
    computed only once.
  - A cold 50-symbol × 4,728-day run takes about 9 minutes on 6 cores, plus about 4 minutes
    for gradient boosting. The 5-ETF forecasts reproduce the earlier results bit for bit.

## 5. Signals and portfolio selection

**Per-symbol rule (v1 and v2):**

- Enter when `p_up ≥ 0.55`, at least 2 of 3 models agree, and price is above its 200-day average.
- Stay in while `p_up ≥ 0.50` and the trend filter holds. This hysteresis cuts churn.

**v2 ranking (`portfolio:` in the config):**

1. Eligible symbols are ranked by `rank_by` (`p_up`, `confidence` or `mu_over_vol`).
2. Current holdings are kept first while they rank within `top_n + hold_buffer`.
   `hold_buffer: null` keeps a holding until its own exit rule fires.
3. Free slots go to the highest-ranked new candidates.
4. A candidate is skipped if its 120-day return correlation with a symbol already picked
   is above `max_corr` (0.85). This stops the agent holding near-duplicates like SPY and MDY.

## 6. Risk engine (authoritative; every order is validated)

| Rule | Setting |
|---|---|
| Volatility sizing | weight = min(`max_position`, 10% ÷ EWMA vol) |
| Max position | 35% (v1), 25% (v2) |
| Sector cap (v2) | ≤ 40% of equity in one `sectors` group |
| Max open positions (v2) | 6 symbols |
| Portfolio vol cap (v2) | ex-ante vol from 120-day covariance ≤ 10%; scales weights down, never up; unknown covariance is treated as perfectly correlated |
| Cash only | gross exposure ≤ 98%, no margin, no shorting, no leveraged or inverse ETFs |
| Drawdown | ≥ 8% → half size; ≥ 12% → flat for 10 business days, then resume at half size with the peak reset |
| Daily loss cap | down ≥ 3% vs the previous close → no new buying that day |
| **Hard floor** | equity ≤ $60 → liquidate, write `state/HALT`, halt permanently |
| PDT guard | under $25k, at most 3 day trades per rolling 5 business days |

**Interaction to know about:** the drawdown brake resets the peak after each cooldown. Two
consecutive 12% drawdowns (0.88² × $77 ≈ $59.6) therefore reach the $60 floor. That is
exactly how v2 hit the floor in March 2011.

## 7. Execution and costs

- Decisions are made at the close and filled at the next open.
- Costs are 5 bps of slippage plus a 1 bp half-spread per side. Alpaca charges $0 commission.
- Orders are fractional and sized in dollars, with a $1 minimum. Sells go first, then buys
  funded by cash. A 5%-of-equity rebalance band skips tiny trades.
- Live orders carry deterministic client order ids, so a re-run can't double-submit.
  Intraday-monitor orders get their own id suffix (for example `-mon1130`), so they never
  collide with the morning's orders.

## 8. Intraday risk monitor

`trading-agent monitor` (code in `execution/monitor.py`, mocked tests in `tests/test_monitor.py`).

- **Equity:** the monitor marks every holding with the latest trade from Alpaca's free IEX
  feed (REST polling). It uses the conservative value, **min(broker-reported equity,
  re-marked equity)**.
  - A quote older than `monitor.max_price_age_sec` (900 s) is ignored, and the broker's own
    mark is used for that symbol.
  - If the feed is down, broker marks are used, so the floor check never depends on the feed.
- **Actions, in priority order:**
  1. **Floor or kill file:** cancel open orders, sell everything, halt permanently. This is
     the **only** case where the PDT guard is overridden, and the override is logged.
     Losing the account is worse than a PDT flag.
  2. **Drawdown ≥ 12%:** start the same 10-day cooldown the daily run uses and block buying.
     With `on_drawdown: flatten` (the default) it also sells positions, except any bought
     today, so it never creates a day trade. The next daily run sells those.
  3. **Daily loss ≥ 3%:** block new buying for the rest of the day. The daily run sees this
     too. `on_daily_loss: flatten` also sells, under the same PDT rule as (2).
- **Concurrency:** a file lock (`state/.lock`) stops the monitor and the daily run from
  editing state at the same time. On GitHub Actions the two workflows share a concurrency group.
- **Free ways to run it:**
  - `.github/workflows/intraday-monitor.yml` runs every 20 minutes on weekdays, and the script
    itself checks the market clock. That's about 570 billed minutes a month, inside a private
    repo's 2,000 free minutes, and unlimited for public repos.
  - Locally, cron runs `scripts/monitor.sh` every 15 minutes, or
    `trading-agent monitor --loop` checks every 60 seconds until the close.
- **Limits:**
  - It is not tick-by-tick. A gap through $60 between checks, or overnight, still happens.
  - GitHub can delay scheduled runs.
  - The monitor is **not** simulated in the daily-bar backtest. Backtest numbers reflect
    close-to-close checks only.

## 9. Backtest protocol

- **Walk-forward and out-of-sample:** every forecast at t uses data up to t only. Trading
  starts on 2008-01-02, after 3 years of history.
- **Periods:**
  - the full period (2008-01-02 to 2026-10-16);
  - 2008–2016 and 2017–present, each restarted from a fresh $77. The split is the same as the
    original RL held-out test.
- **Benchmark:** buy-and-hold SPY with the same entry cost.
- **Same engine:** every config is run by the same engine with the same costs and risk rules.
- **Decision rule, fixed before the comparison:** v2 replaces v1 as the default only if it
  has a higher full-period Sharpe, a max drawdown no more than 2 points worse, and a Sharpe
  that isn't worse in both sub-periods.
- **v2 settings were chosen before any v2 backtest:** top_n 5, hold_buffer 2, max_corr 0.85,
  max_position 25%, sector cap 40%, portfolio vol cap 10% (the same 10% risk budget as v1).

## 10. Results: does the wider universe help? No.

| Full period, $77 start | 5-ETF v1 (default) | Wide-50 v2 | Buy & hold SPY |
|---|---|---|---|
| CAGR | **6.26%** | −1.40% | 11.25% |
| Sharpe | **0.71** | −0.42 | 0.64 |
| Max drawdown | **−13.6%** | −23.2% (floor hit Mar 2011) | −51.9% |
| Ending equity | $240.84 | $59.12 | $564.16 |

| 2017–present, fresh $77 | 5-ETF v1 | Wide-50 v2 | SPY |
|---|---|---|---|
| CAGR / Sharpe | 5.82% / 0.65 | 0.48% / 0.10 | 15.06% / 0.87 |
| Max drawdown | −13.6% | −30.7% | −33.7% |
| Turnover (× equity per year) | 18 | 88 | n/a |

v2 failed the decision rule on every criterion, so **v1 stays the default**. Every variant
tried is reported in `comparison.md`. None of them beat v1:

- **Top-N size** (3/5/8), **no correlation filter**, **no vol cap**, **rank by μ/vol:** all
  lost about 1.4% a year and hit the floor. The new caps weren't the cause.
- **Hold until own exit** (`hold_buffer: null`): about 0% CAGR, Sharpe 0.05, no floor hit.
  Churn was part of the problem but not all of it.
- **ETFs only, 26 of them, hold until exit:** 2.0% CAGR, Sharpe 0.29, max drawdown −22%.
  It did better than with stocks, but still clearly worse than the 5-ETF version.

**Why, as best I can tell:**

1. **Ranking by p_up selects noise.** Among 50 symbols, many have p_up between 0.55 and 0.62
   on any given day, and the ordering reshuffles daily. That turned into turnover (88× a
   year after 2017, so $40 of costs on a $77 account) and whipsaw losses (47% win rate).
2. **The ensemble's edge is trend plus volatility timing on broad, liquid assets.** It
   doesn't pick single stocks well. The ETF-only variant was clearly better than the version
   with stocks.
3. **The floor mechanics:** two 12% drawdowns in 2008–2011 were enough to halt v2 for good.

**Caveat on multiple testing:** 8 variants were tried after seeing the v2 result. They're
diagnostics, not a search for a winner. Even the best of them would be an optimistic,
data-snooped estimate, so none was promoted.

## 11. Limitations

- **18 years of daily data is a small sample of market regimes.** v1 trails SPY on return
  overall, and on both return and Sharpe since 2017.
- **Backtests leave things out:** no taxes, idealized open fills, residual survivorship bias
  (section 3), and no intraday monitor in the simulation.
- **$77 accounts:** the $1 minimum and cent rounding matter. Small orders are a real cost
  of a wide universe.
- **yfinance is unofficial.** The Alpaca fallback exists, and if both fail the agent doesn't trade.
