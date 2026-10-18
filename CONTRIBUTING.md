# Contributing

Thanks for taking a look! This is a personal portfolio project, but issues and PRs are welcome.

1. Create a virtual environment and install: `pip install -e ".[all]"`
2. Run the tests: `pytest` (they must pass; CI runs them on every push).
3. Keep the risk engine authoritative: new execution paths must go through
   `GuardedBroker`, and any change to `src/trading_agent/risk/` needs a test in `tests/`.
4. Never commit secrets (`.env`), data caches, models or runtime state; `.gitignore` covers them.
5. Prefer small, focused commits with descriptive messages.
