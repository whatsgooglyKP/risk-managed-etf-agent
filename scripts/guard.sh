#!/usr/bin/env bash
# Optional intraday safety net: checks only the $60 hard floor / kill file and liquidates if hit.
#   */30 10-15 * * 1-5  /path/to/trading-agent/scripts/guard.sh >> /path/to/trading-agent/logs/cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
exec python -m trading_agent.cli guard "$@"
