#!/usr/bin/env bash
# One-command daily run for cron/launchd. Paper trading unless you pass --live
# (and set execution.mode: live in configs/default.yaml).
#   crontab -e   ->   50 9 * * 1-5  /path/to/trading-agent/scripts/run_daily.sh >> /path/to/trading-agent/logs/cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
mkdir -p logs
exec python -m trading_agent.cli daily "$@"
