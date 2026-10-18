#!/usr/bin/env bash
# Intraday risk monitor (live IEX prices vs $60 floor, daily loss cap, drawdown).
# One check per call, e.g. from cron every 15 minutes during market hours (ET):
#   */15 9-15 * * 1-5  /path/to/trading-agent/scripts/monitor.sh >> /path/to/trading-agent/logs/cron.log 2>&1
# Or keep it running all session on your own computer (checks every monitor.interval_sec):
#   ./scripts/monitor.sh --loop
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
exec python -m trading_agent.cli monitor "$@"
