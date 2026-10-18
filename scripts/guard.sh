#!/usr/bin/env bash
# Kept for old cron lines: now runs one full intraday monitor check (see scripts/monitor.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
exec python -m trading_agent.cli monitor "$@"
