#!/usr/bin/env bash
# First-time setup on macOS/Linux: creates .venv, installs everything, copies .env.example.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU-only build for the RL module
pip install -e ".[all]"
[ -f .env ] || cp .env.example .env
echo "Done. Edit .env with your Alpaca PAPER keys, then run: source .venv/bin/activate && trading-agent check"
