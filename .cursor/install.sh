#!/usr/bin/env bash
# Idempotent dependency setup for the rag_project Cloud Agent environment.
# Safe to run repeatedly: it only installs the missing venv toolchain, then
# (re)creates an isolated virtual environment and installs pinned deps.
set -euo pipefail

cd "$(dirname "$0")/.."

# The base image ships python3 but not the venv/ensurepip module, so add it
# once if it is missing. Guarded so reruns are cheap no-ops.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "rag_project environment ready. Activate with: source .venv/bin/activate"
