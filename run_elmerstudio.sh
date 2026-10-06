#!/usr/bin/env sh
# Launch Elmer Studio (Linux / macOS). Creates the virtual environment on first run.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m elmerstudio "$@"
