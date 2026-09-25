#!/bin/sh
# Launch MsMBG-Fit on Linux / macOS. Installs uv first if it is missing.
set -e
cd "$(dirname "$0")"

# uv installs to ~/.local/bin (older versions: ~/.cargo/bin), which may not be on PATH yet
PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing it..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Error: curl or wget is required to install uv." >&2
        exit 1
    fi
fi

# Update to the latest version if this is a git clone (skipped if offline or with local changes)
if [ -d .git ] && command -v git >/dev/null 2>&1; then
    echo "Checking for updates..."
    git pull --ff-only || echo "Warning: could not update, starting the current version."
fi

# uv run installs Python 3.13 and syncs dependencies if needed before starting
uv run python main.py
