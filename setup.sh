#!/bin/bash
# Autofollow — one-click setup for macOS
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="python3"

echo "=== Autofollow Setup ==="

# Verify Python 3 is available
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found. Install it from https://www.python.org/downloads/ and try again."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PYTHON_VERSION"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists — skipping creation."
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip quietly
pip install --upgrade pip --quiet

# Install dependencies only if any are missing
echo "Checking dependencies..."
if ! pip install -r "$SCRIPT_DIR/requirements.txt" --quiet; then
    echo "ERROR: Dependency installation failed. Check the output above."
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo "Run the app with:  ./run.sh"
