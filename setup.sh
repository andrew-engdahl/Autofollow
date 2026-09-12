#!/bin/bash
# Autofollow — one-click setup for macOS
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="python3"
WITH_RECOGNITION=0
for arg in "$@"; do
    case "$arg" in
        --with-recognition) WITH_RECOGNITION=1 ;;
        -h|--help)
            echo "Usage: ./setup.sh [--with-recognition]"
            echo "  --with-recognition   also install face/speaker/music recognition (large)"
            exit 0 ;;
    esac
done

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

# The opencv-python, opencv-contrib-python and *-headless wheels all install
# the same 'cv2' package on top of each other.  Having more than one present
# produces a mixed install that can crash at import or misbehave at runtime,
# so keep exactly one (opencv-python).
extras=$(pip list --format=freeze 2>/dev/null | grep -iE '^opencv-(contrib-python|python-headless|contrib-python-headless)==' | cut -d= -f1 || true)
if [ -n "$extras" ]; then
    echo "Removing conflicting OpenCV packages: $extras"
    # shellcheck disable=SC2086
    pip uninstall -y $extras --quiet
    pip install --force-reinstall --no-deps "opencv-python>=4.10,<5" --quiet
fi

# Install dependencies only if any are missing
echo "Checking dependencies..."
if ! pip install -r "$SCRIPT_DIR/requirements.txt" --quiet; then
    echo "ERROR: Dependency installation failed. Check the output above."
    exit 1
fi

# Optional recognition stack — never fatal for the core app.
if [ "$WITH_RECOGNITION" = "1" ]; then
    echo "Installing recognition extras (this downloads several hundred MB)..."
    if ! pip install -r "$SCRIPT_DIR/requirements-recognition.txt" --quiet; then
        echo "WARNING: Recognition extras failed to install. The core app still works;"
        echo "         face/speaker/music features will show as unavailable."
    fi
fi

# Rebuild the Dock launcher's signature so macOS accepts the bundle
if [ -d "$SCRIPT_DIR/Autofollow.app" ] && command -v codesign &>/dev/null; then
    codesign --force --deep --sign - "$SCRIPT_DIR/Autofollow.app" 2>/dev/null || true
fi

echo ""
echo "=== Setup complete! ==="
echo "Run the app with:  ./run.sh   (or open Autofollow.app)"
if [ "$WITH_RECOGNITION" != "1" ]; then
    echo "For face/speaker/music recognition:  ./setup.sh --with-recognition"
fi
