#!/bin/bash
# Autofollow launcher script for macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the app with all arguments passed through
python "$SCRIPT_DIR/main.py" "$@"
