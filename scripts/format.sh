#!/bin/bash

# Format Python code with black
# Usage: ./scripts/format.sh [--check]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ "$1" == "--check" ]; then
    echo "Checking code formatting..."
    uv run black --check backend/
else
    echo "Formatting code with black..."
    uv run black backend/
fi

echo "Done!"
