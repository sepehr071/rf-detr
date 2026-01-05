#!/bin/bash
# RF-DETR One-Click Install Script
# Runs setup.sh followed by install-service.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==================================="
echo "RF-DETR One-Click Install"
echo "==================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Run setup
echo "Step 1/2: Running setup..."
echo ""
"$SCRIPT_DIR/setup.sh"

# Check if setup succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2/2: Installing service..."
    echo ""
    "$SCRIPT_DIR/install-service.sh"
else
    echo ""
    echo "ERROR: Setup failed. Service not installed."
    exit 1
fi

echo ""
echo "==================================="
echo "Installation complete!"
echo "==================================="
