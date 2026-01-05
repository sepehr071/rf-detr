#!/bin/bash
# RF-DETR One-Click Install Script
# Runs setup.sh, setup-camera.sh, and install-service.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==================================="
echo "RF-DETR One-Click Install"
echo "==================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Run Python setup
echo "Step 1/3: Running Python setup..."
echo ""
"$SCRIPT_DIR/setup.sh"

# Check if setup succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Python setup failed."
    exit 1
fi

# Step 2: Run camera setup (requires sudo)
echo ""
echo "Step 2/3: Setting up camera permissions..."
echo ""
if [ -f "$SCRIPT_DIR/setup-camera.sh" ]; then
    chmod +x "$SCRIPT_DIR/setup-camera.sh"
    sudo "$SCRIPT_DIR/setup-camera.sh"
else
    echo "WARNING: setup-camera.sh not found, skipping camera setup"
fi

# Step 3: Install service
echo ""
echo "Step 3/3: Installing service..."
echo ""
"$SCRIPT_DIR/install-service.sh"

echo ""
echo "==================================="
echo "Installation complete!"
echo "==================================="
echo ""
echo "The service is now running. Check status with:"
echo "  sudo systemctl status bottle-detection"
echo ""
echo "IMPORTANT: Log out and log back in for group changes to take effect."
