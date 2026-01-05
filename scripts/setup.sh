#!/bin/bash
# RF-DETR Setup Script
# Creates virtual environment and installs dependencies

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==================================="
echo "RF-DETR Setup Script"
echo "==================================="
echo "Project directory: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x "$SCRIPT_DIR/start.sh"
chmod +x "$SCRIPT_DIR/install-service.sh"
chmod +x "$SCRIPT_DIR/uninstall-service.sh"

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To test manually:"
echo "  ./scripts/start.sh"
echo ""
echo "To install as a service:"
echo "  ./scripts/install-service.sh"
