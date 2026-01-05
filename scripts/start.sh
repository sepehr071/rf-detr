#!/bin/bash
# RF-DETR Start Script
# Activates virtual environment and runs detection

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/venv"
    echo "Run ./scripts/setup.sh first"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Run detection with web mode, full inference, NMS, and headless CLI mode
# Logs are written to both stdout (for systemd) and logs/detection.log
exec python3 main.py --web --mode full --nms --cli 2>&1 | tee -a logs/detection.log
