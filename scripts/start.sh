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

# Check camera permissions (warning only, don't exit)
echo "[START] Checking camera permissions..."
if groups | grep -q '\bvideo\b'; then
    echo "[START] User is in 'video' group - camera access OK"
else
    echo "[START] WARNING: User is not in 'video' group"
    echo "[START] Camera access may fail. Run: sudo ./scripts/setup-camera.sh"
fi

# Check if video devices exist
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "[START] Found video devices: $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
else
    echo "[START] WARNING: No /dev/video* devices found"
fi

# Set environment for V4L2 priority
export OPENCV_VIDEOIO_PRIORITY_V4L2=1

echo "[START] Starting detection..."

# Run detection with web mode, full inference, NMS, and headless CLI mode
# Logs are written to both stdout (for systemd) and logs/detection.log
exec python3 main.py --web --mode full --nms --cli 2>&1 | tee -a logs/detection.log
