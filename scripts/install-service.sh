#!/bin/bash
# RF-DETR Service Installation Script
# Generates and installs systemd service for auto-startup

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="bottle-detection"
CURRENT_USER=$(whoami)

echo "==================================="
echo "RF-DETR Service Installation"
echo "==================================="
echo "Project directory: $PROJECT_DIR"
echo "Service name: $SERVICE_NAME"
echo "User: $CURRENT_USER"
echo ""

# Check if venv exists
if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Run ./scripts/setup.sh first"
    exit 1
fi

# Check if start.sh is executable
if [ ! -x "$SCRIPT_DIR/start.sh" ]; then
    chmod +x "$SCRIPT_DIR/start.sh"
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Generate systemd service file
echo "Generating systemd service file..."
cat > /tmp/${SERVICE_NAME}.service << EOF
[Unit]
Description=RF-DETR Bottle Detection Service
After=multi-user.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/scripts/start.sh
Restart=always
RestartSec=5
StandardOutput=append:${PROJECT_DIR}/logs/service.log
StandardError=append:${PROJECT_DIR}/logs/service-error.log

# CPU Performance Settings
Nice=-5
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Use all available CPU cores
Environment="OMP_NUM_THREADS=0"
Environment="MKL_NUM_THREADS=0"

[Install]
WantedBy=multi-user.target
EOF

echo "Service file created at /tmp/${SERVICE_NAME}.service"
echo ""

# Install service (requires sudo)
echo "Installing service (requires sudo)..."
sudo cp /tmp/${SERVICE_NAME}.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo ""
echo "==================================="
echo "Service installed and started!"
echo "==================================="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}    # Check status"
echo "  sudo systemctl stop ${SERVICE_NAME}      # Stop service"
echo "  sudo systemctl start ${SERVICE_NAME}     # Start service"
echo "  sudo systemctl restart ${SERVICE_NAME}   # Restart service"
echo "  sudo systemctl disable ${SERVICE_NAME}   # Disable auto-start"
echo ""
echo "View logs:"
echo "  tail -f ${PROJECT_DIR}/logs/service.log"
echo "  tail -f ${PROJECT_DIR}/logs/detection.log"
echo "  journalctl -u ${SERVICE_NAME} -f"
