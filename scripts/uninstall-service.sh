#!/bin/bash
# RF-DETR Service Uninstall Script
# Stops and removes the systemd service

SERVICE_NAME="bottle-detection"

echo "==================================="
echo "RF-DETR Service Uninstall"
echo "==================================="
echo ""

# Check if service exists
if [ ! -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
    echo "Service ${SERVICE_NAME} is not installed"
    exit 0
fi

echo "Stopping service..."
sudo systemctl stop ${SERVICE_NAME} 2>/dev/null || true

echo "Disabling service..."
sudo systemctl disable ${SERVICE_NAME} 2>/dev/null || true

echo "Removing service file..."
sudo rm -f /etc/systemd/system/${SERVICE_NAME}.service

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo ""
echo "==================================="
echo "Service uninstalled successfully"
echo "==================================="
echo ""
echo "Note: Log files in logs/ directory were NOT removed"
