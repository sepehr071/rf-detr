#!/bin/bash
# AyandAi GUI Service Uninstall Script
# Removes the GUI control panel service

SERVICE_NAME="ayandai-gui"

echo "==================================="
echo "AyandAi GUI Service Uninstall"
echo "==================================="
echo ""

# Check if service exists
if [ ! -f ~/.config/systemd/user/${SERVICE_NAME}.service ]; then
    echo "Service ${SERVICE_NAME} is not installed"
    exit 0
fi

echo "Stopping service..."
systemctl --user stop ${SERVICE_NAME} 2>/dev/null || true

echo "Disabling service..."
systemctl --user disable ${SERVICE_NAME} 2>/dev/null || true

echo "Removing service file..."
rm -f ~/.config/systemd/user/${SERVICE_NAME}.service

echo "Reloading systemd..."
systemctl --user daemon-reload

echo ""
echo "==================================="
echo "GUI Service uninstalled successfully"
echo "==================================="
