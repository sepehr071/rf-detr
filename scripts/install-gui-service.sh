#!/bin/bash
# AyandAi GUI Service Installation Script
# Installs the GUI control panel as a systemd user service with auto-startup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="ayandai-gui"
CURRENT_USER=$(whoami)

echo "==================================="
echo "AyandAi GUI Service Installation"
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

# Check for display
if [ -z "$DISPLAY" ]; then
    echo "WARNING: No DISPLAY set. GUI service requires a graphical session."
fi

# Create user systemd directory if not exists
mkdir -p ~/.config/systemd/user

# Create start script for GUI
cat > "$PROJECT_DIR/scripts/start-gui.sh" << 'STARTEOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source venv/bin/activate
exec python3 gui.py
STARTEOF

chmod +x "$PROJECT_DIR/scripts/start-gui.sh"

# Generate systemd user service file
cat > ~/.config/systemd/user/${SERVICE_NAME}.service << EOF
[Unit]
Description=AyandAi Edge Service Control GUI
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/scripts/start-gui.sh
Restart=on-failure
RestartSec=5
Environment="DISPLAY=:0"

[Install]
WantedBy=default.target
EOF

echo "Service file created at ~/.config/systemd/user/${SERVICE_NAME}.service"
echo ""

# Reload and enable user service
systemctl --user daemon-reload
systemctl --user enable ${SERVICE_NAME}

echo ""
echo "==================================="
echo "GUI Service Installed!"
echo "==================================="
echo ""
echo "The GUI will auto-start when you log in."
echo ""
echo "To start now (must be in graphical session):"
echo "  systemctl --user start ${SERVICE_NAME}"
echo ""
echo "Useful commands:"
echo "  systemctl --user status ${SERVICE_NAME}    # Check status"
echo "  systemctl --user stop ${SERVICE_NAME}      # Stop GUI"
echo "  systemctl --user restart ${SERVICE_NAME}   # Restart GUI"
echo "  systemctl --user disable ${SERVICE_NAME}   # Disable auto-start"
echo ""
echo "To uninstall:"
echo "  ./scripts/uninstall-gui-service.sh"
