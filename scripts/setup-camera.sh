#!/bin/bash
# =============================================================================
# Camera Setup Script for Ubuntu
# =============================================================================
# This script configures the system for stable USB camera operation.
# Run with: sudo ./scripts/setup-camera.sh
# =============================================================================

set -e

echo "======================================"
echo "Camera Setup for Ubuntu"
echo "======================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (sudo ./scripts/setup-camera.sh)"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
echo "Setting up camera for user: $ACTUAL_USER"

# 1. Add user to video group
echo ""
echo "Step 1: Adding user to 'video' group..."
if groups $ACTUAL_USER | grep -q '\bvideo\b'; then
    echo "  User '$ACTUAL_USER' is already in 'video' group"
else
    usermod -aG video $ACTUAL_USER
    echo "  Added '$ACTUAL_USER' to 'video' group"
    echo "  NOTE: You must log out and log back in for this to take effect!"
fi

# 2. Configure uvcvideo module for stability
echo ""
echo "Step 2: Configuring uvcvideo kernel module..."
UVCVIDEO_CONF="/etc/modprobe.d/uvcvideo.conf"

cat > $UVCVIDEO_CONF << 'EOF'
# USB Camera Stability Settings
# Prevents camera disconnects and timeout issues
options uvcvideo nodrop=1 timeout=5000 quirks=0x80
EOF

echo "  Created $UVCVIDEO_CONF"
cat $UVCVIDEO_CONF

# 3. Disable USB autosuspend for cameras
echo ""
echo "Step 3: Disabling USB autosuspend for video devices..."
UDEV_RULES="/etc/udev/rules.d/99-usb-camera.rules"

cat > $UDEV_RULES << 'EOF'
# Disable USB autosuspend for video devices to prevent disconnects
ACTION=="add", SUBSYSTEM=="usb", ATTR{bInterfaceClass}=="0e", TEST=="power/control", ATTR{power/control}="on"
ACTION=="add", SUBSYSTEM=="usb", ATTR{bDeviceClass}=="0e", TEST=="power/control", ATTR{power/control}="on"
# Also for any UVC device
ACTION=="add", SUBSYSTEM=="video4linux", ATTR{power/control}="on"
EOF

echo "  Created $UDEV_RULES"
cat $UDEV_RULES

# 4. Reload udev rules
echo ""
echo "Step 4: Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger
echo "  udev rules reloaded"

# 5. Reload uvcvideo module (if loaded)
echo ""
echo "Step 5: Reloading uvcvideo kernel module..."
if lsmod | grep -q uvcvideo; then
    echo "  Unloading uvcvideo..."
    modprobe -r uvcvideo 2>/dev/null || echo "  Warning: Could not unload uvcvideo (camera might be in use)"
    sleep 1
fi
echo "  Loading uvcvideo with new settings..."
modprobe uvcvideo
echo "  uvcvideo reloaded"

# 6. Verify settings
echo ""
echo "Step 6: Verifying configuration..."
echo "  uvcvideo module parameters:"
cat /sys/module/uvcvideo/parameters/nodrop 2>/dev/null && echo "    nodrop: $(cat /sys/module/uvcvideo/parameters/nodrop)" || echo "    nodrop: (not available)"
cat /sys/module/uvcvideo/parameters/timeout 2>/dev/null && echo "    timeout: $(cat /sys/module/uvcvideo/parameters/timeout)" || echo "    timeout: (not available)"
cat /sys/module/uvcvideo/parameters/quirks 2>/dev/null && echo "    quirks: $(cat /sys/module/uvcvideo/parameters/quirks)" || echo "    quirks: (not available)"

# 7. List video devices
echo ""
echo "Step 7: Available video devices:"
if [ -d "/dev" ]; then
    ls -la /dev/video* 2>/dev/null || echo "  No video devices found"
fi

# 8. Show v4l2 devices if available
echo ""
if command -v v4l2-ctl &> /dev/null; then
    echo "Step 8: v4l2-ctl device list:"
    v4l2-ctl --list-devices 2>/dev/null || echo "  No V4L2 devices found"
else
    echo "Step 8: Installing v4l-utils for v4l2-ctl..."
    apt-get update -qq && apt-get install -y -qq v4l-utils
    echo "  v4l2-ctl device list:"
    v4l2-ctl --list-devices 2>/dev/null || echo "  No V4L2 devices found"
fi

echo ""
echo "======================================"
echo "Camera Setup Complete!"
echo "======================================"
echo ""
echo "IMPORTANT: Please log out and log back in for group changes to take effect."
echo ""
echo "If cameras still disappear, try:"
echo "  1. Use a direct USB port (not a hub)"
echo "  2. Try a different USB port"
echo "  3. Check dmesg for USB errors: dmesg | grep -i usb | tail -20"
echo ""
