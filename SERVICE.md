# RF-DETR Service Setup Guide

This guide explains how to run RF-DETR as a systemd service on Ubuntu for automatic startup at boot.

## Quick Start (One-Click)

```bash
# Make scripts executable and run all-in-one installer
chmod +x scripts/*.sh
./scripts/install-all.sh
```

This will: setup Python, configure camera, and install the service.

## Manual Setup (Step by Step)

```bash
# 1. Make scripts executable
chmod +x scripts/*.sh

# 2. Setup virtual environment and install dependencies
./scripts/setup.sh

# 3. Setup camera permissions and USB stability (IMPORTANT!)
sudo ./scripts/setup-camera.sh

# 4. Log out and log back in (required for video group)

# 5. Test manually first (Ctrl+C to stop)
./scripts/start.sh

# 6. Install as systemd service
./scripts/install-service.sh
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `scripts/setup.sh` | Creates venv, installs pip dependencies |
| `scripts/setup-camera.sh` | Configures camera permissions, uvcvideo module, USB autosuspend |
| `scripts/start.sh` | Activates venv, runs detection in headless mode |
| `scripts/install-service.sh` | Generates and installs systemd service |
| `scripts/install-all.sh` | One-click: setup + camera + service |
| `scripts/uninstall-service.sh` | Removes the systemd service |

## Service Configuration

- **Service Name:** `bottle-detection`
- **Command:** `python3 main.py --web --mode full --nms --cli`
- **Restart Policy:** Always restart after 5 seconds on failure
- **Logs:** `logs/detection.log` and `logs/service.log`

## Service Management Commands

### Start, Stop, Restart

```bash
# Start the service
sudo systemctl start bottle-detection

# Stop the service
sudo systemctl stop bottle-detection

# Restart the service
sudo systemctl restart bottle-detection
```

### Pause & Resume (Freeze Process)

```bash
# Pause (freezes process in memory)
sudo systemctl kill --signal=SIGSTOP bottle-detection

# Resume (continues frozen process)
sudo systemctl kill --signal=SIGCONT bottle-detection
```

### Enable/Disable Auto-Start

```bash
# Enable auto-start at boot
sudo systemctl enable bottle-detection

# Disable auto-start at boot
sudo systemctl disable bottle-detection
```

### Check Status

```bash
# View service status
sudo systemctl status bottle-detection

# Check if service is active
systemctl is-active bottle-detection

# Check if service is enabled
systemctl is-enabled bottle-detection
```

## Viewing Logs

```bash
# Application logs (detection output)
tail -f logs/detection.log

# Service stdout logs
tail -f logs/service.log

# Service error logs
tail -f logs/service-error.log

# Systemd journal logs
journalctl -u bottle-detection -f

# Last 100 lines from journal
journalctl -u bottle-detection -n 100

# Logs since last boot
journalctl -u bottle-detection -b
```

## Uninstalling the Service

```bash
./scripts/uninstall-service.sh
```

This will:
1. Stop the service
2. Disable auto-start
3. Remove the service file
4. Reload systemd

**Note:** Log files in `logs/` are NOT removed automatically.

## Troubleshooting

### Service won't start

```bash
# Check detailed status and recent logs
sudo systemctl status bottle-detection -l

# Check journal for errors
journalctl -u bottle-detection -n 50 --no-pager
```

### Permission issues

```bash
# Ensure scripts are executable
chmod +x scripts/*.sh

# Check file ownership
ls -la scripts/
```

### Camera not detected

```bash
# Run camera setup script (recommended)
sudo ./scripts/setup-camera.sh

# Or manually:

# List video devices
ls -la /dev/video*

# Check camera permissions (user must be in 'video' group)
groups $USER
sudo usermod -aG video $USER
# Log out and back in for group change to take effect

# Check uvcvideo module is loaded
lsmod | grep uvcvideo
sudo modprobe uvcvideo

# List cameras with v4l2-ctl
v4l2-ctl --list-devices
```

### Cameras disappear after scanning

This is often caused by USB bus instability. The setup-camera.sh script configures:

1. **uvcvideo module settings** (`/etc/modprobe.d/uvcvideo.conf`):
   ```
   options uvcvideo nodrop=1 timeout=5000 quirks=0x80
   ```

2. **USB autosuspend disable** (`/etc/udev/rules.d/99-usb-camera.rules`):
   ```
   ACTION=="add", SUBSYSTEM=="usb", ATTR{bInterfaceClass}=="0e", TEST=="power/control", ATTR{power/control}="on"
   ```

Manual fix:
```bash
# Reload uvcvideo module
sudo rmmod uvcvideo
sudo modprobe uvcvideo nodrop=1 timeout=5000 quirks=0x80

# Disable USB autosuspend globally
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend
```

### Camera works manually but not as service

```bash
# Check service has video group access
sudo systemctl status bottle-detection

# Restart service after camera setup
sudo systemctl restart bottle-detection

# Check logs for camera errors
journalctl -u bottle-detection -n 50 | grep -i camera
```

### Virtual environment issues

```bash
# Remove and recreate venv
rm -rf venv
./scripts/setup.sh
```

## Manual Service File Location

The systemd service file is installed at:
```
/etc/systemd/system/bottle-detection.service
```

To view it:
```bash
cat /etc/systemd/system/bottle-detection.service
```
