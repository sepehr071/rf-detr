#!/usr/bin/env python3
"""
RF-DETR Service Control GUI
A simple tkinter GUI to manage the bottle-detection systemd service.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import os
from datetime import datetime

SERVICE_NAME = "bottle-detection"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ServiceControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RF-DETR Service Control")
        self.root.geometry("500x450")
        self.root.resizable(True, True)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.create_widgets()
        self.auto_refresh()

    def create_widgets(self):
        # Title
        title_label = ttk.Label(
            self.root,
            text="RF-DETR Service Control",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=10, sticky="ew")

        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Service Status", padding=10)
        status_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(status_frame, text="Unknown", font=("Arial", 10, "bold"))
        self.status_label.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(status_frame, text="Runtime:").grid(row=1, column=0, sticky="w")
        self.runtime_label = ttk.Label(status_frame, text="--")
        self.runtime_label.grid(row=1, column=1, sticky="w", padx=10)

        # Buttons Frame
        buttons_frame = ttk.Frame(self.root, padding=10)
        buttons_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Configure button columns
        for i in range(4):
            buttons_frame.grid_columnconfigure(i, weight=1)

        # Row 1: Setup, Start, Stop, Status
        ttk.Button(buttons_frame, text="Setup", command=self.run_setup).grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(buttons_frame, text="Start", command=self.start_service).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(buttons_frame, text="Stop", command=self.stop_service).grid(
            row=0, column=2, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(buttons_frame, text="Refresh", command=self.refresh_status).grid(
            row=0, column=3, padx=5, pady=5, sticky="ew"
        )

        # Row 2: Calibrate, Restart
        ttk.Button(buttons_frame, text="Calibrate ROI", command=self.run_calibrate).grid(
            row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(buttons_frame, text="Restart", command=self.restart_service).grid(
            row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Output Log
        log_frame = ttk.LabelFrame(self.root, text="Output Log", padding=5)
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            width=50,
            font=("Consolas", 9)
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Clear Log Button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(
            row=1, column=0, pady=5
        )

    def log(self, message):
        """Add message to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        """Clear the log text area."""
        self.log_text.delete(1.0, tk.END)

    def run_command(self, command, use_pkexec=False, shell=False):
        """Run a command and return output."""
        try:
            if use_pkexec:
                command = ["pkexec"] + command

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=shell,
                cwd=SCRIPT_DIR
            )
            return result.stdout + result.stderr, result.returncode
        except Exception as e:
            return str(e), 1

    def run_command_async(self, command, use_pkexec=False, callback=None):
        """Run command in background thread."""
        def task():
            output, code = self.run_command(command, use_pkexec)
            self.root.after(0, lambda: self.log(output.strip() if output.strip() else "Command completed"))
            if callback:
                self.root.after(0, callback)

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def get_service_status(self):
        """Get service status."""
        output, code = self.run_command(["systemctl", "is-active", SERVICE_NAME])
        return output.strip()

    def get_service_runtime(self):
        """Get service runtime."""
        output, code = self.run_command(
            ["systemctl", "show", SERVICE_NAME, "--property=ActiveEnterTimestamp"]
        )
        if "ActiveEnterTimestamp=" in output:
            timestamp_str = output.split("=")[1].strip()
            if timestamp_str:
                try:
                    # Parse the timestamp
                    start_time = datetime.strptime(
                        timestamp_str,
                        "%a %Y-%m-%d %H:%M:%S %Z"
                    )
                    delta = datetime.now() - start_time
                    hours, remainder = divmod(int(delta.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f"{hours}h {minutes}m {seconds}s"
                except:
                    pass
        return "--"

    def refresh_status(self):
        """Refresh the status display."""
        status = self.get_service_status()

        if status == "active":
            self.status_label.config(text="Active", foreground="green")
            runtime = self.get_service_runtime()
            self.runtime_label.config(text=runtime)
        elif status == "inactive":
            self.status_label.config(text="Inactive", foreground="red")
            self.runtime_label.config(text="--")
        else:
            self.status_label.config(text=status.capitalize(), foreground="orange")
            self.runtime_label.config(text="--")

    def auto_refresh(self):
        """Auto-refresh status every 5 seconds."""
        self.refresh_status()
        self.root.after(5000, self.auto_refresh)

    def run_setup(self):
        """Run setup.sh script."""
        self.log("Running setup.sh...")
        setup_script = os.path.join(SCRIPT_DIR, "scripts", "setup.sh")

        def run():
            # Open terminal to run setup (needs user interaction)
            subprocess.Popen(
                ["gnome-terminal", "--", "bash", "-c",
                 f"cd '{SCRIPT_DIR}' && ./scripts/setup.sh; read -p 'Press Enter to close...'"],
                cwd=SCRIPT_DIR
            )
            self.root.after(0, lambda: self.log("Setup started in new terminal"))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def start_service(self):
        """Start the service using pkexec."""
        self.log("Starting service...")
        self.run_command_async(
            ["systemctl", "start", SERVICE_NAME],
            use_pkexec=True,
            callback=self.refresh_status
        )

    def stop_service(self):
        """Stop the service using pkexec."""
        self.log("Stopping service...")
        self.run_command_async(
            ["systemctl", "stop", SERVICE_NAME],
            use_pkexec=True,
            callback=self.refresh_status
        )

    def restart_service(self):
        """Restart the service using pkexec."""
        self.log("Restarting service...")
        self.run_command_async(
            ["systemctl", "restart", SERVICE_NAME],
            use_pkexec=True,
            callback=self.refresh_status
        )

    def run_calibrate(self):
        """Run calibrate.py for ROI configuration."""
        self.log("Starting ROI calibration...")
        calibrate_script = os.path.join(SCRIPT_DIR, "calibrate.py")

        def run():
            # Activate venv and run calibrate.py
            venv_python = os.path.join(SCRIPT_DIR, "venv", "bin", "python3")
            if os.path.exists(venv_python):
                python_cmd = venv_python
            else:
                python_cmd = "python3"

            subprocess.Popen(
                [python_cmd, calibrate_script],
                cwd=SCRIPT_DIR
            )
            self.root.after(0, lambda: self.log("Calibration window opened"))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = ServiceControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
