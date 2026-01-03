"""
RF-DETR Detection System
========================
Entry point for the bottle detection system.

All logic is delegated to specialized modules:
- cli.py: Argument parsing
- runner.py: Detection orchestration (DetectionRunner class)
- pipeline.py: Core inference pipeline
- utils/: Preprocessing, I/O, networking

Usage:
    python main.py                    # Camera mode (SAHI, default)
    python main.py --mode full        # Full frame mode
    python main.py --mode hybrid      # Hybrid mode (full + SAHI)
    python main.py --image photo.jpg  # Single image inference
    python main.py --track            # With object tracking
    python main.py --cli              # Headless mode (no preview)
    python main.py --mqtt             # Publish via MQTT
    python main.py --web              # Publish via HTTP POST
"""

import os
import sys

from cli import create_argument_parser
from runner import DetectionRunner


def main():
    """Main entry point - thin orchestrator."""
    args = create_argument_parser().parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        _show_available_checkpoints(args.checkpoint)
        sys.exit(1)

    # Run detection
    runner = DetectionRunner(args)
    runner.run()


def _show_available_checkpoints(checkpoint_path: str):
    """Show available checkpoints if requested one not found."""
    checkpoint_dir = os.path.dirname(checkpoint_path) or "runs/rfdetr_seg_training"
    if os.path.exists(checkpoint_dir):
        print("\nAvailable checkpoints:")
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pth'):
                print(f"  - {os.path.join(checkpoint_dir, f)}")


if __name__ == "__main__":
    main()
