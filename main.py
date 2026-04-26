#!/usr/bin/env python3
"""
Autofollow — intelligent virtual PTZ camera using pose detection.

GUI mode (default):
    python main.py

CLI / headless mode:
    python main.py --headless --output output.mp4 --camera 1

List cameras:
    python main.py --list-cameras
"""

import argparse
import sys


def list_cameras():
    import cv2
    cameras = []
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    if cameras:
        print(f"Available cameras: {cameras}")
    else:
        print("No cameras found")


def run_gui():
    from PyQt5.QtWidgets import QApplication
    from control_ui import ControlWindow
    app = QApplication(sys.argv)
    app.setApplicationName("Autofollow")
    window = ControlWindow()
    window.show()
    sys.exit(app.exec_())


def run_headless(args):
    """Fallback CLI mode — no GUI, just OpenCV preview window."""
    import config

    if args.camera is not None:
        config.CAMERA_INDEX = args.camera
    if args.shot_type:
        config.SHOT_TYPE = args.shot_type
    if args.max_zoom is not None:
        config.MAX_ZOOM = args.max_zoom
    if args.deadzone is not None:
        if not 0 <= args.deadzone <= 1:
            print("Error: deadzone must be between 0 and 1", file=sys.stderr)
            sys.exit(1)
        config.DEADZONE = args.deadzone

    from video_processor import VideoProcessor
    processor = VideoProcessor(
        camera_index=config.CAMERA_INDEX,
        output_file=args.output,
    )
    try:
        processor.initialize_camera()
        processor.process_video_stream(
            max_frames=args.max_frames,
            show_preview=not args.no_preview,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        processor.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Autofollow — intelligent virtual PTZ camera"
    )
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available camera devices and exit')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (OpenCV preview only)')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera device index (headless mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (headless mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (headless mode)')
    parser.add_argument('--shot-type', type=str, default=None,
                        choices=['full_body', 'waist_up', 'medium', 'close_up'],
                        help='Shot type (headless mode)')
    parser.add_argument('--max-zoom', type=float, default=None,
                        help='Maximum zoom factor (headless mode)')
    parser.add_argument('--deadzone', type=float, default=None,
                        help='Horizontal deadzone 0–1 (headless mode)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable preview window (headless mode)')

    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    if args.headless:
        run_headless(args)
    else:
        run_gui()


if __name__ == '__main__':
    main()
