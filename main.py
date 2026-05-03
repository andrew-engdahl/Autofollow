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
    from PyQt5.QtWidgets import QApplication, QSplashScreen
    from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QBrush, QLinearGradient
    from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
    from control_ui import ControlWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Autofollow")
    app.setApplicationDisplayName("Autofollow")
    # Ensure the app comes to the foreground when launched from the Dock
    app.setQuitOnLastWindowClosed(True)

    # ── Splash screen ────────────────────────────────────────────────────
    W, H = 520, 260
    px = QPixmap(W, H)
    px.fill(Qt.transparent)

    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    # Background: dark gradient
    grad = QLinearGradient(0, 0, 0, H)
    grad.setColorAt(0.0, QColor(28, 30, 36))
    grad.setColorAt(1.0, QColor(18, 20, 24))
    p.fillRect(0, 0, W, H, QBrush(grad))

    # Lens icon — concentric circles with a highlight arc
    cx, cy, r = 72, H // 2, 44
    # Outer ring
    p.setPen(QPen(QColor(80, 140, 220), 3))
    p.setBrush(QColor(22, 26, 34))
    p.drawEllipse(QPointF(cx, cy), r, r)
    # Inner glass
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(38, 48, 72))
    p.drawEllipse(QPointF(cx, cy), r - 8, r - 8)
    # Iris ring
    p.setPen(QPen(QColor(60, 110, 190), 2))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r - 16, r - 16)
    # Pupil
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(14, 16, 22))
    p.drawEllipse(QPointF(cx, cy), r - 24, r - 24)
    # Specular highlight
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(200, 220, 255, 80))
    p.drawEllipse(QPointF(cx - 10, cy - 12), 10, 7)

    # Wordmark
    font = QFont("SF Pro Display", 36, QFont.Bold)
    font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
    p.setFont(font)
    p.setPen(QColor(230, 235, 245))
    p.drawText(QRectF(130, 60, 370, 70), Qt.AlignLeft | Qt.AlignVCenter, "Autofollow")

    # Tagline
    tag_font = QFont("SF Pro Text", 12, QFont.Normal)
    p.setFont(tag_font)
    p.setPen(QColor(100, 115, 140))
    p.drawText(QRectF(132, 128, 370, 28), Qt.AlignLeft | Qt.AlignVCenter,
               "Intelligent virtual PTZ camera")

    # Thin accent line under wordmark
    p.setPen(QPen(QColor(60, 110, 200), 1.5))
    p.drawLine(132, 122, 132 + 230, 122)

    # Version / loading hint
    ver_font = QFont("SF Mono", 9)
    p.setFont(ver_font)
    p.setPen(QColor(60, 70, 90))
    p.drawText(QRectF(0, H - 28, W, 20), Qt.AlignCenter, "Loading…")

    p.end()

    splash = QSplashScreen(px, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setAttribute(Qt.WA_TranslucentBackground)
    splash.show()
    app.processEvents()

    window = ControlWindow()

    # Close splash and show main window after 1.8 s
    def _finish():
        splash.finish(window)
        window.show()

    QTimer.singleShot(1800, _finish)
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
