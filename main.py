#!/usr/bin/env python3
"""
Autofollow: Intelligent video framing using pose detection.

A macOS app that captures video from a camera device, detects the person using pose estimation,
and automatically crops the video to a 16:9 shot with smooth camera movements.

Usage:
    python main.py                                      # Live preview from default camera
    python main.py --camera 0                           # Specify camera device
    python main.py --output output.mp4                  # Save output to file
    python main.py --shot-type full_body                # Full body shot (other options: waist_up, medium, close_up)
    python main.py --max-zoom 3.0                       # Set maximum zoom factor
    python main.py --no-overlay                         # Hide overlay text
    python main.py --help                               # Show help
"""

import argparse
import sys
import config
from video_processor import VideoProcessor


def find_available_cameras(max_cameras=4):
    """Find available camera devices."""
    import cv2
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Autofollow: Intelligent video framing using pose detection'
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available camera devices')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (e.g., output.mp4)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--shot-type', type=str, default=None,
                       choices=['full_body', 'waist_up', 'medium', 'close_up'],
                       help='Type of shot to frame (default: config value)')
    parser.add_argument('--max-zoom', type=float, default=None,
                       help='Maximum zoom factor (default: config value)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Disable overlay text on preview')

    args = parser.parse_args()

    # Override config with command-line arguments if provided
    if args.shot_type:
        config.SHOT_TYPE = args.shot_type
    
    if args.max_zoom:
        config.MAX_ZOOM = args.max_zoom

    # List available cameras
    if args.list_cameras:
        cameras = find_available_cameras()
        if cameras:
            print(f"Available cameras: {cameras}")
        else:
            print("No cameras found")
        return

    # Initialize processor
    processor = None
    try:
        # Determine overlay setting: --no-overlay flag overrides config, otherwise use config default
        show_overlay_setting = False if args.no_overlay else None  # None will use config default
        processor = VideoProcessor(camera_index=args.camera, output_file=args.output,
                                   show_overlay=show_overlay_setting)
        processor.initialize_camera()

        # Process video
        stats = processor.process_video_stream(
            max_frames=args.max_frames,
            show_preview=not args.no_preview
        )

        print(f"\nProcessing complete!")
        print(f"Total frames processed: {stats['total_frames']}")
        if args.output:
            print(f"Output saved to: {args.output}")

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        cameras = find_available_cameras()
        print(f"Available cameras: {cameras}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if processor:
            processor.cleanup()


if __name__ == '__main__':
    main()
