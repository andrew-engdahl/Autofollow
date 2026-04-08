#!/usr/bin/env python3
"""
Autofollow: Intelligent video framing using pose detection.

A macOS app that captures video from a camera device, detects the person using pose estimation,
and automatically crops the video to a 16:9 medium close-up shot with smooth camera movements.

Usage:
    python main.py                          # Live preview from default camera
    python main.py --camera 0               # Specify camera device
    python main.py --output output.mp4      # Save output to file
    python main.py --help                   # Show help
"""

import argparse
import sys
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
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')

    args = parser.parse_args()

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
        processor = VideoProcessor(camera_index=args.camera, output_file=args.output)
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
