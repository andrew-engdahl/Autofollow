"""Main video processing pipeline."""

import cv2
import numpy as np
from pose_detector import PoseDetector
from framing_engine import FramingEngine
from smoothing import ExponentialSmoother
from config import DETECTION_INTERVAL, OUTPUT_WIDTH, OUTPUT_HEIGHT, SHOW_OVERLAY


class VideoProcessor:
    """Main video processing pipeline combining detection, framing, and smoothing."""

    def __init__(self, camera_index=0, output_file=None, show_overlay=None, show_crosshairs=False):
        """
        Initialize the video processor.

        Args:
            camera_index: Camera device index (0 = default camera)
            output_file: Optional path to save output video
            show_overlay: Whether to show overlay text (uses config default if None)
            show_crosshairs: Whether to overlay crosshairs at center of frame
        """
        self.camera_index = camera_index
        self.output_file = output_file
        self.show_overlay = show_overlay if show_overlay is not None else SHOW_OVERLAY
        self.show_crosshairs = show_crosshairs
        
        # Initialize components
        self.pose_detector = PoseDetector()
        self.framing_engine = None
        self.smoother = ExponentialSmoother()
        
        # Video capture
        self.cap = None
        self.out = None
        self.frame_count = 0
        
    def initialize_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.camera_index}")
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:  # Default FPS if not reported
            fps = 30
            
        print(f"Camera initialized: {width}x{height} @ {fps:.1f}fps")
        
        # Initialize framing engine with actual input dimensions
        self.framing_engine = FramingEngine(width, height)
        
        # Initialize video writer if output file specified
        if self.output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_file, fourcc, fps, 
                                      (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            if not self.out.isOpened():
                print(f"Warning: Could not open video writer for {self.output_file}")
                self.out = None

        return width, height, fps

    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input video frame

        Returns:
            dict: Processed output and metadata
        """
        # Run pose detection every N frames
        if self.frame_count % DETECTION_INTERVAL == 0:
            detection = self.pose_detector.detect(frame)
        else:
            detection = {'detected': False, 'bbox': None}

        # Calculate optimal crop box
        crop_box = self.framing_engine.calculate_crop_box(detection)

        # Apply smoothing to crop parameters
        smoothed_x, smoothed_y, smoothed_zoom = self.smoother.smooth(
            crop_box['x'], crop_box['y'], crop_box['zoom']
        )

        # Update crop box with smoothed values
        crop_box['x'] = smoothed_x
        crop_box['y'] = smoothed_y
        crop_box['zoom'] = smoothed_zoom

        # Apply crop and resize
        cropped_frame = self.framing_engine.apply_crop(frame, crop_box)

        # Prepare output info
        output = {
            'frame': cropped_frame,
            'detection': detection,
            'crop_box': crop_box,
            'frame_num': self.frame_count
        }

        self.frame_count += 1
        return output

    def draw_crosshairs(self, frame):
        """
        Draw crosshairs at the center of the frame.

        Args:
            frame: Input frame to draw on (modified in-place)
        """
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Crosshair line length (20% of frame width)
        line_length = width // 5
        thickness = 2
        color = (0, 255, 0)  # Green
        
        # Draw horizontal line
        cv2.line(frame, (center_x - line_length, center_y), 
                (center_x + line_length, center_y), color, thickness)
        
        # Draw vertical line
        cv2.line(frame, (center_x, center_y - line_length), 
                (center_x, center_y + line_length), color, thickness)
        
        # Draw center circle
        cv2.circle(frame, (center_x, center_y), 5, color, thickness)

    def process_video_stream(self, max_frames=None, show_preview=True):
        """
        Process video stream from camera.

        Args:
            max_frames: Maximum frames to process (None = continuous)
            show_preview: Whether to display preview window

        Returns:
            stats: Processing statistics
        """
        print("Starting video processing... Press 'q' to quit")
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Process frame
            result = self.process_frame(frame)
            cropped_frame = result['frame']

            # Write to output file
            if self.out:
                self.out.write(cropped_frame)

            # Display preview
            if show_preview:
                # Add frame info if overlay is enabled
                if self.show_overlay:
                    info_text = f"Frame: {result['frame_num']} | " \
                               f"Detected: {'Yes' if result['detection']['detected'] else 'No'}"
                    cv2.putText(cropped_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw crosshairs if enabled
                if self.show_crosshairs:
                    self.draw_crosshairs(cropped_frame)

                cv2.imshow('Autofollow - Live Preview', cropped_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Check max frames
            if max_frames and frame_num >= max_frames:
                break

            frame_num += 1

        print(f"Processed {self.frame_count} frames")
        return {'total_frames': self.frame_count}

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        self.pose_detector.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
