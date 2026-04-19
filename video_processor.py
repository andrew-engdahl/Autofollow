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
        self.input_width = None
        self.input_height = None
        
    def initialize_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.camera_index}")
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Store input dimensions for coordinate transformation
        self.input_width = width
        self.input_height = height
        
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

    def get_torso_center(self, detection):
        """
        Calculate the torso center from pose keypoints.

        COCO 17-point format:
        - Index 5: Left shoulder
        - Index 6: Right shoulder
        - Index 11: Left hip
        - Index 12: Right hip

        Args:
            detection: Detection dict from pose_detector

        Returns:
            tuple: (x, y) in original frame coordinates, or None if keypoints unavailable
        """
        if not detection.get('keypoints') or not detection.get('detected'):
            return None

        keypoints = detection['keypoints']
        
        # Torso keypoint indices (COCO format)
        shoulder_indices = [5, 6]  # Left and right shoulders
        hip_indices = [11, 12]     # Left and right hips
        
        torso_points = []
        
        # Collect keypoints with sufficient confidence
        for idx in shoulder_indices + hip_indices:
            if idx < len(keypoints):
                x_norm, y_norm, z_norm, conf = keypoints[idx]
                if conf > 0.3:  # Minimum confidence threshold
                    # Convert from normalized to pixel coordinates
                    x = int(x_norm * self.input_width) if self.input_width else 0
                    y = int(y_norm * self.input_height) if self.input_height else 0
                    torso_points.append((x, y))
        
        if len(torso_points) < 2:
            return None
        
        # Calculate center of torso points
        center_x = int(sum(p[0] for p in torso_points) / len(torso_points))
        center_y = int(sum(p[1] for p in torso_points) / len(torso_points))
        
        return (center_x, center_y)

    def transform_to_cropped_coords(self, point, crop_box):
        """
        Transform a point from original frame coordinates to cropped frame coordinates.

        Args:
            point: (x, y) in original frame coordinates
            crop_box: Crop box dict with 'x', 'y', 'zoom' keys

        Returns:
            tuple: (x, y) in cropped frame coordinates, or None if out of bounds
        """
        if point is None:
            return None
        
        orig_x, orig_y = point
        crop_x = crop_box['x']
        crop_y = crop_box['y']
        zoom = crop_box['zoom']
        
        # Transform: subtract crop origin and scale by zoom
        new_x = int((orig_x - crop_x) * zoom)
        new_y = int((orig_y - crop_y) * zoom)
        
        # Check if point is within output bounds
        if 0 <= new_x < OUTPUT_WIDTH and 0 <= new_y < OUTPUT_HEIGHT:
            return (new_x, new_y)
        
        return None

    def draw_torso_target(self, frame, torso_center_cropped):
        """
        Draw a transparent square with red outline at torso center.

        Args:
            frame: Cropped frame to draw on (modified in-place)
            torso_center_cropped: (x, y) position in cropped frame coordinates
        """
        if torso_center_cropped is None:
            return
        
        x, y = torso_center_cropped
        size = 40  # Half-size of the square
        thickness = 2
        color = (0, 0, 255)  # Red
        alpha = 0.3  # Transparency
        
        # Draw filled transparent square
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - size, y - size), (x + size, y + size), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw red outline
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, thickness)

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
                    
                    # Draw torso target indicator
                    torso_center = self.get_torso_center(result['detection'])
                    torso_center_cropped = self.transform_to_cropped_coords(torso_center, result['crop_box'])
                    self.draw_torso_target(cropped_frame, torso_center_cropped)

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
