"""Framing engine for intelligent video cropping."""

import cv2
import numpy as np
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_ASPECT_RATIO,
    PADDING_RATIO, SHOT_TYPE, SHOT_TYPE_ZOOM, MAX_ZOOM, MIN_FACE_SCALE, MAX_FACE_SCALE, DEADZONE
)


class FramingEngine:
    """Calculates crop region to frame person with configurable shot type."""

    def __init__(self, input_width, input_height):
        """
        Initialize the framing engine.

        Args:
            input_width: Input video width
            input_height: Input video height
        """
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = OUTPUT_WIDTH
        self.output_height = OUTPUT_HEIGHT
        self.prev_crop_x = None  # Track previous X position for deadzone logic

    def calculate_crop_box(self, detection_result):
        """
        Calculate the crop box that frames the person optimally.

        Args:
            detection_result: Output from PoseDetector.detect()

        Returns:
            dict: Contains crop coordinates (x, y, w, h), zoom factor, and center
        """
        if not detection_result['detected'] or detection_result['bbox'] is None:
            # Default to center framing if no detection
            return self._get_default_crop()

        x_min, y_min, x_max, y_max = detection_result['bbox']
        
        # Calculate person's bounding box
        person_width = x_max - x_min
        person_height = y_max - y_min
        person_center_x = (x_min + x_max) / 2
        person_center_y = (y_min + y_max) / 2

        # Add padding around detected person
        padded_x_min = max(0, int(x_min - person_width * PADDING_RATIO))
        padded_y_min = max(0, int(y_min - person_height * PADDING_RATIO))
        padded_x_max = min(self.input_width, int(x_max + person_width * PADDING_RATIO))
        padded_y_max = min(self.input_height, int(y_max + person_height * PADDING_RATIO))

        padded_width = padded_x_max - padded_x_min
        padded_height = padded_y_max - padded_y_min

        # Calculate zoom based on shot type
        # Get target zoom for the selected shot type
        target_zoom = SHOT_TYPE_ZOOM.get(SHOT_TYPE, SHOT_TYPE_ZOOM['medium'])
        # Clamp zoom to maximum allowed
        zoom = min(target_zoom, MAX_ZOOM)

        # Calculate crop size maintaining 16:9 aspect ratio
        crop_width = int(self.output_width / zoom)
        crop_height = int(self.output_height / zoom)

        # Ensure aspect ratio is maintained
        if crop_width / crop_height != OUTPUT_ASPECT_RATIO:
            crop_height = int(crop_width / OUTPUT_ASPECT_RATIO)

        # Center on person horizontally with deadzone consideration
        desired_x = max(0, int(person_center_x - crop_width / 2))
        
        # Apply deadzone: panning speed increases as subject moves away from center
        if self.prev_crop_x is not None:
            # Calculate viewport center and distance from subject
            viewport_center = self.prev_crop_x + crop_width / 2
            distance_from_center = abs(person_center_x - viewport_center)
            half_deadzone = (crop_width * DEADZONE) / 2
            inner_deadzone = half_deadzone * 0.5  # Strict no-pan zone in the center
            
            # Calculate panning speed factor (0 to 1) with quadratic easing
            if distance_from_center <= inner_deadzone:
                # Inner deadzone: absolutely no panning
                pan_speed_factor = 0.0
            elif distance_from_center <= half_deadzone:
                # Outer deadzone: gradual ramp from 0 to 1 with quadratic easing
                normalized_distance = (distance_from_center - inner_deadzone) / (half_deadzone - inner_deadzone)
                pan_speed_factor = normalized_distance ** 2
            else:
                # Outside deadzone: full speed panning
                pan_speed_factor = 1.0
            
            # Interpolate between current and desired position based on speed factor
            crop_x = self.prev_crop_x + pan_speed_factor * (desired_x - self.prev_crop_x)
        else:
            crop_x = desired_x
        
        # Store current crop_x for next frame's deadzone calculation
        self.prev_crop_x = crop_x

        # Determine vertical positioning based on whether full body fits
        if person_height <= crop_height:
            # Full body fits - center on person vertically
            crop_y = max(0, int(person_center_y - crop_height / 2))
        else:
            # Full body doesn't fit - prioritize head near top
            # Position head with padding (20% from top) for breathing room
            head_y_offset = int(crop_height * 0.20)
            crop_y = max(0, int(y_min - head_y_offset))

        # Ensure crop doesn't exceed frame boundaries
        crop_x = min(crop_x, self.input_width - crop_width)
        crop_y = min(crop_y, self.input_height - crop_height)

        # Clamp to valid range
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)

        return {
            'x': crop_x,
            'y': crop_y,
            'width': crop_width,
            'height': crop_height,
            'zoom': zoom,
            'center_x': person_center_x,
            'center_y': person_center_y,
            'detected_width': person_width,
            'detected_height': person_height
        }

    def _get_default_crop(self):
        """Get default center framing when no pose detected."""
        crop_width = int(self.output_width)
        crop_height = int(self.output_height)
        
        if crop_width > self.input_width:
            crop_width = self.input_width
            crop_height = int(crop_width / OUTPUT_ASPECT_RATIO)
        
        if crop_height > self.input_height:
            crop_height = self.input_height
            crop_width = int(crop_height * OUTPUT_ASPECT_RATIO)

        crop_x = (self.input_width - crop_width) // 2
        crop_y = (self.input_height - crop_height) // 2

        return {
            'x': crop_x,
            'y': crop_y,
            'width': crop_width,
            'height': crop_height,
            'zoom': 1.0,
            'center_x': self.input_width / 2,
            'center_y': self.input_height / 2,
            'detected_width': 0,
            'detected_height': 0
        }

    def apply_crop(self, frame, crop_box):
        """
        Apply crop to frame.

        Args:
            frame: Input frame
            crop_box: Output from calculate_crop_box()

        Returns:
            cropped_frame: Cropped and resized to 16:9
        """
        x = int(crop_box['x'])
        y = int(crop_box['y'])
        w = int(crop_box['width'])
        h = int(crop_box['height'])

        # Ensure coordinates are valid
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        # Crop the frame
        cropped = frame[y:y+h, x:x+w]

        # Resize to output resolution
        resized = cv2.resize(cropped, (self.output_width, self.output_height), 
                           interpolation=cv2.INTER_LINEAR)

        return resized

    def draw_crop_box(self, frame, crop_box):
        """
        Draw crop box on frame for visualization.

        Args:
            frame: Input frame
            crop_box: Output from calculate_crop_box()

        Returns:
            frame: Frame with crop box drawn
        """
        x = int(crop_box['x'])
        y = int(crop_box['y'])
        w = int(crop_box['width'])
        h = int(crop_box['height'])

        # Draw outer rect (crop region)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw center point
        cx = int(crop_box['center_x'])
        cy = int(crop_box['center_y'])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return frame
