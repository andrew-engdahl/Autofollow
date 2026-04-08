"""Smoothing algorithms for camera movement."""

import numpy as np
from config import SMOOTHING_FACTOR, MAX_PAN_SPEED, MAX_ZOOM_SPEED


class ExponentialSmoother:
    """Applies exponential smoothing to reduce jittery camera movements."""

    def __init__(self, smoothing_factor=SMOOTHING_FACTOR):
        """
        Initialize the smoother.

        Args:
            smoothing_factor: Smoothing strength (0-1, lower = more smoothing)
        """
        self.smoothing_factor = smoothing_factor
        self.prev_x = None
        self.prev_y = None
        self.prev_zoom = None

    def smooth(self, x, y, zoom):
        """
        Apply exponential smoothing to camera parameters.

        Args:
            x: Camera X position
            y: Camera Y position
            zoom: Camera zoom level

        Returns:
            tuple: (smoothed_x, smoothed_y, smoothed_zoom)
        """
        if self.prev_x is None:
            # First frame, no smoothing
            self.prev_x = x
            self.prev_y = y
            self.prev_zoom = zoom
            return x, y, zoom

        # Apply exponential smoothing
        smoothed_x = self.prev_x + self.smoothing_factor * (x - self.prev_x)
        smoothed_y = self.prev_y + self.smoothing_factor * (y - self.prev_y)
        smoothed_zoom = self.prev_zoom + self.smoothing_factor * (zoom - self.prev_zoom)

        # Clamp movement speed
        smoothed_x = self._clamp_movement(self.prev_x, smoothed_x, MAX_PAN_SPEED)
        smoothed_y = self._clamp_movement(self.prev_y, smoothed_y, MAX_PAN_SPEED)
        smoothed_zoom = self._clamp_zoom_speed(self.prev_zoom, smoothed_zoom, MAX_ZOOM_SPEED)

        self.prev_x = smoothed_x
        self.prev_y = smoothed_y
        self.prev_zoom = smoothed_zoom

        return smoothed_x, smoothed_y, smoothed_zoom

    @staticmethod
    def _clamp_movement(prev_val, curr_val, max_speed):
        """Limit movement speed to prevent sudden jumps."""
        delta = curr_val - prev_val
        if abs(delta) > max_speed:
            return prev_val + np.sign(delta) * max_speed
        return curr_val

    @staticmethod
    def _clamp_zoom_speed(prev_zoom, curr_zoom, max_speed):
        """Limit zoom speed to prevent sudden scale changes."""
        delta = curr_zoom - prev_zoom
        if abs(delta) > max_speed:
            return prev_zoom + np.sign(delta) * max_speed
        return curr_zoom

    def reset(self):
        """Reset the smoother state."""
        self.prev_x = None
        self.prev_y = None
        self.prev_zoom = None
