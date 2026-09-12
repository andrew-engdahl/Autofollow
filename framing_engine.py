"""Framing engine — stateless target calculation for virtual PTZ."""

import cv2
import numpy as np
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
    PADDING_RATIO, SHOT_TYPE, MAX_ZOOM, CONFIDENCE_THRESHOLD,
)
from tracker import TrackedPerson

# COCO keypoint indices used for shot framing
_KP = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_hip': 11,     'right_hip': 12,
    'left_knee': 13,    'right_knee': 14,
    'left_ankle': 15,   'right_ankle': 16,
}

# For each shot type: keypoints that define the bottom of the visible region.
# The zoom is computed so the crop exactly spans from head (+ headroom) to this point (+ padding).
_SHOT_BOTTOM_KPS = {
    'full_body': ('left_ankle', 'right_ankle'),
    'waist_up':  ('left_hip',   'right_hip'),
    'medium':    ('left_knee',  'right_knee'),
    'close_up':  ('left_shoulder', 'right_shoulder'),
}

# Fallback when bottom keypoints are not confidently detected:
# fraction of bbox height below y_min to use as the bottom of the shot.
_SHOT_BOTTOM_FRAC = {
    'full_body': 1.00,
    'waist_up':  0.55,
    'medium':    0.72,
    'close_up':  0.25,
}

# Headroom above y_min (topmost keypoint ≈ eye/nose level) as a fraction of
# the visible shot body height.  Compensates for the crown above the top keypoint.
_SHOT_HEADROOM_FRAC = {
    'full_body': 0.10,
    'waist_up':  0.18,
    'medium':    0.20,
    'close_up':  0.30,
}

class FramingEngine:
    """Computes crop targets for single-person and multi-person (wide) shots.

    All methods are stateless — they return (target_x, target_y, target_zoom)
    without storing previous positions. Smoothing and deadzone logic live in
    PTZSmoother (smoothing.py).

    Zoom is expressed relative to the output size: zoom 1.0 means the crop is
    exactly OUTPUT_WIDTH × OUTPUT_HEIGHT source pixels.  `min_zoom` is the
    zoom at which the crop covers the whole camera frame, so a "wide shot"
    is genuinely wide whether the camera is 720p or 4K.
    """

    def __init__(self, input_width: int, input_height: int):
        self.input_width = max(1, int(input_width))
        self.input_height = max(1, int(input_height))
        self.min_zoom = max(OUTPUT_WIDTH / self.input_width,
                            OUTPUT_HEIGHT / self.input_height)

    # ------------------------------------------------------------------
    # Single-person target
    # ------------------------------------------------------------------

    def calculate_target(self, person: TrackedPerson,
                         shot_type: str | None = None) -> tuple[float, float, float]:
        """Compute the ideal crop origin and zoom for a single person.

        Zoom is derived from the person's actual keypoint positions so the
        framed region matches the requested shot type regardless of how large
        the person appears in the raw camera frame.

        Returns:
            (target_x, target_y, target_zoom)
        """
        x_min, y_min, x_max, y_max = person.bbox
        kps = person.keypoints  # (17, 4) — [x_norm, y_norm, 0, conf]
        stype = shot_type or SHOT_TYPE

        person_h = max(1, y_max - y_min)
        person_center_x = (x_min + x_max) / 2.0

        # --- Determine bottom of the visible region ---
        # Use confident keypoints when available; fall back to a bbox fraction.
        bottom_kp_names = _SHOT_BOTTOM_KPS.get(stype, ('left_ankle', 'right_ankle'))
        bottom_ys = []
        for name in bottom_kp_names:
            idx = _KP[name]
            if idx < len(kps) and kps[idx][3] > CONFIDENCE_THRESHOLD:
                bottom_ys.append(kps[idx][1] * self.input_height)

        if bottom_ys:
            shot_bottom_y = max(bottom_ys)  # lower of the two landmarks
        else:
            shot_bottom_y = y_min + person_h * _SHOT_BOTTOM_FRAC.get(stype, 1.0)

        # --- Compute crop extents ---
        shot_body_h = max(1.0, shot_bottom_y - y_min)

        # Headroom above y_min so the crown of the head is clear of the frame edge
        headroom = shot_body_h * _SHOT_HEADROOM_FRAC.get(stype, 0.15)
        top_y = y_min - headroom

        # Padding below the bottom landmark
        bottom_y = shot_bottom_y + shot_body_h * PADDING_RATIO

        # Zoom that fits this exact body region into the output height
        required_crop_h = max(1.0, bottom_y - top_y)
        zoom = float(np.clip(OUTPUT_HEIGHT / required_crop_h,
                             self.min_zoom, max(self.min_zoom, MAX_ZOOM)))

        crop_w = OUTPUT_WIDTH / zoom
        crop_h = OUTPUT_HEIGHT / zoom

        # Horizontal: center on person
        target_x = person_center_x - crop_w / 2.0
        # Vertical: head with headroom at the top of the crop
        target_y = top_y

        # Clamp so the full crop region stays within the frame
        target_x = float(np.clip(target_x, 0, max(0.0, self.input_width - crop_w)))
        target_y = float(np.clip(target_y, 0, max(0.0, self.input_height - crop_h)))

        return target_x, target_y, zoom

    # ------------------------------------------------------------------
    # Multi-person (wide-shot) target
    # ------------------------------------------------------------------

    def calculate_wide_target(self, persons: list[TrackedPerson]) -> tuple[float, float, float]:
        """Compute a crop that frames all detected persons.

        Zooms out as needed to fit everyone, down to the full frame.

        Returns:
            (target_x, target_y, target_zoom)
        """
        if not persons:
            return self.default_target()

        if len(persons) == 1:
            return self.calculate_target(persons[0])

        # Union bbox of all persons with padding
        x_mins = [p.bbox[0] for p in persons]
        y_mins = [p.bbox[1] for p in persons]
        x_maxs = [p.bbox[2] for p in persons]
        y_maxs = [p.bbox[3] for p in persons]

        union_w = max(x_maxs) - min(x_mins)
        union_h = max(y_maxs) - min(y_mins)
        union_cx = (min(x_mins) + max(x_maxs)) / 2.0
        union_cy = (min(y_mins) + max(y_maxs)) / 2.0

        # Add padding
        padded_w = union_w * (1.0 + PADDING_RATIO * 2)
        padded_h = union_h * (1.0 + PADDING_RATIO * 2)

        # Zoom needed to fit the union box at 16:9
        zoom_by_width = OUTPUT_WIDTH / max(padded_w, 1.0)
        zoom_by_height = OUTPUT_HEIGHT / max(padded_h, 1.0)
        zoom = max(self.min_zoom, min(zoom_by_width, zoom_by_height, MAX_ZOOM))

        crop_w = OUTPUT_WIDTH / zoom
        crop_h = OUTPUT_HEIGHT / zoom

        target_x = float(np.clip(union_cx - crop_w / 2.0, 0, max(0.0, self.input_width - crop_w)))
        target_y = float(np.clip(union_cy - crop_h / 2.0, 0, max(0.0, self.input_height - crop_h)))

        return target_x, target_y, zoom

    # ------------------------------------------------------------------
    # Frame cropping
    # ------------------------------------------------------------------

    def apply_crop(self, frame: np.ndarray, x: float, y: float, zoom: float) -> np.ndarray:
        """Extract the crop region and resize to output resolution.

        The crop origin is clamped so the full crop_w × crop_h region always
        fits inside the frame, preventing aspect-ratio distortion when the
        smoother moves the camera to a position near the frame edges.

        Uses INTER_AREA when shrinking (better quality) and INTER_LINEAR when
        zooming in.
        """
        zoom = max(zoom, self.min_zoom)   # never ask for a crop larger than the frame
        crop_w = min(self.input_width, max(1, int(OUTPUT_WIDTH / zoom)))
        crop_h = min(self.input_height, max(1, int(OUTPUT_HEIGHT / zoom)))

        # Clamp origin so the full crop always fits — prevents edge stretching
        xi = int(np.clip(x, 0, max(0, self.input_width - crop_w)))
        yi = int(np.clip(y, 0, max(0, self.input_height - crop_h)))

        cropped = frame[yi:yi + crop_h, xi:xi + crop_w]

        interp = cv2.INTER_AREA if zoom < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=interp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def default_target(self) -> tuple[float, float, float]:
        """Wide shot: the largest centered 16:9 region of the camera frame."""
        crop_w = OUTPUT_WIDTH / self.min_zoom
        crop_h = OUTPUT_HEIGHT / self.min_zoom
        x = (self.input_width - crop_w) / 2.0
        y = (self.input_height - crop_h) / 2.0
        return x, y, self.min_zoom

    # Backwards-compatible alias
    _default_target = default_target
