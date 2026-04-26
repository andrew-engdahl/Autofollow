"""Unified quadratic PTZ smoother with per-person state and deadzone."""

import numpy as np
from config import SMOOTHING_FACTOR, MAX_PAN_SPEED, MAX_ZOOM_SPEED, DEADZONE

# Adaptive easing: factor grows quadratically from SMOOTHING_FACTOR (near target)
# up to _MAX_FACTOR (far from target), creating natural ease-out on all axes.
_MAX_FACTOR = 0.18

# Reference distances used to normalize the quadratic ramp.
# At these distances the factor is roughly halfway between min and max.
_REF_PAN_DISTANCE = 200.0   # pixels
_REF_ZOOM_DISTANCE = 1.0    # zoom units


def _quadratic_step(current: float, target: float, max_speed: float,
                    reference_distance: float) -> float:
    """Move `current` toward `target` with quadratic ease-out.

    The adaptive factor grows as a quadratic function of normalized distance,
    so the camera moves fast when far away and slows smoothly as it arrives.
    Movement is capped at `max_speed` units per frame.
    """
    error = target - current
    if error == 0.0:
        return current
    normalized = min(1.0, abs(error) / reference_distance)
    factor = SMOOTHING_FACTOR + (_MAX_FACTOR - SMOOTHING_FACTOR) * normalized ** 2
    movement = factor * error
    # Hard speed cap
    movement = max(-max_speed, min(max_speed, movement))
    return current + movement


class PTZSmoother:
    """Per-person camera state with quadratic easing and horizontal deadzone.

    State is keyed by person ID so each tracked person gets smooth, independent
    camera movement. A special key 'wide' is used for multi-person wide shots.
    """

    def __init__(self):
        # person_id → {'x': float, 'y': float, 'zoom': float}
        self._state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, person_id: str, target_x: float, target_y: float,
               target_zoom: float, person_center_x: float | None = None,
               crop_width: float | None = None) -> tuple[float, float, float]:
        """Smooth toward the given target and return the new (x, y, zoom).

        Args:
            person_id:        Key for this person's state ('person1', 'wide', …).
            target_x/y/zoom:  Desired crop origin and zoom from FramingEngine.
            person_center_x:  Subject's X pixel position (for deadzone on X axis).
            crop_width:       Current crop width in pixels (for deadzone calculation).
        """
        if person_id not in self._state:
            self._state[person_id] = {
                'x': target_x, 'y': target_y, 'zoom': target_zoom
            }
            return target_x, target_y, target_zoom

        state = self._state[person_id]
        curr_x, curr_y, curr_zoom = state['x'], state['y'], state['zoom']

        # X-axis: apply deadzone before quadratic step
        effective_target_x = target_x
        if person_center_x is not None and crop_width is not None:
            effective_target_x = self._apply_deadzone(
                curr_x, target_x, person_center_x, crop_width
            )

        new_x = _quadratic_step(curr_x, effective_target_x, MAX_PAN_SPEED, _REF_PAN_DISTANCE)
        new_y = _quadratic_step(curr_y, target_y, MAX_PAN_SPEED, _REF_PAN_DISTANCE)
        new_zoom = _quadratic_step(curr_zoom, target_zoom, MAX_ZOOM_SPEED, _REF_ZOOM_DISTANCE)

        state['x'], state['y'], state['zoom'] = new_x, new_y, new_zoom
        return new_x, new_y, new_zoom

    def reset(self, person_id: str | None = None):
        """Reset state for one person or all persons."""
        if person_id is None:
            self._state.clear()
        else:
            self._state.pop(person_id, None)

    def get_state(self, person_id: str) -> dict | None:
        return self._state.get(person_id)

    # ------------------------------------------------------------------
    # Deadzone
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_deadzone(current_x: float, target_x: float,
                        person_center_x: float, crop_width: float) -> float:
        """Modify target_x based on how far the subject is from the viewport center.

        Inner zone  → no movement (return current_x).
        Outer zone  → quadratic ramp from 0 to full target.
        Outside     → full target_x.
        """
        viewport_center = current_x + crop_width / 2.0
        distance = abs(person_center_x - viewport_center)
        half_deadzone = (crop_width * DEADZONE) / 2.0
        inner_deadzone = half_deadzone * 0.5

        if distance <= inner_deadzone:
            return current_x
        elif distance <= half_deadzone:
            t = (distance - inner_deadzone) / (half_deadzone - inner_deadzone)
            return current_x + t ** 2 * (target_x - current_x)
        else:
            return target_x
