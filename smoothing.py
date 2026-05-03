"""Unified PTZ smoother with per-person state and deadzone.

Axis behaviour
--------------
X (pan)   – primary motion; quadratic ease-out only fires when the subject
             leaves the deadzone.  SMOOTHING controls the baseline lerp rate.
Y (tilt)  – secondary; plain lerp at a much slower rate so the camera barely
             tilts unless the subject is truly out of frame vertically.
Z (zoom)  – secondary; same treatment as tilt, even slower, so zoom changes
             are nearly imperceptible moment-to-moment.
"""

import config

# ── Baseline lerp rates ──────────────────────────────────────────────────────
# SMOOTHING=0 → _PAN_BASE_ALPHA   (gentle, responsive)
# SMOOTHING=1 → _PAN_MIN_ALPHA    (very slow, noticeably delayed)
_PAN_BASE_ALPHA = 0.08   # factor at SMOOTHING=0 — slower baseline to reduce jitter
_PAN_MIN_ALPHA  = 0.02   # factor at SMOOTHING=1

# Tilt and zoom are always heavily dampened, independent of the user dial.
_TILT_ALPHA_SCALE = 0.20  # tilt lerp = pan_alpha * this
_ZOOM_ALPHA_SCALE = 0.10  # zoom lerp = pan_alpha * this

# Quadratic ramp: camera accelerates when subject is far from target.
# Kept modest so fast moves don't look like a whip pan.
_PAN_QUAD_MAX = 0.30
_REF_PAN_DISTANCE = 300.0   # pixels — distance at which quad factor reaches _PAN_QUAD_MAX

# Hard speed caps — pixels (or zoom units) per frame.
# These are the absolute ceiling regardless of the lerp calculation.
_MAX_PAN_SPEED  = 15    # px/frame  (~450px/s at 30fps — smooth but responsive)
_MAX_TILT_SPEED = 3     # px/frame
_MAX_ZOOM_SPEED = 0.015 # zoom units/frame


def _pan_alpha() -> float:
    """Map the 0-1 SMOOTHING dial to a pan lerp factor (read from config each call)."""
    t = max(0.0, min(1.0, config.SMOOTHING))
    return _PAN_BASE_ALPHA + (_PAN_MIN_ALPHA - _PAN_BASE_ALPHA) * t


def _lerp_step(current: float, target: float, alpha: float, max_speed: float) -> float:
    error = target - current
    if error == 0.0:
        return current
    movement = alpha * error
    movement = max(-max_speed, min(max_speed, movement))
    return current + movement


def _pan_step(current: float, target: float) -> float:
    """Quadratic ease-out pan: accelerates when far, gentle when close."""
    error = target - current
    if error == 0.0:
        return current
    base = _pan_alpha()
    normalized = min(1.0, abs(error) / _REF_PAN_DISTANCE)
    factor = base + (_PAN_QUAD_MAX - base) * normalized ** 2
    movement = factor * error
    movement = max(-_MAX_PAN_SPEED, min(_MAX_PAN_SPEED, movement))
    return current + movement


class PTZSmoother:
    """Per-person camera state with separate smoothing for pan, tilt, and zoom."""

    def __init__(self):
        self._state: dict[str, dict] = {}

    def update(self, person_id: str, target_x: float, target_y: float,
               target_zoom: float, person_center_x: float | None = None,
               crop_width: float | None = None) -> tuple[float, float, float]:
        """Smooth toward the given target and return the new (x, y, zoom).

        Args:
            person_id:        Key for this person's state.
            target_x/y/zoom:  Desired crop origin and zoom from FramingEngine.
            person_center_x:  Subject X pixel position (for X-axis deadzone).
            crop_width:       Current crop width in pixels (for deadzone calc).
        """
        if person_id not in self._state:
            self._state[person_id] = {
                'x': target_x, 'y': target_y, 'zoom': target_zoom
            }
            return target_x, target_y, target_zoom

        state = self._state[person_id]
        curr_x, curr_y, curr_zoom = state['x'], state['y'], state['zoom']

        # ── X (pan): quadratic ease-out, only outside the deadzone ──────────
        effective_target_x = target_x
        if person_center_x is not None and crop_width is not None:
            effective_target_x = self._apply_deadzone(
                curr_x, target_x, person_center_x, crop_width
            )
        new_x = _pan_step(curr_x, effective_target_x)

        # ── Y (tilt): slow lerp ──────────────────────────────────────────────
        tilt_alpha = _pan_alpha() * _TILT_ALPHA_SCALE
        new_y = _lerp_step(curr_y, target_y, tilt_alpha, _MAX_TILT_SPEED)

        # ── Z (zoom): even slower lerp ───────────────────────────────────────
        zoom_alpha = _pan_alpha() * _ZOOM_ALPHA_SCALE
        new_zoom = _lerp_step(curr_zoom, target_zoom, zoom_alpha, _MAX_ZOOM_SPEED)

        state['x'], state['y'], state['zoom'] = new_x, new_y, new_zoom
        return new_x, new_y, new_zoom

    def reset(self, person_id: str | None = None):
        if person_id is None:
            self._state.clear()
        else:
            self._state.pop(person_id, None)

    def get_state(self, person_id: str) -> dict | None:
        return self._state.get(person_id)

    @staticmethod
    def _apply_deadzone(current_x: float, target_x: float,
                        person_center_x: float, crop_width: float) -> float:
        """Hold pan while subject stays within the deadzone; ramp outside it."""
        viewport_center = current_x + crop_width / 2.0
        distance = abs(person_center_x - viewport_center)
        half_deadzone = (crop_width * config.DEADZONE) / 2.0
        inner_deadzone = half_deadzone * 0.5

        if distance <= inner_deadzone:
            return current_x
        elif distance <= half_deadzone:
            t = (distance - inner_deadzone) / (half_deadzone - inner_deadzone)
            return current_x + t ** 2 * (target_x - current_x)
        else:
            return target_x
