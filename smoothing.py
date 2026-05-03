"""Unified PTZ smoother with per-person state and deadzone.

Axis behaviour
--------------
X (pan)   – primary motion; quadratic ease-out fires when the subject leaves
             the deadzone.  The quad ramp is based on how far the subject center
             has moved *past* the deadzone edge (0 = just left deadzone,
             1 = at _REF_PAN_DISTANCE beyond it), so the camera starts very
             gently and accelerates smoothly rather than lurching at the start.
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

# Quadratic ramp: camera accelerates when subject is far past the deadzone edge.
# normalized=0 → at deadzone boundary, normalized=1 → _REF_PAN_DISTANCE beyond it.
_PAN_QUAD_MAX = 0.30
_REF_PAN_DISTANCE = 300.0   # pixels of subject overshoot at which quad reaches max

# Hard speed caps — pixels (or zoom units) per frame.
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


def _pan_step(current: float, target: float, subject_overshoot: float = 1.0) -> float:
    """Quadratic ease-out pan.

    subject_overshoot: normalized distance the subject center has moved past
    the deadzone edge, clamped to [0, 1].  0 = just left the deadzone
    (camera barely moves), 1 = fully out at _REF_PAN_DISTANCE (full quad
    factor).  Defaults to 1.0 when no deadzone info is available so the
    behaviour is unchanged for callers that don't pass it.
    """
    error = target - current
    if error == 0.0:
        return current
    base = _pan_alpha()
    normalized = max(0.0, min(1.0, subject_overshoot))
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
        subject_overshoot = 1.0
        if person_center_x is not None and crop_width is not None:
            effective_target_x, subject_overshoot = self._apply_deadzone(
                curr_x, target_x, person_center_x, crop_width
            )
        new_x = _pan_step(curr_x, effective_target_x, subject_overshoot)

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
                        person_center_x: float, crop_width: float
                        ) -> tuple[float, float]:
        """Hold pan while subject stays within the deadzone; ramp outside it.

        Returns:
            (effective_target_x, subject_overshoot)
            subject_overshoot is in [0, 1]: how far past the deadzone edge the
            subject center is, normalized by _REF_PAN_DISTANCE.  Used by
            _pan_step to scale the quadratic factor so motion starts gently
            right at the deadzone boundary.
        """
        viewport_center = current_x + crop_width / 2.0
        distance = abs(person_center_x - viewport_center)
        half_deadzone = (crop_width * config.DEADZONE) / 2.0
        inner_deadzone = half_deadzone * 0.5

        if distance <= inner_deadzone:
            return current_x, 0.0

        if distance <= half_deadzone:
            # Soft inner ramp: partial target blend, low overshoot
            t = (distance - inner_deadzone) / (half_deadzone - inner_deadzone)
            blended_target = current_x + t ** 2 * (target_x - current_x)
            overshoot = min(1.0, (distance - inner_deadzone) / _REF_PAN_DISTANCE)
            return blended_target, overshoot

        # Outside hard deadzone edge: full target, overshoot based on how far past edge
        overshoot = min(1.0, (distance - half_deadzone) / _REF_PAN_DISTANCE)
        return target_x, overshoot
