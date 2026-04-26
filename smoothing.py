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

from config import SMOOTHING, MAX_PAN_SPEED, MAX_TILT_SPEED, MAX_ZOOM_SPEED, DEADZONE

# ── Baseline lerp rates ──────────────────────────────────────────────────────
# SMOOTHING=0 → _PAN_BASE_ALPHA   (already substantial smoothing)
# SMOOTHING=1 → _PAN_MIN_ALPHA    (very slow, noticeably delayed)
_PAN_BASE_ALPHA = 0.18   # factor at SMOOTHING=0
_PAN_MIN_ALPHA  = 0.04   # factor at SMOOTHING=1

# Tilt and zoom are always heavily dampened, independent of the user dial.
# These are fractions of the pan alpha so they remain proportionally slower.
_TILT_ALPHA_SCALE = 0.25  # tilt lerp = pan_alpha * this
_ZOOM_ALPHA_SCALE = 0.15  # zoom lerp = pan_alpha * this

# Quadratic ramp for pan: factor grows from pan_alpha (near target) up to
# _PAN_QUAD_MAX (far from target), creating a natural ease-out.
_PAN_QUAD_MAX = 0.55
_REF_PAN_DISTANCE = 200.0   # pixels — normalises the quadratic ramp


def _pan_alpha() -> float:
    """Map the 0-1 SMOOTHING dial to a baseline pan lerp factor (inverted scale)."""
    t = max(0.0, min(1.0, SMOOTHING))
    return _PAN_BASE_ALPHA + (_PAN_MIN_ALPHA - _PAN_BASE_ALPHA) * t


def _lerp_step(current: float, target: float, alpha: float, max_speed: float) -> float:
    """Simple lerp step capped at max_speed."""
    error = target - current
    if error == 0.0:
        return current
    movement = alpha * error
    movement = max(-max_speed, min(max_speed, movement))
    return current + movement


def _pan_step(current: float, target: float, max_speed: float) -> float:
    """Quadratic ease-out pan step: fast when far, slow when close."""
    error = target - current
    if error == 0.0:
        return current
    base = _pan_alpha()
    normalized = min(1.0, abs(error) / _REF_PAN_DISTANCE)
    factor = base + (_PAN_QUAD_MAX - base) * normalized ** 2
    movement = factor * error
    movement = max(-max_speed, min(max_speed, movement))
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
        new_x = _pan_step(curr_x, effective_target_x, MAX_PAN_SPEED)

        # ── Y (tilt): slow lerp — barely moves ──────────────────────────────
        tilt_alpha = _pan_alpha() * _TILT_ALPHA_SCALE
        new_y = _lerp_step(curr_y, target_y, tilt_alpha, MAX_TILT_SPEED)

        # ── Z (zoom): even slower lerp — nearly static ──────────────────────
        zoom_alpha = _pan_alpha() * _ZOOM_ALPHA_SCALE
        new_zoom = _lerp_step(curr_zoom, target_zoom, zoom_alpha, MAX_ZOOM_SPEED)

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
        """Return an effective target_x based on how far the subject is from center.

        Inner zone (50% of deadzone radius) → hold position.
        Transition band                     → quadratic ramp 0→full.
        Outside deadzone                    → full target_x.
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
