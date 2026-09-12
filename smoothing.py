"""Unified PTZ smoother with per-subject state, per-axis deadzones and
zoom-locked centering.

The camera state is the crop *center* plus zoom — never the crop origin.
That single choice is what keeps a virtual zoom clean: the crop grows or
shrinks around a fixed point instead of collapsing toward its top-left corner
and dragging the subject across the frame while a slow pan catches up.

Axis behaviour
--------------
X (pan)   – primary motion.  A soft deadzone holds the camera while the
             subject drifts near the viewport center; outside it a quadratic
             ease-out ramps the pan up gently with distance.
Y (tilt)  – secondary.  Same soft deadzone (as a fraction of crop height),
             then a much slower lerp, so the camera barely tilts unless the
             framing is genuinely off vertically.
Z (zoom)  – secondary.  Relative hysteresis so keypoint noise never makes the
             shot "breathe"; a slow lerp once the target really changes.

Zoom-locked centering
---------------------
Whenever the zoom is moving, the center is made to converge on *its* target
at least as fast (fractionally) as the zoom converges on its own.  The camera
therefore travels in a straight line through (cx, cy, zoom) space and every
axis arrives together: a push-in is centered on the destination from the
first frame to the last, with no trailing pan or tilt.
"""

import config

# ── Baseline lerp rates ──────────────────────────────────────────────────────
# SMOOTHING=0 → _PAN_BASE_ALPHA   (gentle, responsive)
# SMOOTHING=1 → _PAN_MIN_ALPHA    (very slow, noticeably delayed)
_PAN_BASE_ALPHA = 0.08
_PAN_MIN_ALPHA  = 0.02

# Tilt and zoom are always heavily dampened, independent of the user dial.
_TILT_ALPHA_SCALE = 0.25  # tilt lerp = pan_alpha * this
_ZOOM_ALPHA_SCALE = 0.20  # zoom lerp = pan_alpha * this

# Quadratic ramp: the pan accelerates as the subject gets further past the
# deadzone edge.  normalized=0 → at the edge, 1 → _REF_PAN_DISTANCE beyond it.
_PAN_QUAD_MAX = 0.30
_REF_PAN_DISTANCE = 300.0   # source pixels

# Soft deadzones.  Inside the inner half nothing moves; across the outer half
# the effective target blends in quadratically so motion starts from zero
# rather than snapping on at the boundary.
_TILT_DEADZONE = 0.10       # fraction of crop height (full width of the zone)
_ZOOM_DEADZONE = 0.08       # relative zoom change (full width of the zone)

# Hard speed caps (source pixels, or zoom units, per frame) come from config so
# they can be tuned in one place: MAX_PAN_SPEED, MAX_TILT_SPEED, MAX_ZOOM_SPEED.


def _pan_alpha() -> float:
    """Map the 0-1 SMOOTHING dial to a pan lerp factor (read from config each call)."""
    t = max(0.0, min(1.0, config.SMOOTHING))
    return _PAN_BASE_ALPHA + (_PAN_MIN_ALPHA - _PAN_BASE_ALPHA) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _soft_target(current: float, target: float, half_zone: float,
                 ref_distance: float | None = None) -> tuple[float, float]:
    """Apply a soft deadzone of ±half_zone around `current`.

    Returns (effective_target, overshoot):
      * inside the inner half of the zone → hold (effective_target = current)
      * across the outer half → blend toward `target` quadratically
      * beyond the zone → full target
    `overshoot` is how far past the inner edge the target lies, normalized by
    `ref_distance` and clamped to [0, 1]; the pan uses it to scale its ramp.
    """
    error = target - current
    distance = abs(error)
    inner = half_zone * 0.5
    if distance <= inner:
        return current, 0.0
    ref = ref_distance if ref_distance else max(half_zone, 1e-9)
    overshoot = min(1.0, (distance - inner) / ref)
    if distance <= half_zone:
        t = (distance - inner) / max(half_zone - inner, 1e-9)
        return current + t * t * error, overshoot
    return target, overshoot


def _lerp_step(current: float, target: float, alpha: float, max_speed: float) -> float:
    movement = alpha * (target - current)
    return current + _clamp(movement, -max_speed, max_speed)


def _pan_step(current: float, target: float, subject_overshoot: float) -> float:
    """Quadratic ease-out pan: gentle at the deadzone edge, faster further out."""
    base = _pan_alpha()
    normalized = _clamp(subject_overshoot, 0.0, 1.0)
    factor = base + (_PAN_QUAD_MAX - base) * normalized ** 2
    movement = factor * (target - current)
    cap = config.MAX_PAN_SPEED
    return current + _clamp(movement, -cap, cap)


class PTZSmoother:
    """Per-subject camera state (crop center + zoom) with per-axis smoothing.

    The public API speaks in crop *origins* (x, y, zoom) to match FramingEngine
    and apply_crop(); the center representation is internal.
    """

    def __init__(self, input_width: int | None = None, input_height: int | None = None):
        self._state: dict[str, dict] = {}
        self._bounds: tuple[float, float] | None = None
        if input_width and input_height:
            self.set_bounds(input_width, input_height)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def set_bounds(self, input_width: int, input_height: int):
        """Tell the smoother the camera frame size so crops never leave it."""
        self._bounds = (float(max(1, input_width)), float(max(1, input_height)))

    @staticmethod
    def _crop_size(zoom: float) -> tuple[float, float]:
        z = max(zoom, 1e-6)
        return config.OUTPUT_WIDTH / z, config.OUTPUT_HEIGHT / z

    def _clamp_center(self, cx: float, cy: float, zoom: float) -> tuple[float, float]:
        """Keep the crop at `zoom` inside the frame (centered if it can't fit)."""
        if self._bounds is None:
            return cx, cy
        w, h = self._bounds
        crop_w, crop_h = self._crop_size(zoom)
        cx = w / 2.0 if crop_w >= w else _clamp(cx, crop_w / 2.0, w - crop_w / 2.0)
        cy = h / 2.0 if crop_h >= h else _clamp(cy, crop_h / 2.0, h - crop_h / 2.0)
        return cx, cy

    def _to_center(self, x: float, y: float, zoom: float) -> tuple[float, float]:
        crop_w, crop_h = self._crop_size(zoom)
        return self._clamp_center(x + crop_w / 2.0, y + crop_h / 2.0, zoom)

    def _to_origin(self, state: dict) -> tuple[float, float, float]:
        crop_w, crop_h = self._crop_size(state['zoom'])
        return (state['cx'] - crop_w / 2.0,
                state['cy'] - crop_h / 2.0,
                state['zoom'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, person_id: str, target_x: float, target_y: float,
               target_zoom: float) -> tuple[float, float, float]:
        """Smooth toward the given crop target and return the new (x, y, zoom).

        Args:
            person_id:        Key for this subject's camera state.
            target_x/y/zoom:  Desired crop origin and zoom from FramingEngine.
        """
        target_zoom = max(float(target_zoom), 1e-6)
        target_cx, target_cy = self._to_center(target_x, target_y, target_zoom)

        state = self._state.get(person_id)
        if state is None:
            # A brand-new subject starts exactly on target: the pretravel phase
            # in the switcher relies on this to land a settled shot.
            self._state[person_id] = {'cx': target_cx, 'cy': target_cy, 'zoom': target_zoom}
            return self._to_origin(self._state[person_id])

        cx, cy, zoom = state['cx'], state['cy'], state['zoom']
        crop_w, crop_h = self._crop_size(zoom)

        # ── Z (zoom): relative hysteresis, then a slow lerp ──────────────
        eff_zoom, _ = _soft_target(zoom, target_zoom, zoom * _ZOOM_DEADZONE / 2.0)
        new_zoom = _lerp_step(zoom, eff_zoom,
                              _pan_alpha() * _ZOOM_ALPHA_SCALE, config.MAX_ZOOM_SPEED)
        # Fraction of the *true* zoom error closed this frame (0 when idle).
        zoom_err = target_zoom - zoom
        zoom_progress = _clamp((new_zoom - zoom) / zoom_err, 0.0, 1.0) if abs(zoom_err) > 1e-9 else 0.0

        # ── X (pan): soft deadzone + quadratic ease-out ──────────────────
        eff_cx, overshoot = _soft_target(cx, target_cx, crop_w * config.DEADZONE / 2.0,
                                         _REF_PAN_DISTANCE)
        new_cx = _pan_step(cx, eff_cx, overshoot)

        # ── Y (tilt): soft deadzone + slow lerp ──────────────────────────
        eff_cy, _ = _soft_target(cy, target_cy, crop_h * _TILT_DEADZONE / 2.0)
        new_cy = _lerp_step(cy, eff_cy, _pan_alpha() * _TILT_ALPHA_SCALE, config.MAX_TILT_SPEED)

        # ── Zoom-locked centering ────────────────────────────────────────
        # While the zoom is travelling, the center must cover at least the
        # same fraction of its remaining distance so both arrive together and
        # the zoom stays anchored on the destination.  This bypasses the
        # deadzones on purpose; it is bounded by the zoom's own (capped) pace.
        if zoom_progress > 0.0:
            locked_cx = cx + zoom_progress * (target_cx - cx)
            locked_cy = cy + zoom_progress * (target_cy - cy)
            if abs(locked_cx - cx) > abs(new_cx - cx):
                new_cx = locked_cx
            if abs(locked_cy - cy) > abs(new_cy - cy):
                new_cy = locked_cy

        new_cx, new_cy = self._clamp_center(new_cx, new_cy, new_zoom)
        state['cx'], state['cy'], state['zoom'] = new_cx, new_cy, new_zoom
        return self._to_origin(state)

    def widen(self, person_id: str, amount: float, min_zoom: float) -> bool:
        """Zoom out by `amount` around the current center (used while searching).

        Returns False if the key is unknown or already at `min_zoom`.
        """
        state = self._state.get(person_id)
        if state is None:
            return False
        new_zoom = max(min_zoom, state['zoom'] - amount)
        if new_zoom == state['zoom']:
            return False
        state['zoom'] = new_zoom
        state['cx'], state['cy'] = self._clamp_center(state['cx'], state['cy'], new_zoom)
        return True

    def reset(self, person_id: str | None = None):
        if person_id is None:
            self._state.clear()
        else:
            self._state.pop(person_id, None)

    def get_state(self, person_id: str) -> dict | None:
        """Current camera for `person_id` as {'x', 'y', 'zoom', 'cx', 'cy'}, or None."""
        state = self._state.get(person_id)
        if state is None:
            return None
        x, y, zoom = self._to_origin(state)
        return {'x': x, 'y': y, 'zoom': zoom, 'cx': state['cx'], 'cy': state['cy']}

    def seed(self, person_id: str, x: float, y: float, zoom: float):
        """Set a key's camera position directly from a crop origin (no smoothing)."""
        zoom = max(float(zoom), 1e-6)
        cx, cy = self._to_center(float(x), float(y), zoom)
        self._state[person_id] = {'cx': cx, 'cy': cy, 'zoom': zoom}

    def copy_state(self, src_id: str, dst_id: str, remove_src: bool = False):
        """Make dst_id continue from wherever src_id's camera currently is."""
        src = self._state.get(src_id)
        if src is None:
            return
        self._state[dst_id] = dict(src)
        if remove_src:
            self._state.pop(src_id, None)
