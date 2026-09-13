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

Acceleration limiting
---------------------
Pose detection lands every few frames and every detection carries some
noise, so the framing target is a staircase with wobble on it.  A pure lerp
reproduces every step as a visible jolt because the camera's velocity may
change arbitrarily from one frame to the next.  Each axis therefore keeps its
last per-frame movement and may only change it by a bounded amount per frame:
the camera accelerates into a move, cruises, and eases out, and a target
that jumps around by a few pixels between detections simply isn't followed.

Zoom-proportional speed
-----------------------
A subject moving at a given speed on stage crosses the output ``zoom`` times
faster on a tight shot than on the wide shot, so a fixed lerp gain lags more
and more as the camera pushes in.  Pan and tilt gains, their speed caps and
the pan ramp distance are therefore all scaled by the zoom relative to the
wide shot, which keeps the *on-screen* response the same at every framing.
The zoom axis moves at a rate relative to the current zoom (a constant
fraction per frame) with its own ramp, so a shot-type change is decisive
while keypoint jitter still can't make the shot breathe.
"""

import config
from geometry import wide_zoom, px_scale

# ── Baseline lerp rates ──────────────────────────────────────────────────────
# SMOOTHING=0 → _PAN_BASE_ALPHA   (gentle, responsive)
# SMOOTHING=1 → _PAN_MIN_ALPHA    (very slow, noticeably delayed)
_PAN_BASE_ALPHA = 0.08
_PAN_MIN_ALPHA  = 0.02

# Tilt is always heavily dampened, independent of the user dial.
_TILT_ALPHA_SCALE = 0.25  # tilt lerp = pan_alpha * this
# Zoom idles at the pan gain and ramps up for big changes (see below).
_ZOOM_ALPHA_SCALE = 1.0   # zoom base lerp = pan_alpha * this

# Quadratic ramp: the pan accelerates as the subject gets further past the
# deadzone edge.  normalized=0 → at the edge, 1 → _REF_PAN_DISTANCE beyond it.
_PAN_QUAD_MAX = 0.30
_REF_PAN_DISTANCE = 300.0   # *output* pixels — converted to source px per zoom in update()

# Zoom ramp: normalized=0 → at the hysteresis edge, 1 → a _REF_ZOOM_CHANGE
# relative change beyond it (e.g. waist-up → close-up is a 0.5 change).
_ZOOM_QUAD_MAX = 0.20
_REF_ZOOM_CHANGE = 0.5

# Gains are multiplied by the relative zoom, so cap the per-frame fraction of
# the remaining error a lerp may close — above ~0.5 motion starts to look
# stepped rather than eased.
_MAX_LERP_FACTOR = 0.5

# Acceleration caps: the most an axis's per-frame movement may change from
# one frame to the next.  Pan/tilt are reference (720p wide-shot) pixels per
# frame², scaled like the speed caps; zoom is a fraction of the current zoom
# per frame².  Smaller = smoother but slower to react.
_MAX_PAN_ACCEL = 1.0
_MAX_TILT_ACCEL = 0.3
_MAX_ZOOM_ACCEL = 0.0012

# Soft deadzones.  Inside the inner half nothing moves; across the outer half
# the effective target blends in quadratically so motion starts from zero
# rather than snapping on at the boundary.
_TILT_DEADZONE = 0.10       # fraction of crop height (full width of the zone)
_ZOOM_DEADZONE = 0.08       # relative zoom change (full width of the zone)

# Hard speed caps come from config so they can be tuned in one place:
# MAX_PAN_SPEED / MAX_TILT_SPEED (reference pixels per frame on the wide shot,
# scaled up with zoom) and MAX_ZOOM_SPEED (fraction of the current zoom per
# frame).
#
# Pixel-unit constants in config are *reference* pixels — tuned for a
# 1280×720 input.  set_bounds() records the detected input size and the
# smoother converts them to source pixels on the fly, so a 4K camera pans the
# same distance across the output per frame as a 720p one.  Zoom-unit
# constants used by widen() are scaled the other way, because zoom values
# themselves shrink as the input grows (the wide shot is zoom 1.0 at 720p but
# 0.33 at 4K).


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
    movement = _clamp(alpha, 0.0, _MAX_LERP_FACTOR) * (target - current)
    return current + _clamp(movement, -max_speed, max_speed)


def _slew(prev_velocity: float, desired_movement: float, max_accel: float) -> float:
    """Limit how much this frame's movement may differ from last frame's."""
    return _clamp(desired_movement, prev_velocity - max_accel, prev_velocity + max_accel)


def _ramp_step(current: float, target: float, overshoot: float, max_speed: float,
               base: float, quad_max: float, gain: float = 1.0) -> float:
    """Quadratic ease-out: gentle at the deadzone edge, faster further out.

    ``gain`` multiplies the whole ramp (the zoom-proportional factor).
    """
    normalized = _clamp(overshoot, 0.0, 1.0)
    factor = (base + (quad_max - base) * normalized ** 2) * gain
    movement = _clamp(factor, 0.0, _MAX_LERP_FACTOR) * (target - current)
    return current + _clamp(movement, -max_speed, max_speed)


class PTZSmoother:
    """Per-subject camera state (crop center + zoom) with per-axis smoothing.

    The public API speaks in crop *origins* (x, y, zoom) to match FramingEngine
    and apply_crop(); the center representation is internal.
    """

    def __init__(self, input_width: int | None = None, input_height: int | None = None):
        self._state: dict[str, dict] = {}
        self._bounds: tuple[float, float] | None = None
        self._px_scale = 1.0     # source px per reference px (1.0 until bounds are known)
        self._zoom_scale = 1.0   # zoom units per reference zoom unit
        if input_width and input_height:
            self.set_bounds(input_width, input_height)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def set_bounds(self, input_width: int, input_height: int):
        """Tell the smoother the detected camera frame size.

        Crops are clamped to it, and the pixel / zoom speed constants are
        rescaled so motion looks the same at any input resolution.
        """
        self._bounds = (float(max(1, input_width)), float(max(1, input_height)))
        self._px_scale = px_scale(input_width, input_height)
        self._zoom_scale = wide_zoom(input_width, input_height)

    @property
    def px_scale(self) -> float:
        """Source pixels per reference (720p) pixel for the current bounds."""
        return self._px_scale

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
            self._state[person_id] = self._new_state(target_cx, target_cy, target_zoom)
            return self._to_origin(self._state[person_id])

        cx, cy, zoom = state['cx'], state['cy'], state['zoom']
        crop_w, crop_h = self._crop_size(zoom)
        alpha = _pan_alpha()

        # How far the shot is pushed in relative to the wide shot (1.0 = wide).
        # Pan/tilt gains and caps scale with it so the response on screen is
        # the same at every framing; the ramp distance is a fixed number of
        # output pixels, i.e. it shrinks in source pixels as the zoom grows.
        z_rel = max(1.0, zoom / self._zoom_scale)
        max_pan = config.MAX_PAN_SPEED * self._px_scale * z_rel
        max_tilt = config.MAX_TILT_SPEED * self._px_scale * z_rel
        ref_pan_distance = _REF_PAN_DISTANCE * self._px_scale / z_rel
        # Zoom moves a fraction of itself per frame at most, so a push-in
        # feels the same whether it starts wide or already tight.
        max_zoom_step = config.MAX_ZOOM_SPEED * zoom
        # Acceleration caps.  Unlike the speed caps these are *not* scaled by
        # z_rel: in source pixels they are the same at every zoom (so on
        # screen they grow linearly with it, not quadratically).  Simulation
        # against a noisy, late, every-other-frame detection showed that
        # scaling them up with zoom only lets the camera hunt on a tight shot,
        # with more jerk *and* more lag.
        pan_accel = _MAX_PAN_ACCEL * self._px_scale
        tilt_accel = _MAX_TILT_ACCEL * self._px_scale
        zoom_accel = _MAX_ZOOM_ACCEL * zoom

        # ── Z (zoom): relative hysteresis, then a ramped lerp ───────────
        eff_zoom, zoom_overshoot = _soft_target(
            zoom, target_zoom, zoom * _ZOOM_DEADZONE / 2.0, zoom * _REF_ZOOM_CHANGE)
        new_zoom = _ramp_step(zoom, eff_zoom, zoom_overshoot, max_zoom_step,
                              alpha * _ZOOM_ALPHA_SCALE, _ZOOM_QUAD_MAX)
        new_zoom = zoom + _slew(state['vz'], new_zoom - zoom, zoom_accel)
        # Fraction of the *true* zoom error closed this frame (0 when idle).
        zoom_err = target_zoom - zoom
        zoom_progress = _clamp((new_zoom - zoom) / zoom_err, 0.0, 1.0) if abs(zoom_err) > 1e-9 else 0.0

        # ── X (pan): soft deadzone + quadratic ease-out ──────────────────
        eff_cx, overshoot = _soft_target(cx, target_cx, crop_w * config.DEADZONE / 2.0,
                                         ref_pan_distance)
        new_cx = _ramp_step(cx, eff_cx, overshoot, max_pan, alpha, _PAN_QUAD_MAX, gain=z_rel)
        new_cx = cx + _slew(state['vx'], new_cx - cx, pan_accel)

        # ── Y (tilt): soft deadzone + slow lerp ──────────────────────────
        eff_cy, _ = _soft_target(cy, target_cy, crop_h * _TILT_DEADZONE / 2.0)
        new_cy = _lerp_step(cy, eff_cy, alpha * _TILT_ALPHA_SCALE * z_rel, max_tilt)
        new_cy = cy + _slew(state['vy'], new_cy - cy, tilt_accel)

        # ── Zoom-locked centering ────────────────────────────────────────
        # While the zoom is travelling, the center must cover at least the
        # same fraction of its remaining distance so both arrive together and
        # the zoom stays anchored on the destination.  This bypasses the
        # deadzones and the pan/tilt slew on purpose; it is bounded by the
        # zoom's own (capped, acceleration-limited) pace, so it stays smooth.
        if zoom_progress > 0.0:
            locked_cx = cx + zoom_progress * (target_cx - cx)
            locked_cy = cy + zoom_progress * (target_cy - cy)
            if abs(locked_cx - cx) > abs(new_cx - cx):
                new_cx = locked_cx
            if abs(locked_cy - cy) > abs(new_cy - cy):
                new_cy = locked_cy

        new_cx, new_cy = self._clamp_center(new_cx, new_cy, new_zoom)
        state['vx'], state['vy'], state['vz'] = new_cx - cx, new_cy - cy, new_zoom - zoom
        state['cx'], state['cy'], state['zoom'] = new_cx, new_cy, new_zoom
        return self._to_origin(state)

    @staticmethod
    def _new_state(cx: float, cy: float, zoom: float) -> dict:
        return {'cx': cx, 'cy': cy, 'zoom': zoom, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0}

    def widen(self, person_id: str, amount: float, min_zoom: float) -> bool:
        """Zoom out by `amount` around the current center (used while searching).

        `amount` is in reference (720p) zoom units and is rescaled to the
        current input so the search widens at the same visible rate everywhere.
        Returns False if the key is unknown or already at `min_zoom`.
        """
        state = self._state.get(person_id)
        if state is None:
            return False
        new_zoom = max(min_zoom, state['zoom'] - amount * self._zoom_scale)
        if new_zoom == state['zoom']:
            return False
        state['zoom'] = new_zoom
        state['cx'], state['cy'] = self._clamp_center(state['cx'], state['cy'], new_zoom)
        state['vx'] = state['vy'] = state['vz'] = 0.0
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
        self._state[person_id] = self._new_state(cx, cy, zoom)

    def copy_state(self, src_id: str, dst_id: str, remove_src: bool = False):
        """Make dst_id continue from wherever src_id's camera currently is."""
        src = self._state.get(src_id)
        if src is None:
            return
        self._state[dst_id] = dict(src)
        if remove_src:
            self._state.pop(src_id, None)
