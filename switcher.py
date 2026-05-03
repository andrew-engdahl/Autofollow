"""Virtual switcher — manages shot selection and cut/crossfade transitions."""

import time
import cv2
import numpy as np
from tracker import TrackedPerson
from config import SWITCH_MODE, SWITCH_TRIGGER, SWITCH_INTERVAL, CROSSFADE_DURATION, SWITCHER_MIN_DISPLACEMENT_RATIO

# Seconds to warm up the pending person's smoother before committing to the transition.
# This gives the virtual camera time to travel to the new subject's position so the
# cut/crossfade always lands on a settled, well-framed shot rather than a cold start.
_PRETRAVEL_DURATION = 0.5


class VirtualSwitcher:
    """Decides when and how to switch between tracked persons.

    Supports two trigger modes:
        'time'   — auto-switch on a configurable interval
        'manual' — only switch when force_switch() is called explicitly

    Supports two transition modes:
        'cut'       — immediate switch
        'crossfade' — linear blend over CROSSFADE_DURATION seconds

    State machine
    -------------
    idle → (trigger fires) → pretraveling → (settled) → transitioning → idle
                                                      → (cut) → idle directly
    During pretraveling the pending person's smoother is warmed up off-screen;
    the output still shows the active person's frame.
    """

    def __init__(self):
        self.switch_mode: str = SWITCH_MODE
        self.trigger: str = SWITCH_TRIGGER
        self.interval: float = SWITCH_INTERVAL
        self.crossfade_duration: float = CROSSFADE_DURATION

        self.active_id: str | None = None
        self._pending_id: str | None = None   # target during pretravel / crossfade
        self._pretraveling: bool = False       # True while camera is warming up off-screen
        self._pretravel_start: float | None = None
        self._fade_start: float | None = None  # time.monotonic() when crossfade began
        self._last_switch_time: float = time.monotonic()
        self._manual_request: str | None = None
        self.current_crop_width: float = 0.0  # set by caller each frame for displacement gating

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_pretraveling(self) -> bool:
        """True while the pending camera is warming up but not yet shown."""
        return self._pretraveling

    @property
    def is_transitioning(self) -> bool:
        """True only during the visible cut/crossfade phase (after pretravel)."""
        return self._pending_id is not None and not self._pretraveling

    @property
    def fade_progress(self) -> float:
        """0.0 = fully on active, 1.0 = fully on pending."""
        if self._fade_start is None or self._pending_id is None:
            return 0.0
        elapsed = time.monotonic() - self._fade_start
        return min(1.0, elapsed / max(self.crossfade_duration, 0.001))

    def force_switch(self, person_id: str):
        """Immediately initiate a switch to the given person (manual trigger)."""
        if person_id != self.active_id:
            self._manual_request = person_id

    def decide(self, persons: list[TrackedPerson]) -> str | None:
        """Evaluate trigger conditions; return person_id to activate (or None).

        Call once per frame. If a switch is warranted this method sets internal
        state and returns the new active_id. Callers should check is_transitioning
        to know whether to render two frames (crossfade).
        """
        if not persons:
            return self.active_id

        now = time.monotonic()
        ids = [p.id for p in persons]

        # Initialise active_id on first call
        if self.active_id is None or self.active_id not in ids:
            self.active_id = persons[0].id
            self._last_switch_time = now
            return self.active_id

        # Phase 1 — pretravel: camera is warming up off-screen
        if self._pretraveling:
            if now - self._pretravel_start >= _PRETRAVEL_DURATION:
                # Camera has settled; commit to the transition
                self._pretraveling = False
                if self.switch_mode == 'cut':
                    self.active_id = self._pending_id
                    self._pending_id = None
                    self._last_switch_time = now
                else:
                    self._fade_start = now  # begin crossfade
            return self.active_id

        # Phase 2 — crossfade in progress
        if self._pending_id is not None:
            if self.fade_progress >= 1.0:
                self.active_id = self._pending_id
                self._pending_id = None
                self._fade_start = None
                self._last_switch_time = now
            return self.active_id

        # Determine target
        target = None
        is_manual = False

        if self._manual_request is not None:
            if self._manual_request in ids and self._manual_request != self.active_id:
                target = self._manual_request
                is_manual = True
            self._manual_request = None

        if target is None and self.trigger == 'time':
            if now - self._last_switch_time >= self.interval and len(persons) > 1:
                # Cycle to the next person in the list
                try:
                    current_idx = ids.index(self.active_id)
                    target = ids[(current_idx + 1) % len(ids)]
                except ValueError:
                    target = ids[0]

        if target is not None and not is_manual:
            # Only auto-switch if the candidate is significantly displaced from the
            # current shot center — prevents flickering between nearby subjects.
            if SWITCHER_MIN_DISPLACEMENT_RATIO > 0.0 and self.current_crop_width > 0.0:
                active_person = next((p for p in persons if p.id == self.active_id), None)
                candidate_person = next((p for p in persons if p.id == target), None)
                if active_person and candidate_person:
                    active_cx = (active_person.bbox[0] + active_person.bbox[2]) / 2.0
                    candidate_cx = (candidate_person.bbox[0] + candidate_person.bbox[2]) / 2.0
                    displacement = abs(candidate_cx - active_cx)
                    if displacement < self.current_crop_width * SWITCHER_MIN_DISPLACEMENT_RATIO:
                        target = None  # too close — skip this switch

        if target is not None:
            self._initiate_switch(target, now)

        return self.active_id

    # ------------------------------------------------------------------
    # Frame blending
    # ------------------------------------------------------------------

    def blend(self, frame_active: np.ndarray, frame_pending: np.ndarray) -> np.ndarray:
        """Blend the active and pending frames based on current fade progress.

        For 'cut' mode this always returns frame_pending once a switch is
        triggered. For 'crossfade' it returns a weighted blend.
        """
        t = self.fade_progress
        if self.switch_mode == 'cut' or t >= 1.0:
            return frame_pending
        if t <= 0.0:
            return frame_active
        return cv2.addWeighted(frame_active, 1.0 - t, frame_pending, t, 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initiate_switch(self, target_id: str, now: float):
        # Always enter pretravel first so the camera can settle on the new subject
        # before the transition becomes visible.
        self._pending_id = target_id
        self._pretraveling = True
        self._pretravel_start = now
        self._fade_start = None
