"""Background face-recognition worker.

InsightFace detection + embedding costs 50–150 ms per pass on CPU.  Running it
inline on the video thread (as the first integration did, every 6th frame)
stalls the output for that long several times a second.  This worker owns a
single "latest job" slot: the video thread hands it a frame plus a snapshot of
the tracked bodies and carries on; results are picked up with poll() on a
later frame and applied to the tracker by ID.

Recognition runs on crops rather than the full frame.  InsightFace letterboxes
its input to 640×640, so on a wide 1080p/4K stage shot a distant face shrinks
to a few pixels and is never detected.  Cropping to each person's upper body
(few people) or to the band containing everyone (many people) keeps faces at a
usable size for one or a few detector passes.
"""

from __future__ import annotations

import threading
import time
import numpy as np

from profiles import ProfileStore
from tracker import PersonTracker

# Per-person upper-body crops up to this many people; beyond that a single
# union crop bounds the cost.
_MAX_PER_PERSON_CROPS = 4
# Fraction of the body bbox (from the top) that contains the face.
_UPPER_BODY_FRAC = 0.5
# Margin added around crops, as a fraction of the crop size.
_CROP_MARGIN = 0.25
# Never crop smaller than this (px) — tiny crops upscale badly in the detector.
_MIN_CROP = 160


class FaceRecognitionWorker:
    """Runs FaceRecognizer.identify() on a daemon thread with a one-slot queue."""

    def __init__(self, store: ProfileStore):
        self._store = store
        self._recognizer = None
        self._failed = False            # import/model load failed — never retry
        self.error_message: str | None = None   # why it failed, for the status bar
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job: tuple | None = None  # (frame, bodies, frame_shape)
        self._result: list[dict] | None = None
        self._busy = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_duration = 0.0

    # ------------------------------------------------------------------
    # Public API (video thread)
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return not self._failed

    @property
    def busy(self) -> bool:
        return self._busy or self._job is not None

    @property
    def last_duration(self) -> float:
        """Seconds the most recent recognition pass took (for diagnostics)."""
        return self._last_duration

    def has_profiles(self) -> bool:
        """True if any profile has face embeddings (cheap; cached in the recognizer)."""
        if self._failed:
            return False
        if self._recognizer is None:
            # Don't load the model just to answer this — check the store directly.
            return any(p.embeddings is not None and len(p.embeddings) > 0
                       for p in self._store.list())
        return self._recognizer.has_profiles

    def submit(self, frame: np.ndarray, bodies: list[tuple[str, tuple]]) -> bool:
        """Queue a recognition pass. Returns False if the worker is busy or disabled.

        ``bodies`` is [(track_id, bbox)] as of this frame; the face→body match
        is done against this snapshot when the pass completes.
        """
        if self._failed or not bodies:
            return False
        with self._cv:
            if self._busy or self._job is not None:
                return False
            self._job = (frame, list(bodies), frame.shape[:2])
            if not self._running:
                self._running = True
                self._thread = threading.Thread(
                    target=self._loop, name="face-recognition", daemon=True)
                self._thread.start()
            self._cv.notify()
        return True

    def poll(self) -> list[dict] | None:
        """Return the latest finished results (once), or None."""
        with self._lock:
            res, self._result = self._result, None
            return res

    def mark_index_dirty(self):
        if self._recognizer is not None:
            self._recognizer.mark_index_dirty()

    def stop(self):
        with self._cv:
            self._running = False
            self._job = None
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _loop(self):
        while True:
            with self._cv:
                while self._running and self._job is None:
                    self._cv.wait()
                if not self._running:
                    return
                frame, bodies, shape = self._job
                self._job = None
                self._busy = True
            try:
                t0 = time.monotonic()
                results = self._recognize(frame, bodies, shape)
                self._last_duration = time.monotonic() - t0
            except Exception as e:
                print(f"Face recognition failed: {e}")
                results = []
            with self._lock:
                self._result = results
                self._busy = False

    def _ensure_recognizer(self) -> bool:
        if self._recognizer is not None:
            return True
        try:
            from face_recognizer import get_shared_recognizer
            self._recognizer = get_shared_recognizer(self._store)
            return True
        except Exception as e:
            self._fail(e)
            return False

    def _fail(self, exc: Exception):
        msg = str(exc)
        if isinstance(exc, ImportError):
            msg += " — run ./setup.sh --with-recognition"
        self.error_message = f"Face recognition unavailable: {msg}"
        print(self.error_message)
        self._failed = True

    def _recognize(self, frame, bodies, shape) -> list[dict]:
        if not self._ensure_recognizer():
            return []
        if not self._recognizer.has_profiles:
            return []
        fh, fw = shape
        crops = self._crop_regions(bodies, fw, fh)

        matches: dict[str, dict] = {}
        for (x0, y0, x1, y1) in crops:
            sub = np.ascontiguousarray(frame[y0:y1, x0:x1])
            if sub.size == 0:
                continue
            try:
                faces = self._recognizer.identify(sub)
            except Exception as e:
                if not self._recognizer.is_loaded:
                    # Import / model-load errors surface here on first use;
                    # those are permanent.
                    self._fail(e)
                    return []
                print(f"Face identify failed: {e}")   # transient — try again next pass
                continue
            for face in faces:
                if face.get("profile_id") is None:
                    continue
                bx1, by1, bx2, by2 = face["bbox"]
                full_bbox = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                tid = PersonTracker.match_face_to_bbox(full_bbox, bodies)
                if tid is None:
                    continue
                prev = matches.get(tid)
                if prev is None or face["score"] > prev["score"]:
                    matches[tid] = {
                        "track_id": tid,
                        "profile_id": face["profile_id"],
                        "name": face["name"],
                        "priority": face["priority"],
                        "score": face["score"],
                    }
        return list(matches.values())

    @staticmethod
    def _crop_regions(bodies, fw, fh) -> list[tuple[int, int, int, int]]:
        """Upper-body crop per person (few people) or one union crop (many)."""
        def expand(x0, y0, x1, y1):
            w = max(x1 - x0, 1)
            h = max(y1 - y0, 1)
            mx = max(w * _CROP_MARGIN, (_MIN_CROP - w) / 2)
            my = max(h * _CROP_MARGIN, (_MIN_CROP - h) / 2)
            return (int(max(0, x0 - mx)), int(max(0, y0 - my)),
                    int(min(fw, x1 + mx)), int(min(fh, y1 + my)))

        if len(bodies) <= _MAX_PER_PERSON_CROPS:
            crops = []
            for _, (x0, y0, x1, y1) in bodies:
                crops.append(expand(x0, y0, x1, y0 + (y1 - y0) * _UPPER_BODY_FRAC))
            return crops

        x0 = min(b[1][0] for b in bodies)
        y0 = min(b[1][1] for b in bodies)
        x1 = max(b[1][2] for b in bodies)
        y1 = max(b[1][1] + (b[1][3] - b[1][1]) * _UPPER_BODY_FRAC for b in bodies)
        return [expand(x0, y0, x1, y1)]
