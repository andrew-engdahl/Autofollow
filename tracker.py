"""Multi-person tracker with stable IDs, foreground scoring, and activity scoring."""

import time
import numpy as np
from dataclasses import dataclass
import config
from config import MAX_PERSONS, FOREGROUND_EXCLUSION_Y as _DEFAULT_EXCLUSION_Y

_IOU_THRESHOLD = 0.10   # minimum IoU to match; lowered to tolerate fast movement
_BBOX_ALPHA = 0.5       # bbox smoothing: higher = follows detection more closely (less lag)
_ACTIVITY_EMA_ALPHA = 0.3   # EMA smoothing for activity score — damps single-frame spikes
_ACTIVITY_DX_WEIGHT = 2.0   # horizontal displacement weight (heavier — indicates engagement)
_ACTIVITY_DY_WEIGHT = 1.0   # vertical displacement weight

# Center-distance fallback matching: if IoU is too low (person moved fast), accept a
# match when the detection center is within this fraction of the frame's diagonal.
_CENTER_DIST_FALLBACK = 0.25   # fraction of frame diagonal — generous to handle fast movers

_FG_SCORE_EMA_ALPHA = 0.15   # heavy smoothing on bbox-area score to prevent fg_ratio oscillation

# A face match is forgotten if it hasn't been re-confirmed within this long.
# Recognition runs asynchronously at a few Hz, so a 5 s hold rides out the
# occasional missed detection (head turned, brief occlusion) without letting a
# stale identity stick to a track after a hand-off.
FACE_MATCH_STALE_SECONDS = 5.0


def priority_weight(priority: int) -> float:
    """Score multiplier from a profile's priority (0–10).

    priority  0 -> 1.0  (no boost — same as an unmatched person)
    priority  5 -> 2.0
    priority 10 -> 3.0

    Used to bias candidate selection toward higher-priority profiles in both
    primary-focus and time-switcher modes.
    """
    return 1.0 + max(0, min(10, int(priority))) / 5.0


@dataclass
class TrackedPerson:
    id: str                          # 'person1', 'person2', …
    bbox: tuple                      # (x_min, y_min, x_max, y_max) in pixel coords
    keypoints: np.ndarray            # (17, 4) — [x_norm, y_norm, 0, conf]
    confidence: float
    foreground_score: float = 0.0    # EMA-smoothed bbox_area / frame_area — larger = closer
    activity_score: float = 0.0      # EMA-smoothed weighted center displacement
    frames_unseen: int = 0           # consecutive detector passes without a match
    last_seen: float = 0.0           # time.monotonic() of the last matched detection
    # Face-recognition match (populated by FaceRecognizer via PersonTracker.set_profile_match)
    profile_id: str | None = None    # ID of matched People profile, if any
    profile_name: str | None = None  # display name of matched profile
    profile_priority: int = 0        # 0–10 priority of matched profile
    profile_score: float = 0.0       # cosine similarity at last match
    face_seen_at: float = 0.0        # time.monotonic() of the last confirmed face match

    # Transient voice priority boost — applied when speaker recognition matches
    # this person's profile. Decays back to 0 after a few seconds of silence.
    voice_boost: float = 0.0         # 0.0 – 5.0 priority units; added to profile_priority
    voice_boost_expires_at: float = 0.0  # time.monotonic() expiry for the boost

    @property
    def effective_priority(self) -> float:
        """Profile priority plus any active voice boost.

        Clamped to ≤15 so a recognized speaker with priority 10 caps at 15
        (instead of unbounded growth as boosts overlap).
        """
        return min(15.0, float(self.profile_priority) + float(self.voice_boost))


def _iou(a, b):
    """Intersection-over-Union of two bboxes (x_min, y_min, x_max, y_max)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


# COCO keypoint indices that define the torso
_TORSO_KP_INDICES = (5, 6, 11, 12)  # left shoulder, right shoulder, left hip, right hip


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _torso_center(keypoints: np.ndarray, frame_w: float, frame_h: float,
                  bbox=None) -> tuple[float, float]:
    """Return the median (cx, cy) of visible torso keypoints in pixel coords.

    Falls back to bbox center if no torso keypoints are confident enough.
    keypoints: (17, 4) array — [x_norm, y_norm, 0, conf]
    """
    pts = []
    for idx in _TORSO_KP_INDICES:
        kp = keypoints[idx]
        if kp[3] > 0.3:  # confidence threshold
            pts.append((kp[0] * frame_w, kp[1] * frame_h))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(np.median(xs)), float(np.median(ys)))
    if bbox is not None:
        return _center(bbox)
    return (frame_w / 2.0, frame_h / 2.0)


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


class PersonTracker:
    """Maintains stable person identities across frames using greedy IoU matching.

    Call update() once per *detector pass* (not per video frame): tracks expire
    on wall-clock time, so skipping detection on some frames doesn't shorten
    how long a briefly-occluded person is remembered.
    """

    def __init__(self):
        self._tracks: dict[str, TrackedPerson] = {}   # id → TrackedPerson
        self._next_index = 1                           # for assigning 'person1', 'person2', …
        self._frame_area = 1.0

    def update(self, detections: list[dict], frame_shape: tuple,
               foreground_exclusion_y: float | None = None,
               max_persons: int | None = None) -> list[TrackedPerson]:
        """Match new detections to existing tracks; assign stable IDs.

        Args:
            detections: List of dicts from PoseDetector.detect().
            frame_shape: (height, width) of the source frame.

        Returns:
            List of TrackedPerson sorted by foreground_score descending (primary first).
        """
        fh, fw = frame_shape[:2]
        now = time.monotonic()
        self._frame_area = max(1.0, float(fw * fh))

        # Drop detections whose torso center falls in the foreground exclusion zone.
        # Using the torso center (median of shoulders + hips) means a performer on stage
        # whose legs extend into the exclusion zone is still tracked correctly.
        excl = foreground_exclusion_y if foreground_exclusion_y is not None else _DEFAULT_EXCLUSION_Y
        if excl > 0.0:
            exclusion_threshold = fh * (1.0 - excl)
            detections = [
                d for d in detections
                if _torso_center(d['keypoints'], fw, fh, d['bbox'])[1] <= exclusion_threshold
            ]

        for t in self._tracks.values():
            t.frames_unseen += 1

        det_to_track = self._match(detections, fw, fh)

        for det_idx, tid in det_to_track.items():
            det = detections[det_idx]
            track = self._tracks[tid]
            prev_cx, prev_cy = _torso_center(track.keypoints, fw, fh, track.bbox)
            curr_cx, curr_cy = _torso_center(det['keypoints'], fw, fh, det['bbox'])
            dx = abs(curr_cx - prev_cx)
            dy = abs(curr_cy - prev_cy)
            raw_weighted = dx * _ACTIVITY_DX_WEIGHT + dy * _ACTIVITY_DY_WEIGHT
            activity = (_ACTIVITY_EMA_ALPHA * raw_weighted
                        + (1.0 - _ACTIVITY_EMA_ALPHA) * track.activity_score)
            # Smooth bbox toward new detection to suppress frame-to-frame jitter
            track.bbox = tuple(
                int(old * (1 - _BBOX_ALPHA) + new * _BBOX_ALPHA)
                for old, new in zip(track.bbox, det['bbox'])
            )
            track.keypoints = det['keypoints']
            track.confidence = det['confidence']
            raw_fg = _area(track.bbox) / self._frame_area
            track.foreground_score = (_FG_SCORE_EMA_ALPHA * raw_fg
                                      + (1.0 - _FG_SCORE_EMA_ALPHA) * track.foreground_score)
            track.activity_score = activity
            track.frames_unseen = 0
            track.last_seen = now

        # Create new tracks for unmatched detections (up to max_persons limit)
        limit = max_persons if max_persons is not None else MAX_PERSONS
        for det_idx, det in enumerate(detections):
            if det_idx in det_to_track:
                continue
            if len(self._tracks) >= limit:
                break
            new_id = f'person{self._next_index}'
            self._next_index += 1
            self._tracks[new_id] = TrackedPerson(
                id=new_id,
                bbox=det['bbox'],
                keypoints=det['keypoints'],
                confidence=det['confidence'],
                foreground_score=_area(det['bbox']) / self._frame_area,
                activity_score=0.0,
                frames_unseen=0,
                last_seen=now,
            )

        # Drop tracks that haven't been seen recently
        to_drop = [tid for tid, t in self._tracks.items()
                   if now - t.last_seen > config.TRACK_DROPOUT_SECONDS]
        for tid in to_drop:
            del self._tracks[tid]

        return self.get_all()

    def _match(self, detections: list[dict], fw: int, fh: int) -> dict[int, str]:
        """Return {detection index → track id} for all matched pairs.

        Phase 1 collects every (IoU, det, track) pair above threshold and assigns
        greedily best-first, so a weak pair can't steal a track from a better
        match further down the list.  Phase 2 gives the leftovers a
        center-distance fallback to survive fast movement between detector passes.
        """
        det_to_track: dict[int, str] = {}
        used_tracks: set[str] = set()
        existing_ids = list(self._tracks.keys())

        iou_pairs = []
        for det_idx, det in enumerate(detections):
            for tid in existing_ids:
                score = _iou(det['bbox'], self._tracks[tid].bbox)
                if score >= _IOU_THRESHOLD:
                    iou_pairs.append((score, det_idx, tid))
        iou_pairs.sort(reverse=True)

        for _, det_idx, tid in iou_pairs:
            if det_idx in det_to_track or tid in used_tracks:
                continue
            det_to_track[det_idx] = tid
            used_tracks.add(tid)

        max_center_dist = (fh ** 2 + fw ** 2) ** 0.5 * _CENTER_DIST_FALLBACK
        for det_idx, det in enumerate(detections):
            if det_idx in det_to_track:
                continue
            best_id = None
            best_dist = max_center_dist
            dcx, dcy = _center(det['bbox'])
            for tid in existing_ids:
                if tid in used_tracks:
                    continue
                tcx, tcy = _center(self._tracks[tid].bbox)
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                det_to_track[det_idx] = best_id
                used_tracks.add(best_id)

        return det_to_track

    # ------------------------------------------------------------------
    # Face-recognition match plumbing
    # ------------------------------------------------------------------

    def set_profile_match(self, person_id: str, profile_id: str | None,
                          profile_name: str | None, priority: int,
                          score: float):
        """Attach (or clear) a face-recognition match to a tracked person.

        A profile can only belong to one track at a time: if it was previously
        attached to a different track (e.g. an ID swap after occlusion), that
        track is cleared so priority doesn't get counted twice.
        """
        track = self._tracks.get(person_id)
        if track is None:
            return
        if profile_id is not None:
            for other in self._tracks.values():
                if other is not track and other.profile_id == profile_id:
                    self._clear_profile(other)
        track.profile_id = profile_id
        track.profile_name = profile_name
        track.profile_priority = int(priority)
        track.profile_score = float(score)
        track.face_seen_at = time.monotonic()

    @staticmethod
    def _clear_profile(track: TrackedPerson):
        track.profile_id = None
        track.profile_name = None
        track.profile_priority = 0
        track.profile_score = 0.0
        track.voice_boost = 0.0
        track.voice_boost_expires_at = 0.0

    def apply_voice_boost(self, profile_id: str, boost: float, hold_seconds: float) -> int:
        """Raise voice_boost on every tracked person matched to ``profile_id``.

        Called when the SpeakerRecognizer matches an enrolled voice. Returns
        the number of tracks that received the boost (0 if no tracked person
        is currently matched to that profile — e.g. the speaker isn't on
        camera, only their voice is on the mic).
        """
        expiry = time.monotonic() + hold_seconds
        boosted = 0
        for t in self._tracks.values():
            if t.profile_id == profile_id:
                t.voice_boost = max(t.voice_boost, float(boost))
                t.voice_boost_expires_at = max(t.voice_boost_expires_at, expiry)
                boosted += 1
        return boosted

    def expire_transients(self):
        """Drop expired voice boosts and stale face matches. Call once per frame."""
        now = time.monotonic()
        for t in self._tracks.values():
            if t.voice_boost > 0.0 and now >= t.voice_boost_expires_at:
                t.voice_boost = 0.0
                t.voice_boost_expires_at = 0.0
            if (t.profile_id is not None
                    and now - t.face_seen_at > FACE_MATCH_STALE_SECONDS):
                self._clear_profile(t)

    @staticmethod
    def match_face_to_bbox(face_bbox: tuple[int, int, int, int],
                           bodies: list[tuple[str, tuple]]) -> str | None:
        """Pick the body bbox that contains the given face bbox.

        ``bodies`` is a list of (track_id, (x1, y1, x2, y2)) — a snapshot taken
        when the frame was handed to the recognizer, so this stays correct even
        though recognition finishes some time later.  The face center must lie
        inside the body bbox and in its upper 60% (faces are above the waist).
        Ties go to the smaller body bbox (nearer person occluding background).
        """
        fx1, fy1, fx2, fy2 = face_bbox
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0
        best: tuple[float, str] | None = None
        for tid, (bx1, by1, bx2, by2) in bodies:
            if not (bx1 <= fcx <= bx2 and by1 <= fcy <= by2):
                continue
            if fcy > by1 + (by2 - by1) * 0.6:
                continue
            area = max(1.0, (bx2 - bx1) * (by2 - by1))
            if best is None or area < best[0]:
                best = (area, tid)
        return best[1] if best else None

    def get_primary(self) -> TrackedPerson | None:
        """Return the most-foreground (largest bbox) tracked person."""
        if not self._tracks:
            return None
        return max(self._tracks.values(), key=lambda t: t.foreground_score)

    def get_all(self) -> list[TrackedPerson]:
        """Return all active tracks sorted foreground-first."""
        return sorted(self._tracks.values(),
                      key=lambda t: t.foreground_score, reverse=True)

    def reset(self):
        self._tracks.clear()
        self._next_index = 1
