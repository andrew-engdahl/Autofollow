"""Multi-person tracker with stable IDs, foreground scoring, and activity scoring."""

import numpy as np
from dataclasses import dataclass, field
from config import MAX_PERSONS, FOREGROUND_EXCLUSION_Y as _DEFAULT_EXCLUSION_Y

_DROPOUT_FRAMES = 60    # drop a track after this many consecutive unmatched frames (~2s at 30fps)
_IOU_THRESHOLD = 0.10   # minimum IoU to match; lowered to tolerate fast movement
_BBOX_ALPHA = 0.5       # bbox smoothing: higher = follows detection more closely (less lag)
_ACTIVITY_EMA_ALPHA = 0.3   # EMA smoothing for activity score — damps single-frame spikes
_ACTIVITY_DX_WEIGHT = 2.0   # horizontal displacement weight (heavier — indicates engagement)
_ACTIVITY_DY_WEIGHT = 1.0   # vertical displacement weight

# Center-distance fallback matching: if IoU is too low (person moved fast), accept a
# match when the detection center is within this fraction of the frame's diagonal.
_CENTER_DIST_FALLBACK = 0.25   # fraction of frame diagonal — generous to handle fast movers


_FG_SCORE_EMA_ALPHA = 0.15   # heavy smoothing on bbox-area score to prevent fg_ratio oscillation


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
    frames_unseen: int = 0
    # Face-recognition match (populated by FaceRecognizer via PersonTracker.set_profile_match)
    profile_id: str | None = None    # ID of matched People profile, if any
    profile_name: str | None = None  # display name of matched profile
    profile_priority: int = 0        # 0–10 priority of matched profile
    profile_score: float = 0.0       # cosine similarity at last match
    frames_since_face: int = 999     # frames since the face was last matched (high = stale)

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
    """Maintains stable person identities across frames using greedy IoU matching."""

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

        # Increment unseen counter for all existing tracks
        for t in self._tracks.values():
            t.frames_unseen += 1
            t.frames_since_face += 1

        # Matching: build a full IoU matrix, then greedily assign best pairs
        # (highest IoU first) to reduce the "greedy order" artifacts where a
        # low-confidence pair steals a track from a better match further down.
        frame_diag = (fh ** 2 + fw ** 2) ** 0.5
        max_center_dist = frame_diag * _CENTER_DIST_FALLBACK

        existing_ids = list(self._tracks.keys())
        matched_ids = set()
        matched_det_indices = set()

        # Phase 1: IoU matching — collect all (iou, det_idx, track_id) pairs,
        # sort descending by IoU, then greedily assign best-first.
        iou_pairs = []
        for det_idx, det in enumerate(detections):
            for tid in existing_ids:
                score = _iou(det['bbox'], self._tracks[tid].bbox)
                if score >= _IOU_THRESHOLD:
                    iou_pairs.append((score, det_idx, tid))
        iou_pairs.sort(reverse=True)

        for score, det_idx, tid in iou_pairs:
            if det_idx in matched_det_indices or tid in matched_ids:
                continue
            matched_ids.add(tid)
            matched_det_indices.add(det_idx)

        # Phase 2: center-distance fallback for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            best_id = None
            best_dist = max_center_dist
            dcx, dcy = _center(det['bbox'])
            for tid in existing_ids:
                if tid in matched_ids:
                    continue
                tcx, tcy = _center(self._tracks[tid].bbox)
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                matched_ids.add(best_id)
                matched_det_indices.add(det_idx)

        # Phase 3: apply updates for all matched pairs
        # Re-derive the (det_idx → track_id) mapping from the two match phases.
        # Rebuild by re-scanning (cheap — small N).
        det_to_track: dict[int, str] = {}
        # IoU matches
        seen_d: set[int] = set()
        seen_t: set[str] = set()
        for score, det_idx, tid in iou_pairs:
            if det_idx not in seen_d and tid not in seen_t:
                det_to_track[det_idx] = tid
                seen_d.add(det_idx)
                seen_t.add(tid)
        # Center-distance matches (re-run to get the actual assignments)
        for det_idx, det in enumerate(detections):
            if det_idx in det_to_track:
                continue
            best_id = None
            best_dist = max_center_dist
            dcx, dcy = _center(det['bbox'])
            for tid in existing_ids:
                if tid in seen_t:
                    continue
                tcx, tcy = _center(self._tracks[tid].bbox)
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                det_to_track[det_idx] = best_id
                seen_t.add(best_id)

        matched_det_indices = set(det_to_track.keys())

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

        # Create new tracks for unmatched detections (up to max_persons limit)
        limit = max_persons if max_persons is not None else MAX_PERSONS
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
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
            )

        # Drop tracks that haven't been seen recently
        to_drop = [tid for tid, t in self._tracks.items()
                   if t.frames_unseen > _DROPOUT_FRAMES]
        for tid in to_drop:
            del self._tracks[tid]

        return sorted(self._tracks.values(),
                      key=lambda t: t.foreground_score, reverse=True)

    # ------------------------------------------------------------------
    # Face-recognition match plumbing
    # ------------------------------------------------------------------

    # Stale matches are cleared after this many frames without re-confirmation.
    # At ~5fps face recognition cadence and 30fps capture, ~5s is a comfortable hold.
    _FACE_MATCH_STALE_FRAMES = 150

    def set_profile_match(self, person_id: str, profile_id: str | None,
                          profile_name: str | None, priority: int,
                          score: float):
        """Attach (or clear) a face-recognition match to a tracked person."""
        track = self._tracks.get(person_id)
        if track is None:
            return
        track.profile_id = profile_id
        track.profile_name = profile_name
        track.profile_priority = int(priority)
        track.profile_score = float(score)
        track.frames_since_face = 0

    def apply_voice_boost(self, profile_id: str, boost: float, hold_seconds: float) -> int:
        """Raise voice_boost on every tracked person matched to ``profile_id``.

        Called when the SpeakerRecognizer matches an enrolled voice. Returns
        the number of tracks that received the boost (0 if no tracked person
        is currently matched to that profile — e.g. the speaker isn't on
        camera, only their voice is on the mic).
        """
        import time
        expiry = time.monotonic() + hold_seconds
        boosted = 0
        for t in self._tracks.values():
            if t.profile_id == profile_id:
                t.voice_boost = max(t.voice_boost, float(boost))
                t.voice_boost_expires_at = max(t.voice_boost_expires_at, expiry)
                boosted += 1
        return boosted

    def decay_voice_boosts(self):
        """Clear voice_boost on tracks whose boost has expired. Call per frame."""
        import time
        now = time.monotonic()
        for t in self._tracks.values():
            if t.voice_boost > 0.0 and now >= t.voice_boost_expires_at:
                t.voice_boost = 0.0
                t.voice_boost_expires_at = 0.0

    def expire_stale_face_matches(self):
        """Forget profile matches that haven't been re-confirmed recently."""
        for t in self._tracks.values():
            if (t.profile_id is not None
                    and t.frames_since_face > self._FACE_MATCH_STALE_FRAMES):
                t.profile_id = None
                t.profile_name = None
                t.profile_priority = 0
                t.profile_score = 0.0

    def find_by_face_bbox(self, face_bbox: tuple[int, int, int, int],
                          frame_shape: tuple) -> str | None:
        """Locate the tracked person whose body contains the given face bbox.

        The match heuristic is: the face center must lie inside a tracked
        body bbox, *and* be in the upper 60% of that body bbox (faces are
        above the waist). Ties are broken by smaller body bbox area —
        nearer person occluding background.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0
        candidates: list[tuple[float, str]] = []
        for tid, track in self._tracks.items():
            bx1, by1, bx2, by2 = track.bbox
            if not (bx1 <= fcx <= bx2):
                continue
            if not (by1 <= fcy <= by2):
                continue
            # Face must be in the upper portion of the body bbox
            upper_limit = by1 + (by2 - by1) * 0.6
            if fcy > upper_limit:
                continue
            area = max(1.0, (bx2 - bx1) * (by2 - by1))
            candidates.append((area, tid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

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
