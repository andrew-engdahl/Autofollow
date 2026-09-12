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
