"""Multi-person tracker with stable IDs, foreground scoring, and activity scoring."""

import numpy as np
from dataclasses import dataclass, field
from config import MAX_PERSONS, FOREGROUND_EXCLUSION_Y as _DEFAULT_EXCLUSION_Y

_DROPOUT_FRAMES = 30    # drop a track after this many consecutive unmatched frames (~1s at 30fps)
_IOU_THRESHOLD = 0.15   # minimum IoU to match; lowered to tolerate fast movement
_BBOX_ALPHA = 0.5       # bbox smoothing: higher = follows detection more closely (less lag)
_ACTIVITY_EMA_ALPHA = 0.3   # EMA smoothing for activity score — damps single-frame spikes
_ACTIVITY_DX_WEIGHT = 2.0   # horizontal displacement weight (heavier — indicates engagement)
_ACTIVITY_DY_WEIGHT = 1.0   # vertical displacement weight

# Center-distance fallback matching: if IoU is too low (person moved fast), accept a
# match when the detection center is within this fraction of the frame's diagonal.
_CENTER_DIST_FALLBACK = 0.15   # fraction of frame diagonal (~0.15 = generous for fast movers)


_FG_SCORE_EMA_ALPHA = 0.15   # heavy smoothing on bbox-area score to prevent fg_ratio oscillation

@dataclass
class TrackedPerson:
    id: str                          # 'person1', 'person2', …
    bbox: tuple                      # (x_min, y_min, x_max, y_max) in pixel coords
    keypoints: np.ndarray            # (17, 4) — [x_norm, y_norm, 0, conf]
    confidence: float
    foreground_score: float = 0.0    # EMA-smoothed bbox_area / frame_area — larger = closer
    activity_score: float = 0.0      # EMA-smoothed weighted center displacement
    frames_unseen: int = 0


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


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


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

        # Drop detections whose bottom edge falls in the foreground exclusion zone.
        # Audience members standing in front of the stage appear in the lower part of
        # the frame; performers on stage are higher up.
        excl = foreground_exclusion_y if foreground_exclusion_y is not None else _DEFAULT_EXCLUSION_Y
        if excl > 0.0:
            exclusion_threshold = fh * (1.0 - excl)
            detections = [d for d in detections if d['bbox'][3] <= exclusion_threshold]

        # Increment unseen counter for all existing tracks
        for t in self._tracks.values():
            t.frames_unseen += 1

        # Greedy matching: IoU first, center-distance fallback for fast movers.
        # The fallback prevents ID churn when a person moves quickly enough that
        # their new detection has little overlap with their previous bbox.
        frame_diag = (fh ** 2 + fw ** 2) ** 0.5
        max_center_dist = frame_diag * _CENTER_DIST_FALLBACK

        existing_ids = list(self._tracks.keys())
        matched_ids = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detections):
            best_iou, best_id = 0.0, None
            for tid in existing_ids:
                if tid in matched_ids:
                    continue
                score = _iou(det['bbox'], self._tracks[tid].bbox)
                if score > best_iou:
                    best_iou, best_id = score, tid

            # Primary match: IoU above threshold
            if best_iou >= _IOU_THRESHOLD and best_id is not None:
                pass  # accepted — fall through to update block below
            else:
                # Fallback: find the nearest unmatched track by center distance
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
                # Update existing track
                track = self._tracks[best_id]
                prev_cx, prev_cy = _center(track.bbox)
                curr_cx, curr_cy = _center(det['bbox'])
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
                matched_ids.add(best_id)
                matched_det_indices.add(det_idx)

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
