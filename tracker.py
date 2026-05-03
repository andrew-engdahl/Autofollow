"""Multi-person tracker with stable IDs, foreground scoring, and activity scoring."""

import numpy as np
from dataclasses import dataclass, field
from config import MAX_PERSONS, FOREGROUND_EXCLUSION_Y as _DEFAULT_EXCLUSION_Y

_DROPOUT_FRAMES = 10   # drop a track after this many consecutive unmatched frames
_IOU_THRESHOLD = 0.3   # minimum IoU to match a detection to an existing track
_BBOX_ALPHA = 0.35     # bbox smoothing: fraction of new detection blended each frame


@dataclass
class TrackedPerson:
    id: str                          # 'person1', 'person2', …
    bbox: tuple                      # (x_min, y_min, x_max, y_max) in pixel coords
    keypoints: np.ndarray            # (17, 4) — [x_norm, y_norm, 0, conf]
    confidence: float
    foreground_score: float = 0.0    # bbox_area / frame_area — larger = closer
    activity_score: float = 0.0      # bbox-center displacement since last frame
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
               foreground_exclusion_y: float | None = None) -> list[TrackedPerson]:
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

        # Greedy IoU matching: pair each detection with the best matching track
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

            if best_iou >= _IOU_THRESHOLD and best_id is not None:
                # Update existing track
                track = self._tracks[best_id]
                prev_center = _center(track.bbox)
                curr_center = _center(det['bbox'])
                activity = float(np.linalg.norm(
                    np.array(curr_center) - np.array(prev_center)
                ))
                # Smooth bbox toward new detection to suppress frame-to-frame jitter
                track.bbox = tuple(
                    int(old * (1 - _BBOX_ALPHA) + new * _BBOX_ALPHA)
                    for old, new in zip(track.bbox, det['bbox'])
                )
                track.keypoints = det['keypoints']
                track.confidence = det['confidence']
                track.foreground_score = _area(track.bbox) / self._frame_area
                track.activity_score = activity
                track.frames_unseen = 0
                matched_ids.add(best_id)
                matched_det_indices.add(det_idx)

        # Create new tracks for unmatched detections (up to MAX_PERSONS)
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            if len(self._tracks) >= MAX_PERSONS:
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
