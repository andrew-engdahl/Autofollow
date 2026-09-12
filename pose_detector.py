"""Pose detection using YOLOv8 — returns all detected persons."""

import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD, YOLO_MODEL, DETECTION_SCALE

# COCO keypoint indices for the hips
_HIP_INDICES = (11, 12)


def _best_device():
    """Pick the fastest available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


class PoseDetector:
    """Detects human poses in video frames using YOLOv8 Pose.

    Returns all detected persons as a list of dicts, not just the first.

    torch/ultralytics are imported here rather than at module level so the
    GUI can come up immediately and load the model on the video thread.
    """

    def __init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLOv8 is required. Install with: pip install ultralytics"
            ) from e

        print(f"Loading {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)
        self.device = _best_device()
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.conf_threshold = CONFIDENCE_THRESHOLD

    def warmup(self, width: int = 640, height: int = 360):
        """Run one throwaway inference so the first real frame isn't slow."""
        try:
            self.model(np.zeros((height, width, 3), dtype=np.uint8),
                       conf=self.conf_threshold, verbose=False)
        except Exception:
            pass

    def detect(self, frame):
        """Detect all persons in a frame.

        Args:
            frame: BGR frame from OpenCV.

        Returns:
            list[dict]: One dict per detected person:
                {
                    'bbox': (x_min, y_min, x_max, y_max),  # pixel coords in original frame
                    'keypoints': np.array,                   # shape (17, 4) — [x_norm, y_norm, 0, conf]
                    'confidence': float,
                }
            Empty list if no persons detected.
        """
        h, w = frame.shape[:2]

        # Optionally downscale frame for faster inference
        if DETECTION_SCALE < 1.0:
            det_w = max(1, int(w * DETECTION_SCALE))
            det_h = max(1, int(h * DETECTION_SCALE))
            det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_frame = frame
            det_w, det_h = w, h

        results = self.model(det_frame, conf=self.conf_threshold, verbose=False)

        persons = []
        if not results or results[0].keypoints is None:
            return persons

        # Pull everything off the GPU in one transfer.  Per-element .item()
        # calls each force a device sync, which on MPS costs more than the
        # inference itself once a few people are in frame.
        kpts_all = results[0].keypoints.data.cpu().numpy()   # (N, 17, 3) — x, y, conf in det_frame coords
        boxes = results[0].boxes
        box_conf = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else None

        if kpts_all.size == 0:
            return persons

        scale = np.array([w / det_w, h / det_h], dtype=np.float32)

        for i, kpts in enumerate(kpts_all):
            px = kpts[:, :2] * scale                     # (17, 2) pixel coords in original frame
            conf = kpts[:, 2]
            visible = conf > self.conf_threshold
            if not visible.any():
                continue

            vis_px = px[visible]
            x_min, y_min = vis_px.min(axis=0)
            x_max, y_max = vis_px.max(axis=0)

            # Reject poses where any visible keypoint lies outside the input frame.
            # This filters people whose body extends beyond the camera's field of view —
            # their partially-clipped poses would produce unreliable framing targets.
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                continue

            # Require at least one hip keypoint.  This filters foreground audience
            # members whose body is cut off at the waist — they appear as large,
            # high-confidence face detections with no lower body.
            if not any(visible[idx] for idx in _HIP_INDICES if idx < len(visible)):
                continue

            landmarks = np.zeros((len(kpts), 4), dtype=np.float32)
            landmarks[:, 0] = px[:, 0] / w
            landmarks[:, 1] = px[:, 1] / h
            landmarks[:, 3] = conf

            if box_conf is not None and i < len(box_conf):
                person_conf = float(box_conf[i])
            else:
                person_conf = float(conf.mean())

            persons.append({
                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'keypoints': landmarks,
                'confidence': person_conf,
            })

        return persons

    def release(self):
        pass  # YOLO handles cleanup automatically
