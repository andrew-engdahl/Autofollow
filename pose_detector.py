"""Pose detection using YOLOv8 — returns all detected persons."""

import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD, YOLO_MODEL, DETECTION_SCALE

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Install with: pip install ultralytics")


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
    """

    def __init__(self):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 is required. Install with: pip install ultralytics")

        print(f"Loading {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)
        device = _best_device()
        self.model.to(device)
        print(f"Using device: {device}")
        self.conf_threshold = CONFIDENCE_THRESHOLD

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

        kpts_data = results[0].keypoints.data    # shape: (N, 17, 3) — x, y, conf in det_frame coords
        boxes_data = results[0].boxes             # detection boxes

        scale_x = w / det_w
        scale_y = h / det_h

        for i, kpts in enumerate(kpts_data):
            # Build keypoints array normalized to original frame
            landmarks = []
            x_coords, y_coords = [], []

            for kpt in kpts:
                kx, ky, kconf = kpt[0].item(), kpt[1].item(), kpt[2].item()
                # Scale back to original frame coords
                px, py = kx * scale_x, ky * scale_y
                landmarks.append([px / w, py / h, 0.0, kconf])
                if kconf > self.conf_threshold:
                    x_coords.append(px)
                    y_coords.append(py)

            if not x_coords:
                continue

            # Reject poses where any visible keypoint lies outside the input frame.
            # This filters people whose body extends beyond the camera's field of view —
            # their partially-clipped poses would produce unreliable framing targets.
            if (min(x_coords) < 0 or min(y_coords) < 0
                    or max(x_coords) > w or max(y_coords) > h):
                continue

            # Require at least one hip keypoint (COCO indices 11=left_hip, 12=right_hip).
            # This filters foreground audience members whose body is cut off at the waist —
            # they appear as large, high-confidence face detections with no lower body.
            _HIP_INDICES = [11, 12]
            has_hips = any(
                landmarks[idx][3] > self.conf_threshold
                for idx in _HIP_INDICES
                if idx < len(landmarks)
            )
            if not has_hips:
                continue

            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))

            # Person confidence: use box confidence when available, else mean keypoint conf
            if boxes_data is not None and i < len(boxes_data):
                conf = float(boxes_data.conf[i])
            else:
                conf = float(np.mean([lm[3] for lm in landmarks]))

            persons.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'keypoints': np.array(landmarks, dtype=np.float32),
                'confidence': conf,
            })

        return persons

    def release(self):
        pass  # YOLO handles cleanup automatically
