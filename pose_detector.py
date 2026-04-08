"""Pose detection using YOLOv8 Pose Detection."""

import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available, some functionality will be limited")


class PoseDetector:
    """Detects human pose in video frames using YOLOv8 Pose."""

    def __init__(self):
        """Initialize the YOLOv8 Pose detector."""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 is required. Install with: pip install ultralytics")
        
        # Load YOLOv8 Pose model (auto-downloads if needed)
        print("Loading YOLOv8 Pose model (this may take a moment on first run)...")
        self.model = YOLO('yolov8n-pose.pt')  # nano model for speed
        self.model.to('cpu')  # Use CPU (or 'cuda' for GPU if available)
        self.conf_threshold = CONFIDENCE_THRESHOLD

    def detect(self, frame):
        """
        Detect pose landmarks in a frame.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            dict: Contains 'landmarks', 'bbox', and 'detected' keys
        """
        h, w, c = frame.shape
        
        # Run YOLO Pose detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        output = {
            'detected': False,
            'landmarks': None,
            'bbox': None,
            'keypoints': None
        }
        
        if len(results) > 0 and results[0].keypoints is not None:
            # Get first detected person
            keypoints = results[0].keypoints
            
            if keypoints.data is not None and len(keypoints.data) > 0:
                # YOLOv8 keypoints format: [x, y, confidence] for each of 17 points
                kpts = keypoints.data[0]  # First person
                
                landmarks = []
                for kpt in kpts:
                    x, y, conf = kpt[0].item(), kpt[1].item(), kpt[2].item()
                    # Normalize to 0-1 range
                    landmarks.append([x / w, y / h, 0, conf])  # z=0 since YOLO doesn't provide z
                
                output['landmarks'] = np.array(landmarks)
                output['detected'] = True
                
                # Calculate bounding box from visible keypoints
                x_coords = []
                y_coords = []
                
                for idx, (x, y, conf) in enumerate(kpts):
                    if conf > CONFIDENCE_THRESHOLD:
                        x_coords.append(x.item())
                        y_coords.append(y.item())
                
                if x_coords and y_coords:
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Ensure bbox is within frame
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    output['bbox'] = (x_min, y_min, x_max, y_max)
                    output['keypoints'] = output['landmarks']
        
        return output

    def draw_pose(self, frame, detection_result):
        """
        Draw pose landmarks and skeleton on the frame.

        Args:
            frame: Input video frame
            detection_result: Output from detect() method

        Returns:
            frame: Frame with pose drawn
        """
        if detection_result['detected'] and detection_result['landmarks'] is not None:
            h, w, c = frame.shape
            landmarks = detection_result['landmarks']
            
            # Convert normalized landmarks to pixel coordinates
            keypoints = landmarks[:, :2]
            keypoints[:, 0] *= w
            keypoints[:, 1] *= h
            
            # Draw circles for each keypoint
            for idx, (x, y) in enumerate(keypoints):
                confidence = landmarks[idx, 3]
                if confidence > CONFIDENCE_THRESHOLD:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # Draw bounding box
            if detection_result['bbox']:
                x_min, y_min, x_max, y_max = detection_result['bbox']
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return frame

    def release(self):
        """Clean up resources."""
        pass  # YOLO handles cleanup automatically
