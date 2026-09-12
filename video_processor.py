"""Headless video processor — usable standalone (CLI) or embedded in VideoThread."""

import cv2
import numpy as np
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, DETECTION_INTERVAL, SHOW_OVERLAY,
)
from pose_detector import PoseDetector
from tracker import PersonTracker
from framing_engine import FramingEngine
from smoothing import PTZSmoother
from camera import open_capture, describe_capture


class VideoProcessor:
    """Processes a camera stream and produces cropped output frames.

    This class handles the pure processing logic. For the GUI path the
    VideoThread in control_ui.py drives it; for CLI use, call
    process_video_stream() directly.
    """

    def __init__(self, camera_index: int = 0, output_file: str | None = None,
                 show_overlay: bool | None = None):
        self.camera_index = camera_index
        self.output_file = output_file
        self.show_overlay = show_overlay if show_overlay is not None else SHOW_OVERLAY

        self._detector = PoseDetector()
        self._tracker = PersonTracker()
        self._framing: FramingEngine | None = None
        self._smoother = PTZSmoother()

        self._cap = None
        self._out = None
        self._frame_count = 0
        self.input_width: int | None = None
        self.input_height: int | None = None

    def initialize_camera(self):
        """Open camera and prepare output writer."""
        self._cap = open_capture(self.camera_index)
        if self._cap is None:
            raise RuntimeError(f"Failed to open camera device {self.camera_index}")

        w, h, fps = describe_capture(self._cap)

        self.input_width = w
        self.input_height = h
        self._framing = FramingEngine(w, h)
        self._smoother.set_bounds(w, h)

        print(f"Camera: {w}×{h} @ {fps:.1f} fps")

        if self.output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._out = cv2.VideoWriter(self.output_file, fourcc, fps,
                                        (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        return w, h, fps

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run the full pipeline on a single frame.

        Returns a dict with 'frame' (cropped output), 'persons', 'active_id'.
        """
        # Detection only runs every DETECTION_INTERVAL frames; in between, keep
        # following the last known track positions (tracks expire on wall time).
        if self._frame_count % DETECTION_INTERVAL == 0:
            detections = self._detector.detect(frame)
            persons = self._tracker.update(detections, frame.shape)
        else:
            persons = self._tracker.get_all()

        output_frame, active_id = self._render_primary(frame, persons)

        if self.show_overlay:
            n = len(persons)
            cv2.putText(output_frame,
                        f"Frame {self._frame_count} | {n} person(s) | {active_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self._frame_count += 1
        return {'frame': output_frame, 'persons': persons, 'active_id': active_id}

    def _render_primary(self, frame, persons):
        if not persons:
            tx, ty, tz = self._framing.default_target()
            sx, sy, sz = self._smoother.update('primary', tx, ty, tz)
        else:
            primary = persons[0]
            tx, ty, tz = self._framing.calculate_target(primary)
            sx, sy, sz = self._smoother.update('primary', tx, ty, tz)
        return self._framing.apply_crop(frame, sx, sy, sz), \
               persons[0].id if persons else 'none'

    def process_video_stream(self, max_frames: int | None = None,
                             show_preview: bool = True) -> dict:
        """CLI entry point: run the capture loop until 'q' or max_frames."""
        print("Starting… press 'q' to quit")
        frame_num = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            cropped = result['frame']

            if self._out:
                self._out.write(cropped)

            if show_preview:
                cv2.imshow('Autofollow', cropped)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1
            if max_frames and frame_num >= max_frames:
                break

        print(f"Processed {self._frame_count} frames")
        return {'total_frames': self._frame_count}

    def cleanup(self):
        if self._cap:
            self._cap.release()
        if self._out:
            self._out.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
