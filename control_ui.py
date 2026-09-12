"""PyQt5 control panel with integrated video thread and fullscreen output window."""

import time
import threading
from collections import deque
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

import config
from camera import scan_cameras, open_capture, describe_capture
from tracker import PersonTracker
from framing_engine import FramingEngine
from smoothing import PTZSmoother
from switcher import VirtualSwitcher, _PRETRAVEL_DURATION
from profiles import ProfileStore
from audio_thread import AudioThread, SPEAKER_BOOST_HOLD_S
from audio_capture import list_input_devices
from face_worker import FaceRecognitionWorker


# Hand a frame to the face-recognition worker at most every N captured frames
# (~5 Hz at 30 fps).  The worker runs on its own thread and drops the request
# if it is still busy, so this is a ceiling, not a fixed cost.
FACE_RECOGNITION_INTERVAL = 6

# Priority units added transiently to a profile whose voice is being recognized.
# Combined with profile.priority via TrackedPerson.effective_priority.
VOICE_PRIORITY_BOOST = 5.0

# Primary Focus dwell while music mode is active (performances want snappier
# hand-offs than a sermon).
_MUSIC_MODE_DWELL_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Diagnostics constants
# ---------------------------------------------------------------------------

# Standard COCO 17-point skeleton connections (0-indexed)
_SKELETON = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # shoulder bar
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # torso sides
    (11, 12),                 # hip bar
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# The diagnostics preview is small, so annotate a downscaled copy of the
# camera frame rather than a full 1080p/4K copy every frame.
_DIAG_MAX_WIDTH = 960

# How fast to zoom out each frame while searching for a lost subject
# (zoom units/frame).  At 30fps this goes from zoom 3 to the full frame in ~3s.
_SEARCH_ZOOM_OUT_RATE = 0.008

# Camera watchdog: if no frame arrives for this long, try to re-open the device.
_CAMERA_STALL_SECONDS = 3.0

# Maximum output frames handed to the UI thread but not yet painted.  Anything
# beyond this is dropped so a slow UI can't build up a growing latency queue.
_MAX_FRAMES_IN_FLIGHT = 2

# Per-person colors: evenly-spaced hues around the HSV wheel (BGR).
def _person_color(index: int) -> tuple[int, int, int]:
    """Return a vivid BGR color for person at position `index` (0-based)."""
    hue = int((index * 137.5) % 180)   # golden-angle step keeps neighbours distinct
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_BGR888 = hasattr(QImage, 'Format_BGR888')


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    h, w, ch = frame.shape
    if _HAS_BGR888:
        # Qt swaps channels while copying — saves a separate cvtColor pass.
        frame = np.ascontiguousarray(frame)
        return QImage(frame.data, w, h, ch * w, QImage.Format_BGR888).copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def _bbox_center_x(person) -> float:
    return (person.bbox[0] + person.bbox[2]) / 2.0


# ---------------------------------------------------------------------------
# App-wide shared state (written by UI thread, read by video thread)
# ---------------------------------------------------------------------------

class AppState:
    # Fields persisted between launches (QSettings key → attribute).
    _PERSISTED = (
        'camera_index', 'tracking_mode', 'shot_type', 'switch_mode',
        'switch_trigger', 'switch_interval', 'crossfade_duration',
        'auto_follow_enabled', 'foreground_exclusion_y', 'max_persons',
        'diag_overlays', 'display_index', 'audio_enabled', 'audio_device_name',
    )

    def __init__(self):
        self._lock = QMutex()
        self.camera_index: int = config.CAMERA_INDEX
        self.tracking_mode: str = config.TRACKING_MODE   # 'primary' | 'switcher'
        self.shot_type: str = config.SHOT_TYPE
        self.switch_mode: str = config.SWITCH_MODE       # 'cut' | 'crossfade'
        self.switch_trigger: str = config.SWITCH_TRIGGER # 'time' | 'manual'
        self.switch_interval: float = config.SWITCH_INTERVAL
        self.crossfade_duration: float = config.CROSSFADE_DURATION
        self.manual_switch_id: str | None = None         # set by UI, consumed by video thread
        self.camera_change_requested: bool = False
        self.auto_follow_enabled: bool = True
        self.foreground_exclusion_y: float = config.FOREGROUND_EXCLUSION_Y
        self.max_persons: int = config.MAX_PERSONS
        self.diag_visible: bool = False                  # diagnostics window is on screen
        self.diag_overlays: bool = config.SHOW_DIAGNOSTICS
        self.display_index: int = 0

        # Audio-driven state (written by AudioThread handlers on the UI thread)
        self.audio_enabled: bool = False                 # user wants audio analysis on
        self.audio_device_name: str = ''                 # '' = system default
        self.music_mode: bool = False
        self.audio_music_score: float = 0.0
        self.audio_speech_score: float = 0.0
        # Most-recent recognized speaker (sticky until a different one matches
        # or the boost window expires; shown in the diagnostics panel).
        self.audio_speaker_name: str | None = None
        self.audio_speaker_score: float = 0.0
        self.audio_speaker_expires_at: float = 0.0
        # Pending voice boost: (profile_id, boost, hold_seconds). Consumed by
        # VideoThread next frame so AudioThread → tracker handoff happens on the
        # right thread.
        self.pending_voice_boost: tuple | None = None

    def read(self):
        """Return a snapshot of current settings (thread-safe)."""
        with QMutexLocker(self._lock):
            return {
                'camera_index': self.camera_index,
                'tracking_mode': self.tracking_mode,
                'shot_type': self.shot_type,
                'switch_mode': self.switch_mode,
                'switch_trigger': self.switch_trigger,
                'switch_interval': self.switch_interval,
                'crossfade_duration': self.crossfade_duration,
                'auto_follow_enabled': self.auto_follow_enabled,
                'foreground_exclusion_y': self.foreground_exclusion_y,
                'max_persons': self.max_persons,
                'diag_visible': self.diag_visible,
                'diag_overlays': self.diag_overlays,
                'music_mode': self.music_mode,
                'audio_state': {
                    'enabled': self.audio_enabled,
                    'music_mode': self.music_mode,
                    'music_score': self.audio_music_score,
                    'speech_score': self.audio_speech_score,
                    'speaker_name': self.audio_speaker_name,
                    'speaker_score': self.audio_speaker_score,
                    'speaker_expires_at': self.audio_speaker_expires_at,
                },
            }

    def set(self, **kwargs):
        """Thread-safe attribute update from the UI thread."""
        with QMutexLocker(self._lock):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def consume_voice_boost(self) -> tuple | None:
        with QMutexLocker(self._lock):
            val = self.pending_voice_boost
            self.pending_voice_boost = None
            return val

    def consume_manual_switch(self) -> str | None:
        with QMutexLocker(self._lock):
            val = self.manual_switch_id
            self.manual_switch_id = None
            return val

    def consume_camera_change(self) -> bool:
        with QMutexLocker(self._lock):
            val = self.camera_change_requested
            self.camera_change_requested = False
            return val

    # --- persistence -------------------------------------------------------

    def load(self, settings: QSettings):
        """Restore persisted fields, keeping each attribute's current type."""
        for key in self._PERSISTED:
            if not settings.contains(key):
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    val = settings.value(key, current, type=bool)
                elif isinstance(current, int):
                    val = settings.value(key, current, type=int)
                elif isinstance(current, float):
                    val = settings.value(key, current, type=float)
                else:
                    val = settings.value(key, current, type=str)
            except (TypeError, ValueError):
                continue
            setattr(self, key, val)
        # Sanity-clamp anything a stale settings file could have corrupted
        if self.shot_type not in ('full_body', 'waist_up', 'medium', 'close_up'):
            self.shot_type = config.SHOT_TYPE
        if self.tracking_mode not in ('primary', 'switcher'):
            self.tracking_mode = config.TRACKING_MODE
        if self.switch_mode not in ('cut', 'crossfade'):
            self.switch_mode = config.SWITCH_MODE
        if self.switch_trigger not in ('time', 'manual'):
            self.switch_trigger = 'time'
        self.foreground_exclusion_y = min(1.0, max(0.0, self.foreground_exclusion_y))
        self.max_persons = min(30, max(1, self.max_persons))
        self.switch_interval = min(60.0, max(0.5, self.switch_interval))
        self.crossfade_duration = min(5.0, max(0.1, self.crossfade_duration))

    def save(self, settings: QSettings):
        with QMutexLocker(self._lock):
            for key in self._PERSISTED:
                settings.setValue(key, getattr(self, key))


# ---------------------------------------------------------------------------
# Video processing thread
# ---------------------------------------------------------------------------

class VideoThread(QThread):
    """Captures, processes, and emits frames without blocking the UI."""

    frame_ready = pyqtSignal(QImage, dict)   # (processed output frame, metadata)
    diag_frame_ready = pyqtSignal(QImage)    # raw input frame with color-coded overlays
    camera_info = pyqtSignal(str)             # e.g. "1920x1080 @ 30fps"
    status = pyqtSignal(str)                  # human-readable pipeline state / errors
    model_ready = pyqtSignal()                # pose model loaded; pipeline about to start
    persons_updated = pyqtSignal(list)        # list of person IDs currently tracked

    def __init__(self, state: AppState, profile_store: ProfileStore | None = None,
                 parent=None):
        super().__init__(parent)
        self._state = state
        self._running = False

        # Pipeline components.  The detector is created in run() so the heavy
        # torch/ultralytics import and model load happen off the UI thread.
        self._cap = None
        self._detector = None
        self._tracker = PersonTracker()
        self._framing: FramingEngine | None = None
        self._smoother = PTZSmoother()
        self._switcher = VirtualSwitcher()
        self._frame_count = 0

        # Face recognition runs on its own worker thread (see face_worker.py);
        # results are polled and applied to the tracker by ID.
        self._profile_store = profile_store
        self._face_worker = (FaceRecognitionWorker(profile_store)
                             if profile_store is not None else None)
        self._face_error_reported = False

        # Latest raw camera frame — used by the People UI to capture reference
        # images.  cap.read() hands us a fresh buffer every frame and nothing in
        # the pipeline writes into it, so holding a reference costs nothing;
        # grab_latest_frame() copies on demand.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self._person_index_map: dict[str, int] = {}   # stable color index per person ID
        self._next_person_color_idx: int = 0
        self._last_person_ids: list[str] = []

        # Primary Focus state
        self._primary_id: str | None = None
        self._primary_pending_id: str | None = None
        self._primary_pretraveling: bool = False
        self._primary_pretravel_start: float = 0.0
        self._primary_fade_start: float | None = None
        self._primary_last_switch_time: float = time.monotonic()
        # Search state: entered when the primary is lost; camera holds position and
        # slowly zooms out until they reappear or someone else is adopted.
        self._searching: bool = False
        self._search_start: float = 0.0
        # Disabled-mode transition state
        self._disabled_transitioning: bool = False
        self._disabled_transition_start: float | None = None

        # UI backpressure + real frame-rate measurement
        self._inflight_lock = threading.Lock()
        self._frames_in_flight = 0
        self._frame_times: deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------

    def run(self):
        self._running = True

        self.status.emit("Loading pose model…")
        try:
            from pose_detector import PoseDetector
            self._detector = PoseDetector()
            self._detector.warmup()
        except Exception as e:   # missing package, bad weights file, etc.
            self.status.emit(f"Could not load pose model: {e}")
            return
        self.status.emit(f"Model ready on {self._detector.device}")
        self.model_ready.emit()

        settings = self._state.read()
        self._open_camera(settings['camera_index'])
        last_frame_time = time.monotonic()
        last_reopen_attempt = 0.0

        while self._running:
            settings = self._state.read()

            # Handle camera change
            if self._state.consume_camera_change():
                self._open_camera(settings['camera_index'])
                self._reset_tracking()
                last_frame_time = time.monotonic()

            now = time.monotonic()
            if self._cap is None:
                # Camera unavailable — retry periodically instead of giving up.
                if now - last_reopen_attempt >= _CAMERA_STALL_SECONDS:
                    last_reopen_attempt = now
                    self._open_camera(settings['camera_index'], quiet=True)
                    if self._cap is not None:
                        self._reset_tracking()
                        last_frame_time = now
                self.msleep(100)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if now - last_frame_time >= _CAMERA_STALL_SECONDS:
                    self.status.emit(
                        f"No signal from camera {settings['camera_index']} — reconnecting…")
                    self._release_camera()
                    last_reopen_attempt = now
                else:
                    self.msleep(10)
                continue
            last_frame_time = now
            self._frame_times.append(now)
            with self._latest_frame_lock:
                self._latest_frame = frame

            result = self._process_frame(frame, settings)
            self._frame_count += 1
            if result is None:
                continue
            qimg, meta = result

            # Drop frames the UI hasn't caught up with instead of queueing them.
            with self._inflight_lock:
                if self._frames_in_flight >= _MAX_FRAMES_IN_FLIGHT:
                    continue
                self._frames_in_flight += 1
            self.frame_ready.emit(qimg, meta)

    def frame_consumed(self):
        """Called by the UI thread after it has painted a frame."""
        with self._inflight_lock:
            self._frames_in_flight = max(0, self._frames_in_flight - 1)

    def grab_latest_frame(self) -> np.ndarray | None:
        """Return a copy of the most recent raw camera frame, or None."""
        with self._latest_frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def reindex_profiles(self):
        """Tell the face recognizer to rebuild its profile index on next pass."""
        if self._face_worker is not None:
            self._face_worker.mark_index_dirty()

    def stop(self):
        self._running = False
        if self._face_worker is not None:
            self._face_worker.stop()
        # cap.read() can block indefinitely on a stalled device; don't let that
        # hang application shutdown.
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
        self._release_camera()

    # ------------------------------------------------------------------
    # Camera open / reset
    # ------------------------------------------------------------------

    def _release_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _open_camera(self, index: int, quiet: bool = False):
        self._release_camera()
        self._cap = open_capture(index)
        if self._cap is None:
            self.camera_info.emit("not available")
            if not quiet:
                self.status.emit(f"Could not open camera {index}")
            return
        w, h, fps = describe_capture(self._cap)
        self._framing = FramingEngine(w, h)
        self.camera_info.emit(f"{w}×{h} @ {fps:.0f} fps")
        self.status.emit(f"Camera {index} ready")

    def _reset_tracking(self):
        """Forget everything about the previous scene (camera change / reconnect)."""
        self._tracker.reset()
        self._smoother.reset()
        self._switcher = VirtualSwitcher()
        self._primary_id = None
        self._clear_primary_transition()
        self._searching = False
        self._person_index_map.clear()
        self._next_person_color_idx = 0
        self._last_person_ids = []
        self._disabled_transitioning = False
        self._disabled_transition_start = None

    def _measured_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, settings: dict):
        # Detection only runs every DETECTION_INTERVAL frames; in between we keep
        # following the last known track positions (tracks expire on wall time).
        if self._frame_count % config.DETECTION_INTERVAL == 0:
            detections = self._detector.detect(frame)
            persons = self._tracker.update(
                detections, frame.shape,
                foreground_exclusion_y=settings.get('foreground_exclusion_y', 0.0),
                max_persons=settings.get('max_persons', config.MAX_PERSONS))
        else:
            persons = self._tracker.get_all()

        # Face recognition: apply results from the last pass, then hand this
        # frame to the worker if it's idle and there is anything to match.
        if self._face_worker is not None and persons:
            for m in self._face_worker.poll() or ():
                self._tracker.set_profile_match(
                    m['track_id'], m['profile_id'], m['name'], m['priority'], m['score'])
            if not self._face_worker.available:
                if not self._face_error_reported:
                    self._face_error_reported = True
                    self.status.emit(self._face_worker.error_message or
                                     "Face recognition unavailable")
            elif (self._frame_count % FACE_RECOGNITION_INTERVAL == 0
                    and not self._face_worker.busy
                    and self._face_worker.has_profiles()):
                self._face_worker.submit(frame, [(p.id, p.bbox) for p in persons])

        # Voice boost queued by the AudioThread, then expire boosts / stale faces.
        voice_boost = self._state.consume_voice_boost()
        if voice_boost is not None:
            profile_id, boost, hold = voice_boost
            self._tracker.apply_voice_boost(profile_id, boost, hold)
        self._tracker.expire_transients()

        # Sync switcher settings from UI state
        self._switcher.music_mode = bool(settings.get('music_mode', False))
        self._switcher.switch_mode = settings['switch_mode']
        self._switcher.trigger = settings['switch_trigger']
        self._switcher.interval = settings['switch_interval']
        self._switcher.crossfade_duration = settings['crossfade_duration']

        # Manual switch request from UI
        manual_id = self._state.consume_manual_switch()
        if manual_id:
            self._switcher.force_switch(manual_id)

        # Tell the UI which people exist — only when the set actually changes
        ids = [p.id for p in persons]
        if ids != self._last_person_ids:
            self._last_person_ids = ids
            self.persons_updated.emit(ids)

        # Assign stable color indices to new person IDs
        for p in persons:
            if p.id not in self._person_index_map:
                self._person_index_map[p.id] = self._next_person_color_idx
                self._next_person_color_idx += 1

        mode = settings['tracking_mode']
        shot_type = settings['shot_type']
        auto_enabled = settings.get('auto_follow_enabled', True)

        if not auto_enabled:
            output_frame = self._render_disabled(frame, settings)
            active_id = 'disabled'
            mode = 'disabled'
        else:
            # Clear disabled transition state so the next disable starts fresh
            self._disabled_transitioning = False
            self._disabled_transition_start = None

            if mode == 'primary' or not persons:
                output_frame = self._render_primary(frame, persons, shot_type, settings)
                active_id = self._primary_id or 'none'
            else:
                output_frame, active_id = self._render_switcher(frame, persons, shot_type)

        # Diagnostics preview — only built when someone is looking at it.
        if settings.get('diag_visible'):
            diag_frame = self._annotate_diag_frame(
                frame, persons, active_id, self._person_index_map,
                settings.get('foreground_exclusion_y', 0.0),
                overlays=settings.get('diag_overlays', True))
            self.diag_frame_ready.emit(_bgr_to_qimage(diag_frame))

        # Build per-person diagnostics list for the diagnostics panel
        persons_diag = []
        for p in persons:
            cx = (p.bbox[0] + p.bbox[2]) / 2.0
            cy = (p.bbox[1] + p.bbox[3]) / 2.0
            smoother_state = self._smoother.get_state(p.id) or {}
            persons_diag.append({
                'id': p.id,
                'fg_score': p.foreground_score,
                'activity': p.activity_score,
                'bbox': p.bbox,
                'center': (cx, cy),
                'smoother_x': smoother_state.get('x'),
                'smoother_zoom': smoother_state.get('zoom'),
                'frames_unseen': p.frames_unseen,
                'profile_id': p.profile_id,
                'profile_name': p.profile_name,
                'profile_priority': p.profile_priority,
                'profile_score': p.profile_score,
                'voice_boost': p.voice_boost,
                'effective_priority': p.effective_priority,
            })

        # Audio snapshot for the diagnostics panel; expire the sticky speaker
        # name once its hold window passes so the panel doesn't lie.
        audio_state = dict(settings.get('audio_state') or {})
        if time.monotonic() >= audio_state.get('speaker_expires_at', 0.0):
            audio_state['speaker_name'] = None
            audio_state['speaker_score'] = 0.0

        music_mode = bool(settings.get('music_mode', False))
        meta = {
            'fps': self._measured_fps(),
            'n_persons': len(persons),
            'active_id': active_id,
            'persons': persons_diag,
            'primary_id': self._primary_id,
            'pending_id': self._primary_pending_id,
            'pretraveling': self._primary_pretraveling,
            'dwell_elapsed': time.monotonic() - self._primary_last_switch_time,
            'dwell_threshold': self._primary_dwell(music_mode),
            'mode': mode,
            'smoother_primary': self._smoother.get_state('primary'),
            'searching': self._searching,
            'person_index_map': dict(self._person_index_map),
            'audio_state': audio_state,
            'face_ms': (self._face_worker.last_duration * 1000.0
                        if self._face_worker is not None else 0.0),
        }
        return _bgr_to_qimage(output_frame), meta

    # ------------------------------------------------------------------
    # Diagnostic overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_diag_frame(frame: np.ndarray, persons, primary_id: str,
                              person_index_map: dict,
                              foreground_exclusion_y: float = 0.0,
                              overlays: bool = True) -> np.ndarray:
        """Render color-coded skeleton overlays on a downscaled copy of the input.

        Each person gets a unique hue (golden-angle spacing).  The body
        silhouette — convex hull of all visible keypoints — is filled with a
        semi-transparent wash of that color.  Skeleton lines and joint dots are
        drawn on top in full color.  The primary person's bbox is outlined with
        a brighter border and an ID label.

        This output is only ever sent to the diagnostics preview; it never
        touches the main output pipeline.
        """
        src_h, src_w = frame.shape[:2]
        if src_w > _DIAG_MAX_WIDTH:
            s = _DIAG_MAX_WIDTH / src_w
            out = cv2.resize(frame, (_DIAG_MAX_WIDTH, max(1, int(src_h * s))),
                             interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
            out = frame.copy()
        h, w = out.shape[:2]

        if not overlays:
            return out

        overlay = out.copy()

        for person in persons:
            idx = person_index_map.get(person.id, 0)
            color = _person_color(idx)
            kps = person.keypoints  # (17, 4) — [x_norm, y_norm, 0, conf]

            pts: dict[int, tuple[int, int]] = {}
            for ki, kp in enumerate(kps):
                if kp[3] > config.CONFIDENCE_THRESHOLD:
                    pts[ki] = (int(kp[0] * w), int(kp[1] * h))

            # ── Filled silhouette: convex hull of visible keypoints ──────
            if len(pts) >= 3:
                hull_pts = np.array(list(pts.values()), dtype=np.int32)
                hull = cv2.convexHull(hull_pts)
                cv2.fillConvexPoly(overlay, hull, color)

            # ── Skeleton lines ───────────────────────────────────────────
            for a, b in _SKELETON:
                if a in pts and b in pts:
                    cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)

            # ── Joint dots ───────────────────────────────────────────────
            for pt in pts.values():
                cv2.circle(out, pt, 4, color, -1, cv2.LINE_AA)
                cv2.circle(out, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Bbox outline ─────────────────────────────────────────────
            bx1, by1, bx2, by2 = (int(v * s) for v in person.bbox)
            is_primary = person.id == primary_id
            border_thickness = 3 if is_primary else 1
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, border_thickness, cv2.LINE_AA)

            # ── Primary indicator: downward arrow above bbox + centroid halo
            if is_primary:
                cx_p = (bx1 + bx2) // 2
                arrow_tip_y = by1 - 6
                arrow_base_y = arrow_tip_y - 18
                arrow_half_w = 10
                tri = np.array([
                    [cx_p,                arrow_tip_y],
                    [cx_p - arrow_half_w, arrow_base_y],
                    [cx_p + arrow_half_w, arrow_base_y],
                ], dtype=np.int32)
                cv2.polylines(out, [tri], True, color, 2, cv2.LINE_AA)
                cv2.fillConvexPoly(out, tri, color)
                cy_p = (by1 + by2) // 2
                cv2.circle(out, (cx_p, cy_p), 10, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(out, (cx_p, cy_p), 10, color,           2, cv2.LINE_AA)

            # ── ID + score label ─────────────────────────────────────────
            # Prefer the matched profile name over the generic personN ID;
            # ★N shows static priority, ● shows an active voice boost.
            id_text = person.profile_name or person.id
            if person.profile_name and person.profile_priority > 0:
                id_text = f"★{person.profile_priority} {id_text}"
            if person.voice_boost > 0:
                id_text = f"● {id_text}"
            label = f"{'▶ ' if is_primary else ''}{id_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            lscale, lthick = 0.5, 1
            (tw, th), _ = cv2.getTextSize(label, font, lscale, lthick)
            lx, ly = bx1, max(by1 - (30 if is_primary else 4), th + 2)
            cv2.rectangle(out, (lx, ly - th - 2), (lx + tw + 4, ly + 2), color, -1)
            cv2.putText(out, label, (lx + 2, ly), font, lscale,
                        (0, 0, 0), lthick + 1, cv2.LINE_AA)
            cv2.putText(out, label, (lx + 2, ly), font, lscale,
                        (255, 255, 255), lthick, cv2.LINE_AA)

        # Blend silhouette overlay at 35% opacity
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # ── Audience exclusion zone ───────────────────────────────────────
        if foreground_exclusion_y > 0.0:
            excl_y = int(h * (1.0 - foreground_exclusion_y))
            yellow = (0, 220, 220)  # BGR yellow

            # Semi-transparent tint + diagonal hatching, drawn only on the band
            # itself rather than blending the whole frame.
            band = out[excl_y:h, :]
            if band.size:
                tinted = band.copy()
                tinted[:] = yellow
                cv2.addWeighted(tinted, 0.15, band, 0.85, 0, band)
                stripes = band.copy()
                band_h = h - excl_y
                for x_start in range(-band_h, w, 18):
                    cv2.line(stripes, (x_start, 0), (x_start + band_h, band_h),
                             yellow, 1, cv2.LINE_AA)
                cv2.addWeighted(stripes, 0.45, band, 0.55, 0, band)
            cv2.line(out, (0, excl_y), (w, excl_y), yellow, 2, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _follow(self, key: str, person, shot_type: str) -> tuple[float, float, float]:
        """Advance smoother `key` toward `person`'s framing target; return (x, y, zoom)."""
        tx, ty, tz = self._framing.calculate_target(person, shot_type)
        return self._smoother.update(
            key, tx, ty, tz,
            person_center_x=_bbox_center_x(person),
            crop_width=config.OUTPUT_WIDTH / tz,
        )

    def _held(self, key: str) -> tuple[float, float, float]:
        s = self._smoother.get_state(key)
        return s['x'], s['y'], s['zoom']

    def _render_disabled(self, frame, settings: dict):
        """Auto-follow off: ease out to the full-frame wide shot and stay there."""
        sw_mode = settings['switch_mode']
        xfade_dur = settings['crossfade_duration']
        tx, ty, tz = self._framing.default_target()

        # On the first frame of a disable, seed the from-smoother with the
        # current live camera position so the transition starts from there.
        if not self._disabled_transitioning:
            self._disabled_transitioning = True
            live = self._smoother.get_state('primary')
            if live:
                self._smoother.seed('__disabled_from__', live['x'], live['y'], live['zoom'])
            else:
                self._smoother.seed('__disabled_from__', tx, ty, tz)
            self._smoother.seed('__passthrough__', tx, ty, tz)
            self._disabled_transition_start = (
                time.monotonic() if sw_mode == 'crossfade' else None)

        px, py, pz = self._smoother.update('__passthrough__', tx, ty, tz)
        frame_to = self._framing.apply_crop(frame, px, py, pz)

        if self._disabled_transition_start is not None:
            t = min(1.0, (time.monotonic() - self._disabled_transition_start)
                    / max(xfade_dur, 0.001))
            if t < 1.0:
                fx, fy, fz = self._smoother.update('__disabled_from__', tx, ty, tz)
                frame_from = self._framing.apply_crop(frame, fx, fy, fz)
                return cv2.addWeighted(frame_from, 1.0 - t, frame_to, t, 0)
            self._disabled_transition_start = None
        return frame_to

    # ------------------------------------------------------------------
    # Primary Focus mode
    # ------------------------------------------------------------------

    @staticmethod
    def _primary_dwell(music_mode: bool) -> float:
        return _MUSIC_MODE_DWELL_SECONDS if music_mode else config.PRIMARY_DWELL_SECONDS

    @staticmethod
    def _pick_subject(persons, music_mode: bool):
        """Best person to adopt when we have no primary.

        Foreground area is the base signal, with a mild bias toward recognized
        high-priority people (and anyone whose voice is currently recognized).
        The bias is capped at 1.75× so an obviously closer unmatched person
        still wins over a low-priority profile at the back.  In music mode the
        most active performer wins outright.
        """
        if music_mode:
            return max(persons, key=lambda p: (p.activity_score, p.foreground_score))
        return max(persons,
                   key=lambda p: p.foreground_score * (1.0 + p.effective_priority / 20.0))

    def _clear_primary_transition(self):
        self._primary_pending_id = None
        self._primary_pretraveling = False
        self._primary_fade_start = None

    def _begin_primary_transition(self, target_id: str, now: float):
        self._primary_pending_id = target_id
        self._primary_pretraveling = True
        self._primary_pretravel_start = now
        self._primary_fade_start = None

    def _commit_primary(self, now: float):
        """Make the pending subject the primary, carrying its camera position over.

        Without copying the smoother state the 'primary' camera would still be
        parked on the old subject after the cut/crossfade and visibly pan across.
        """
        pending = self._primary_pending_id
        self._smoother.copy_state(pending, 'primary', remove_src=True)
        self._primary_id = pending
        self._primary_last_switch_time = now
        self._searching = False
        self._clear_primary_transition()

    def _primary_shot(self, frame, persons, shot_type):
        """Crop for the current primary: follow them if visible, else hold position."""
        person = next((p for p in persons if p.id == self._primary_id), None)
        if person is not None:
            x, y, z = self._follow('primary', person, shot_type)
        else:
            x, y, z = self._held('primary')
        return self._framing.apply_crop(frame, x, y, z)

    def _search_zoom_out(self):
        """Nudge the held camera wider, keeping the crop centered on the same spot."""
        state = self._smoother.get_state('primary')
        old_zoom = state['zoom']
        new_zoom = max(self._framing.min_zoom, old_zoom - _SEARCH_ZOOM_OUT_RATE)
        if new_zoom == old_zoom:
            return
        new_w = config.OUTPUT_WIDTH / new_zoom
        new_h = config.OUTPUT_HEIGHT / new_zoom
        x = state['x'] - (new_w - config.OUTPUT_WIDTH / old_zoom) / 2.0
        y = state['y'] - (new_h - config.OUTPUT_HEIGHT / old_zoom) / 2.0
        # Keep the stored position inside the frame so the next pan starts moving
        # immediately instead of first "travelling" through clamped-off space.
        state['x'] = min(max(0.0, x), max(0.0, self._framing.input_width - new_w))
        state['y'] = min(max(0.0, y), max(0.0, self._framing.input_height - new_h))
        state['zoom'] = new_zoom

    def _render_primary(self, frame, persons, shot_type, settings: dict):
        """Track the nearest (largest bbox) person as primary.

        Primary persistence:
          - Stays on the current subject while they are visible.
          - Switches to a closer subject when their foreground_score is ≥1.5×
            the current one, or when they are ≥2× more active — but only after
            PRIMARY_DWELL_SECONDS on the current subject.
          - All switches use the same pretravel → cut/crossfade pipeline as
            VirtualSwitcher, and the new subject's camera position is carried
            over so nothing pans after the transition.

        Losing the subject:
          - The camera holds its position and slowly zooms out (search).
          - If the subject is back within PRIMARY_REACQUIRE_DELAY nothing changes;
            otherwise the best remaining person is adopted via a transition.
        """
        sw_mode = settings.get('switch_mode', 'crossfade')
        xfade_dur = settings.get('crossfade_duration', 1.0)
        music_mode = bool(settings.get('music_mode', False))
        now = time.monotonic()

        # Start on the wide shot so the first acquisition is a push-in, not a snap.
        if self._smoother.get_state('primary') is None:
            self._smoother.seed('primary', *self._framing.default_target())

        by_id = {p.id: p for p in persons}
        primary_present = self._primary_id in by_id

        # ── Search / reacquire ───────────────────────────────────────────
        if not primary_present and self._primary_pending_id is None:
            if not self._searching:
                self._searching = True
                self._search_start = now
            candidate = self._pick_subject(persons, music_mode) if persons else None
            waited = now - self._search_start
            if candidate is not None and (
                    self._primary_id is None or waited >= config.PRIMARY_REACQUIRE_DELAY):
                self._begin_primary_transition(candidate.id, now)
            else:
                self._search_zoom_out()
                return self._primary_shot(frame, persons, shot_type)
        elif primary_present and self._searching:
            self._searching = False

        # ── Candidate selection (steady state only) ──────────────────────
        # persons[] is sorted foreground_score desc (nearest = persons[0]).
        # Raw bbox area decides who is "nearest" — profile priority never
        # changes the pick, only how reluctant we are to switch, so unmatched
        # guests stay fully eligible whenever they're clearly closer.  Both
        # scores are EMA-smoothed in the tracker so single-frame noise can't
        # trigger a switch, and the dwell time prevents ping-ponging.
        others = [p for p in persons if p.id != self._primary_id]
        if (primary_present and self._primary_pending_id is None and others
                and now - self._primary_last_switch_time >= self._primary_dwell(music_mode)):
            current_p = by_id[self._primary_id]
            if music_mode:
                # The most active performer (singing, soloing, leading) is the
                # right primary during music; fall back to area when nobody moves.
                nearest = max(others, key=lambda p: (p.activity_score, p.foreground_score))
            else:
                nearest = max(others, key=lambda p: p.foreground_score)
            fg_ratio = nearest.foreground_score / max(current_p.foreground_score, 1e-6)
            cand_act = nearest.activity_score
            curr_act = current_p.activity_score
            # Priority modulates the proximity threshold:
            #   diff +10 → 1.0 (any closer wins), 0 → 1.5, -10 → 2.25.
            priority_diff = nearest.effective_priority - current_p.effective_priority
            switch_threshold = max(1.0, 1.5 - priority_diff * 0.075)
            fg_wins = fg_ratio >= switch_threshold
            # Strictly higher-priority candidate who is about as close — switch now.
            priority_override = priority_diff > 0 and fg_ratio >= 0.9
            # Activity switch: candidate must clear a noise floor AND be 2× more active.
            activity_wins = cand_act >= 5.0 and cand_act > curr_act * 2.0
            if priority_override or fg_wins or activity_wins:
                self._begin_primary_transition(nearest.id, now)

        # ── Transition: pretravel → cut / crossfade ──────────────────────
        if self._primary_pending_id is not None:
            pending = by_id.get(self._primary_pending_id)
            if pending is None:
                # Pending subject vanished before we got there — abort; if the
                # primary is gone too, the next frame re-enters search.
                self._smoother.reset(self._primary_pending_id)
                self._clear_primary_transition()
                return self._primary_shot(frame, persons, shot_type)

            px, py, pz = self._follow(self._primary_pending_id, pending, shot_type)
            frame_active = self._primary_shot(frame, persons, shot_type)

            if self._primary_pretraveling:
                if now - self._primary_pretravel_start >= _PRETRAVEL_DURATION:
                    self._primary_pretraveling = False
                    if sw_mode == 'cut':
                        self._commit_primary(now)
                        return self._framing.apply_crop(frame, px, py, pz)
                    self._primary_fade_start = now
                return frame_active

            t = min(1.0, (now - self._primary_fade_start) / max(xfade_dur, 0.001))
            frame_pending = self._framing.apply_crop(frame, px, py, pz)
            if t >= 1.0:
                self._commit_primary(now)
                return frame_pending
            return cv2.addWeighted(frame_active, 1.0 - t, frame_pending, t, 0)

        # ── Steady state: follow primary ─────────────────────────────────
        return self._primary_shot(frame, persons, shot_type)

    # ------------------------------------------------------------------
    # Virtual switcher mode
    # ------------------------------------------------------------------

    def _render_switcher(self, frame, persons, shot_type):
        """Virtual switcher: cut or crossfade between tracked persons.

        When a switch is queued the pending person's smoother is advanced every
        frame (pretravel phase) so the virtual camera has already arrived at the
        new subject's position by the time the cut or crossfade fires.
        """
        by_id = {p.id: p for p in persons}

        # Give the switcher the current crop width so it can gate switches by displacement.
        current_person = by_id.get(self._switcher.active_id)
        if current_person is not None:
            _, _, cur_zoom = self._framing.calculate_target(current_person, shot_type)
            self._switcher.current_crop_width = config.OUTPUT_WIDTH / cur_zoom

        active_id = self._switcher.decide(persons)
        active_person = by_id.get(active_id, persons[0])
        ax, ay, az = self._follow(active_id, active_person, shot_type)
        frame_active = self._framing.apply_crop(frame, ax, ay, az)

        pending_person = by_id.get(self._switcher._pending_id)
        if pending_person is None:
            return frame_active, active_id

        px, py, pz = self._follow(pending_person.id, pending_person, shot_type)

        # Pretravel: advance the pending smoother off-screen so it's settled
        # before the transition becomes visible; keep showing the active frame.
        if self._switcher.is_pretraveling:
            return frame_active, active_id

        # Crossfade: blend settled active and pending frames
        if self._switcher.is_transitioning:
            frame_pending = self._framing.apply_crop(frame, px, py, pz)
            return self._switcher.blend(frame_active, frame_pending), active_id

        return frame_active, active_id


# ---------------------------------------------------------------------------
# Diagnostics panel
# ---------------------------------------------------------------------------

_LOG_MAX_LINES = 200        # keep last N switch events in the log
_DIAG_TABLE_INTERVAL = 0.1  # seconds between table/label refreshes (log is per-frame)


class DiagnosticsWindow(QMainWindow):
    """Floating panel showing live tracking state and a switch-event log.

    Updated every frame via update_diagnostics(); never touches the video thread.
    """

    overlays_changed = pyqtSignal(bool)     # emitted when "Show Overlays" is toggled
    visibility_changed = pyqtSignal(bool)   # emitted on show / hide / close

    def __init__(self, show_overlays: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofollow Diagnostics")
        self.setMinimumSize(640, 640)
        self._last_active_id: str | None = None   # for detecting new switch events
        self._last_table_update: float = 0.0

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(6)

        # ── Show Overlays checkbox ───────────────────────────────────────
        overlay_row = QHBoxLayout()
        self._overlay_checkbox = QCheckBox("Show Overlays")
        self._overlay_checkbox.setChecked(show_overlays)
        self._overlay_checkbox.setToolTip(
            "Draw skeletons, bounding boxes and the exclusion zone on the camera preview.")
        self._overlay_checkbox.stateChanged.connect(
            lambda state: self.overlays_changed.emit(bool(state))
        )
        overlay_row.addWidget(self._overlay_checkbox)
        overlay_row.addStretch()
        layout.addLayout(overlay_row)

        # ── Video preview (raw input with color-coded overlays) ──────────
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(320, 180)
        self._video_label.setStyleSheet("background: #111; border: 1px solid #333;")
        self._video_label.setSizePolicy(
            self._video_label.sizePolicy().Expanding,
            self._video_label.sizePolicy().Expanding,
        )
        layout.addWidget(self._video_label, stretch=3)

        # ── State summary row ────────────────────────────────────────────
        state_box = QGroupBox("Tracking State")
        state_grid = QHBoxLayout(state_box)

        self._lbl_mode     = self._make_field("Mode", state_grid)
        self._lbl_active   = self._make_field("Active", state_grid)
        self._lbl_pending  = self._make_field("Pending", state_grid)
        self._lbl_phase    = self._make_field("Phase", state_grid)
        self._lbl_dwell    = self._make_field("Dwell", state_grid)
        self._lbl_fps      = self._make_field("FPS", state_grid)
        self._lbl_face_ms  = self._make_field("Face", state_grid)
        layout.addWidget(state_box)

        # ── Audio state row ──────────────────────────────────────────────
        audio_box = QGroupBox("Audio")
        audio_grid = QHBoxLayout(audio_box)
        self._lbl_audio_mode    = self._make_field("Mode", audio_grid)
        self._lbl_audio_speaker = self._make_field("Speaker", audio_grid)
        self._lbl_audio_scores  = self._make_field("Music/Speech", audio_grid)
        layout.addWidget(audio_box)

        # ── Per-person table ─────────────────────────────────────────────
        persons_box = QGroupBox("Tracked Persons")
        persons_layout = QVBoxLayout(persons_box)
        self._table = QTableWidget(0, 11)
        self._table.setHorizontalHeaderLabels(
            ["", "ID / Profile", "Face", "Voice",
             "FG(sm)", "Act(sm)", "FG Ratio", "Act Ratio",
             "Unseen", "Center X,Y", "Zoom(sm)"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setFixedHeight(150)
        persons_layout.addWidget(self._table)
        layout.addWidget(persons_box)

        # ── Switch event log ─────────────────────────────────────────────
        log_box = QGroupBox("Switch Event Log")
        log_layout = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 10))
        self._log.setStyleSheet("background:#1e1e1e; color:#d4d4d4;")
        log_layout.addWidget(self._log)

        btn_clear = QPushButton("Clear Log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._log.clear)
        log_layout.addWidget(btn_clear, alignment=Qt.AlignRight)
        layout.addWidget(log_box)

    # ------------------------------------------------------------------

    @staticmethod
    def _make_field(label: str, row: QHBoxLayout) -> QLabel:
        """Add a label+value pair to a horizontal layout; return the value label."""
        lbl = QLabel(f"{label}:")
        lbl.setStyleSheet("font-weight: bold;")
        val = QLabel("—")
        val.setMinimumWidth(70)
        row.addWidget(lbl)
        row.addWidget(val)
        return val

    # ------------------------------------------------------------------

    def update_video(self, qimg: QImage):
        """Display a new annotated frame in the diagnostics video preview."""
        if not self.isVisible():
            return
        pix = QPixmap.fromImage(qimg)
        self._video_label.setPixmap(
            pix.scaled(self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ------------------------------------------------------------------

    def update_diagnostics(self, meta: dict):
        """Called from the UI thread every frame with the metadata dict.

        Always processes the switch-event log (so history is captured even
        when the window is hidden). The table and labels are refreshed at most
        every _DIAG_TABLE_INTERVAL seconds, and only while visible.
        """
        mode       = meta.get('mode', '?')
        active_id  = meta.get('active_id', '?')
        primary_id = meta.get('primary_id') or active_id
        pending_id = meta.get('pending_id')
        pretrav    = meta.get('pretraveling', False)
        dwell      = meta.get('dwell_elapsed', 0.0)
        fps        = meta.get('fps', 0.0)
        persons    = meta.get('persons', [])

        # Phase string
        if mode == 'disabled':
            phase = 'disabled'
        elif meta.get('searching'):
            phase = 'searching'
        elif pretrav:
            phase = 'pretravel'
        elif pending_id:
            phase = 'crossfade'
        else:
            phase = 'steady'

        # ── Per-person scores (needed for log too) ───────────────────────
        current_p = next((p for p in persons if p['id'] == primary_id), None)
        curr_fg  = current_p['fg_score'] if current_p else 1e-6
        curr_act = current_p['activity'] if current_p else 1e-6

        dwell_threshold = meta.get('dwell_threshold', 3.0)
        gate_open = dwell >= dwell_threshold

        now = time.monotonic()
        if self.isVisible() and now - self._last_table_update >= _DIAG_TABLE_INTERVAL:
            self._last_table_update = now
            self._lbl_mode.setText(mode)
            self._lbl_active.setText(primary_id or '—')
            self._lbl_pending.setText(pending_id or '—')
            self._lbl_phase.setText(phase)
            self._lbl_dwell.setText(f"{dwell:.2f}/{dwell_threshold:.1f}s")
            self._lbl_fps.setText(f"{fps:.1f}")
            face_ms = meta.get('face_ms', 0.0)
            self._lbl_face_ms.setText(f"{face_ms:.0f} ms" if face_ms > 0 else "—")

            audio_state = meta.get('audio_state') or {}
            if audio_state.get('enabled'):
                music_on = bool(audio_state.get('music_mode'))
                self._lbl_audio_mode.setText("MUSIC" if music_on else "Speech")
                self._lbl_audio_mode.setStyleSheet(
                    "color: #c678dd; font-weight: bold;" if music_on
                    else "color: #98c379; font-weight: bold;")
            else:
                self._lbl_audio_mode.setText("off")
                self._lbl_audio_mode.setStyleSheet("color: gray;")
            spk_name = audio_state.get('speaker_name')
            if spk_name:
                self._lbl_audio_speaker.setText(
                    f"● {spk_name} ({audio_state.get('speaker_score', 0.0):.2f})")
                self._lbl_audio_speaker.setStyleSheet("color: #61afef; font-weight: bold;")
            else:
                self._lbl_audio_speaker.setText("—")
                self._lbl_audio_speaker.setStyleSheet("color: gray;")
            self._lbl_audio_scores.setText(
                f"m={audio_state.get('music_score', 0.0):.2f} / "
                f"s={audio_state.get('speech_score', 0.0):.2f}")

            # Red = gate closed (can't switch yet), green = gate open
            if not gate_open:
                self._lbl_dwell.setStyleSheet("color: #e06c75; font-weight: bold;")
            else:
                self._lbl_dwell.setStyleSheet("color: #98c379; font-weight: bold;")

            self._refresh_table(persons, meta.get('person_index_map', {}),
                                primary_id, pending_id, curr_fg, curr_act, gate_open)

        # ── Switch event log: append a line when active_id changes ───────
        if active_id != self._last_active_id and active_id not in ('none', 'disabled', None):
            ts = time.strftime("%H:%M:%S")
            reason = ''
            if persons:
                nearest = persons[0]
                fg_r  = nearest['fg_score'] / max(curr_fg,  1e-6)
                act_r = nearest['activity'] / max(curr_act, 1e-6)
                # Show which threshold was met (or neither — means forced recovery)
                why = []
                if fg_r >= 1.5:
                    why.append(f"fg={fg_r:.2f}≥1.5")
                if nearest['activity'] >= 5.0 and act_r >= 2.0:
                    why.append(f"act={act_r:.2f}≥2.0")
                if not why:
                    why.append('reacquired')
                reason = f"  [{', '.join(why)}]"
            line = (f"[{ts}]  {self._last_active_id or '—'} → {active_id}"
                    f"  dwell={dwell:.2f}s{reason}")
            self._log.append(line)
            # Trim to max lines
            doc = self._log.document()
            while doc.blockCount() > _LOG_MAX_LINES:
                cursor = self._log.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()   # remove the trailing newline
            self._log.ensureCursorVisible()
        self._last_active_id = active_id

    def _refresh_table(self, persons, person_index_map, primary_id, pending_id,
                       curr_fg, curr_act, gate_open):
        self._table.setRowCount(len(persons))
        for row, p in enumerate(persons):
            pid       = p['id']
            fg        = p['fg_score']
            act       = p['activity']
            fg_ratio  = fg  / max(curr_fg,  1e-6)
            act_ratio = act / max(curr_act, 1e-6)
            cx, cy    = p.get('center', (0, 0))
            unseen    = p.get('frames_unseen', 0)
            sz        = p.get('smoother_zoom')

            is_active  = pid == primary_id
            is_pending = pid == pending_id

            # Flag cells that would trigger a switch (for easy reading)
            fg_trigger  = fg_ratio >= 1.5 and gate_open and not is_active
            act_trigger = act >= 5.0 and act_ratio >= 2.0 and gate_open and not is_active

            # Person's skeleton color (matches the video overlay).
            # _person_color uses OpenCV HSV hue 0-179 with step 137.5,
            # Qt fromHsv uses 0-359, so multiply the same step by 2.
            p_idx = person_index_map.get(pid, 0)
            skel_qcolor = QColor.fromHsv(int((p_idx * 275) % 360), 200, 220)
            swatch_bg = QColor(
                skel_qcolor.red()   // 4,
                skel_qcolor.green() // 4,
                skel_qcolor.blue()  // 4,
            )

            # Column 0: color swatch (solid person color, narrow)
            swatch = QTableWidgetItem()
            swatch.setBackground(skel_qcolor)
            if is_active:
                swatch.setText("▶")
                swatch.setForeground(QColor(0, 0, 0))
            swatch.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, swatch)

            # Face column: profile name + cosine score; voice column: active boost
            profile_name = p.get('profile_name')
            profile_priority = p.get('profile_priority', 0)
            if profile_name:
                face_cell = f"{profile_name} ★{profile_priority} ({p.get('profile_score', 0.0):.2f})"
                id_cell = f"{pid} → {profile_name}"
            else:
                face_cell = "—"
                id_cell = pid
            voice_boost = p.get('voice_boost', 0.0)
            if voice_boost > 0:
                voice_cell = f"● +{voice_boost:.0f} (eff {p.get('effective_priority', profile_priority):.0f})"
            else:
                voice_cell = "—"

            cells = [
                id_cell,
                face_cell,
                voice_cell,
                f"{fg:.4f}",
                f"{act:.2f}",
                f"{'!' if fg_trigger  else ''}{fg_ratio:.2f}",
                f"{'!' if act_trigger else ''}{act_ratio:.2f}",
                str(unseen),
                f"{cx:.0f},{cy:.0f}",
                f"{sz:.2f}" if sz is not None else "—",
            ]
            for col, text in enumerate(cells, start=1):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if is_active:
                    item.setBackground(QColor(40, 80, 40))
                elif is_pending:
                    item.setBackground(QColor(80, 60, 20))
                elif fg_trigger or act_trigger:
                    item.setBackground(QColor(80, 40, 40))
                else:
                    item.setBackground(swatch_bg)
                item.setForeground(skel_qcolor)
                self._table.setItem(row, col, item)

        # Fix swatch column to a narrow fixed width
        self._table.setColumnWidth(0, 22)

    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.hide()


# ---------------------------------------------------------------------------
# Fullscreen output window
# ---------------------------------------------------------------------------

class OutputWindow(QMainWindow):
    """Borderless fullscreen window displaying the processed video output."""

    visibility_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofollow Output")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setCursor(Qt.BlankCursor)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background: black;")
        self.setCentralWidget(self._label)

    def update_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pix.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.hide()

    def mouseDoubleClickEvent(self, event):
        self.hide()


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlWindow(QMainWindow):
    """Main control panel window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autofollow")
        self.setMinimumWidth(340)

        self._settings = QSettings("Autofollow", "Autofollow")
        self._state = AppState()
        self._state.load(self._settings)

        self._profile_store = ProfileStore()
        self._people_win = None  # lazy-created when user opens "Manage People"

        self._output_win = OutputWindow()
        self._output_win.visibility_changed.connect(self._on_output_visibility)
        self._diag_win = DiagnosticsWindow(show_overlays=self._state.diag_overlays)
        self._diag_win.overlays_changed.connect(self._on_diagnostics_changed)
        self._diag_win.visibility_changed.connect(self._on_diag_visibility)

        self._video_thread = VideoThread(self._state, profile_store=self._profile_store)
        self._video_thread.frame_ready.connect(self._on_frame)
        self._video_thread.diag_frame_ready.connect(self._diag_win.update_video)
        self._video_thread.camera_info.connect(self._on_camera_info)
        self._video_thread.status.connect(self._on_status)
        self._video_thread.persons_updated.connect(self._on_persons_updated)

        # Audio analysis (capture + VAD + speaker recognition + music classification).
        # Starts idle; capture only begins when the user enables it.
        self._audio_thread = AudioThread(self._profile_store)
        self._audio_thread.audio_state_changed.connect(self._on_audio_state_changed)
        self._audio_thread.speaker_detected.connect(self._on_speaker_detected)
        self._audio_thread.error.connect(self._on_audio_error)
        self._audio_thread.status.connect(self._on_status)
        self._audio_status_text = "Audio: off"

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(320, 180)
        self._preview_label.setStyleSheet("background: black;")

        self._last_status: str = ""
        self._build_ui()
        self._video_thread.start()
        self._audio_thread.start()
        # If audio analysis was on last time, bring it back only once the pose
        # model is loaded: importing TensorFlow / loading YAMNet at the same
        # time starves the video thread and delays the first frame by seconds.
        if self._state.audio_enabled:
            self._video_thread.model_ready.connect(
                lambda: self._apply_audio_settings(enable=self._state.audio_enabled))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)

        layout.addWidget(self._build_camera_section())
        layout.addWidget(self._build_audio_section())
        layout.addWidget(self._build_shot_section())
        layout.addWidget(self._build_switcher_section())
        layout.addWidget(self._preview_label)
        layout.addWidget(self._build_output_section())
        layout.addWidget(self._build_status_bar())

    # --- Camera ---

    def _build_camera_section(self):
        box = QGroupBox("Camera")
        row = QHBoxLayout(box)
        self._cam_combo = QComboBox()
        self._refresh_cameras(initial=True)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setToolTip("Rescan for connected cameras")
        btn_refresh.clicked.connect(self._refresh_cameras)
        self._cam_info_label = QLabel("")
        self._cam_info_label.setStyleSheet("color: gray; font-size: 10px;")
        row.addWidget(self._cam_combo)
        row.addWidget(btn_refresh)
        row.addWidget(self._cam_info_label)
        return box

    def _refresh_cameras(self, initial: bool = False):
        # Never re-probe the camera the video thread is streaming from — opening
        # a device twice can stall the capture.  It is listed without probing.
        active = set() if initial else {self._state.camera_index}
        cameras = scan_cameras(skip=active)
        if initial and self._state.camera_index not in cameras:
            # Remembered camera is gone (unplugged) — fall back to the first one.
            if cameras:
                self._state.camera_index = cameras[0]

        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for idx in cameras:
            self._cam_combo.addItem(f"Camera {idx}", idx)
        if not cameras:
            self._cam_combo.addItem("No cameras found", None)
        for i in range(self._cam_combo.count()):
            if self._cam_combo.itemData(i) == self._state.camera_index:
                self._cam_combo.setCurrentIndex(i)
                break
        self._cam_combo.blockSignals(False)

    # --- Audio ---

    def _build_audio_section(self):
        """Audio input controls — Enabled toggle + input device picker.

        The device is remembered by *name* (USB indices shuffle between boots)
        and audio analysis is re-enabled on launch if it was on last time.
        """
        box = QGroupBox("Audio (speaker + music detection)")
        row = QHBoxLayout(box)
        self._audio_enable_cb = QCheckBox("Enabled")
        self._audio_enable_cb.setChecked(self._state.audio_enabled)
        self._audio_enable_cb.setToolTip(
            "Listen on the selected input to recognize enrolled speakers and detect music.\n"
            "Models download on first use (see Manage People).")
        self._audio_enable_cb.toggled.connect(self._on_audio_enabled_toggled)
        row.addWidget(self._audio_enable_cb)

        row.addWidget(QLabel("Input:"))
        self._audio_combo = QComboBox()
        self._populate_audio_devices_combo()
        self._audio_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        row.addWidget(self._audio_combo, stretch=1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(70)
        btn_refresh.clicked.connect(self._populate_audio_devices_combo)
        row.addWidget(btn_refresh)
        return box

    def _populate_audio_devices_combo(self):
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()
        self._audio_combo.addItem("(System default)", None)
        for dev in list_input_devices():
            self._audio_combo.addItem(
                f"[{dev['index']}] {dev['name']} ({dev['max_channels']}ch)", dev)
        # Re-select the remembered device by name
        wanted = self._state.audio_device_name
        if wanted:
            for i in range(1, self._audio_combo.count()):
                if self._audio_combo.itemData(i)['name'] == wanted:
                    self._audio_combo.setCurrentIndex(i)
                    break
        self._audio_combo.blockSignals(False)

    def _selected_audio_device(self) -> tuple[int | None, str]:
        data = self._audio_combo.currentData()
        if not data:
            return None, ''
        return data['index'], data['name']

    def _apply_audio_settings(self, enable: bool):
        index, _ = self._selected_audio_device()
        self._audio_thread.set_device(index)
        self._audio_thread.set_enabled(enable)

    def _on_audio_enabled_toggled(self, on: bool):
        _, name = self._selected_audio_device()
        self._state.set(audio_enabled=on, audio_device_name=name)
        self._persist()
        self._apply_audio_settings(enable=on)
        if self._people_win is not None:
            self._people_win._refresh_audio_hint()

    def _on_audio_device_changed(self, idx: int):
        index, name = self._selected_audio_device()
        self._state.set(audio_device_name=name)
        self._persist()
        self._audio_thread.set_device(index)

    # --- Shot type ---

    def _build_shot_section(self):
        box = QGroupBox("Shot Type")
        row = QHBoxLayout(box)
        self._shot_combo = QComboBox()
        for label, key in [("Full Body", "full_body"), ("Waist Up", "waist_up"),
                            ("Medium", "medium"), ("Close-Up", "close_up")]:
            self._shot_combo.addItem(label, key)
        for i in range(self._shot_combo.count()):
            if self._shot_combo.itemData(i) == self._state.shot_type:
                self._shot_combo.setCurrentIndex(i)
                break
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        row.addWidget(self._shot_combo)
        return box

    # --- Virtual Switcher / Primary Focus settings ---

    def _build_switcher_section(self):
        self._switcher_box = QGroupBox("Virtual Switcher")
        layout = QVBoxLayout(self._switcher_box)

        # Trigger row — Primary Focus first, then switcher triggers
        trig_row = QHBoxLayout()
        trig_label = QLabel("Mode:")
        self._trig_group = QButtonGroup()
        triggers = [
            ("Disabled", "disabled", "Show the full camera frame; no tracking."),
            ("Primary",  "primary",  "Follow the closest person; hand off when someone closer or more active appears."),
            ("Time",     "time",     "Rotate between tracked people on a fixed interval."),
            ("Manual",   "manual",   "Only switch when you press a person button below."),
        ]
        _valid_triggers = {t[1] for t in triggers}
        current_trigger = (
            "disabled" if not self._state.auto_follow_enabled
            else "primary" if self._state.tracking_mode == 'primary'
            else self._state.switch_trigger
                if self._state.switch_trigger in _valid_triggers else "time"
        )
        for label, key, tip in triggers:
            rb = QRadioButton(label)
            rb.setProperty("trigger_key", key)
            rb.setToolTip(tip)
            rb.setChecked(key == current_trigger)
            self._trig_group.addButton(rb)
            trig_row.addWidget(rb)
        self._trig_group.buttonClicked.connect(self._on_trigger_changed)
        trig_row.insertWidget(0, trig_label)
        layout.addLayout(trig_row)

        # Interval (time trigger only)
        int_row = QHBoxLayout()
        self._interval_label = QLabel("Interval (s):")
        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.5, 60.0)
        self._interval_spin.setSingleStep(0.5)
        self._interval_spin.setValue(self._state.switch_interval)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        int_row.addWidget(self._interval_label)
        int_row.addWidget(self._interval_spin)
        layout.addLayout(int_row)

        # Transition mode
        sm_row = QHBoxLayout()
        sm_label = QLabel("Transition:")
        self._sm_group = QButtonGroup()
        for label, key in [("Cut", "cut"), ("Crossfade", "crossfade")]:
            rb = QRadioButton(label)
            rb.setProperty("sm_key", key)
            rb.setChecked(key == self._state.switch_mode)
            self._sm_group.addButton(rb)
            sm_row.addWidget(rb)
        self._sm_group.buttonClicked.connect(self._on_switch_mode_changed)
        sm_row.insertWidget(0, sm_label)
        layout.addLayout(sm_row)

        # Crossfade duration
        cf_row = QHBoxLayout()
        self._cf_label = QLabel("Fade (s):")
        self._cf_spin = QDoubleSpinBox()
        self._cf_spin.setRange(0.1, 5.0)
        self._cf_spin.setSingleStep(0.1)
        self._cf_spin.setValue(self._state.crossfade_duration)
        self._cf_spin.valueChanged.connect(self._on_crossfade_changed)
        cf_row.addWidget(self._cf_label)
        cf_row.addWidget(self._cf_spin)
        layout.addLayout(cf_row)

        # Manual person buttons (populated dynamically)
        self._manual_label = QLabel("Manual Switch:")
        layout.addWidget(self._manual_label)
        self._persons_row = QHBoxLayout()
        layout.addLayout(self._persons_row)
        self._person_buttons: dict[str, QPushButton] = {}

        self._update_trigger_ui(current_trigger)
        return self._switcher_box

    # --- Output ---

    def _build_output_section(self):
        box = QGroupBox("Output")
        layout = QVBoxLayout(box)

        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("Display:"))
        self._display_combo = QComboBox()
        screens = QApplication.screens()
        for i, screen in enumerate(screens):
            geo = screen.geometry()
            self._display_combo.addItem(
                f"Display {i + 1}  ({geo.width()}×{geo.height()})", i
            )
        if 0 <= self._state.display_index < len(screens):
            self._display_combo.setCurrentIndex(self._state.display_index)
        self._display_combo.currentIndexChanged.connect(self._on_display_changed)
        display_row.addWidget(self._display_combo)
        display_row.addStretch()
        layout.addLayout(display_row)

        btn_row = QHBoxLayout()
        self._btn_fullscreen = QPushButton("Open Fullscreen Output")
        self._btn_fullscreen.setToolTip(
            "Show the program output fullscreen on the selected display.\n"
            "Press Esc or double-click the output to close it.")
        self._btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        btn_row.addWidget(self._btn_fullscreen)
        btn_diag = QPushButton("Open Diagnostics")
        btn_diag.clicked.connect(self._open_diagnostics)
        btn_row.addWidget(btn_diag)
        btn_people = QPushButton("Manage People")
        btn_people.setToolTip("Enroll people by face and voice, and set their priority.")
        btn_people.clicked.connect(self._open_people)
        btn_row.addWidget(btn_people)
        layout.addLayout(btn_row)

        # Foreground exclusion zone slider
        excl_row = QHBoxLayout()
        excl_row.addWidget(QLabel("Audience Exclusion:"))
        self._excl_slider = QSlider(Qt.Horizontal)
        self._excl_slider.setRange(0, 100)
        self._excl_slider.setValue(int(round(self._state.foreground_exclusion_y * 100)))
        self._excl_slider.setToolTip(
            "Ignore people whose torso is in the bottom N% of the frame (foreground audience filter).\n"
            "0 = disabled. Increase until stage-front audience members are no longer tracked.\n"
            "The zone is shown in yellow in the Diagnostics window."
        )
        self._excl_value_label = QLabel(f"{self._excl_slider.value()}%")
        self._excl_value_label.setFixedWidth(32)
        self._excl_slider.valueChanged.connect(self._on_exclusion_changed)
        excl_row.addWidget(self._excl_slider)
        excl_row.addWidget(self._excl_value_label)
        layout.addLayout(excl_row)

        # Max tracked persons spinbox
        mp_row = QHBoxLayout()
        mp_row.addWidget(QLabel("Max Tracked Persons:"))
        self._max_persons_spin = QSpinBox()
        self._max_persons_spin.setRange(1, 30)
        self._max_persons_spin.setValue(self._state.max_persons)
        self._max_persons_spin.setToolTip(
            "Maximum number of people tracked simultaneously.\n"
            "Higher values let the switcher follow more performers but use more CPU."
        )
        self._max_persons_spin.valueChanged.connect(self._on_max_persons_changed)
        mp_row.addWidget(self._max_persons_spin)
        mp_row.addStretch()
        layout.addLayout(mp_row)

        return box

    # --- Status bar ---

    def _build_status_bar(self):
        self._status_label = QLabel("Initializing…")
        self._status_label.setStyleSheet("color: gray; font-size: 11px; padding: 2px;")
        return self._status_label

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame(self, qimg: QImage, meta: dict):
        try:
            pix = QPixmap.fromImage(qimg)
            self._preview_label.setPixmap(
                pix.scaled(self._preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            if self._output_win.isVisible():
                self._output_win.update_frame(qimg)
            # Always feed the diagnostics panel so the log captures events even when hidden.
            self._diag_win.update_diagnostics(meta)
            self._status_label.setText(
                f"FPS: {meta['fps']:.1f}  |  "
                f"Tracking: {meta['n_persons']} person(s)  |  "
                f"Active: {meta['active_id']}"
                + ("  |  searching…" if meta.get('searching') else "")
                + f"  |  {self._audio_status_text}"
            )
        finally:
            self._video_thread.frame_consumed()

    def _on_camera_info(self, info: str):
        self._cam_info_label.setText(info)

    def _on_status(self, text: str):
        self._last_status = text
        self._status_label.setText(text)

    def _on_persons_updated(self, person_ids: list):
        # Rebuild manual person buttons to match currently tracked IDs
        existing = set(self._person_buttons.keys())
        current = set(person_ids)

        for pid in existing - current:
            btn = self._person_buttons.pop(pid)
            self._persons_row.removeWidget(btn)
            btn.deleteLater()

        manual_active = self._manual_mode_active()
        for pid in current - existing:
            btn = QPushButton(pid.replace('person', 'P'))
            btn.setFixedWidth(40)
            btn.setToolTip(f"Switch to {pid}")
            btn.setVisible(manual_active)
            btn.clicked.connect(lambda checked, p=pid: self._manual_switch(p))
            self._person_buttons[pid] = btn
            self._persons_row.addWidget(btn)

    def _manual_mode_active(self) -> bool:
        return (self._state.auto_follow_enabled
                and self._state.tracking_mode == 'switcher'
                and self._state.switch_trigger == 'manual')

    def _on_camera_changed(self, idx: int):
        cam_idx = self._cam_combo.itemData(idx)
        if cam_idx is not None:
            self._state.set(camera_index=cam_idx, camera_change_requested=True)
            self._persist()

    def _on_shot_changed(self, idx: int):
        self._state.set(shot_type=self._shot_combo.itemData(idx))
        self._persist()

    def _on_trigger_changed(self, button):
        key = button.property("trigger_key")
        if key == 'disabled':
            self._state.set(auto_follow_enabled=False)
        elif key == 'primary':
            self._state.set(auto_follow_enabled=True, tracking_mode='primary')
        else:
            self._state.set(auto_follow_enabled=True, tracking_mode='switcher',
                            switch_trigger=key)
        self._update_trigger_ui(key)
        self._persist()

    def _update_trigger_ui(self, trigger_key: str):
        """Show/hide controls based on selected trigger."""
        show_interval = trigger_key == 'time'
        self._interval_label.setVisible(show_interval)
        self._interval_spin.setVisible(show_interval)
        manual_active = trigger_key == 'manual'
        self._manual_label.setVisible(manual_active)
        for btn in self._person_buttons.values():
            btn.setVisible(manual_active)

    def _on_interval_changed(self, val: float):
        self._state.set(switch_interval=val)
        self._persist()

    def _on_switch_mode_changed(self, button):
        self._state.set(switch_mode=button.property("sm_key"))
        self._persist()

    def _on_crossfade_changed(self, val: float):
        self._state.set(crossfade_duration=val)
        self._persist()

    def _manual_switch(self, person_id: str):
        self._state.set(manual_switch_id=person_id)

    def _on_diagnostics_changed(self, enabled: bool):
        self._state.set(diag_overlays=bool(enabled))
        self._persist()

    def _on_diag_visibility(self, visible: bool):
        self._state.set(diag_visible=bool(visible))

    def _on_output_visibility(self, visible: bool):
        self._btn_fullscreen.setText(
            "Close Fullscreen Output" if visible else "Open Fullscreen Output")

    def _on_exclusion_changed(self, value: int):
        self._excl_value_label.setText(f"{value}%")
        self._state.set(foreground_exclusion_y=value / 100.0)
        self._persist()

    def _on_max_persons_changed(self, value: int):
        self._state.set(max_persons=value)
        self._persist()

    def _on_display_changed(self, idx: int):
        self._state.set(display_index=idx)
        self._persist()
        if self._output_win.isVisible():
            self._show_output_on_selected_display()

    def _toggle_fullscreen(self):
        if self._output_win.isVisible():
            self._output_win.hide()
        else:
            self._show_output_on_selected_display()

    def _show_output_on_selected_display(self):
        screen_index = self._display_combo.currentData()
        screens = QApplication.screens()
        if screen_index is None or not (0 <= screen_index < len(screens)):
            screen_index = 0
        screen = screens[screen_index]
        geo = screen.geometry()
        # Create the native window first so it can be bound to the target screen;
        # otherwise macOS may pull a fullscreen window back to the primary display.
        self._output_win.winId()
        handle = self._output_win.windowHandle()
        if handle is not None:
            handle.setScreen(screen)
        self._output_win.setGeometry(geo)
        self._output_win.showFullScreen()
        self._output_win.raise_()

    def _open_diagnostics(self):
        self._diag_win.show()
        self._diag_win.raise_()

    # ------------------------------------------------------------------
    # Audio signal handlers
    # ------------------------------------------------------------------

    def _on_audio_state_changed(self, music_mode: bool, music_score: float,
                                speech_score: float):
        enabled_now = self._audio_thread.is_capturing
        self._state.set(music_mode=music_mode, audio_music_score=music_score,
                        audio_speech_score=speech_score)
        mode_text = "MUSIC" if music_mode else "Speech"
        self._audio_status_text = (
            f"Audio: {mode_text} (m={music_score:.2f} s={speech_score:.2f})"
            if enabled_now else "Audio: off")
        # Reflect the real capture state (a device error turns it off).
        if self._audio_enable_cb.isChecked() != enabled_now:
            self._audio_enable_cb.blockSignals(True)
            self._audio_enable_cb.setChecked(enabled_now)
            self._audio_enable_cb.blockSignals(False)

    def _on_speaker_detected(self, profile_id: str, name: str, score: float):
        # Queue a voice boost for the VideoThread to apply on the next frame.
        self._state.set(
            pending_voice_boost=(profile_id, VOICE_PRIORITY_BOOST, SPEAKER_BOOST_HOLD_S),
            audio_speaker_name=name,
            audio_speaker_score=float(score),
            audio_speaker_expires_at=time.monotonic() + SPEAKER_BOOST_HOLD_S,
        )
        self._audio_status_text = f"Audio: ● {name} ({score:.2f})"

    def _on_audio_error(self, msg: str):
        self._audio_status_text = f"Audio error: {msg}"
        self._on_status(f"Audio error: {msg}")

    def _open_people(self):
        if self._people_win is None:
            from people_ui import PeopleWindow
            self._people_win = PeopleWindow(
                self._profile_store,
                frame_grabber=self._video_thread.grab_latest_frame,
                audio_thread=self._audio_thread,
                parent=self,
            )
            # Rebuild the live embedding indices whenever profiles change.
            self._people_win.profiles_changed.connect(self._video_thread.reindex_profiles)
            self._people_win.voice_profiles_changed.connect(
                self._audio_thread.reindex_voice_profiles)
        self._people_win.show()
        self._people_win.raise_()
        self._people_win.activateWindow()

    def _persist(self):
        self._state.save(self._settings)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._persist()
        self._settings.sync()
        self._video_thread.stop()
        self._audio_thread.stop()
        self._output_win.close()
        self._diag_win.close()
        if self._people_win is not None:
            self._people_win.close()
        event.accept()
