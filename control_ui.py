"""PyQt5 control panel with integrated video thread and fullscreen output window."""

import sys
import time
import threading
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

import config
from pose_detector import PoseDetector
from tracker import PersonTracker
from framing_engine import FramingEngine
from smoothing import PTZSmoother
from switcher import VirtualSwitcher
from profiles import ProfileStore
from audio_thread import AudioThread, SPEAKER_BOOST_HOLD_S


# Run face recognition every N captured frames. ~5 Hz at 30 fps capture is enough
# to keep identities sticky without dominating the per-frame budget.
FACE_RECOGNITION_INTERVAL = 6

# Priority units added transiently to a profile whose voice is being recognized.
# Combined with profile.priority via TrackedPerson.effective_priority.
VOICE_PRIORITY_BOOST = 5.0


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

# Per-person colors: evenly-spaced hues around the HSV wheel (BGR).
# Up to 12 persons (matching MAX_PERSONS); cycles if more.
def _person_color(index: int) -> tuple[int, int, int]:
    """Return a vivid BGR color for person at position `index` (0-based)."""
    hue = int((index * 137.5) % 180)   # golden-angle step keeps neighbours distinct
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_cameras(max_cams: int = 8) -> list[int]:
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# App-wide shared state (written by UI thread, read by video thread)
# ---------------------------------------------------------------------------

class AppState:
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
        self.show_diagnostics: bool = config.SHOW_DIAGNOSTICS
        self.auto_follow_enabled: bool = True
        self.foreground_exclusion_y: float = config.FOREGROUND_EXCLUSION_Y
        self.max_persons: int = config.MAX_PERSONS
        # Phase 2: audio-driven state
        self.music_mode: bool = False
        self.audio_enabled: bool = False
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
                'manual_switch_id': self.manual_switch_id,
                'camera_change_requested': self.camera_change_requested,
                'show_diagnostics': self.show_diagnostics,
                'auto_follow_enabled': self.auto_follow_enabled,
                'foreground_exclusion_y': self.foreground_exclusion_y,
                'max_persons': self.max_persons,
                'music_mode': self.music_mode,
            }

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


# ---------------------------------------------------------------------------
# Video processing thread
# ---------------------------------------------------------------------------

class VideoThread(QThread):
    """Captures, processes, and emits frames without blocking the UI."""

    frame_ready = pyqtSignal(QImage, dict)   # (processed output frame, metadata)
    diag_frame_ready = pyqtSignal(QImage)    # raw input frame with color-coded overlays
    camera_info = pyqtSignal(str)             # e.g. "1920x1080 @ 30fps"
    persons_updated = pyqtSignal(list)        # list of person IDs currently tracked

    def __init__(self, state: AppState, profile_store: ProfileStore | None = None,
                 parent=None):
        super().__init__(parent)
        self._state = state
        self._running = False

        # Pipeline components (re-created on camera change)
        self._cap = None
        self._detector = PoseDetector()
        self._tracker = PersonTracker()
        self._framing: FramingEngine | None = None
        self._smoother = PTZSmoother()
        self._switcher = VirtualSwitcher()
        self._frame_count = 0

        # Face recognition (lazy — model loads on first use)
        self._profile_store = profile_store
        self._face_recognizer = None
        self._face_recognition_failed = False  # True if init failed; don't retry
        self._face_cadence_offset = 0

        # Latest raw camera frame (BGR) — used by the People UI to capture
        # reference images of someone currently in front of the camera.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self._person_index_map: dict[str, int] = {}   # stable color index per person ID
        self._next_person_color_idx: int = 0

        self._primary_id: str | None = None
        # Transition state for Primary Focus mode (mirrors switcher logic)
        self._primary_pending_id: str | None = None
        self._primary_pretraveling: bool = False
        self._primary_pretravel_start: float = 0.0
        self._primary_fade_start: float | None = None
        self._primary_last_switch_time: float = time.monotonic()
        # Search state: entered when primary ID is lost; camera holds position and
        # slowly zooms out until the primary reappears.  No cut/crossfade occurs.
        self._searching: bool = False
        # Disabled-mode transition state
        self._disabled_transitioning: bool = False
        self._disabled_transition_start: float | None = None

    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        settings = self._state.read()
        self._open_camera(settings['camera_index'])

        while self._running:
            settings = self._state.read()

            # Handle camera change
            if self._state.consume_camera_change():
                self._open_camera(settings['camera_index'])
                self._tracker.reset()
                self._smoother.reset()
                self._primary_id = None

            if self._cap is None or not self._cap.isOpened():
                self.msleep(100)
                continue

            ret, frame = self._cap.read()
            if not ret:
                self.msleep(10)
                continue

            # Stash a copy of the raw frame for "capture from camera" in People UI.
            with self._latest_frame_lock:
                self._latest_frame = frame.copy()

            result = self._process_frame(frame, settings)
            if result is not None:
                qimg, meta = result
                self.frame_ready.emit(qimg, meta)

            self._frame_count += 1

    def stop(self):
        self._running = False
        self.wait()
        if self._cap:
            self._cap.release()

    def grab_latest_frame(self) -> np.ndarray | None:
        """Return a copy of the most recent raw camera frame, or None."""
        with self._latest_frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    # ------------------------------------------------------------------
    # Camera open
    # ------------------------------------------------------------------

    def _open_camera(self, index: int):
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap = None
            return
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._framing = FramingEngine(w, h)
        self.camera_info.emit(f"{w}×{h} @ {fps:.0f} fps")

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, settings: dict):
        h, w = frame.shape[:2]
        t0 = time.monotonic()

        # Pose detection (every DETECTION_INTERVAL frames)
        if self._frame_count % config.DETECTION_INTERVAL == 0:
            detections = self._detector.detect(frame)
        else:
            detections = []

        persons = self._tracker.update(detections, frame.shape,
                                       foreground_exclusion_y=settings.get('foreground_exclusion_y', 0.0),
                                       max_persons=settings.get('max_persons', config.MAX_PERSONS))

        # Face recognition (slow cadence). Tracker has already updated bboxes,
        # so we can spatially match detected faces back to tracked person IDs.
        self._run_face_recognition(frame, persons)
        # Forget profile matches that haven't been re-confirmed recently.
        self._tracker.expire_stale_face_matches()
        # Apply any pending voice boost queued by the AudioThread, then decay
        # any boosts whose hold window has expired.
        voice_boost = self._state.consume_voice_boost()
        if voice_boost is not None:
            profile_id, boost, hold = voice_boost
            self._tracker.apply_voice_boost(profile_id, boost, hold)
        self._tracker.decay_voice_boosts()

        # Mirror the latest music_mode flag into the switcher so it picks the
        # right dwell + target-selection strategy.
        self._switcher.music_mode = bool(settings.get('music_mode', False))

        # Sync switcher settings from UI state
        self._switcher.switch_mode = settings['switch_mode']
        self._switcher.trigger = settings['switch_trigger']
        self._switcher.interval = settings['switch_interval']
        self._switcher.crossfade_duration = settings['crossfade_duration']

        # Manual switch request from UI
        manual_id = self._state.consume_manual_switch()
        if manual_id:
            self._switcher.force_switch(manual_id)

        # Emit tracked person IDs for UI person buttons
        self.persons_updated.emit([p.id for p in persons])

        # Assign stable color indices to new person IDs
        for p in persons:
            if p.id not in self._person_index_map:
                self._person_index_map[p.id] = self._next_person_color_idx
                self._next_person_color_idx += 1

        mode = settings['tracking_mode']
        shot_type = settings['shot_type']
        auto_enabled = settings.get('auto_follow_enabled', True)

        if not auto_enabled:
            sw_mode = settings['switch_mode']
            xfade_dur = settings['crossfade_duration']
            tx, ty, tz = self._framing._default_target()

            # On the first frame of a disable, seed the from-smoother with the
            # current live camera position so the transition starts from there.
            if not self._disabled_transitioning:
                self._disabled_transitioning = True
                live = self._smoother.get_state('primary')
                seed = dict(live) if live else {'x': tx, 'y': ty, 'zoom': tz}
                self._smoother._state['__disabled_from__'] = seed
                if sw_mode == 'crossfade':
                    self._disabled_transition_start = time.monotonic()
                else:
                    self._disabled_transition_start = None  # cut: skip blend

            if self._disabled_transition_start is not None:
                elapsed_t = time.monotonic() - self._disabled_transition_start
                t = min(1.0, elapsed_t / max(xfade_dur, 0.001))
                px, py, pz = self._smoother.update('__passthrough__', tx, ty, tz)
                frame_to = self._framing.apply_crop(frame, px, py, pz)
                if t < 1.0:
                    fx, fy, fz = self._smoother.update('__disabled_from__', tx, ty, tz)
                    frame_from = self._framing.apply_crop(frame, fx, fy, fz)
                    output_frame = cv2.addWeighted(frame_from, 1.0 - t, frame_to, t, 0)
                else:
                    self._disabled_transition_start = None
                    output_frame = frame_to
            else:
                px, py, pz = self._smoother.update('__passthrough__', tx, ty, tz)
                output_frame = self._framing.apply_crop(frame, px, py, pz)

            # Emit diagnostics frame even in disabled mode
            diag_frame = self._annotate_diag_frame(frame, persons,
                                                    self._primary_id or '',
                                                    self._person_index_map,
                                                    settings.get('foreground_exclusion_y', 0.0))
            self.diag_frame_ready.emit(_bgr_to_qimage(diag_frame))

            elapsed = time.monotonic() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            return _bgr_to_qimage(output_frame), {
                'fps': fps, 'n_persons': len(persons), 'active_id': 'disabled'
            }

        # Auto-follow enabled — clear disabled transition state so next disable starts fresh
        self._disabled_transitioning = False
        self._disabled_transition_start = None

        if mode == 'primary' or not persons:
            output_frame = self._render_primary(frame, persons, shot_type, settings)
            active_id = self._primary_id if self._primary_id else (persons[0].id if persons else 'none')
        else:
            output_frame, active_id = self._render_switcher(frame, persons, shot_type)

        # Emit the color-coded diagnostics preview (raw input + overlays, never cropped)
        diag_frame = self._annotate_diag_frame(frame, persons, active_id,
                                               self._person_index_map,
                                               settings.get('foreground_exclusion_y', 0.0))
        self.diag_frame_ready.emit(_bgr_to_qimage(diag_frame))

        elapsed = time.monotonic() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

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

        dwell_elapsed = time.monotonic() - self._primary_last_switch_time
        # Snapshot audio state for the diagnostics panel. Read under the state
        # lock so we get a consistent view of speaker_name + scores together.
        now = time.monotonic()
        with QMutexLocker(self._state._lock):
            spk_name = self._state.audio_speaker_name
            spk_score = self._state.audio_speaker_score
            spk_expires = self._state.audio_speaker_expires_at
            audio_state = {
                'enabled': self._state.audio_enabled,
                'music_mode': self._state.music_mode,
                'music_score': self._state.audio_music_score,
                'speech_score': self._state.audio_speech_score,
                # Expire the sticky speaker name once the hold window passes
                # so the panel doesn't lie about who's currently talking.
                'speaker_name': spk_name if now < spk_expires else None,
                'speaker_score': spk_score if now < spk_expires else 0.0,
            }
        meta = {
            'fps': fps,
            'n_persons': len(persons),
            'active_id': active_id,
            'persons': persons_diag,
            'primary_id': self._primary_id,
            'pending_id': self._primary_pending_id,
            'pretraveling': self._primary_pretraveling,
            'dwell_elapsed': dwell_elapsed,
            'dwell_threshold': 3.0,
            'mode': mode,
            'smoother_primary': self._smoother.get_state('primary'),
            'searching': self._searching,
            'person_index_map': dict(self._person_index_map),
            'audio_state': audio_state,
        }
        return _bgr_to_qimage(output_frame), meta

    # ------------------------------------------------------------------
    # Face recognition
    # ------------------------------------------------------------------

    def _run_face_recognition(self, frame, persons):
        """Run face recognition on the current frame and tag tracker persons.

        Skipped when:
          - No profile store is wired (face recognition disabled at startup).
          - Earlier init failed (we never retry inside the loop).
          - No profiles with embeddings exist (nothing to match against).
          - The current frame isn't on the FACE_RECOGNITION_INTERVAL cadence.
          - No persons are tracked (nothing to attach a match to).
        """
        if self._profile_store is None or self._face_recognition_failed:
            return
        if not persons:
            return
        # Cadence: only run every Nth frame
        if (self._frame_count + self._face_cadence_offset) % FACE_RECOGNITION_INTERVAL != 0:
            return
        # No profiles with embeddings? skip the (expensive) detect-and-match call.
        has_indexed = any(p.embeddings is not None and len(p.embeddings) > 0
                          for p in self._profile_store.list())
        if not has_indexed:
            return

        # Lazy-init the recognizer on first use so app startup is fast.
        if self._face_recognizer is None:
            try:
                from face_recognizer import FaceRecognizer
                self._face_recognizer = FaceRecognizer(self._profile_store)
            except Exception as e:
                print(f"FaceRecognizer init failed: {e}")
                self._face_recognition_failed = True
                return

        try:
            faces = self._face_recognizer.identify(frame)
        except Exception as e:
            print(f"Face identify failed: {e}")
            return

        for face in faces:
            if face.get("profile_id") is None:
                continue
            tid = self._tracker.find_by_face_bbox(face["bbox"], frame.shape)
            if tid is None:
                continue
            self._tracker.set_profile_match(
                tid,
                face["profile_id"],
                face["name"],
                face["priority"],
                face["score"],
            )

    def reindex_profiles(self):
        """Tell the face recognizer to rebuild its profile index on next call."""
        if self._face_recognizer is not None:
            self._face_recognizer.mark_index_dirty()

    # ------------------------------------------------------------------
    # Diagnostic overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_diag_frame(frame: np.ndarray, persons, primary_id: str,
                              person_index_map: dict,
                              foreground_exclusion_y: float = 0.0) -> np.ndarray:
        """Render color-coded skeleton overlays on the raw input frame.

        Each person gets a unique hue (golden-angle spacing).  The body
        silhouette — convex hull of all visible keypoints — is filled with a
        semi-transparent wash of that color.  Skeleton lines and joint dots are
        drawn on top in full color.  The primary person's bbox is outlined with
        a brighter border and an ID label.

        This output is only ever sent to the diagnostics preview; it never
        touches the main output pipeline.
        """
        h, w = frame.shape[:2]
        out = frame.copy()
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
            bx1, by1, bx2, by2 = person.bbox
            is_primary = person.id == primary_id
            border_thickness = 3 if is_primary else 1
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, border_thickness, cv2.LINE_AA)

            # ── Primary indicator: downward arrow above bbox + centroid halo
            if is_primary:
                cx_p = (bx1 + bx2) // 2
                arrow_tip_y = by1 - 6
                arrow_base_y = arrow_tip_y - 18
                arrow_half_w = 10
                # Filled downward-pointing triangle (arrow head)
                tri = np.array([
                    [cx_p,              arrow_tip_y],
                    [cx_p - arrow_half_w, arrow_base_y],
                    [cx_p + arrow_half_w, arrow_base_y],
                ], dtype=np.int32)
                cv2.fillConvexPoly(overlay, tri, (255, 255, 255))
                cv2.polylines(out, [tri], True, color, 2, cv2.LINE_AA)
                cv2.fillConvexPoly(out, tri, color)
                # Bright halo ring at centroid
                cy_p = (by1 + by2) // 2
                cv2.circle(out, (cx_p, cy_p), 10, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(out, (cx_p, cy_p), 10, color,           2, cv2.LINE_AA)

            # ── ID + score label ─────────────────────────────────────────
            # Prefer the matched profile name (e.g. "Pastor Bob") over the
            # generic personN ID. Adds ★ for static priority, and a mic glyph
            # (●) when speaker recognition is currently boosting this person.
            id_text = person.profile_name if person.profile_name else person.id
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

            # Semi-transparent filled region with diagonal hatching
            excl_overlay = out.copy()
            cv2.rectangle(excl_overlay, (0, excl_y), (w, h), yellow, -1)
            cv2.addWeighted(excl_overlay, 0.15, out, 0.85, 0, out)

            # Diagonal stripes over the exclusion region
            stripe_overlay = out.copy()
            stripe_gap = 18
            for x_start in range(-h, w, stripe_gap):
                pt1 = (x_start, excl_y)
                pt2 = (x_start + (h - excl_y), h)
                cv2.line(stripe_overlay, pt1, pt2, yellow, 1, cv2.LINE_AA)
            cv2.addWeighted(stripe_overlay, 0.45, out, 0.55, 0, out)

            # Solid border line along the top edge of the exclusion zone
            cv2.line(out, (0, excl_y), (w, excl_y), yellow, 2, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Render modes
    # ------------------------------------------------------------------

    def _render_primary(self, frame, persons, shot_type, settings: dict | None = None):
        """Track the nearest (largest bbox) person as primary.

        Primary persistence:
          - Stays on current subject until it leaves the frame entirely.
          - Switches to a closer subject when foreground_score > current * 1.15
            (i.e. ~15% bigger bbox area), OR same-distance but 2× more active.
          - All transitions use the same cut/crossfade pipeline as VirtualSwitcher.

        No-subject wide-shot:
          - When no persons are detected, crossfade to a full-width wide shot.
          - After a subject reappears, hold the wide shot for at least xfade_dur
            seconds before re-acquiring, giving the transition time to breathe.
        """
        sw_mode = (settings or {}).get('switch_mode', 'crossfade')
        xfade_dur = (settings or {}).get('crossfade_duration', self._switcher.crossfade_duration)
        now = time.monotonic()

        from switcher import _PRETRAVEL_DURATION

        # How fast to zoom out each frame while searching (zoom units/frame).
        # At 30fps this reaches _MIN_ZOOM=0.5 from zoom=3 in ~83 frames (~2.8s).
        _SEARCH_ZOOM_OUT_RATE = 0.008

        # ----------------------------------------------------------------
        # Search mode: primary lost — hold pan/tilt, zoom out slowly
        # ----------------------------------------------------------------
        current_ids = {p.id for p in persons}

        primary_present = self._primary_id is not None and self._primary_id in current_ids

        if not primary_present:
            # Enter search if not already in it
            if not self._searching:
                self._searching = True
                self._primary_pending_id = None
                self._primary_pretraveling = False
                self._primary_fade_start = None

            # Adopt the largest-in-frame person as initial primary, with a small
            # priority bias so a profile-matched person beats a similarly-sized
            # unmatched bystander.  The bias is capped at 1.5x so an obviously
            # closer/larger unmatched person (e.g. a guest speaker at the pulpit)
            # still wins over a low-priority profile match seated in the back.
            if persons and self._primary_id is None:
                # effective_priority includes any active voice_boost so a
                # recognized speaker who's currently talking is preferred even
                # if their bbox is slightly smaller than another candidate.
                pick = max(persons,
                           key=lambda p: p.foreground_score * (1.0 + p.effective_priority / 20.0))
                self._primary_id = pick.id
                self._primary_last_switch_time = now
                self._searching = False
            else:
                # Hold current position; nudge zoom out toward _MIN_ZOOM
                state = self._smoother.get_state('primary')
                if state is not None:
                    from framing_engine import _MIN_ZOOM
                    state['zoom'] = max(_MIN_ZOOM, state['zoom'] - _SEARCH_ZOOM_OUT_RATE)
                    return self._framing.apply_crop(frame, state['x'], state['y'], state['zoom'])
                else:
                    # No smoother state yet — nothing to show
                    return frame

        # Primary reappeared — exit search mode
        if self._searching:
            self._searching = False
            self._primary_last_switch_time = now

        current_p = next((p for p in persons if p.id == self._primary_id), persons[0])

        music_mode = getattr(self._switcher, 'music_mode', False)

        # --- Candidate selection: nearest (by fg_score) or significantly more active ---
        # persons[] is sorted foreground_score desc (nearest = persons[0]). The
        # raw bbox area is the source of truth for "who is closest" — priority
        # never *changes* the nearest pick, only how reluctant we are to switch.
        # This keeps unmatched subjects (guest speakers, readers, anyone whose
        # face isn't in the profile store) fully eligible to become primary
        # whenever they're clearly closer to the camera than the current one.
        # fg_ratio thresholds use hysteresis: a higher bar to initiate a switch than
        # was required to arrive at the current primary, preventing oscillation when two
        # people have similar sizes.  Both fg_score and activity_score are EMA-smoothed
        # in the tracker so single-frame noise doesn't trigger a switch.
        if music_mode:
            # In music mode the most-active performer is the right primary —
            # they're singing, soloing, or leading the band. Fall back to
            # foreground area when nobody is significantly moving.
            nearest = max(persons,
                          key=lambda p: (p.activity_score, p.foreground_score))
        else:
            nearest = persons[0]
        # Music mode halves the minimum dwell so we can follow song dynamics.
        min_dwell = 1.5 if music_mode else 3.0
        if nearest.id != self._primary_id and self._primary_pending_id is None:
            if now - self._primary_last_switch_time >= min_dwell:
                fg_ratio = nearest.foreground_score / max(current_p.foreground_score, 1e-6)
                cand_act = nearest.activity_score
                curr_act = current_p.activity_score

                # Priority modulates the switch threshold but never gates the
                # candidate. priority_diff > 0 means the candidate has higher
                # priority than the current primary (e.g. pastor entering a
                # frame currently following an unmatched person) and we should
                # switch eagerly; priority_diff < 0 means we should stick with
                # the priority person against a bystander a bit longer.
                priority_diff = nearest.effective_priority - current_p.effective_priority
                # Threshold range: priority_diff= +10 -> 1.0 (any closer wins),
                #                  priority_diff=   0 -> 1.5 (50% closer),
                #                  priority_diff= -10 -> 2.25 (must be 125% closer).
                switch_threshold = max(1.0, 1.5 - priority_diff * 0.075)

                # Eager priority override: strictly higher-priority candidate
                # and at least as close — switch immediately.
                priority_override = (priority_diff > 0 and fg_ratio >= 1.0)
                fg_wins = fg_ratio >= switch_threshold
                # Activity switch: candidate must clear a noise floor AND be 2× more active.
                activity_wins = cand_act >= 5.0 and cand_act > curr_act * 2.0
                if priority_override or fg_wins or activity_wins:
                    self._primary_pending_id = nearest.id
                    self._primary_pretraveling = True
                    self._primary_pretravel_start = now
                    self._primary_fade_start = None

        # --- Phase 1: pretravel ---
        if self._primary_pretraveling:
            pending_person = next((p for p in persons if p.id == self._primary_pending_id), None)
            if pending_person:
                ptx, pty, ptz = self._framing.calculate_target(pending_person, shot_type)
                cx_p = (pending_person.bbox[0] + pending_person.bbox[2]) / 2.0
                cw_p = config.OUTPUT_WIDTH / ptz
                self._smoother.update(self._primary_pending_id, ptx, pty, ptz,
                                      person_center_x=cx_p, crop_width=cw_p)
            if now - self._primary_pretravel_start >= _PRETRAVEL_DURATION:
                self._primary_pretraveling = False
                if sw_mode == 'cut':
                    self._primary_id = self._primary_pending_id
                    self._primary_pending_id = None
                    self._primary_last_switch_time = now
                else:
                    self._primary_fade_start = now
            primary = next((p for p in persons if p.id == self._primary_id), persons[0])
            tx, ty, tz = self._framing.calculate_target(primary, shot_type)
            cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
            cw = config.OUTPUT_WIDTH / tz
            sx, sy, sz = self._smoother.update('primary', tx, ty, tz,
                                               person_center_x=cx, crop_width=cw)
            return self._framing.apply_crop(frame, sx, sy, sz)

        # --- Phase 2: crossfade to new primary ---
        if self._primary_pending_id is not None and self._primary_fade_start is not None:
            elapsed = now - self._primary_fade_start
            t = min(1.0, elapsed / max(xfade_dur, 0.001))

            primary = next((p for p in persons if p.id == self._primary_id), persons[0])
            atx, aty, atz = self._framing.calculate_target(primary, shot_type)
            cx_a = (primary.bbox[0] + primary.bbox[2]) / 2.0
            cw_a = config.OUTPUT_WIDTH / atz
            ax, ay, az = self._smoother.update('primary', atx, aty, atz,
                                               person_center_x=cx_a, crop_width=cw_a)
            frame_active = self._framing.apply_crop(frame, ax, ay, az)

            pending_person = next((p for p in persons if p.id == self._primary_pending_id), None)
            if pending_person:
                ptx, pty, ptz = self._framing.calculate_target(pending_person, shot_type)
                cx_p = (pending_person.bbox[0] + pending_person.bbox[2]) / 2.0
                cw_p = config.OUTPUT_WIDTH / ptz
                px, py, pz = self._smoother.update(self._primary_pending_id, ptx, pty, ptz,
                                                   person_center_x=cx_p, crop_width=cw_p)
                frame_pending = self._framing.apply_crop(frame, px, py, pz)
                blended = cv2.addWeighted(frame_active, 1.0 - t, frame_pending, t, 0)
            else:
                blended = frame_active
                t = 1.0

            if t >= 1.0:
                self._primary_id = self._primary_pending_id
                self._primary_pending_id = None
                self._primary_fade_start = None
                self._primary_last_switch_time = now

            return blended

        # --- Steady state: follow primary ---
        primary = next((p for p in persons if p.id == self._primary_id), persons[0])
        tx, ty, tz = self._framing.calculate_target(primary, shot_type)
        center_x = (primary.bbox[0] + primary.bbox[2]) / 2.0
        crop_w = config.OUTPUT_WIDTH / tz
        sx, sy, sz = self._smoother.update(
            'primary', tx, ty, tz,
            person_center_x=center_x, crop_width=crop_w
        )
        return self._framing.apply_crop(frame, sx, sy, sz)

    def _render_switcher(self, frame, persons, shot_type):
        """Virtual switcher: cut or crossfade between tracked persons.

        When a switch is queued the pending person's smoother is advanced every
        frame (pretravel phase) so the virtual camera has already arrived at the
        new subject's position by the time the cut or crossfade fires.
        """
        # Give the switcher the current crop width so it can gate switches by displacement.
        if self._switcher.active_id:
            current_person = next((p for p in persons if p.id == self._switcher.active_id), None)
            if current_person:
                _, _, cur_zoom = self._framing.calculate_target(current_person, shot_type)
                self._switcher.current_crop_width = config.OUTPUT_WIDTH / cur_zoom

        active_id = self._switcher.decide(persons)

        # Render active person
        active_person = next((p for p in persons if p.id == active_id), persons[0])
        atx, aty, atz = self._framing.calculate_target(active_person, shot_type)
        cx_a = (active_person.bbox[0] + active_person.bbox[2]) / 2.0
        cw_a = config.OUTPUT_WIDTH / atz
        ax, ay, az = self._smoother.update(
            active_id, atx, aty, atz,
            person_center_x=cx_a, crop_width=cw_a
        )
        frame_active = self._framing.apply_crop(frame, ax, ay, az)

        # Pretravel: advance the pending smoother off-screen so it's settled
        # before the transition becomes visible; keep showing the active frame.
        if self._switcher.is_pretraveling:
            pending_id = self._switcher._pending_id
            pending_person = next((p for p in persons if p.id == pending_id), None)
            if pending_person:
                ptx, pty, ptz = self._framing.calculate_target(pending_person, shot_type)
                cx_p = (pending_person.bbox[0] + pending_person.bbox[2]) / 2.0
                cw_p = config.OUTPUT_WIDTH / ptz
                self._smoother.update(
                    pending_id, ptx, pty, ptz,
                    person_center_x=cx_p, crop_width=cw_p
                )
            return frame_active, active_id

        # Crossfade: blend settled active and pending frames
        if self._switcher.is_transitioning:
            pending_id = self._switcher._pending_id
            pending_person = next((p for p in persons if p.id == pending_id), None)
            if pending_person:
                ptx, pty, ptz = self._framing.calculate_target(pending_person, shot_type)
                cx_p = (pending_person.bbox[0] + pending_person.bbox[2]) / 2.0
                cw_p = config.OUTPUT_WIDTH / ptz
                px, py, pz = self._smoother.update(
                    pending_id, ptx, pty, ptz,
                    person_center_x=cx_p, crop_width=cw_p
                )
                frame_pending = self._framing.apply_crop(frame, px, py, pz)
                return self._switcher.blend(frame_active, frame_pending), active_id

        return frame_active, active_id


# ---------------------------------------------------------------------------
# Diagnostics panel
# ---------------------------------------------------------------------------

_LOG_MAX_LINES = 200   # keep last N switch events in the log


class DiagnosticsWindow(QMainWindow):
    """Floating panel showing live tracking state and a switch-event log.

    Updated every frame via update_diagnostics(); never touches the video thread.
    """

    overlays_changed = pyqtSignal(bool)   # emitted when "Show Overlays" is toggled

    def __init__(self, show_overlays: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofollow Diagnostics")
        self.setMinimumSize(640, 640)
        self._last_active_id: str | None = None   # for detecting new switch events

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(6)

        # ── Show Overlays checkbox ───────────────────────────────────────
        overlay_row = QHBoxLayout()
        self._overlay_checkbox = QCheckBox("Show Overlays")
        self._overlay_checkbox.setChecked(show_overlays)
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
        self._lbl_persons  = self._make_field("Persons", state_grid)
        layout.addWidget(state_box)

        # ── Audio state row ──────────────────────────────────────────────
        audio_box = QGroupBox("Audio")
        audio_grid = QHBoxLayout(audio_box)
        self._lbl_audio_mode    = self._make_field("Mode", audio_grid)
        self._lbl_audio_speaker = self._make_field("Speaker", audio_grid)
        self._lbl_audio_scores  = self._make_field("Music/Speech", audio_grid)
        layout.addWidget(audio_box)

        # ── Per-person table ─────────────────────────────────────────────
        # Columns: swatch, ID/profile name, face match score, voice boost,
        # foreground score, activity score, ratios, unseen, position, zoom.
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
        self._log.setFont(QFont("Courier", 10))
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
        when the window is hidden). Skips the table and label updates when
        the window is not visible to avoid wasted work.
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
        if meta.get('mode') == 'disabled':
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

        if self.isVisible():
            self._lbl_mode.setText(mode)
            self._lbl_active.setText(primary_id or '—')
            self._lbl_pending.setText(pending_id or '—')
            self._lbl_phase.setText(phase)
            self._lbl_dwell.setText(f"{dwell:.2f}/{dwell_threshold:.0f}s")
            self._lbl_fps.setText(f"{fps:.1f}")
            self._lbl_persons.setText(str(meta.get('n_persons', len(persons))))

            # Audio panel: surfaces the most recent state pushed by AudioThread.
            audio_state = meta.get('audio_state') or {}
            music_mode = bool(audio_state.get('music_mode', False))
            spk_name = audio_state.get('speaker_name')
            spk_score = audio_state.get('speaker_score', 0.0)
            music_s = audio_state.get('music_score', 0.0)
            speech_s = audio_state.get('speech_score', 0.0)
            if audio_state.get('enabled'):
                self._lbl_audio_mode.setText("MUSIC" if music_mode else "Speech")
                self._lbl_audio_mode.setStyleSheet(
                    "color: #c678dd; font-weight: bold;" if music_mode
                    else "color: #98c379; font-weight: bold;"
                )
            else:
                self._lbl_audio_mode.setText("off")
                self._lbl_audio_mode.setStyleSheet("color: gray;")
            if spk_name:
                self._lbl_audio_speaker.setText(f"● {spk_name} ({spk_score:.2f})")
                self._lbl_audio_speaker.setStyleSheet("color: #61afef; font-weight: bold;")
            else:
                self._lbl_audio_speaker.setText("—")
                self._lbl_audio_speaker.setStyleSheet("color: gray;")
            self._lbl_audio_scores.setText(f"m={music_s:.2f} / s={speech_s:.2f}")

            # Red = gate closed (can't switch yet), green = gate open
            if not gate_open:
                self._lbl_dwell.setStyleSheet("color: #e06c75; font-weight: bold;")
            else:
                self._lbl_dwell.setStyleSheet("color: #98c379; font-weight: bold;")

            # ── Per-person table ─────────────────────────────────────────
            self._table.setRowCount(len(persons))
            person_index_map = meta.get('person_index_map', {})
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

                # Face match column: profile name + cosine score, or em-dash
                profile_name = p.get('profile_name')
                profile_score = p.get('profile_score', 0.0)
                profile_priority = p.get('profile_priority', 0)
                if profile_name:
                    face_cell = f"{profile_name} ★{profile_priority} ({profile_score:.2f})"
                    id_cell = f"{pid} → {profile_name}"
                else:
                    face_cell = "—"
                    id_cell = pid

                # Voice column: active boost indicator + effective priority
                voice_boost = p.get('voice_boost', 0.0)
                effective_prio = p.get('effective_priority', profile_priority)
                if voice_boost > 0:
                    voice_cell = f"● +{voice_boost:.0f} (eff {effective_prio:.0f})"
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
                        item.setForeground(skel_qcolor)
                    elif is_pending:
                        item.setBackground(QColor(80, 60, 20))
                        item.setForeground(skel_qcolor)
                    elif fg_trigger or act_trigger:
                        item.setBackground(QColor(80, 40, 40))
                        item.setForeground(skel_qcolor)
                    else:
                        item.setBackground(swatch_bg)
                        item.setForeground(skel_qcolor)
                    self._table.setItem(row, col, item)

            # Fix swatch column to a narrow fixed width
            self._table.setColumnWidth(0, 22)

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

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.hide()


# ---------------------------------------------------------------------------
# Fullscreen output window
# ---------------------------------------------------------------------------

class OutputWindow(QMainWindow):
    """Borderless fullscreen window displaying the processed video output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofollow Output")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background: black;")
        self.setCentralWidget(self._label)

    def update_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pix.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
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

        self._state = AppState()
        self._profile_store = ProfileStore()
        self._people_win = None  # lazy-created when user opens "Manage People"
        self._latest_frame_bgr: np.ndarray | None = None  # last raw frame, for capture
        self._output_win = OutputWindow()
        self._diag_win = DiagnosticsWindow(show_overlays=self._state.show_diagnostics)
        self._diag_win.overlays_changed.connect(self._on_diagnostics_changed)
        self._video_thread = VideoThread(self._state, profile_store=self._profile_store)
        # Audio analysis (capture + VAD + speaker recognition + music classification)
        self._audio_thread = AudioThread(self._profile_store)
        self._audio_thread.audio_state_changed.connect(self._on_audio_state_changed)
        self._audio_thread.speaker_detected.connect(self._on_speaker_detected)
        self._audio_thread.error.connect(self._on_audio_error)
        # Surface the latest audio state to the UI label
        self._audio_status_text = "Audio: off"
        self._video_thread.frame_ready.connect(self._on_frame)
        self._video_thread.diag_frame_ready.connect(self._diag_win.update_video)
        self._video_thread.camera_info.connect(self._on_camera_info)
        self._video_thread.persons_updated.connect(self._on_persons_updated)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(320, 180)
        self._preview_label.setStyleSheet("background: black;")

        self._build_ui()
        self._video_thread.start()
        # Audio analysis starts disabled. The user enables it from the People
        # window by picking an input device; this avoids grabbing the mic
        # on startup (and skips the YAMNet/SpeechBrain downloads until needed).
        self._audio_thread.start()

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
        self._refresh_cameras()
        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_cameras)
        self._cam_info_label = QLabel("")
        self._cam_info_label.setStyleSheet("color: gray; font-size: 10px;")
        row.addWidget(self._cam_combo)
        row.addWidget(btn_refresh)
        row.addWidget(self._cam_info_label)
        return box

    def _refresh_cameras(self):
        cameras = _scan_cameras()
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for idx in cameras:
            self._cam_combo.addItem(f"Camera {idx}", idx)
        # Try to select configured default
        for i in range(self._cam_combo.count()):
            if self._cam_combo.itemData(i) == self._state.camera_index:
                self._cam_combo.setCurrentIndex(i)
                break
        self._cam_combo.blockSignals(False)

    # --- Audio ---

    def _build_audio_section(self):
        """Audio input controls — Enabled toggle + input device picker.

        Drives the AudioThread. Same controls exist in the People window
        because that's where you record voice samples, but having them here
        means the user doesn't have to open Manage People just to turn audio
        analysis on or off.
        """
        box = QGroupBox("Audio (speaker + music detection)")
        row = QHBoxLayout(box)
        self._audio_enable_cb = QCheckBox("Enabled")
        self._audio_enable_cb.setChecked(self._audio_thread.is_capturing)
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
        from audio_capture import list_input_devices
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()
        self._audio_combo.addItem("(System default)", None)
        for dev in list_input_devices():
            self._audio_combo.addItem(
                f"[{dev['index']}] {dev['name']} ({dev['max_channels']}ch)",
                dev['index'],
            )
        # Reflect current selection if any
        if self._audio_thread.device_index is not None:
            for i in range(self._audio_combo.count()):
                if self._audio_combo.itemData(i) == self._audio_thread.device_index:
                    self._audio_combo.setCurrentIndex(i)
                    break
        self._audio_combo.blockSignals(False)

    def _on_audio_enabled_toggled(self, on: bool):
        if on:
            device_index = self._audio_combo.itemData(self._audio_combo.currentIndex())
            self._audio_thread.set_device(device_index)
            self._audio_thread.set_enabled(True)
        else:
            self._audio_thread.set_enabled(False)
        # Keep the People window's checkbox/dropdown in sync if it's open.
        if self._people_win is not None:
            people_cb = getattr(self._people_win, '_audio_enable', None)
            if people_cb is not None and people_cb.isChecked() != on:
                people_cb.blockSignals(True)
                people_cb.setChecked(on)
                people_cb.blockSignals(False)

    def _on_audio_device_changed(self, idx: int):
        device_index = self._audio_combo.itemData(idx)
        self._audio_thread.set_device(device_index)

    # --- Shot type ---

    def _build_shot_section(self):
        box = QGroupBox("Shot Type")
        row = QHBoxLayout(box)
        self._shot_combo = QComboBox()
        for label, key in [("Full Body", "full_body"), ("Waist Up", "waist_up"),
                            ("Medium", "medium"), ("Close-Up", "close_up")]:
            self._shot_combo.addItem(label, key)
        # Set default
        for i in range(self._shot_combo.count()):
            if self._shot_combo.itemData(i) == self._state.shot_type:
                self._shot_combo.setCurrentIndex(i)
                break
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        row.addWidget(self._shot_combo)
        return box

    # --- Virtual Switcher / Primary Focus settings ---

    def _build_switcher_section(self):
        # No section label — title is implicit from context
        self._switcher_box = QGroupBox("Virtual Switcher")
        layout = QVBoxLayout(self._switcher_box)

        # Trigger row — Primary Focus first, then switcher triggers
        trig_row = QHBoxLayout()
        trig_label = QLabel("Mode:")
        self._trig_group = QButtonGroup()
        triggers = [
            ("Disabled", "disabled"),
            ("Primary",  "primary"),
            ("Time",     "time"),
            ("Manual",   "manual"),
        ]
        # Determine initial checked state
        _valid_triggers = {t[1] for t in triggers}
        current_trigger = (
            "disabled" if not self._state.auto_follow_enabled
            else "primary" if self._state.tracking_mode == 'primary'
            else self._state.switch_trigger
                if self._state.switch_trigger in _valid_triggers else "time"
        )
        for label, key in triggers:
            rb = QRadioButton(label)
            rb.setProperty("trigger_key", key)
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
        manual_label = QLabel("Manual Switch:")
        layout.addWidget(manual_label)
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
        display_row.addWidget(self._display_combo)
        display_row.addStretch()
        layout.addLayout(display_row)

        btn_row = QHBoxLayout()
        btn_fullscreen = QPushButton("Open Fullscreen Output")
        btn_fullscreen.clicked.connect(self._open_fullscreen)
        btn_row.addWidget(btn_fullscreen)
        btn_diag = QPushButton("Open Diagnostics")
        btn_diag.clicked.connect(self._open_diagnostics)
        btn_row.addWidget(btn_diag)
        btn_people = QPushButton("Manage People")
        btn_people.clicked.connect(self._open_people)
        btn_row.addWidget(btn_people)
        layout.addLayout(btn_row)

        # Foreground exclusion zone slider
        excl_row = QHBoxLayout()
        excl_row.addWidget(QLabel("Audience Exclusion:"))
        self._excl_slider = QSlider(Qt.Horizontal)
        self._excl_slider.setRange(0, 100)
        self._excl_slider.setValue(int(self._state.foreground_exclusion_y * 100))
        self._excl_slider.setToolTip(
            "Ignore detections in the bottom N% of the frame (foreground audience filter).\n"
            "0 = disabled. Increase until stage-front audience members are no longer tracked."
        )
        self._excl_value_label = QLabel(f"{int(self._state.foreground_exclusion_y * 100)}%")
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
        # Update preview
        pix = QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(
            pix.scaled(self._preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        # Update fullscreen output if open
        if self._output_win.isVisible():
            self._output_win.update_frame(qimg)
        # Always feed the diagnostics panel so the log captures events even when hidden.
        # update_diagnostics() skips the table rebuild when the window is not visible.
        self._diag_win.update_diagnostics(meta)
        # Status bar
        self._status_label.setText(
            f"FPS: {meta['fps']:.1f}  |  "
            f"Detected: {meta['n_persons']} person(s)  |  "
            f"Active: {meta['active_id']}  |  "
            f"{self._audio_status_text}"
        )

    def _on_camera_info(self, info: str):
        self._cam_info_label.setText(info)

    def _on_persons_updated(self, person_ids: list):
        # Rebuild manual person buttons to match currently tracked IDs
        existing = set(self._person_buttons.keys())
        current = set(person_ids)

        for pid in existing - current:
            btn = self._person_buttons.pop(pid)
            self._persons_row.removeWidget(btn)
            btn.deleteLater()

        manual_active = self._state.switch_trigger == 'manual' and self._state.tracking_mode == 'switcher'
        for pid in current - existing:
            btn = QPushButton(pid.replace('person', 'P'))
            btn.setFixedWidth(40)
            btn.setVisible(manual_active)
            btn.clicked.connect(lambda checked, p=pid: self._manual_switch(p))
            self._person_buttons[pid] = btn
            self._persons_row.addWidget(btn)

    def _on_camera_changed(self, idx: int):
        cam_idx = self._cam_combo.itemData(idx)
        if cam_idx is not None:
            with QMutexLocker(self._state._lock):
                self._state.camera_index = cam_idx
                self._state.camera_change_requested = True

    def _on_shot_changed(self, idx: int):
        key = self._shot_combo.itemData(idx)
        with QMutexLocker(self._state._lock):
            self._state.shot_type = key

    def _on_trigger_changed(self, button):
        key = button.property("trigger_key")
        with QMutexLocker(self._state._lock):
            if key == 'disabled':
                self._state.auto_follow_enabled = False
            elif key == 'primary':
                self._state.auto_follow_enabled = True
                self._state.tracking_mode = 'primary'
            else:
                self._state.auto_follow_enabled = True
                self._state.tracking_mode = 'switcher'
                self._state.switch_trigger = key
        self._update_trigger_ui(key)

    def _update_trigger_ui(self, trigger_key: str):
        """Show/hide controls based on selected trigger."""
        show_interval = trigger_key == 'time'
        self._interval_label.setVisible(show_interval)
        self._interval_spin.setVisible(show_interval)
        manual_active = trigger_key == 'manual'
        for btn in self._person_buttons.values():
            btn.setVisible(manual_active)

    def _on_interval_changed(self, val: float):
        with QMutexLocker(self._state._lock):
            self._state.switch_interval = val

    def _on_switch_mode_changed(self, button):
        key = button.property("sm_key")
        with QMutexLocker(self._state._lock):
            self._state.switch_mode = key

    def _on_crossfade_changed(self, val: float):
        with QMutexLocker(self._state._lock):
            self._state.crossfade_duration = val

    def _manual_switch(self, person_id: str):
        with QMutexLocker(self._state._lock):
            self._state.manual_switch_id = person_id

    def _on_diagnostics_changed(self, state: int):
        with QMutexLocker(self._state._lock):
            self._state.show_diagnostics = bool(state)

    def _on_exclusion_changed(self, value: int):
        self._excl_value_label.setText(f"{value}%")
        with QMutexLocker(self._state._lock):
            self._state.foreground_exclusion_y = value / 100.0

    def _on_max_persons_changed(self, value: int):
        with QMutexLocker(self._state._lock):
            self._state.max_persons = value

    def _open_fullscreen(self):
        screen_index = self._display_combo.currentData()
        screens = QApplication.screens()
        if 0 <= screen_index < len(screens):
            geo = screens[screen_index].geometry()
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
        with QMutexLocker(self._state._lock):
            self._state.music_mode = music_mode
            self._state.audio_music_score = music_score
            self._state.audio_speech_score = speech_score
            self._state.audio_enabled = enabled_now
        mode_text = "MUSIC" if music_mode else "Speech"
        self._audio_status_text = (
            f"Audio: {mode_text}  (music={music_score:.2f}, speech={speech_score:.2f})"
            if enabled_now else "Audio: off"
        )
        # Reflect enabled state in the main-panel checkbox so external
        # changes (e.g. toggling from the People window, or a device error)
        # are visible without the user needing to click anywhere.
        cb = getattr(self, '_audio_enable_cb', None)
        if cb is not None and cb.isChecked() != enabled_now:
            cb.blockSignals(True)
            cb.setChecked(enabled_now)
            cb.blockSignals(False)

    def _on_speaker_detected(self, profile_id: str, name: str, score: float):
        import time
        # Queue a voice boost for the VideoThread to apply on the next frame.
        with QMutexLocker(self._state._lock):
            self._state.pending_voice_boost = (
                profile_id, VOICE_PRIORITY_BOOST, SPEAKER_BOOST_HOLD_S
            )
            self._state.audio_speaker_name = name
            self._state.audio_speaker_score = float(score)
            self._state.audio_speaker_expires_at = time.monotonic() + SPEAKER_BOOST_HOLD_S
            self._state.audio_enabled = self._audio_thread.is_capturing
        self._audio_status_text = f"Audio: ● {name}  ({score:.2f})"

    def _on_audio_error(self, msg: str):
        self._audio_status_text = f"Audio error: {msg}"

    def _open_people(self):
        if self._people_win is None:
            from people_ui import PeopleWindow
            self._people_win = PeopleWindow(
                self._profile_store,
                frame_grabber=self._video_thread.grab_latest_frame,
                audio_thread=self._audio_thread,
                parent=self,
            )
            # Tell the video thread to rebuild its embedding index whenever
            # the profile set changes (add/edit/delete/re-embed).
            self._people_win.profiles_changed.connect(self._video_thread.reindex_profiles)
            # Voice profile changes invalidate the SpeakerRecognizer's index.
            self._people_win.voice_profiles_changed.connect(
                self._audio_thread.reindex_voice_profiles
            )
        self._people_win.show()
        self._people_win.raise_()
        self._people_win.activateWindow()

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._video_thread.stop()
        self._audio_thread.stop()
        self._output_win.close()
        self._diag_win.close()
        if self._people_win is not None:
            self._people_win.close()
        event.accept()
