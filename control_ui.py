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
            }

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

    frame_ready = pyqtSignal(QImage, dict)   # (frame, metadata)
    camera_info = pyqtSignal(str)             # e.g. "1920x1080 @ 30fps"
    persons_updated = pyqtSignal(list)        # list of person IDs currently tracked

    def __init__(self, state: AppState, parent=None):
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

        self._primary_id: str | None = None
        # Transition state for Primary Focus mode (mirrors switcher logic)
        self._primary_pending_id: str | None = None
        self._primary_pretraveling: bool = False
        self._primary_pretravel_start: float = 0.0
        self._primary_fade_start: float | None = None
        self._primary_last_switch_time: float = time.monotonic()  # enforces 1s minimum dwell
        # Wide-shot state: entered when no subjects detected
        self._wide_shot_active: bool = False
        self._wide_shot_fade_start: float | None = None   # None = cut already applied
        self._wide_shot_reacquire_at: float = 0.0         # earliest time to leave wide-shot
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

        mode = settings['tracking_mode']
        shot_type = settings['shot_type']
        diagnostics = settings.get('show_diagnostics', False)
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

            elapsed = time.monotonic() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            return _bgr_to_qimage(output_frame), {
                'fps': fps, 'n_persons': len(persons), 'active_id': 'disabled'
            }

        # Auto-follow enabled — clear disabled transition state so next disable starts fresh
        self._disabled_transitioning = False
        self._disabled_transition_start = None

        if mode == 'primary' or not persons:
            output_frame = self._render_primary(frame, persons, shot_type, diagnostics, settings)
            active_id = self._primary_id if self._primary_id else (persons[0].id if persons else 'none')
        else:
            output_frame, active_id = self._render_switcher(frame, persons, shot_type, diagnostics)

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
            })

        dwell_elapsed = time.monotonic() - self._primary_last_switch_time
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
        }
        return _bgr_to_qimage(output_frame), meta

    # ------------------------------------------------------------------
    # Diagnostic overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_frame(frame: np.ndarray, persons, primary_id: str) -> np.ndarray:
        """Draw skeleton and torso position box on the input frame.

        Primary person: green.  All others: red.
        Operates in input-pixel space so the annotation survives any crop/zoom.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        for person in persons:
            is_primary = person.id == primary_id
            skel_color = (0, 220, 0) if is_primary else (0, 0, 220)
            box_color  = (0, 220, 0) if is_primary else (0, 0, 220)
            kps = person.keypoints  # (17, 4) — [x_norm, y_norm, 0, conf]

            pts: dict[int, tuple[int, int]] = {}
            for idx, kp in enumerate(kps):
                if kp[3] > config.CONFIDENCE_THRESHOLD:
                    pts[idx] = (int(kp[0] * w), int(kp[1] * h))

            for a, b in _SKELETON:
                if a in pts and b in pts:
                    cv2.line(out, pts[a], pts[b], skel_color, 2)

            # --- Torso position box ---
            # Use shoulder/hip keypoints to define the torso rect; fall back to bbox.
            torso_kp_indices = [5, 6, 11, 12]  # L-shoulder, R-shoulder, L-hip, R-hip
            torso_pts = [pts[i] for i in torso_kp_indices if i in pts]

            if len(torso_pts) >= 2:
                txs = [p[0] for p in torso_pts]
                tys = [p[1] for p in torso_pts]
                tx1, ty1, tx2, ty2 = min(txs), min(tys), max(txs), max(tys)
            else:
                # Fallback: upper half of bbox
                bx1, by1, bx2, by2 = person.bbox
                tx1, ty1, tx2 = bx1, by1, bx2
                ty2 = by1 + (by2 - by1) // 2

            # Expand the torso rect slightly for readability
            pad = 8
            tx1 = max(0, tx1 - pad)
            ty1 = max(0, ty1 - pad)
            tx2 = min(w - 1, tx2 + pad)
            ty2 = min(h - 1, ty2 + pad)

            # Semi-transparent fill
            overlay = out.copy()
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), box_color, -1)
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), box_color, 2)

            # X, Y, Z axis values
            # X: normalised horizontal center of bbox (-1 = left, +1 = right)
            bx1_, by1_, bx2_, by2_ = person.bbox
            px_norm = ((bx1_ + bx2_) / 2.0 / max(w, 1) - 0.5) * 2.0
            # Y: normalised vertical center (−1 = top, +1 = bottom)
            py_norm = ((by1_ + by2_) / 2.0 / max(h, 1) - 0.5) * 2.0
            # Z: foreground proxy — bbox area relative to frame (0→1, bigger = closer)
            pz = person.foreground_score  # already computed in tracker

            label_lines = [
                f"X:{px_norm:+.2f}",
                f"Y:{py_norm:+.2f}",
                f"Z:{pz:.3f}",
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.45, 1
            line_h = 14
            text_x = tx1 + 4
            text_y = ty1 + line_h
            for ln in label_lines:
                cv2.putText(out, ln, (text_x, text_y), font, font_scale,
                            (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(out, ln, (text_x, text_y), font, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)
                text_y += line_h

        return out

    # ------------------------------------------------------------------
    # Render modes
    # ------------------------------------------------------------------

    def _render_primary(self, frame, persons, shot_type, diagnostics: bool = False,
                        settings: dict | None = None):
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

        # ----------------------------------------------------------------
        # Wide-shot fade-in/hold (no subjects detected, or hold period active)
        # ----------------------------------------------------------------
        wide_target_x, wide_target_y, wide_target_z = self._framing._default_target()

        entering_wide = not persons and not self._wide_shot_active
        if entering_wide:
            self._wide_shot_active = True
            self._primary_id = None
            self._primary_pending_id = None
            self._primary_pretraveling = False
            self._primary_fade_start = None
            if sw_mode == 'crossfade':
                self._wide_shot_fade_start = now
            else:
                self._wide_shot_fade_start = None  # immediate cut

        if self._wide_shot_active:
            # Set earliest re-acquire time whenever we (re-)enter wide-shot
            if entering_wide:
                self._wide_shot_reacquire_at = now + xfade_dur

            # Crossfade into the wide shot
            if self._wide_shot_fade_start is not None:
                elapsed = now - self._wide_shot_fade_start
                t = min(1.0, elapsed / max(xfade_dur, 0.001))
                # Advance the wide-shot smoother
                wx, wy, wz = self._smoother.update('__wide__', wide_target_x, wide_target_y, wide_target_z)
                frame_wide = self._framing.apply_crop(frame, wx, wy, wz)
                if t >= 1.0:
                    self._wide_shot_fade_start = None  # fade complete
                    if persons:
                        # Subject appeared during fade — check hold
                        if now >= self._wide_shot_reacquire_at:
                            self._wide_shot_active = False
                    return frame_wide
                # Still fading: blend from wherever the primary smoother is
                px, py, pz = self._smoother.update('primary', wide_target_x, wide_target_y, wide_target_z)
                frame_from = self._framing.apply_crop(frame, px, py, pz)
                return cv2.addWeighted(frame_from, 1.0 - t, frame_wide, t, 0)

            # Wide-shot is fully active (fade done)
            wx, wy, wz = self._smoother.update('__wide__', wide_target_x, wide_target_y, wide_target_z)
            frame_wide = self._framing.apply_crop(frame, wx, wy, wz)

            if persons and now >= self._wide_shot_reacquire_at:
                # Hold period elapsed — leave wide-shot and fall through to normal tracking
                self._wide_shot_active = False
            else:
                # Still in hold or no subjects; keep extending the reacquire deadline
                if not persons:
                    self._wide_shot_reacquire_at = now + xfade_dur
                return frame_wide

        # ----------------------------------------------------------------
        # Normal subject tracking
        # ----------------------------------------------------------------
        current_ids = {p.id for p in persons}

        # Initialise or recover if primary truly left the frame.
        # Only stamp _primary_last_switch_time when we are forced to pick a
        # different person — not on every frame the primary is briefly unseen.
        if self._primary_id not in current_ids:
            new_primary = persons[0].id
            self._primary_id = new_primary
            self._primary_pending_id = None
            self._primary_pretraveling = False
            self._primary_fade_start = None
            # Reset dwell only when forced to a genuinely different person,
            # so the 1s gate fires correctly after recovery.
            self._primary_last_switch_time = now

        current_p = next((p for p in persons if p.id == self._primary_id), persons[0])

        # --- Candidate selection: nearest (by fg_score) or significantly more active ---
        # persons[] is sorted foreground_score desc (nearest = persons[0])
        # fg_ratio thresholds use hysteresis: a higher bar to initiate a switch than
        # was required to arrive at the current primary, preventing oscillation when two
        # people have similar sizes.  Both fg_score and activity_score are EMA-smoothed
        # in the tracker so single-frame noise doesn't trigger a switch.
        nearest = persons[0]
        if nearest.id != self._primary_id and self._primary_pending_id is None:
            if now - self._primary_last_switch_time >= 3.0:
                fg_ratio = nearest.foreground_score / max(current_p.foreground_score, 1e-6)
                cand_act = nearest.activity_score
                curr_act = current_p.activity_score
                # Proximity switch: candidate must be substantially closer (50% more area).
                fg_wins = fg_ratio >= 1.5
                # Activity switch: candidate must clear a noise floor AND be 2× more active.
                activity_wins = cand_act >= 5.0 and cand_act > curr_act * 2.0
                if fg_wins or activity_wins:
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
            src = self._annotate_frame(frame, persons, self._primary_id) if diagnostics else frame
            primary = next((p for p in persons if p.id == self._primary_id), persons[0])
            tx, ty, tz = self._framing.calculate_target(primary, shot_type)
            cx = (primary.bbox[0] + primary.bbox[2]) / 2.0
            cw = config.OUTPUT_WIDTH / tz
            sx, sy, sz = self._smoother.update('primary', tx, ty, tz,
                                               person_center_x=cx, crop_width=cw)
            return self._framing.apply_crop(src, sx, sy, sz)

        # --- Phase 2: crossfade to new primary ---
        if self._primary_pending_id is not None and self._primary_fade_start is not None:
            elapsed = now - self._primary_fade_start
            t = min(1.0, elapsed / max(xfade_dur, 0.001))

            src = self._annotate_frame(frame, persons, self._primary_id) if diagnostics else frame

            primary = next((p for p in persons if p.id == self._primary_id), persons[0])
            atx, aty, atz = self._framing.calculate_target(primary, shot_type)
            cx_a = (primary.bbox[0] + primary.bbox[2]) / 2.0
            cw_a = config.OUTPUT_WIDTH / atz
            ax, ay, az = self._smoother.update('primary', atx, aty, atz,
                                               person_center_x=cx_a, crop_width=cw_a)
            frame_active = self._framing.apply_crop(src, ax, ay, az)

            pending_person = next((p for p in persons if p.id == self._primary_pending_id), None)
            if pending_person:
                ptx, pty, ptz = self._framing.calculate_target(pending_person, shot_type)
                cx_p = (pending_person.bbox[0] + pending_person.bbox[2]) / 2.0
                cw_p = config.OUTPUT_WIDTH / ptz
                px, py, pz = self._smoother.update(self._primary_pending_id, ptx, pty, ptz,
                                                   person_center_x=cx_p, crop_width=cw_p)
                frame_pending = self._framing.apply_crop(src, px, py, pz)
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
        src = self._annotate_frame(frame, persons, self._primary_id) if diagnostics else frame
        return self._framing.apply_crop(src, sx, sy, sz)

    def _render_switcher(self, frame, persons, shot_type, diagnostics: bool = False):
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

        # Annotate input frame once; both active and pending crops share the same overlay
        src = self._annotate_frame(frame, persons, active_id) if diagnostics else frame

        # Render active person
        active_person = next((p for p in persons if p.id == active_id), persons[0])
        atx, aty, atz = self._framing.calculate_target(active_person, shot_type)
        cx_a = (active_person.bbox[0] + active_person.bbox[2]) / 2.0
        cw_a = config.OUTPUT_WIDTH / atz
        ax, ay, az = self._smoother.update(
            active_id, atx, aty, atz,
            person_center_x=cx_a, crop_width=cw_a
        )
        frame_active = self._framing.apply_crop(src, ax, ay, az)

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
                frame_pending = self._framing.apply_crop(src, px, py, pz)
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
        self.setMinimumSize(540, 460)
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

        # ── State summary row ────────────────────────────────────────────
        state_box = QGroupBox("Tracking State")
        state_grid = QHBoxLayout(state_box)

        self._lbl_mode     = self._make_field("Mode", state_grid)
        self._lbl_active   = self._make_field("Active", state_grid)
        self._lbl_pending  = self._make_field("Pending", state_grid)
        self._lbl_phase    = self._make_field("Phase", state_grid)
        self._lbl_dwell    = self._make_field("Dwell", state_grid)
        self._lbl_fps      = self._make_field("FPS", state_grid)
        layout.addWidget(state_box)

        # ── Per-person table ─────────────────────────────────────────────
        persons_box = QGroupBox("Tracked Persons")
        persons_layout = QVBoxLayout(persons_box)
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["ID", "FG(sm)", "Act(sm)", "FG Ratio", "Act Ratio", "Unseen", "Center X,Y", "Zoom(sm)"]
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

            # Red = gate closed (can't switch yet), green = gate open
            if not gate_open:
                self._lbl_dwell.setStyleSheet("color: #e06c75; font-weight: bold;")
            else:
                self._lbl_dwell.setStyleSheet("color: #98c379; font-weight: bold;")

            # ── Per-person table ─────────────────────────────────────────
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

                cells = [
                    pid,
                    f"{fg:.4f}",
                    f"{act:.2f}",
                    f"{'!' if fg_trigger  else ''}{fg_ratio:.2f}",
                    f"{'!' if act_trigger else ''}{act_ratio:.2f}",
                    str(unseen),
                    f"{cx:.0f},{cy:.0f}",
                    f"{sz:.2f}" if sz is not None else "—",
                ]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    if is_active:
                        item.setBackground(QColor(40, 80, 40))
                    elif is_pending:
                        item.setBackground(QColor(80, 60, 20))
                    elif fg_trigger or act_trigger:
                        item.setBackground(QColor(80, 40, 40))
                    self._table.setItem(row, col, item)

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
                reason = f"  [{', '.join(why) if why else 'recovery'}]"
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
        self._output_win = OutputWindow()
        self._diag_win = DiagnosticsWindow(show_overlays=self._state.show_diagnostics)
        self._diag_win.overlays_changed.connect(self._on_diagnostics_changed)
        self._video_thread = VideoThread(self._state)
        self._video_thread.frame_ready.connect(self._on_frame)
        self._video_thread.camera_info.connect(self._on_camera_info)
        self._video_thread.persons_updated.connect(self._on_persons_updated)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(320, 180)
        self._preview_label.setStyleSheet("background: black;")

        self._build_ui()
        self._video_thread.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)

        layout.addWidget(self._build_camera_section())
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
            f"Active: {meta['active_id']}"
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

    def closeEvent(self, event):
        self._video_thread.stop()
        self._output_win.close()
        self._diag_win.close()
        event.accept()
