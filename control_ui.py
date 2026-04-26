"""PyQt5 control panel with integrated video thread and fullscreen output window."""

import sys
import time
import threading
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QDoubleSpinBox, QSlider, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont

import config
from pose_detector import PoseDetector
from tracker import PersonTracker
from framing_engine import FramingEngine
from smoothing import PTZSmoother
from switcher import VirtualSwitcher


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
        self.switch_trigger: str = config.SWITCH_TRIGGER # 'time' | 'activity' | 'manual'
        self.switch_interval: float = config.SWITCH_INTERVAL
        self.crossfade_duration: float = config.CROSSFADE_DURATION
        self.manual_switch_id: str | None = None         # set by UI, consumed by video thread
        self.camera_change_requested: bool = False

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

        # Sticky primary: only change the tracked person when the candidate is
        # significantly more foreground, preventing rapid toggling between two
        # similarly-sized people in frame.
        self._primary_id: str | None = None

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

        persons = self._tracker.update(detections, frame.shape)

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

        if mode == 'primary' or not persons:
            output_frame = self._render_primary(frame, persons, shot_type)
            active_id = persons[0].id if persons else 'none'
        else:
            output_frame, active_id = self._render_switcher(frame, persons, shot_type)

        elapsed = time.monotonic() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        meta = {
            'fps': fps,
            'n_persons': len(persons),
            'active_id': active_id,
        }
        return _bgr_to_qimage(output_frame), meta

    # ------------------------------------------------------------------
    # Render modes
    # ------------------------------------------------------------------

    def _render_primary(self, frame, persons, shot_type):
        """Track the most-foreground person with a single virtual camera.

        Uses hysteresis on primary selection: a new candidate must be at least
        30% larger (more foreground) than the current primary before we switch,
        preventing rapid toggling between similarly-sized people.
        """
        if not persons:
            tx, ty, tz = self._framing._default_target()
            sx, sy, sz = self._smoother.update('primary', tx, ty, tz)
            return self._framing.apply_crop(frame, sx, sy, sz)

        current_ids = {p.id for p in persons}
        candidate = persons[0]  # highest foreground_score

        if self._primary_id not in current_ids:
            # Current primary left the frame — switch immediately
            self._primary_id = candidate.id
        elif candidate.id != self._primary_id:
            # Current primary still present; only switch if candidate is meaningfully larger
            current_p = next(p for p in persons if p.id == self._primary_id)
            if candidate.foreground_score > current_p.foreground_score * 1.30:
                self._primary_id = candidate.id

        primary = next((p for p in persons if p.id == self._primary_id), candidate)
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
        layout.addWidget(self._build_mode_section())
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

    # --- Mode ---

    def _build_mode_section(self):
        box = QGroupBox("Tracking Mode")
        row = QHBoxLayout(box)
        self._mode_group = QButtonGroup()
        rb_primary = QRadioButton("Primary Focus")
        rb_switcher = QRadioButton("Virtual Switcher")
        rb_primary.setChecked(self._state.tracking_mode == 'primary')
        rb_switcher.setChecked(self._state.tracking_mode == 'switcher')
        self._mode_group.addButton(rb_primary, 0)
        self._mode_group.addButton(rb_switcher, 1)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        row.addWidget(rb_primary)
        row.addWidget(rb_switcher)
        return box

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

    # --- Switcher settings ---

    def _build_switcher_section(self):
        self._switcher_box = QGroupBox("Virtual Switcher Settings")
        layout = QVBoxLayout(self._switcher_box)

        # Trigger
        trig_row = QHBoxLayout()
        trig_label = QLabel("Trigger:")
        self._trig_group = QButtonGroup()
        for label, key in [("Time", "time"), ("Activity", "activity"), ("Manual", "manual")]:
            rb = QRadioButton(label)
            rb.setProperty("trigger_key", key)
            rb.setChecked(key == self._state.switch_trigger)
            self._trig_group.addButton(rb)
            trig_row.addWidget(rb)
        self._trig_group.buttonClicked.connect(self._on_trigger_changed)
        trig_row.insertWidget(0, trig_label)
        layout.addLayout(trig_row)

        # Interval (time trigger)
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

        # Switch mode
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

        self._switcher_box.setVisible(self._state.tracking_mode == 'switcher')
        return self._switcher_box

    # --- Output ---

    def _build_output_section(self):
        box = QGroupBox("Output")
        row = QHBoxLayout(box)
        btn_fullscreen = QPushButton("Open Fullscreen Output")
        btn_fullscreen.clicked.connect(self._open_fullscreen)
        row.addWidget(btn_fullscreen)
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

        for pid in current - existing:
            btn = QPushButton(pid.replace('person', 'P'))
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda checked, p=pid: self._manual_switch(p))
            self._person_buttons[pid] = btn
            self._persons_row.addWidget(btn)

    def _on_camera_changed(self, idx: int):
        cam_idx = self._cam_combo.itemData(idx)
        if cam_idx is not None:
            with QMutexLocker(self._state._lock):
                self._state.camera_index = cam_idx
                self._state.camera_change_requested = True

    def _on_mode_changed(self, button):
        mode = 'primary' if self._mode_group.id(button) == 0 else 'switcher'
        with QMutexLocker(self._state._lock):
            self._state.tracking_mode = mode
        self._switcher_box.setVisible(mode == 'switcher')

    def _on_shot_changed(self, idx: int):
        key = self._shot_combo.itemData(idx)
        with QMutexLocker(self._state._lock):
            self._state.shot_type = key

    def _on_trigger_changed(self, button):
        key = button.property("trigger_key")
        with QMutexLocker(self._state._lock):
            self._state.switch_trigger = key

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

    def _open_fullscreen(self):
        self._output_win.showFullScreen()
        self._output_win.raise_()

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._video_thread.stop()
        self._output_win.close()
        event.accept()
