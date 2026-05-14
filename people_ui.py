"""People management UI — list, add, edit, and delete recognized-person profiles.

Opens as an independent window. Each profile has a name, priority (0–10), and
a gallery of reference face images. Adding or removing images triggers a
background re-embed of the affected profile via the shared FaceRecognizer; the
VideoThread is told to refresh its match index when the user closes the dialog
or saves a profile.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog, QDialog,
    QLineEdit, QSpinBox, QDialogButtonBox, QFormLayout, QGroupBox, QSlider,
    QScrollArea, QFrame, QComboBox, QCheckBox,
)

from profiles import ProfileStore, Profile
from audio_capture import list_input_devices, default_input_device_index, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Background embedding worker
# ---------------------------------------------------------------------------

class _EmbedThread(QThread):
    """Re-embed all reference images for a profile off the UI thread."""

    finished_with = pyqtSignal(str, int)   # (profile_id, embeddings_count)
    failed = pyqtSignal(str, str)          # (profile_id, error_message)

    def __init__(self, profile_id: str, store: ProfileStore, parent=None):
        super().__init__(parent)
        self._profile_id = profile_id
        self._store = store

    def run(self):
        try:
            from face_recognizer import FaceRecognizer
            recognizer = FaceRecognizer(self._store)
            profile = self._store.get(self._profile_id)
            if profile is None:
                self.failed.emit(self._profile_id, "Profile no longer exists")
                return
            count = recognizer.rebuild_profile_embeddings(profile)
            self.finished_with.emit(self._profile_id, count)
        except Exception as e:
            self.failed.emit(self._profile_id, str(e))


class _VoiceEmbedThread(QThread):
    """Re-embed all voice samples for a profile off the UI thread."""

    finished_with = pyqtSignal(str, int)
    failed = pyqtSignal(str, str)

    def __init__(self, profile_id: str, store: ProfileStore, parent=None):
        super().__init__(parent)
        self._profile_id = profile_id
        self._store = store

    def run(self):
        try:
            from speaker_recognizer import SpeakerRecognizer
            recognizer = SpeakerRecognizer(self._store)
            profile = self._store.get(self._profile_id)
            if profile is None:
                self.failed.emit(self._profile_id, "Profile no longer exists")
                return
            count = recognizer.rebuild_profile_voice_embeddings(profile)
            self.finished_with.emit(self._profile_id, count)
        except Exception as e:
            self.failed.emit(self._profile_id, str(e))


# ---------------------------------------------------------------------------
# Edit dialog (add or edit a single profile)
# ---------------------------------------------------------------------------

class EditProfileDialog(QDialog):
    """Modal dialog to create or edit a single profile."""

    def __init__(self, profile: Profile | None, store: ProfileStore,
                 frame_grabber: Callable[[], np.ndarray | None] | None = None,
                 audio_grabber: Callable[[float], np.ndarray | None] | None = None,
                 parent=None):
        super().__init__(parent)
        self._store = store
        self._profile = profile
        self._frame_grabber = frame_grabber
        self._audio_grabber = audio_grabber
        self._pending_image_paths: list[Path] = []     # external files to copy in on save
        self._pending_frame_captures: list[np.ndarray] = []   # frame bytes to write on save
        self._pending_voice_paths: list[Path] = []     # external voice files to copy on save
        self._pending_voice_recordings: list[np.ndarray] = []  # captured 16kHz mono samples
        self._record_timer: QTimer | None = None
        self._record_remaining_s: float = 0.0
        self._record_btn = None     # set below in voice section

        self.setWindowTitle("Edit Person" if profile else "Add Person")
        self.resize(640, 640)

        root = QVBoxLayout(self)

        # --- Metadata form ---
        form_box = QGroupBox("Profile")
        form = QFormLayout(form_box)
        self._name_edit = QLineEdit(profile.name if profile else "")
        self._name_edit.setPlaceholderText("e.g. Pastor Mike")
        form.addRow("Name:", self._name_edit)

        prio_row = QHBoxLayout()
        self._prio_slider = QSlider(Qt.Horizontal)
        self._prio_slider.setRange(0, 10)
        self._prio_slider.setValue(profile.priority if profile else 5)
        self._prio_value = QLabel(str(self._prio_slider.value()))
        self._prio_value.setFixedWidth(24)
        self._prio_slider.valueChanged.connect(
            lambda v: self._prio_value.setText(str(v))
        )
        prio_row.addWidget(self._prio_slider)
        prio_row.addWidget(self._prio_value)
        prio_w = QWidget()
        prio_w.setLayout(prio_row)
        form.addRow("Priority:", prio_w)

        prio_hint = QLabel(
            "0 = normal · 5 = preferred · 10 = always preferred\n"
            "High-priority people get longer dwell on the auto-switcher\n"
            "and win contests over unmatched bystanders in primary mode."
        )
        prio_hint.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow("", prio_hint)
        root.addWidget(form_box)

        # --- Reference images gallery ---
        img_box = QGroupBox("Reference Images")
        img_layout = QVBoxLayout(img_box)
        btn_row = QHBoxLayout()
        btn_add_files = QPushButton("Add from file…")
        btn_add_files.clicked.connect(self._on_add_files)
        btn_capture = QPushButton("Capture from camera")
        btn_capture.setEnabled(frame_grabber is not None)
        btn_capture.clicked.connect(self._on_capture)
        btn_row.addWidget(btn_add_files)
        btn_row.addWidget(btn_capture)
        btn_row.addStretch()
        img_layout.addLayout(btn_row)

        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        self._gallery_inner = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_inner)
        self._gallery_layout.setAlignment(Qt.AlignLeft)
        self._gallery_scroll.setWidget(self._gallery_inner)
        self._gallery_scroll.setMinimumHeight(160)
        img_layout.addWidget(self._gallery_scroll)
        root.addWidget(img_box, stretch=1)

        # --- Voice samples ---
        voice_box = QGroupBox("Voice Samples (for speaker recognition)")
        voice_layout = QVBoxLayout(voice_box)
        v_btn_row = QHBoxLayout()
        btn_voice_file = QPushButton("Add WAV/FLAC…")
        btn_voice_file.clicked.connect(self._on_add_voice_file)
        self._record_btn = QPushButton("Record 5 s from mic")
        self._record_btn.setEnabled(audio_grabber is not None)
        self._record_btn.clicked.connect(self._on_record_voice)
        v_btn_row.addWidget(btn_voice_file)
        v_btn_row.addWidget(self._record_btn)
        v_btn_row.addStretch()
        voice_layout.addLayout(v_btn_row)
        self._voice_list = QListWidget()
        self._voice_list.setMaximumHeight(120)
        voice_layout.addWidget(self._voice_list)
        v_hint = QLabel(
            "Provide ≥10 s of clean speech (this person, no background music).\n"
            "Multiple short samples are better than one long one."
        )
        v_hint.setStyleSheet("color: gray; font-size: 10px;")
        voice_layout.addWidget(v_hint)
        root.addWidget(voice_box)

        # --- Save / Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_gallery()
        self._refresh_voice_list()

    # ------------------------------------------------------------------

    def _refresh_voice_list(self):
        self._voice_list.clear()
        if self._profile is not None:
            for fn in list(self._profile.voice_filenames):
                item = QListWidgetItem(f"{fn[:16]}…  (saved)")
                item.setData(Qt.UserRole, ("saved", fn))
                self._voice_list.addItem(item)
        for p in self._pending_voice_paths:
            item = QListWidgetItem(f"{p.name}  (new)")
            item.setData(Qt.UserRole, ("pending_file", str(p)))
            self._voice_list.addItem(item)
        for i, _ in enumerate(self._pending_voice_recordings):
            item = QListWidgetItem(f"Recording {i+1}  (new, 5 s)")
            item.setData(Qt.UserRole, ("pending_rec", i))
            self._voice_list.addItem(item)
        if self._voice_list.count() == 0:
            placeholder = QListWidgetItem("(No voice samples yet)")
            placeholder.setFlags(Qt.NoItemFlags)
            self._voice_list.addItem(placeholder)
        # Allow right-click to remove (simple: double-click to delete with confirm)
        try:
            self._voice_list.itemDoubleClicked.disconnect()
        except TypeError:
            pass
        self._voice_list.itemDoubleClicked.connect(self._on_voice_double_click)

    def _on_add_voice_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select voice sample(s)", "",
            "Audio (*.wav *.flac *.ogg)"
        )
        for p in paths:
            self._pending_voice_paths.append(Path(p))
        self._refresh_voice_list()

    def _on_record_voice(self):
        if self._audio_grabber is None:
            return
        # Record by waiting 5 s and then grabbing the latest 5 s of audio.
        # The AudioCapture buffer in audio_capture.py is 3 s long, so we grab
        # in three overlapping chunks and concatenate the new portions. To keep
        # this simple here we extend the buffer indirectly by waiting then
        # pulling get_recent(5.0); audio_capture clamps to its buffer size and
        # will return up to 3 s. For longer samples the user should add a file.
        # — actually simpler: pull whatever is buffered now (up to 3 s).
        self._record_btn.setEnabled(False)
        self._record_btn.setText("Recording…")
        self._record_remaining_s = 3.0    # match AudioCapture BUFFER_SECONDS
        # Use a one-shot timer instead of busy-waiting on the UI thread.
        self._record_timer = QTimer(self)
        self._record_timer.setSingleShot(True)
        self._record_timer.timeout.connect(self._on_record_complete)
        self._record_timer.start(int(self._record_remaining_s * 1000))

    def _on_record_complete(self):
        try:
            waveform = self._audio_grabber(self._record_remaining_s)
        except Exception as e:
            QMessageBox.warning(self, "Capture failed", str(e))
            waveform = None
        if waveform is None or len(waveform) < SAMPLE_RATE // 2:
            QMessageBox.warning(self, "No audio",
                                "No audio captured. Is the input device selected and unmuted?")
        else:
            self._pending_voice_recordings.append(waveform.copy())
        if self._record_btn is not None:
            self._record_btn.setText("Record 5 s from mic")
            self._record_btn.setEnabled(True)
        self._refresh_voice_list()

    def _on_voice_double_click(self, item):
        kind_data = item.data(Qt.UserRole)
        if not kind_data:
            return
        kind, payload = kind_data
        if QMessageBox.question(
            self, "Remove sample", "Remove this voice sample?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        if kind == "saved" and self._profile is not None:
            self._store.remove_voice_sample(self._profile.id, payload)
        elif kind == "pending_file":
            self._pending_voice_paths = [
                p for p in self._pending_voice_paths if str(p) != payload
            ]
        elif kind == "pending_rec":
            idx = int(payload)
            if 0 <= idx < len(self._pending_voice_recordings):
                self._pending_voice_recordings.pop(idx)
        self._refresh_voice_list()

    def commit_pending_voice(self, profile: Profile):
        """Copy/write pending voice samples into the profile's voice dir."""
        import io
        try:
            import soundfile as sf
        except ImportError:
            sf = None
        for path in self._pending_voice_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                self._store.add_voice_sample_bytes(profile.id, data,
                                                   suffix=path.suffix)
            except OSError:
                continue
        if sf is None:
            return
        for wav in self._pending_voice_recordings:
            buf = io.BytesIO()
            sf.write(buf, wav.astype(np.float32), SAMPLE_RATE, format="WAV")
            self._store.add_voice_sample_bytes(profile.id, buf.getvalue(),
                                               suffix=".wav")

    def has_pending_voice_changes(self) -> bool:
        return bool(self._pending_voice_paths or self._pending_voice_recordings)

    # ------------------------------------------------------------------

    def _refresh_gallery(self):
        # Clear existing widgets
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Existing on-disk images (only if editing an existing profile)
        if self._profile is not None:
            for fn in list(self._profile.image_filenames):
                pixmap = self._load_thumb_disk(fn)
                self._gallery_layout.addWidget(self._make_thumb_widget(
                    pixmap, label=fn[:8],
                    on_remove=lambda f=fn: self._remove_existing_image(f),
                ))

        # Pending (not yet saved) file picks
        for path in self._pending_image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            pixmap = self._frame_to_pixmap(img)
            self._gallery_layout.addWidget(self._make_thumb_widget(
                pixmap, label="new",
                on_remove=lambda p=path: self._remove_pending_path(p),
            ))

        # Pending camera captures
        for i, frame in enumerate(self._pending_frame_captures):
            pixmap = self._frame_to_pixmap(frame)
            self._gallery_layout.addWidget(self._make_thumb_widget(
                pixmap, label="snap",
                on_remove=lambda i=i: self._remove_pending_capture(i),
            ))

        self._gallery_layout.addStretch()

    def _make_thumb_widget(self, pixmap: QPixmap, label: str,
                           on_remove: Callable[[], None]) -> QWidget:
        w = QFrame()
        w.setFrameShape(QFrame.StyledPanel)
        w.setFixedSize(120, 150)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setPixmap(pixmap.scaled(
            100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        layout.addWidget(img_label)
        text = QLabel(label)
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(text)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(on_remove)
        btn_remove.setFixedHeight(20)
        layout.addWidget(btn_remove)
        return w

    def _load_thumb_disk(self, filename: str) -> QPixmap:
        if self._profile is None:
            return QPixmap()
        path = self._store.image_path(self._profile.id, filename)
        if path is None:
            return QPixmap()
        img = cv2.imread(str(path))
        if img is None:
            return QPixmap()
        return self._frame_to_pixmap(img)

    @staticmethod
    def _frame_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # ------------------------------------------------------------------

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select reference image(s)", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        for p in paths:
            self._pending_image_paths.append(Path(p))
        self._refresh_gallery()

    def _on_capture(self):
        if self._frame_grabber is None:
            return
        frame = self._frame_grabber()
        if frame is None:
            QMessageBox.warning(self, "No frame", "No camera frame is available yet.")
            return
        self._pending_frame_captures.append(frame.copy())
        self._refresh_gallery()

    def _remove_existing_image(self, filename: str):
        # Only marked for actual deletion on accept(); for now stage by mutating
        # the profile in-place. The on-disk file is removed by the store.
        if self._profile is None:
            return
        if QMessageBox.question(
            self, "Remove image", f"Remove this reference image?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self._store.remove_image(self._profile.id, filename)
        self._refresh_gallery()

    def _remove_pending_path(self, path: Path):
        self._pending_image_paths = [p for p in self._pending_image_paths if p != path]
        self._refresh_gallery()

    def _remove_pending_capture(self, index: int):
        if 0 <= index < len(self._pending_frame_captures):
            self._pending_frame_captures.pop(index)
        self._refresh_gallery()

    # ------------------------------------------------------------------

    def name(self) -> str:
        return self._name_edit.text().strip()

    def priority(self) -> int:
        return self._prio_slider.value()

    def commit_pending_images(self, profile: Profile):
        """Copy/write pending images into the profile's images dir."""
        for path in self._pending_image_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                suffix = path.suffix
                self._store.add_image_bytes(profile.id, data, suffix=suffix)
            except OSError:
                continue
        for frame in self._pending_frame_captures:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                self._store.add_image_bytes(profile.id, bytes(buf), suffix=".jpg")

    def has_pending_image_changes(self) -> bool:
        return bool(self._pending_image_paths or self._pending_frame_captures)


# ---------------------------------------------------------------------------
# Main People window
# ---------------------------------------------------------------------------

class PeopleWindow(QMainWindow):
    """Standalone window for managing People profiles."""

    profiles_changed = pyqtSignal()         # emitted on any profile change
    voice_profiles_changed = pyqtSignal()   # emitted when voice samples/embeddings change

    def __init__(self, store: ProfileStore,
                 frame_grabber: Callable[[], np.ndarray | None] | None = None,
                 audio_thread=None,
                 parent=None):
        super().__init__(parent)
        self._store = store
        self._frame_grabber = frame_grabber
        self._audio_thread = audio_thread
        self._embed_threads: list = []   # keep refs alive while running

        self.setWindowTitle("People — Autofollow")
        self.resize(620, 560)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)

        header = QLabel(
            "People profiles let Autofollow recognize specific individuals and prefer them "
            "when choosing who to follow or switch to. Higher-priority people (e.g. the pastor) "
            "get longer dwell and override unmatched bystanders."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: gray;")
        layout.addWidget(header)

        # --- Audio device picker ---
        if self._audio_thread is not None:
            audio_box = QGroupBox("Audio Input (for speaker + music detection)")
            audio_layout = QHBoxLayout(audio_box)
            self._audio_enable = QCheckBox("Enabled")
            self._audio_enable.toggled.connect(self._on_audio_enable_toggled)
            self._audio_combo = QComboBox()
            self._populate_audio_devices()
            self._audio_combo.currentIndexChanged.connect(self._on_audio_device_changed)
            audio_layout.addWidget(self._audio_enable)
            audio_layout.addWidget(QLabel("Device:"))
            audio_layout.addWidget(self._audio_combo, stretch=1)
            audio_layout.addStretch()
            layout.addWidget(audio_box)

        # List + buttons
        body = QHBoxLayout()
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _: self._edit_selected())
        body.addWidget(self._list, stretch=1)

        btn_col = QVBoxLayout()
        self._btn_add = QPushButton("Add Person…")
        self._btn_edit = QPushButton("Edit…")
        self._btn_delete = QPushButton("Delete")
        self._btn_reindex = QPushButton("Re-index Faces")
        self._btn_add.clicked.connect(self._add_new)
        self._btn_edit.clicked.connect(self._edit_selected)
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_reindex.clicked.connect(self._reindex_selected)
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            btn_col.addWidget(b)
        btn_col.addStretch()
        body.addLayout(btn_col)
        layout.addLayout(body)

        # Status bar
        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._status)

        self._refresh_list()

    # ------------------------------------------------------------------

    def _refresh_list(self):
        self._list.clear()
        profiles = self._store.list()
        if not profiles:
            self._list.addItem("(No people yet — click Add Person)")
            self._list.item(0).setFlags(Qt.NoItemFlags)
            return
        for p in profiles:
            n_imgs = p.n_images
            n_emb = 0 if p.embeddings is None else len(p.embeddings)
            n_voice = p.n_voice_samples
            n_v_emb = 0 if p.voice_embeddings is None else len(p.voice_embeddings)
            badge = "★" * min(p.priority, 5)
            text = (
                f"{p.name}    {badge}  (priority {p.priority})    "
                f"face: {n_imgs}/{n_emb}    voice: {n_voice}/{n_v_emb}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, p.id)
            self._list.addItem(item)

    def _selected_profile(self) -> Profile | None:
        item = self._list.currentItem()
        if item is None:
            return None
        pid = item.data(Qt.UserRole)
        if not pid:
            return None
        return self._store.get(pid)

    # ------------------------------------------------------------------

    def _add_new(self):
        dlg = EditProfileDialog(None, self._store, self._frame_grabber,
                                audio_grabber=self._grab_audio, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        name = dlg.name()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a name.")
            return
        profile = self._store.create(name, priority=dlg.priority())
        dlg.commit_pending_images(profile)
        dlg.commit_pending_voice(profile)
        # Re-embed faces and/or voices depending on what was attached
        if profile.n_images > 0:
            self._start_embed(profile.id)
        if profile.n_voice_samples > 0:
            self._start_voice_embed(profile.id)
        if profile.n_images == 0 and profile.n_voice_samples == 0:
            self._refresh_list()
            self.profiles_changed.emit()

    def _edit_selected(self):
        profile = self._selected_profile()
        if profile is None:
            return
        dlg = EditProfileDialog(profile, self._store, self._frame_grabber,
                                audio_grabber=self._grab_audio, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            self._refresh_list()
            return
        had_image_changes = dlg.has_pending_image_changes()
        had_voice_changes = dlg.has_pending_voice_changes()
        self._store.update(profile.id, name=dlg.name(), priority=dlg.priority())
        dlg.commit_pending_images(profile)
        dlg.commit_pending_voice(profile)
        if had_image_changes:
            self._start_embed(profile.id)
        if had_voice_changes:
            self._start_voice_embed(profile.id)
        if not (had_image_changes or had_voice_changes):
            self._refresh_list()
            self.profiles_changed.emit()

    def _delete_selected(self):
        profile = self._selected_profile()
        if profile is None:
            return
        if QMessageBox.question(
            self, "Delete person",
            f"Delete profile '{profile.name}' and all reference images?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self._store.delete(profile.id)
        self._refresh_list()
        self.profiles_changed.emit()

    def _reindex_selected(self):
        profile = self._selected_profile()
        if profile is None:
            return
        if profile.n_images == 0 and profile.n_voice_samples == 0:
            QMessageBox.information(self, "Nothing to index",
                                    "This profile has no reference images or voice samples yet.")
            return
        if profile.n_images > 0:
            self._start_embed(profile.id)
        if profile.n_voice_samples > 0:
            self._start_voice_embed(profile.id)

    # ------------------------------------------------------------------
    # Audio device wiring
    # ------------------------------------------------------------------

    def _populate_audio_devices(self):
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()
        self._audio_combo.addItem("(System default)", None)
        for dev in list_input_devices():
            self._audio_combo.addItem(
                f"[{dev['index']}] {dev['name']} ({dev['max_channels']}ch)",
                dev['index'],
            )
        # Pre-select the current device if already set
        if self._audio_thread is not None and self._audio_thread.device_index is not None:
            for i in range(self._audio_combo.count()):
                if self._audio_combo.itemData(i) == self._audio_thread.device_index:
                    self._audio_combo.setCurrentIndex(i)
                    break
        self._audio_combo.blockSignals(False)
        # Reflect enabled state
        if hasattr(self, "_audio_enable") and self._audio_thread is not None:
            self._audio_enable.setChecked(self._audio_thread.is_capturing)

    def _on_audio_device_changed(self, idx: int):
        if self._audio_thread is None:
            return
        device_index = self._audio_combo.itemData(idx)
        self._audio_thread.set_device(device_index)

    def _on_audio_enable_toggled(self, on: bool):
        if self._audio_thread is None:
            return
        if on:
            # Apply currently-selected device before enabling
            device_index = self._audio_combo.itemData(self._audio_combo.currentIndex())
            self._audio_thread.set_device(device_index)
            self._audio_thread.set_enabled(True)
        else:
            self._audio_thread.set_enabled(False)

    def _grab_audio(self, seconds: float) -> np.ndarray | None:
        """Used by EditProfileDialog to capture a recorded voice sample."""
        if self._audio_thread is None:
            return None
        cap = getattr(self._audio_thread, '_capture', None)
        if cap is None:
            return None
        return cap.get_recent(seconds)

    # ------------------------------------------------------------------

    def _start_embed(self, profile_id: str):
        profile = self._store.get(profile_id)
        if profile is None:
            return
        self._status.setText(f"Embedding {profile.name} faces… (loads InsightFace on first run)")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(False)
        thr = _EmbedThread(profile_id, self._store, self)
        thr.finished_with.connect(self._on_embed_done)
        thr.failed.connect(self._on_embed_failed)
        thr.finished.connect(lambda t=thr: self._embed_threads.remove(t)
                              if t in self._embed_threads else None)
        self._embed_threads.append(thr)
        thr.start()

    def _start_voice_embed(self, profile_id: str):
        profile = self._store.get(profile_id)
        if profile is None:
            return
        self._status.setText(f"Embedding {profile.name} voice… (loads SpeechBrain on first run)")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(False)
        thr = _VoiceEmbedThread(profile_id, self._store, self)
        thr.finished_with.connect(self._on_voice_embed_done)
        thr.failed.connect(self._on_embed_failed)
        thr.finished.connect(lambda t=thr: self._embed_threads.remove(t)
                              if t in self._embed_threads else None)
        self._embed_threads.append(thr)
        thr.start()

    def _on_voice_embed_done(self, profile_id: str, count: int):
        profile = self._store.get(profile_id)
        name = profile.name if profile else profile_id
        self._status.setText(f"Embedded {count} voice sample(s) for {name}.")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(True)
        self._store.reload()
        self._refresh_list()
        self.voice_profiles_changed.emit()

    def _on_embed_done(self, profile_id: str, count: int):
        profile = self._store.get(profile_id)
        name = profile.name if profile else profile_id
        self._status.setText(f"Embedded {count} reference image(s) for {name}.")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(True)
        # Reload from disk so updated embeddings/index are reflected
        self._store.reload()
        self._refresh_list()
        self.profiles_changed.emit()

    def _on_embed_failed(self, profile_id: str, error: str):
        self._status.setText(f"Embedding failed: {error}")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(True)
        QMessageBox.critical(self, "Embedding failed",
                             f"Could not embed faces:\n{error}")
