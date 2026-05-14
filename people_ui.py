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
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog, QDialog,
    QLineEdit, QSpinBox, QDialogButtonBox, QFormLayout, QGroupBox, QSlider,
    QScrollArea, QFrame,
)

from profiles import ProfileStore, Profile


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


# ---------------------------------------------------------------------------
# Edit dialog (add or edit a single profile)
# ---------------------------------------------------------------------------

class EditProfileDialog(QDialog):
    """Modal dialog to create or edit a single profile."""

    def __init__(self, profile: Profile | None, store: ProfileStore,
                 frame_grabber: Callable[[], np.ndarray | None] | None = None,
                 parent=None):
        super().__init__(parent)
        self._store = store
        self._profile = profile
        self._frame_grabber = frame_grabber
        self._pending_image_paths: list[Path] = []     # external files to copy in on save
        self._pending_frame_captures: list[np.ndarray] = []   # frame bytes to write on save

        self.setWindowTitle("Edit Person" if profile else "Add Person")
        self.resize(560, 480)

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

        # --- Save / Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_gallery()

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

    profiles_changed = pyqtSignal()   # emitted when the store contents change

    def __init__(self, store: ProfileStore,
                 frame_grabber: Callable[[], np.ndarray | None] | None = None,
                 parent=None):
        super().__init__(parent)
        self._store = store
        self._frame_grabber = frame_grabber
        self._embed_threads: list[_EmbedThread] = []   # keep refs alive while running

        self.setWindowTitle("People — Autofollow")
        self.resize(620, 460)

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
            badge = "★" * min(p.priority, 5)
            text = f"{p.name}    {badge}  (priority {p.priority})    {n_imgs} image(s), {n_emb} embedded"
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
        dlg = EditProfileDialog(None, self._store, self._frame_grabber, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        name = dlg.name()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a name.")
            return
        profile = self._store.create(name, priority=dlg.priority())
        dlg.commit_pending_images(profile)
        # Re-embed if any images were attached
        if profile.n_images > 0:
            self._start_embed(profile.id)
        else:
            self._refresh_list()
            self.profiles_changed.emit()

    def _edit_selected(self):
        profile = self._selected_profile()
        if profile is None:
            return
        dlg = EditProfileDialog(profile, self._store, self._frame_grabber, self)
        if dlg.exec_() != QDialog.Accepted:
            self._refresh_list()
            return
        had_pending = dlg.has_pending_image_changes()
        self._store.update(profile.id, name=dlg.name(), priority=dlg.priority())
        dlg.commit_pending_images(profile)
        if had_pending:
            self._start_embed(profile.id)
        else:
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
        if profile.n_images == 0:
            QMessageBox.information(self, "No images",
                                    "This profile has no reference images yet.")
            return
        self._start_embed(profile.id)

    # ------------------------------------------------------------------

    def _start_embed(self, profile_id: str):
        profile = self._store.get(profile_id)
        if profile is None:
            return
        self._status.setText(f"Embedding {profile.name}… (loads InsightFace on first run)")
        for b in (self._btn_add, self._btn_edit, self._btn_delete, self._btn_reindex):
            b.setEnabled(False)
        thr = _EmbedThread(profile_id, self._store, self)
        thr.finished_with.connect(self._on_embed_done)
        thr.failed.connect(self._on_embed_failed)
        # Cleanup the reference when the thread finishes
        thr.finished.connect(lambda t=thr: self._embed_threads.remove(t)
                              if t in self._embed_threads else None)
        self._embed_threads.append(thr)
        thr.start()

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
