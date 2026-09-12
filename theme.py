"""App-wide look and feel: Fusion style, dark palette, stylesheet and small
reusable widgets (cards, segmented buttons, video surfaces).

Everything visual that more than one window needs lives here so the control
panel, diagnostics and people windows read as one application.
"""

import os
import tempfile

from PyQt5.QtCore import Qt, QRectF, QSize, QPointF
from PyQt5.QtGui import QColor, QPalette, QPainter, QImage, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QButtonGroup, QSizePolicy,
)

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#121419"   # window background
PANEL   = "#1a1d24"   # cards
RAISED  = "#232730"   # inputs, hover
BORDER  = "#2b303b"
TEXT    = "#e8ebf1"
MUTED   = "#8b93a5"
DIM     = "#5c6373"
ACCENT  = "#4f8cff"
ACCENT_HOVER = "#6ea0ff"
ACCENT_TEXT  = "#0b1020"
GREEN   = "#3ddc97"
RED     = "#ff6b6b"
AMBER   = "#f5c249"
PURPLE  = "#c084fc"
CYAN    = "#5ad4e6"

FONT_FAMILY = '"SF Pro Text", "Helvetica Neue", "Segoe UI", sans-serif'
MONO_FAMILY = '"SF Mono", Menlo, Consolas, monospace'


def _rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha:.2f})"


def _build_stylesheet(icons: str) -> str:
    return f"""
    * {{
        font-family: {FONT_FAMILY};
        font-size: 13px;
    }}
    QMainWindow, QDialog, QWidget#root {{
        background: {BG};
    }}
    QWidget {{
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QToolTip {{
        background: {RAISED};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 6px 8px;
        border-radius: 6px;
    }}

    /* ── Cards ─────────────────────────────────────────────────────────── */
    QFrame#card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QLabel#cardTitle {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    QLabel#muted {{
        color: {MUTED};
        font-size: 12px;
    }}
    QLabel#dim {{
        color: {DIM};
        font-size: 11px;
    }}
    QLabel#fieldLabel, QCheckBox#fieldLabel {{
        color: {MUTED};
        font-size: 12px;
    }}
    QLabel#value {{
        font-size: 14px;
        font-weight: 600;
    }}
    QLabel#mono {{
        font-family: {MONO_FAMILY};
        font-size: 12px;
    }}
    QLabel#statusBar {{
        color: {MUTED};
        font-size: 12px;
        padding: 2px 4px;
    }}

    /* ── Group boxes (dialogs still use them) ──────────────────────────── */
    QGroupBox {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-top: 18px;
        padding: 10px 8px 6px 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        color: {MUTED};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }}

    /* ── Buttons ───────────────────────────────────────────────────────── */
    QPushButton {{
        background: {RAISED};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 7px;
        padding: 6px 14px;
        min-height: 18px;
    }}
    QPushButton:hover {{
        background: #2b3039;
        border-color: #3a404c;
    }}
    QPushButton:pressed {{
        background: #1e222a;
    }}
    QPushButton:disabled {{
        color: {DIM};
        background: {PANEL};
        border-color: {BORDER};
    }}
    QPushButton#primary {{
        background: {ACCENT};
        color: {ACCENT_TEXT};
        border: none;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background: {ACCENT_HOVER};
    }}
    QPushButton#primary:checked {{
        background: {RED};
    }}
    QPushButton#ghost {{
        background: transparent;
        border: 1px solid {BORDER};
    }}
    QPushButton#ghost:hover {{
        background: {RAISED};
    }}
    QPushButton#icon {{
        padding: 4px 8px;
        min-width: 0;
    }}

    /* Segmented control */
    QPushButton#segment {{
        background: transparent;
        border: none;
        border-radius: 6px;
        padding: 5px 12px;
        color: {MUTED};
    }}
    QPushButton#segment:hover {{
        color: {TEXT};
        background: {_rgba(TEXT, 0.05)};
    }}
    QPushButton#segment:checked {{
        background: {RAISED};
        color: {TEXT};
        font-weight: 600;
    }}
    QFrame#segmentTrack {{
        background: {BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* Person buttons in manual mode */
    QPushButton#personChip {{
        padding: 4px 10px;
        border-radius: 12px;
        background: {RAISED};
        font-family: {MONO_FAMILY};
        font-size: 12px;
    }}
    QPushButton#personChip:hover {{
        border-color: {ACCENT};
        color: {ACCENT_HOVER};
    }}

    /* ── Inputs ────────────────────────────────────────────────────────── */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {RAISED};
        border: 1px solid {BORDER};
        border-radius: 7px;
        padding: 5px 8px;
        min-height: 18px;
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
        border-color: #3a404c;
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
    }}
    QComboBox::down-arrow {{
        image: url({icons}/chevron-down.png);
        width: 10px; height: 6px;
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background: {RAISED};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px;
        outline: 0;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        width: 16px;
        border: none;
        background: transparent;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({icons}/chevron-up.png);
        width: 8px; height: 5px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({icons}/chevron-down.png);
        width: 8px; height: 5px;
    }}
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
        image: none;
    }}

    /* ── Check / radio ─────────────────────────────────────────────────── */
    QCheckBox, QRadioButton {{
        spacing: 8px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid #3a404c;
        background: {RAISED};
    }}
    QCheckBox::indicator {{
        border-radius: 4px;
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked {{
        image: url({icons}/check.png);
    }}
    QRadioButton::indicator:checked {{
        image: url({icons}/radio.png);
    }}

    /* ── Slider ────────────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {BORDER};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
        background: {TEXT};
        border: 2px solid {PANEL};
    }}
    QSlider::handle:horizontal:hover {{
        background: #ffffff;
    }}

    /* ── Tables / lists / text ─────────────────────────────────────────── */
    QTableWidget, QListWidget, QTextEdit, QPlainTextEdit {{
        background: {BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        gridline-color: {BORDER};
        outline: 0;
    }}
    QTableWidget::item {{
        padding: 3px 6px;
    }}
    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 6px;
    }}
    QListWidget::item:selected {{
        background: {_rgba(ACCENT, 0.25)};
        color: {TEXT};
    }}
    QListWidget::item:hover {{
        background: {_rgba(TEXT, 0.05)};
    }}
    QHeaderView::section {{
        background: {PANEL};
        color: {MUTED};
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
        padding: 5px 6px;
        font-size: 11px;
        font-weight: 600;
    }}
    QTableCornerButton::section {{
        background: {PANEL};
        border: none;
    }}

    /* ── Scrollbars ────────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #353b47;
        border-radius: 3px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #444b59;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: #353b47;
        border-radius: 3px;
        min-width: 24px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line,
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: none;
        border: none;
        width: 0; height: 0;
    }}

    /* ── Splitter ──────────────────────────────────────────────────────── */
    QSplitter::handle {{
        background: transparent;
    }}
    QSplitter::handle:horizontal {{
        width: 8px;
    }}
    QSplitter::handle:vertical {{
        height: 8px;
    }}
    QSplitter::handle:hover {{
        background: {_rgba(ACCENT, 0.25)};
    }}

    /* ── Menus ─────────────────────────────────────────────────────────── */
    QMenu {{
        background: {RAISED};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 18px;
        border-radius: 5px;
    }}
    QMenu::item:selected {{
        background: {ACCENT};
        color: {ACCENT_TEXT};
    }}
"""


def _paint_icons() -> str:
    """Rasterise the few glyphs the stylesheet needs (chevrons, check mark).

    Qt stylesheets can't draw vector shapes or take data: URIs, so the icons
    are painted once into a temp folder; @2x variants are picked up on Retina.
    """
    icon_dir = os.path.join(tempfile.gettempdir(), "autofollow-theme")
    os.makedirs(icon_dir, exist_ok=True)

    def render(name: str, w: int, h: int, draw):
        for scale, suffix in ((1, ""), (2, "@2x")):
            pm = QPixmap(w * scale, h * scale)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.scale(scale, scale)
            draw(p, w, h)
            p.end()
            pm.save(os.path.join(icon_dir, f"{name}{suffix}.png"))

    def chevron(direction: int):
        def _draw(p: QPainter, w: int, h: int):
            pen = QPen(QColor(MUTED), 1.6)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            top, bottom = (1.0, h - 1.0) if direction > 0 else (h - 1.0, 1.0)
            p.drawPolyline(QPointF(1.0, top), QPointF(w / 2.0, bottom), QPointF(w - 1.0, top))
        return _draw

    def check(p: QPainter, w: int, h: int):
        pen = QPen(QColor(ACCENT_TEXT), 2.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline(QPointF(w * 0.22, h * 0.53), QPointF(w * 0.43, h * 0.74),
                       QPointF(w * 0.80, h * 0.30))

    def radio(p: QPainter, w: int, h: int):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT_TEXT))
        p.drawEllipse(QPointF(w / 2.0, h / 2.0), w * 0.22, h * 0.22)

    render("chevron-down", 10, 6, chevron(+1))
    render("chevron-up", 10, 6, chevron(-1))
    render("check", 16, 16, check)
    render("radio", 16, 16, radio)
    return icon_dir.replace("\\", "/")


def apply_theme(app: QApplication):
    """Install the Fusion style, a matching dark palette and the stylesheet."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(BG))
    pal.setColor(QPalette.WindowText,      QColor(TEXT))
    pal.setColor(QPalette.Base,            QColor(PANEL))
    pal.setColor(QPalette.AlternateBase,   QColor(RAISED))
    pal.setColor(QPalette.ToolTipBase,     QColor(RAISED))
    pal.setColor(QPalette.ToolTipText,     QColor(TEXT))
    pal.setColor(QPalette.Text,            QColor(TEXT))
    pal.setColor(QPalette.Button,          QColor(RAISED))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT))
    pal.setColor(QPalette.BrightText,      QColor("#ffffff"))
    pal.setColor(QPalette.Link,            QColor(ACCENT))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text,       QColor(DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(DIM))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(DIM))
    app.setPalette(pal)
    app.setStyleSheet(_build_stylesheet(_paint_icons()))


# ── Reusable widgets ─────────────────────────────────────────────────────────

class Card(QFrame):
    """Rounded panel with a small uppercase title and a vertical body layout."""

    def __init__(self, title: str | None = None, parent=None, spacing: int = 8):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(spacing)
        self.header = QHBoxLayout()
        self.header.setSpacing(6)
        self.title_label: QLabel | None = None
        if title:
            self.title_label = QLabel(title.upper())
            self.title_label.setObjectName("cardTitle")
            self.header.addWidget(self.title_label)
        self.header.addStretch()
        outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(spacing)
        outer.addLayout(self.body)

    def add_row(self, label: str | None, *widgets, stretch_last: bool = False) -> QHBoxLayout:
        """Append a labelled row of widgets to the body; returns the row layout."""
        row = QHBoxLayout()
        row.setSpacing(8)
        if label is not None:
            lbl = QLabel(label)
            lbl.setObjectName("fieldLabel")
            lbl.setMinimumWidth(64)
            row.addWidget(lbl)
        for i, w in enumerate(widgets):
            if isinstance(w, QWidget):
                row.addWidget(w, 1 if (stretch_last and i == len(widgets) - 1) else 0)
            else:
                row.addLayout(w)
        if not stretch_last:
            row.addStretch()
        self.body.addLayout(row)
        return row


class Segmented(QFrame):
    """A row of mutually exclusive, checkable buttons (like a macOS segmented control).

    Each option is (label, key, tooltip).  `group.buttonClicked` fires with the
    QPushButton; its "key" property holds the option key.
    """

    def __init__(self, options, current_key: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("segmentTrack")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for label, key, tip in options:
            btn = QPushButton(label)
            btn.setObjectName("segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("key", key)
            if tip:
                btn.setToolTip(tip)
            btn.setChecked(key == current_key)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.group.addButton(btn)
            lay.addWidget(btn)
            self._buttons[key] = btn

    def set_current(self, key: str):
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

    def current_key(self) -> str | None:
        btn = self.group.checkedButton()
        return btn.property("key") if btn is not None else None


class VideoSurface(QWidget):
    """Paints a QImage letterboxed inside the widget, keeping aspect ratio.

    Cheaper than QLabel.setPixmap(pix.scaled(...)) per frame, and it never
    fights the layout for size the way a pixmap-holding label does.
    """

    def __init__(self, aspect: float = 16 / 9, radius: int = 10,
                 background: str = "#000000", parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._aspect = aspect
        self._radius = radius
        self._bg = QColor(background)
        self._placeholder = "No signal"
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 90)

    # Aspect-ratio-aware sizing so layouts allocate a sensible height.
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(w / self._aspect)

    def sizeHint(self) -> QSize:
        return QSize(640, int(640 / self._aspect))

    def set_placeholder(self, text: str):
        self._placeholder = text
        if self._image is None:
            self.update()

    def set_image(self, img: QImage | None):
        self._image = img
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        p.setClipPath(path)
        p.fillRect(rect, self._bg)

        if self._image is None or self._image.isNull():
            p.setPen(QColor(DIM))
            f = p.font()
            f.setPointSize(12)
            p.setFont(f)
            p.drawText(rect, Qt.AlignCenter, self._placeholder)
            return

        iw, ih = self._image.width(), self._image.height()
        scale = min(rect.width() / iw, rect.height() / ih)
        dw, dh = iw * scale, ih * scale
        target = QRectF(rect.x() + (rect.width() - dw) / 2.0,
                        rect.y() + (rect.height() - dh) / 2.0, dw, dh)
        p.drawImage(target, self._image)


class Dot(QLabel):
    """A small colored status dot."""

    def __init__(self, color: str = DIM, size: int = 8, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.set_color(color)

    def set_color(self, color: str):
        self.setStyleSheet(
            f"background: {color}; border-radius: {self._size // 2}px;")


def pill(text: str, color: str = MUTED, bg: str | None = None) -> QLabel:
    """Small rounded status label (e.g. 'PRIMARY', 'MUSIC')."""
    lbl = QLabel(text)
    bg = bg or _rgba(color, 0.14)
    lbl.setStyleSheet(
        f"color: {color}; background: {bg}; border-radius: 9px; padding: 2px 8px;"
        f"font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
    return lbl


def set_pill(lbl: QLabel, text: str, color: str):
    lbl.setText(text)
    lbl.setStyleSheet(
        f"color: {color}; background: {_rgba(color, 0.14)}; border-radius: 9px;"
        f"padding: 2px 8px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
