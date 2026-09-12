"""Launch splash: a frameless card with a live tracking-viewfinder motif,
the wordmark and an indeterminate loader that reports what is being loaded.

Shown before the heavy imports (Qt widgets, OpenCV, torch) so the user gets
immediate feedback; `set_status()` updates the line under the loader and
`finish(window)` dismisses it once the control panel is up.
"""

import math
import time

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QLinearGradient, QRadialGradient, QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QApplication

from theme import ACCENT, CYAN, TEXT, MUTED, DIM, BORDER

_W, _H = 640, 360
_RADIUS = 20


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(_W, _H)
        self._status = "Starting…"
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)
        self._center_on_screen()

    # ------------------------------------------------------------------

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(geo.center().x() - _W // 2, geo.center().y() - _H // 2)

    def set_status(self, text: str):
        self._status = text
        self.update()
        QApplication.processEvents()

    def finish(self, window: QWidget):
        """Dismiss the splash once `window` is on screen."""
        self._timer.stop()
        self.close()
        self.deleteLater()

    # ------------------------------------------------------------------

    def paintEvent(self, event):
        t = time.monotonic() - self._t0
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        rect = QRectF(0, 0, _W, _H)
        card = QPainterPath()
        card.addRoundedRect(rect, _RADIUS, _RADIUS)
        p.setClipPath(card)

        # Background: deep gradient with a soft accent glow in the top-left
        bg = QLinearGradient(0, 0, _W, _H)
        bg.setColorAt(0.0, QColor("#171a21"))
        bg.setColorAt(1.0, QColor("#0f1115"))
        p.fillPath(card, bg)
        glow = QRadialGradient(QPointF(150, 120), 320)
        c0 = QColor(ACCENT); c0.setAlpha(60)
        c1 = QColor(ACCENT); c1.setAlpha(0)
        glow.setColorAt(0.0, c0)
        glow.setColorAt(1.0, c1)
        p.fillRect(rect, glow)

        # Faint grid so it reads as a camera/monitor surface
        grid_pen = QPen(QColor(255, 255, 255, 9), 1)
        p.setPen(grid_pen)
        for x in range(0, _W, 32):
            p.drawLine(x, 0, x, _H)
        for y in range(0, _H, 32):
            p.drawLine(0, y, _W, y)

        self._draw_viewfinder(p, t)
        self._draw_wordmark(p)
        self._draw_loader(p, t)

        # Card border
        p.setClipping(False)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), _RADIUS, _RADIUS)
        p.end()

    # ------------------------------------------------------------------

    def _draw_viewfinder(self, p: QPainter, t: float):
        """Camera frame with tracking brackets 'breathing' around a subject."""
        ox, oy, w, h = 48, 72, 220, 150

        # Camera frame
        frame = QRectF(ox, oy, w, h)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.setBrush(QColor(0, 0, 0, 70))
        p.drawRoundedRect(frame, 10, 10)

        # Subject silhouette (head + shoulders), slowly swaying
        sway = math.sin(t * 0.9) * 6.0
        cx, cy = ox + w / 2 + sway, oy + h * 0.58
        body = QPainterPath()
        body.moveTo(cx - 44, oy + h)
        body.cubicTo(cx - 44, cy + 6, cx - 22, cy - 2, cx, cy - 2)
        body.cubicTo(cx + 22, cy - 2, cx + 44, cy + 6, cx + 44, oy + h)
        body.closeSubpath()
        body.addEllipse(QPointF(cx, cy - 26), 17, 19)
        sil = QLinearGradient(0, cy - 45, 0, oy + h)
        sil.setColorAt(0.0, QColor(255, 255, 255, 70))
        sil.setColorAt(1.0, QColor(255, 255, 255, 18))
        p.setPen(Qt.NoPen)
        p.setBrush(sil)
        p.drawPath(body)

        # Tracking box: eased breathing around the subject, following the sway
        breathe = (math.sin(t * 1.6) + 1) / 2          # 0..1
        bw, bh = 96 + 10 * breathe, 92 + 8 * breathe
        bx, by = cx - bw / 2, cy - 48 - 4 * breathe
        box = QRectF(bx, by, bw, bh)
        accent = QColor(ACCENT)
        pen = QPen(accent, 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        arm = 14
        for (x, y, dx, dy) in (
            (box.left(),  box.top(),     1,  1),
            (box.right(), box.top(),    -1,  1),
            (box.left(),  box.bottom(),  1, -1),
            (box.right(), box.bottom(), -1, -1),
        ):
            p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))

        # Center reticle
        p.setPen(QPen(QColor(CYAN), 1.5))
        rc = box.center()
        p.drawLine(QPointF(rc.x() - 6, rc.y()), QPointF(rc.x() + 6, rc.y()))
        p.drawLine(QPointF(rc.x(), rc.y() - 6), QPointF(rc.x(), rc.y() + 6))

        # Tag above the box
        tag_font = QFont("SF Mono", 8)
        tag_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        p.setFont(tag_font)
        tag_rect = QRectF(box.left(), box.top() - 20, 84, 14)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 40))
        p.drawRoundedRect(tag_rect, 4, 4)
        p.setPen(QColor(ACCENT))
        p.drawText(tag_rect, Qt.AlignCenter, "PRIMARY  ●")

        # Frame corner ticks
        p.setPen(QPen(QColor(255, 255, 255, 60), 1.5))
        tick = 10
        for (x, y, dx, dy) in (
            (frame.left() + 8,  frame.top() + 8,     1,  1),
            (frame.right() - 8, frame.top() + 8,    -1,  1),
            (frame.left() + 8,  frame.bottom() - 8,  1, -1),
            (frame.right() - 8, frame.bottom() - 8, -1, -1),
        ):
            p.drawLine(QPointF(x, y), QPointF(x + dx * tick, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * tick))

    def _draw_wordmark(self, p: QPainter):
        x = 310
        # Small gradient mark
        mark = QRectF(x, 92, 22, 22)
        grad = QLinearGradient(mark.topLeft(), mark.bottomRight())
        grad.setColorAt(0.0, QColor(ACCENT))
        grad.setColorAt(1.0, QColor(CYAN))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(mark, 6, 6)

        font = QFont("SF Pro Display", 34, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, -0.5)
        p.setFont(font)
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(x + 32, 74, 300, 56), Qt.AlignLeft | Qt.AlignVCenter, "Autofollow")

        tag = QFont("SF Pro Text", 13)
        p.setFont(tag)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(x, 134, 300, 24), Qt.AlignLeft | Qt.AlignVCenter,
                   "Intelligent virtual PTZ camera")

        small = QFont("SF Pro Text", 11)
        p.setFont(small)
        p.setPen(QColor(DIM))
        p.drawText(QRectF(x, 160, 300, 44), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                   "Pose tracking · smart switching · program output")

    def _draw_loader(self, p: QPainter, t: float):
        track = QRectF(48, _H - 62, _W - 96, 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BORDER))
        p.drawRoundedRect(track, 2, 2)

        # Sweeping highlight
        span = 0.28
        pos = (t * 0.45) % (1.0 + span) - span
        sweep = QLinearGradient(track.left() + pos * track.width(), 0,
                                track.left() + (pos + span) * track.width(), 0)
        a0 = QColor(ACCENT); a0.setAlpha(0)
        a1 = QColor(ACCENT)
        sweep.setColorAt(0.0, a0)
        sweep.setColorAt(0.5, a1)
        sweep.setColorAt(1.0, a0)
        p.setBrush(sweep)
        p.drawRoundedRect(track, 2, 2)

        font = QFont("SF Pro Text", 11)
        p.setFont(font)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(48, _H - 46, _W - 96, 22), Qt.AlignLeft | Qt.AlignVCenter, self._status)
        p.setPen(QColor(DIM))
        p.drawText(QRectF(48, _H - 46, _W - 96, 22), Qt.AlignRight | Qt.AlignVCenter, "v1")
