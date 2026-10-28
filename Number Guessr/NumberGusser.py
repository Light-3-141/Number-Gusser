"""Draw a digit and watch your trained MNIST model predict it live.

pip install PySide6 torch numpy
Shortcuts: C / Esc = clear, Ctrl+Z = undo, [ and ] = brush size
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

MODEL_PATH = Path("./mnist_Number_Image_model.pth")

# Set to None if you did NOT normalise during training.
NORMALIZE = (0.1307, 0.3081)

CANVAS = 300
BG, CARD, BORDER = "#12141a", "#1a1d26", "#262a36"
ACCENT, ACCENT2, MUTED = "#8b6cff", "#4cc3ff", "#3a3f52"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class MyCNN(nn.Module):
    """Only used if your .pth is a state_dict of a CNN.
    EDIT THIS so it matches the class you trained (layer names must match)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def load_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]

    weights = [(k, v) for k, v in obj.items() if k.endswith("weight")]
    if weights and all(v.ndim == 2 for _, v in weights):  # plain MLP
        layers = []
        for i, (_, w) in enumerate(weights):
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if i < len(weights) - 1:
                layers.append(nn.ReLU())
        model = nn.Sequential(nn.Flatten(), *layers)
        remapped, idx = {}, 1
        for k, _ in weights:
            base = k[:-len("weight")]
            remapped[f"{idx}.weight"] = obj[base + "weight"]
            remapped[f"{idx}.bias"] = obj[base + "bias"]
            idx += 2
        model.load_state_dict(remapped)
        return model.eval()

    model = MyCNN()
    model.load_state_dict(obj)
    return model.eval()


# ---------------------------------------------------------------------------
# Preprocessing (MNIST style: 20x20 box, centred by centre of mass in 28x28)
# ---------------------------------------------------------------------------
def qimage_to_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_Grayscale8)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.bytesPerLine() * h)
    return arr.reshape(h, img.bytesPerLine())[:, :w].copy()


def preprocess(canvas: QImage):
    arr = qimage_to_array(canvas)
    ys, xs = np.nonzero(arr > 20)
    if len(xs) == 0:
        return None
    crop = np.ascontiguousarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))

    small = QImage(crop.data, w, h, w, QImage.Format_Grayscale8)
    small = small.scaled(nw, nh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    small = qimage_to_array(small).astype(np.float32) / 255.0

    out = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    total = out.sum()
    cy = (out.sum(1) * np.arange(28)).sum() / total
    cx = (out.sum(0) * np.arange(28)).sum() / total
    return np.roll(out, (round(14 - cy), round(14 - cx)), axis=(0, 1))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class Canvas(QWidget):
    def __init__(self, on_change):
        super().__init__()
        self.on_change = on_change
        self.brush = 24
        self.last = None
        self.empty = True
        self.undo_stack = []
        self.setFixedSize(CANVAS, CANVAS)
        self.setCursor(Qt.CrossCursor)
        self.image = QImage(CANVAS, CANVAS, QImage.Format_RGB32)
        self.image.fill(Qt.black)

    def clear(self):
        self.undo_stack.clear()
        self.image.fill(Qt.black)
        self.empty = True
        self.update()
        self.on_change()

    def undo(self):
        if self.undo_stack:
            self.image = self.undo_stack.pop()
            self.empty = not np.any(qimage_to_array(self.image) > 20)
            self.update()
            self.on_change()

    def _line(self, a: QPoint, b: QPoint):
        p = QPainter(self.image)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(Qt.white, self.brush, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(a, b)
        p.end()
        self.empty = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.undo_stack.append(self.image.copy())
            del self.undo_stack[:-30]
            self.last = e.position().toPoint()
            self._line(self.last, self.last)
            self.on_change()

    def mouseMoveEvent(self, e):
        if self.last is not None:
            pos = e.position().toPoint()
            self._line(self.last, pos)
            self.last = pos
            self.on_change()

    def mouseReleaseEvent(self, e):
        self.last = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 20, 20)
        p.setClipPath(path)
        p.drawImage(0, 0, self.image)
        p.setClipping(False)
        if self.empty:
            p.setPen(QColor("#4a4f63"))
            f = QFont(self.font())
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "Draw a digit here")
        p.setPen(QPen(QColor(BORDER), 2))
        p.drawRoundedRect(rect, 20, 20)


class BarRow(QWidget):
    """One animated probability bar."""

    def __init__(self, digit):
        super().__init__()
        self.digit, self.value, self.target, self.top = digit, 0.0, 0.0, False
        self.setFixedHeight(30)

    def step(self):
        d = self.target - self.value
        if abs(d) > 0.0005:
            self.value += d * 0.28
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        f = QFont(self.font())
        f.setPointSize(11)
        f.setBold(self.top)
        p.setFont(f)
        p.setPen(QColor("#e8eaf2") if self.top else QColor("#7d8399"))
        p.drawText(0, 0, 22, h, Qt.AlignVCenter | Qt.AlignLeft, str(self.digit))

        track = QRectF(26, 8, w - 26 - 62, h - 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#222634"))
        p.drawRoundedRect(track, 7, 7)
        fill = QRectF(track.x(), track.y(), max(0.0, track.width() * self.value), track.height())
        if fill.width() > 1:
            if self.top:
                g = QLinearGradient(fill.topLeft(), fill.topRight())
                g.setColorAt(0, QColor(ACCENT))
                g.setColorAt(1, QColor(ACCENT2))
                p.setBrush(g)
            else:
                p.setBrush(QColor(MUTED))
            p.drawRoundedRect(fill, 7, 7)

        p.setPen(QColor("#e8eaf2") if self.top else QColor("#7d8399"))
        p.drawText(QRectF(w - 56, 0, 56, h), Qt.AlignVCenter | Qt.AlignRight,
                   f"{self.target * 100:.1f}%")


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


STYLE = f"""
QWidget {{ background: {BG}; color: #e8eaf2; font-family: 'Inter','Segoe UI','Helvetica Neue',sans-serif; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 22px; }}
QFrame#card QLabel, QFrame#card QWidget {{ background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 700; }}
QLabel#sub {{ color: #7d8399; font-size: 12px; }}
QLabel#section {{ color: #7d8399; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QPushButton {{ background: #232736; border: 1px solid {BORDER}; border-radius: 12px;
              padding: 10px 18px; font-weight: 600; }}
QPushButton:hover {{ background: #2c3144; }}
QPushButton:pressed {{ background: #1f2230; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; }}
QPushButton#primary:hover {{ background: #9d82ff; }}
QSlider::groove:horizontal {{ height: 6px; background: #232736; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; background: white; }}
"""


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Digit Painter")
        self.setStyleSheet(STYLE)
        self.model = load_model(MODEL_PATH)
        self.dirty = False

        # ---- left: canvas card
        self.canvas = Canvas(self.mark_dirty)
        self.size_label = QLabel(f"{self.canvas.brush}px")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(10, 44)
        slider.setValue(self.canvas.brush)
        slider.valueChanged.connect(self.set_brush)
        self.slider = slider

        undo = QPushButton("Undo")
        undo.clicked.connect(self.canvas.undo)
        clear = QPushButton("Clear")
        clear.setObjectName("primary")
        clear.clicked.connect(self.canvas.clear)

        brush_row = QHBoxLayout()
        lbl = QLabel("BRUSH")
        lbl.setObjectName("section")
        brush_row.addWidget(lbl)
        brush_row.addWidget(slider, 1)
        brush_row.addWidget(self.size_label)
        btn_row = QHBoxLayout()
        btn_row.addWidget(undo)
        btn_row.addWidget(clear)

        left = card()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(22, 22, 22, 22)
        lv.setSpacing(16)
        lv.addWidget(self.canvas, alignment=Qt.AlignCenter)
        lv.addLayout(brush_row)
        lv.addLayout(btn_row)

        # ---- right: results card
        self.big = QLabel("–")
        self.big.setAlignment(Qt.AlignCenter)
        self.big.setFixedWidth(150)
        self.big.setStyleSheet("font-size: 112px; font-weight: 800; color: #4a4f63;")
        self.conf = QLabel("Waiting for input")
        self.conf.setAlignment(Qt.AlignCenter)
        self.conf.setObjectName("sub")
        big_col = QVBoxLayout()
        big_col.addWidget(self.big)
        big_col.addWidget(self.conf)

        self.preview = QLabel()
        self.preview.setFixedSize(112, 112)
        self.preview.setStyleSheet(f"border: 1px solid {BORDER}; border-radius: 10px; background: black;")
        pv_lbl = QLabel("MODEL INPUT 28×28")
        pv_lbl.setObjectName("section")
        pv_col = QVBoxLayout()
        pv_col.addStretch()
        pv_col.addWidget(self.preview)
        pv_col.addWidget(pv_lbl)
        pv_col.addStretch()

        top = QHBoxLayout()
        top.addLayout(big_col, 1)
        top.addLayout(pv_col)

        self.rows = [BarRow(d) for d in range(10)]
        bars = QVBoxLayout()
        bars.setSpacing(2)
        for r in self.rows:
            bars.addWidget(r)
        pl = QLabel("PROBABILITIES")
        pl.setObjectName("section")

        right = card()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(26, 22, 26, 22)
        rv.setSpacing(14)
        rv.addLayout(top)
        rv.addWidget(pl)
        rv.addLayout(bars)

        # ---- header + root
        title = QLabel("MNIST Digit Painter")
        title.setObjectName("title")
        sub = QLabel("Live predictions from your trained model")
        sub.setObjectName("sub")
        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(title)
        header.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(20)
        body.addWidget(left)
        body.addWidget(right, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(20)
        root.addLayout(header)
        root.addLayout(body)
        self.setMinimumSize(780, 600)

        # ---- shortcuts
        for key in ("C", "Esc"):
            QShortcut(QKeySequence(key), self, activated=self.canvas.clear)
        QShortcut(QKeySequence.Undo, self, activated=self.canvas.undo)
        QShortcut(QKeySequence("["), self, activated=lambda: self.slider.setValue(self.slider.value() - 3))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.slider.setValue(self.slider.value() + 3))

        # ---- timers: ~60 fps animation, ~30 fps continuous prediction
        self.anim = QTimer(self, interval=16, timeout=self.animate)
        self.anim.start()
        self.infer = QTimer(self, interval=33, timeout=self.maybe_predict)
        self.infer.start()

    def set_brush(self, v):
        self.canvas.brush = v
        self.size_label.setText(f"{v}px")

    def mark_dirty(self):
        self.dirty = True

    def animate(self):
        for r in self.rows:
            r.step()

    def maybe_predict(self):
        if self.dirty:
            self.dirty = False
            self.predict()

    def show_empty(self):
        self.big.setText("–")
        self.big.setStyleSheet("font-size: 112px; font-weight: 800; color: #4a4f63;")
        self.conf.setText("Waiting for input")
        self.preview.clear()
        for r in self.rows:
            r.target, r.top = 0.0, False

    @torch.no_grad()
    def predict(self):
        arr = preprocess(self.canvas.image)
        if arr is None:
            self.show_empty()
            return

        # preview of exactly what the model sees
        img = QImage((arr * 255).astype(np.uint8).tobytes(), 28, 28, 28, QImage.Format_Grayscale8)
        self.preview.setPixmap(QPixmap.fromImage(
            img.scaled(112, 112, Qt.IgnoreAspectRatio, Qt.FastTransformation)))

        if NORMALIZE:
            arr = (arr - NORMALIZE[0]) / NORMALIZE[1]
        x = torch.from_numpy(arr).float().view(1, 1, 28, 28)
        try:
            out = self.model(x)
        except RuntimeError:  # model may want a flat 784 vector
            out = self.model(x.view(1, 784))
        out = out.flatten()
        if out.min() >= 0 and abs(out.sum().item() - 1) < 1e-3:
            probs = out
        else:
            probs = F.softmax(out, dim=0)
        probs = probs.numpy()
        best = int(probs.argmax())

        self.big.setText(str(best))
        self.big.setStyleSheet(f"font-size: 112px; font-weight: 800; color: {ACCENT};")
        self.conf.setText(f"{probs[best] * 100:.1f}% confident")
        for i, r in enumerate(self.rows):
            r.target, r.top = float(probs[i]), i == best
            r.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())