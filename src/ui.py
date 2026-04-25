"""
PyQt5 application window for the hybrid BSL translator.

Layout
------
  ┌───────────────────────────────┬──────────────────────────┐
  │  Live webcam feed             │  SENTENCE                │
  │  (hand skeleton overlay)      │  ─────────────────────   │
  │  (face bounding box + label)  │  assembled text here     │
  │                               │                          │
  │                               │  COMPOSING               │
  │                               │  ─────────────────────   │
  │                               │  in-progress letters     │
  │                               │                          │
  │                               │  EMOTION                 │
  │                               │  ─────────────────────   │
  │                               │  😊  Positive  87%       │
  │                               │                          │
  │                               │  [Clear]                 │
  └───────────────────────────────┴──────────────────────────┘

COMPOSING box shows letters currently being fingerspelled.
When a pause is detected the composed word moves to SENTENCE.
"""

import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame,
)
from PyQt5.QtCore  import QThread, pyqtSignal, Qt
from PyQt5.QtGui   import QImage, QPixmap, QFont

from src.hand_pipeline import HandPipeline
from src.face_pipeline import FacePipeline
from src.sign_model    import SignModel
from src.translator    import Translator

_ICON  = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
_COLOR = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}


# ─────────────────────────────────────────────────────────────────────────────
class VideoThread(QThread):
    frame_ready       = pyqtSignal(np.ndarray)
    sentence_ready    = pyqtSignal(str)
    composing_ready   = pyqtSignal(str)
    emotion_ready     = pyqtSignal(str, float)
    sign_detected     = pyqtSignal(str, float, str)   # label, conf, kind

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._hand    = HandPipeline()
        self._face    = FacePipeline()
        self._model   = SignModel()
        self._trans   = Translator()

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("[VideoThread] ERROR: Cannot open webcam.")
            return

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            keypoints, has_hands, frame = self._hand.process(frame)
            emotion,   emo_conf, frame  = self._face.process(frame)

            if self._model.is_loaded:
                label, conf, kind = self._trans.predict(keypoints, has_hands, self._model)
                if label:
                    self.sign_detected.emit(label, conf, kind)
                self.sentence_ready.emit(self._trans.get_sentence())
                self.composing_ready.emit(self._trans.get_composing())
            else:
                cv2.putText(
                    frame,
                    "No model — run train/collect_data.py then train/train_sign.py",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 90, 255), 2,
                )

            self.emotion_ready.emit(emotion, emo_conf)
            self.frame_ready.emit(frame)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

    def clear_translation(self):
        self._trans.clear()


# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSL Hybrid — Bangla Sign Language Translator")
        self.setMinimumSize(1060, 580)
        self._build_ui()
        self._start_thread()

    def _build_ui(self):
        self.setStyleSheet("background-color: #12122a;")
        root   = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._video_panel(), stretch=3)
        layout.addWidget(self._sidebar(),     stretch=1)
        self.setCentralWidget(root)

    def _video_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: #0d0d1f;")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setMinimumSize(640, 480)
        v.addWidget(self.video_lbl)
        return panel

    def _sidebar(self):
        sidebar = QFrame()
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(400)
        sidebar.setStyleSheet("background-color: #1a1a3a;")

        v = QVBoxLayout(sidebar)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)

        # ── Sentence (committed output) ───────────────────────────────────────
        v.addWidget(self._section_lbl("SENTENCE"))
        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setFont(QFont("Nirmala UI", 22))
        self.sentence_box.setMinimumHeight(120)
        self.sentence_box.setStyleSheet("""
            QTextEdit {
                background-color : #0d1040;
                color            : #d8e0ff;
                border           : 1px solid #2a2a5a;
                border-radius    : 8px;
                padding          : 10px;
            }
        """)
        v.addWidget(self.sentence_box)

        # ── Composing (in-progress fingerspelling) ────────────────────────────
        v.addWidget(self._section_lbl("COMPOSING"))
        self.composing_box = QTextEdit()
        self.composing_box.setReadOnly(True)
        self.composing_box.setFont(QFont("Nirmala UI", 18))
        self.composing_box.setMaximumHeight(60)
        self.composing_box.setStyleSheet("""
            QTextEdit {
                background-color : #0a1a30;
                color            : #60c0ff;
                border           : 1px solid #1a3a5a;
                border-radius    : 6px;
                padding          : 6px;
            }
        """)
        v.addWidget(self.composing_box)

        self.last_sign_lbl = QLabel("Waiting for sign…")
        self.last_sign_lbl.setFont(QFont("Segoe UI", 9))
        self.last_sign_lbl.setStyleSheet("color:#4a5a90; font-style:italic;")
        self.last_sign_lbl.setAlignment(Qt.AlignRight)
        v.addWidget(self.last_sign_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#2a2a5a;")
        v.addWidget(sep)

        # ── Emotion ───────────────────────────────────────────────────────────
        v.addWidget(self._section_lbl("EMOTION"))
        emo_card = QFrame()
        emo_card.setStyleSheet("""
            QFrame {
                background-color : #0d1040;
                border           : 1px solid #2a2a5a;
                border-radius    : 8px;
            }
        """)
        ev = QVBoxLayout(emo_card)
        ev.setContentsMargins(12, 12, 12, 12)
        ev.setSpacing(4)

        self.emo_icon_lbl = QLabel("😐")
        self.emo_icon_lbl.setFont(QFont("Segoe UI Emoji", 30))
        self.emo_icon_lbl.setAlignment(Qt.AlignCenter)
        ev.addWidget(self.emo_icon_lbl)

        self.emo_text_lbl = QLabel("Neutral")
        self.emo_text_lbl.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.emo_text_lbl.setAlignment(Qt.AlignCenter)
        self.emo_text_lbl.setStyleSheet(f"color: {_COLOR['neutral']};")
        ev.addWidget(self.emo_text_lbl)

        self.emo_conf_lbl = QLabel("Confidence: —")
        self.emo_conf_lbl.setFont(QFont("Segoe UI", 9))
        self.emo_conf_lbl.setAlignment(Qt.AlignCenter)
        self.emo_conf_lbl.setStyleSheet("color:#4a5a80;")
        ev.addWidget(self.emo_conf_lbl)

        v.addWidget(emo_card)
        v.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFont(QFont("Segoe UI", 10))
        clear_btn.setStyleSheet("""
            QPushButton         { background:#c0392b; color:white; border:none;
                                  border-radius:6px; padding:9px; }
            QPushButton:hover   { background:#a93226; }
            QPushButton:pressed { background:#922b21; }
        """)
        clear_btn.clicked.connect(self._clear)
        v.addWidget(clear_btn)

        return sidebar

    @staticmethod
    def _section_lbl(text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lbl.setStyleSheet("color:#7080c0; letter-spacing:2px;")
        return lbl

    def _start_thread(self):
        self._thread = VideoThread()
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.sentence_ready.connect(self._on_sentence)
        self._thread.composing_ready.connect(self._on_composing)
        self._thread.emotion_ready.connect(self._on_emotion)
        self._thread.sign_detected.connect(self._on_sign)
        self._thread.start()

    def _on_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self.video_lbl.setPixmap(
            pix.scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_sentence(self, text):
        self.sentence_box.setPlainText(text)

    def _on_composing(self, text):
        self.composing_box.setPlainText(text)

    def _on_sign(self, label, conf, kind):
        tag = 'WORD' if kind == 'word' else 'LETTER'
        self.last_sign_lbl.setText(f"{tag}: {label}  ({conf:.0%})")

    def _on_emotion(self, emotion, conf):
        self.emo_icon_lbl.setText(_ICON.get(emotion, '😐'))
        color = _COLOR.get(emotion, '#f39c12')
        self.emo_text_lbl.setText(emotion.capitalize())
        self.emo_text_lbl.setStyleSheet(f"color: {color};")
        self.emo_conf_lbl.setText(f"Confidence: {conf:.0%}")

    def _clear(self):
        self._thread.clear_translation()
        self.sentence_box.clear()
        self.composing_box.clear()
        self.last_sign_lbl.setText("Waiting for sign…")

    def closeEvent(self, event):
        self._thread.stop()
        event.accept()
