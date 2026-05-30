"""
BSL Hybrid Translator
=====================
Hybrid Bangla Sign Language translator combining:
  - Word-level Transformer for common BSL lexical signs
  - Avro-style fingerspelling for unlimited vocabulary

Run from the project root:
    python main.py
"""

import sys

# Must be imported before QApplication to avoid segfault on Windows
import mediapipe  # noqa: F401
import cv2        # noqa: F401

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt
from src.ui          import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
