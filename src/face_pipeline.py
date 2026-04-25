"""
Facial emotion detection using the FER library.

Raw FER emotions are bucketed into three categories:
  positive  →  happy, surprise
  negative  →  angry, disgust, fear, sad
  neutral   →  neutral

A short history buffer smooths out per-frame flicker.
"""

import cv2
from collections import Counter, deque

# ── mappings ──────────────────────────────────────────────────────────────────
EMOTION_MAP = {
    'happy':    'positive',
    'surprise': 'positive',
    'angry':    'negative',
    'disgust':  'negative',
    'fear':     'negative',
    'sad':      'negative',
    'neutral':  'neutral',
}

# BGR colours for each category
EMOTION_COLORS = {
    'positive': (80,  220, 60),
    'negative': (50,  50,  230),
    'neutral':  (0,   165, 255),
}

HISTORY_SIZE = 6    # frames kept for temporal smoothing


class FacePipeline:
    def __init__(self):
        self._available       = False
        self._detector        = None
        self._history         = deque(maxlen=HISTORY_SIZE)
        self._last_emotion    = 'neutral'
        self._last_confidence = 0.0

        try:
            from fer import FER
            # mtcnn=False uses OpenCV Haar cascades — lighter, no extra dep
            self._detector  = FER(mtcnn=False)
            self._available = True
            print("[FacePipeline] FER loaded successfully.")
        except ImportError:
            print("[FacePipeline] 'fer' package not found — emotion detection disabled.")
        except Exception as exc:
            print(f"[FacePipeline] Could not initialise FER: {exc}")

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Detect dominant emotion from the first face in the frame.

        Returns
        -------
        emotion    : str    ('positive' | 'negative' | 'neutral')
        confidence : float  raw confidence of dominant raw emotion (0–1)
        annotated  : BGR frame with bounding box + label drawn
        """
        if not self._available:
            return self._last_emotion, self._last_confidence, frame

        annotated = frame.copy()

        try:
            results = self._detector.detect_emotions(frame)
        except Exception:
            return self._last_emotion, self._last_confidence, annotated

        if not results:
            return self._last_emotion, self._last_confidence, annotated

        face     = results[0]
        emotions = face.get('emotions', {})
        box      = face.get('box')

        if not emotions:
            return self._last_emotion, self._last_confidence, annotated

        neg_score = (emotions.get('angry', 0) + emotions.get('disgust', 0)
                     + emotions.get('fear', 0) + emotions.get('sad', 0))
        pos_score = emotions.get('happy', 0) + emotions.get('surprise', 0)
        neu_score = emotions.get('neutral', 0)

        if neg_score > pos_score and neg_score > neu_score and neg_score >= 0.25:
            mapped = 'negative'
            raw_conf = float(neg_score)
        elif pos_score > neu_score and pos_score >= 0.25:
            mapped = 'positive'
            raw_conf = float(pos_score)
        elif neu_score >= 0.25:
            mapped = 'neutral'
            raw_conf = float(neu_score)
        else:
            # all scores too low — keep last known emotion
            return self._last_emotion, self._last_confidence, annotated

        dominant_raw = max(emotions, key=emotions.get)  # for label display only

        # temporal smoothing
        self._history.append(mapped)
        smoothed = Counter(self._history).most_common(1)[0][0]

        self._last_emotion    = smoothed
        self._last_confidence = raw_conf

        # draw bounding box + label
        if box is not None:
            x, y, w, h = box
            color = EMOTION_COLORS.get(smoothed, (200, 200, 200))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label_txt = f"{smoothed}  {raw_conf:.0%}"
            cv2.putText(
                annotated, label_txt,
                (x, max(y - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            )

        return smoothed, raw_conf, annotated
