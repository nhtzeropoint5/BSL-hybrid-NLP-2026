"""
Hand + upper-body landmark extraction using MediaPipe Hands and Pose.

Each frame produces a 258-dimensional feature vector matching the paper:
  [0:63]    left  hand  — 21 landmarks × (x, y, z)          normalised to wrist
  [63:126]  right hand  — 21 landmarks × (x, y, z)          normalised to wrist
  [126:258] upper body  — 33 pose landmarks × (x, y, z, vis) normalised to shoulders

Zeros fill any absent hand or absent pose.

Why 258?
  126 (hands) + 132 (pose) = 258
  Pose adds arm position, shoulder orientation, and body lean —
  crucial for BSL word signs that differ by arm location, not just hand shape.
"""

import cv2
import mediapipe as mp
import numpy as np

# Pose landmark indices used for normalisation
_L_SHOULDER = 11
_R_SHOULDER = 12

_WHITE = (255, 255, 255)


class HandPipeline:
    def __init__(self):
        self._mp_hands   = mp.solutions.hands
        self._mp_pose    = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils

        try:
            self._styles     = mp.solutions.drawing_styles
            self._has_styles = True
        except AttributeError:
            self._has_styles = False

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1   # 1 = landscape model, more accurate
        )

    # ── background removal ────────────────────────────────────────────────────

    def _remove_bg(self, frame):
        """Replace background with white using MediaPipe Selfie Segmentation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._seg.process(rgb)
        mask = (result.segmentation_mask > 0.5).astype(np.uint8)
        white = np.full_like(frame, 255)
        return np.where(mask[:, :, np.newaxis], frame, white)

    # ── hand helpers ──────────────────────────────────────────────────────────

    def _normalise_hand(self, arr):
        """arr: (21, 3). Position- and scale-normalised relative to wrist."""
        arr   = arr - arr[0]
        scale = np.linalg.norm(arr[9])
        if scale > 1e-6:
            arr = arr / scale
        return arr

    def _extract_hands(self, hand_results):
        """Return (126,) from MediaPipe Hands results."""
        lh = np.zeros(63)
        rh = np.zeros(63)

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for lms, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                arr   = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
                arr   = self._normalise_hand(arr)
                label = handedness.classification[0].label
                if label == 'Left':
                    lh = arr.flatten()
                else:
                    rh = arr.flatten()

        return np.concatenate([lh, rh])   # (126,)

    # ── pose helpers ──────────────────────────────────────────────────────────

    def _extract_pose(self, pose_results):
        """
        Return (132,) from MediaPipe Pose results.
        33 landmarks × 4 values (x, y, z, visibility).
        Normalised relative to shoulder midpoint and shoulder width.
        """
        pose = np.zeros(33 * 4, dtype=np.float32)

        if not pose_results.pose_landmarks:
            return pose

        lms = pose_results.pose_landmarks.landmark

        # reference point: midpoint between shoulders
        ls, rs  = lms[_L_SHOULDER], lms[_R_SHOULDER]
        mid_x   = (ls.x + rs.x) / 2
        mid_y   = (ls.y + rs.y) / 2
        mid_z   = (ls.z + rs.z) / 2
        scale   = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
        if scale < 1e-6:
            scale = 1.0

        for i, lm in enumerate(lms):
            pose[i * 4]     = (lm.x - mid_x) / scale
            pose[i * 4 + 1] = (lm.y - mid_y) / scale
            pose[i * 4 + 2] = (lm.z - mid_z) / scale
            pose[i * 4 + 3] = lm.visibility

        return pose   # (132,)

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Process one BGR frame.

        Returns
        -------
        keypoints  : np.ndarray (258,)  — [hands (126) | pose (132)]
        has_hands  : bool
        annotated  : BGR frame with hand + pose skeleton overlay
        """
        clean = self._remove_bg(frame)
        rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = self._hands.process(rgb)
        pose_results = self._pose.process(rgb)
        rgb.flags.writeable = True

        annotated = clean.copy()
        has_hands = bool(hand_results.multi_hand_landmarks)

        # draw hand skeletons
        if has_hands:
            for lms in hand_results.multi_hand_landmarks:
                if self._has_styles:
                    self._mp_drawing.draw_landmarks(
                        annotated, lms,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._styles.get_default_hand_landmarks_style(),
                        self._styles.get_default_hand_connections_style(),
                    )
                else:
                    self._mp_drawing.draw_landmarks(
                        annotated, lms, self._mp_hands.HAND_CONNECTIONS)

        # draw pose skeleton (upper body only — less visual clutter)
        if pose_results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                pose_results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(80, 200, 80), thickness=1, circle_radius=2),
                connection_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(80, 200, 80), thickness=1),
            )

        hand_kp = self._extract_hands(hand_results)   # (126,)
        pose_kp = self._extract_pose(pose_results)     # (132,)
        keypoints = np.concatenate([hand_kp, pose_kp]) # (258,)

        return keypoints, has_hands, annotated
