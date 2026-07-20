"""
Owns the MediaPipe HandLandmarker lifecycle and the raw detection call.
Deliberately knows NOTHING about OpenCV drawing — only responsible for
turning a raw BGR frame into a detection_result object.

This is the module future scripts (data_collector.py, live_predictor.py)
will import directly, since they need landmark coordinates, not visuals.
"""

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


class HandLandmarkerService:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO
        )

        # create_from_options gives us the actual usable landmarker object.
        # We hold onto it as self.landmarker so detect() can reuse it every
        # frame instead of re-creating the model (which would be very slow).
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame):
        """
        Takes a raw BGR frame straight from cv2.VideoCapture.
        Returns the MediaPipe detection_result object.

        Conversion + timestamping lives here so every caller (main.py,
        data_collector.py, live_predictor.py) gets identical, correct
        preprocessing — no risk of one script forgetting the RGB conversion.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return detection_result

    def close(self):
        """Explicit cleanup — releases the underlying MediaPipe graph resources."""
        self.landmarker.close()