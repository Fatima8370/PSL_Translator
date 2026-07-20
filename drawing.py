"""
Pure rendering logic. Takes already-detected landmark data and draws it
onto a frame. Has zero knowledge of MediaPipe's detection API — if we ever
swap detection backends, this file doesn't change at all.
"""

import cv2
import config


def draw_hand_landmarks(frame, hand_landmarks, frame_width, frame_height):
    """
    hand_landmarks: a single hand's list of 21 normalized landmark points
    (one element from detection_result.hand_landmarks).
    """
    # Draw connections (bone lines) first, so landmark dots render on top
    # of the lines rather than being partially covered by them.
    for start_idx, end_idx in config.HAND_CONNECTIONS:
        pt1 = hand_landmarks[start_idx]
        pt2 = hand_landmarks[end_idx]

        x1, y1 = int(pt1.x * frame_width), int(pt1.y * frame_height)
        x2, y2 = int(pt2.x * frame_width), int(pt2.y * frame_height)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw nodes (knuckle points)
    for landmark in hand_landmarks:
        cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    return frame