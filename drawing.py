"""
Pure rendering logic. Takes already-detected landmark data and draws it
onto a frame. Has zero knowledge of MediaPipe's detection API — if we ever
swap detection backends, this file doesn't change at all.
"""

import cv2
import config

def draw_recording_overlay(frame, label_text, seconds_remaining):
    """
    Clean, fixed-position overlay shown only DURING an active burst capture.
    Separate from draw_hand_landmarks — this is status text, not skeleton data.
    """
    padding = 10
    bar_height = 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"RECORDING: {label_text}", (padding, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    cv2.putText(frame, f"Time left: {seconds_remaining:.1f}s", (padding, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame

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


def draw_prediction_box(frame, predicted_label, confidence):
    """
    Semi-transparent box, fixed to the top-right corner, showing the
    model's current predicted letter/number and its confidence.
    Kept separate from draw_recording_overlay (which is collection-only UI).
    """
    box_width = 160
    box_height = 90
    margin = 15

    frame_h, frame_w, _ = frame.shape
    x1 = frame_w - box_width - margin
    y1 = margin
    x2 = frame_w - margin
    y2 = margin + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    # 0.45 opacity keeps the camera feed clearly visible behind the box.
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Border for definition against busy backgrounds.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    label_text = predicted_label if predicted_label is not None else "-"

    # Centered-ish letter, large and clear.
    cv2.putText(frame, label_text, (x1 + 55, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 120), 3)

    if confidence is not None:
        conf_text = f"{confidence * 100:.0f}%"
        cv2.putText(frame, conf_text, (x1 + 10, y1 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        pass

    return frame