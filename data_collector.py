"""
Timed-burst dataset collector.

Workflow:
1. You specify dialect + label once at startup.
2. Press 'c' to start a 5-second recording burst (timer shown on screen).
3. Every frame during the burst, if at least one hand is detected, a 128-length
   feature vector is appended to a buffer.
4. When the burst ends, the buffer is flushed to
   dataset/dialect_X/LABEL.csv (append mode).
5. Press 'c' again to record another burst for the same label, or 'q' to quit.
"""

import os
import csv
import time
import cv2

import config
from landmarker_service import HandLandmarkerService
from drawing import draw_hand_landmarks, draw_recording_overlay
from feature_extraction import extract_two_hand_feature_vector


def get_csv_path(dialect, label):
    folder = os.path.join(config.DATASET_ROOT, f"dialect_{dialect}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{label}.csv")


def append_rows_to_csv(csv_path, rows):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [f"f{i}" for i in range(128)]
            writer.writerow(header)
        else:
            pass

        for row in rows:
            writer.writerow(row.tolist())


def run_burst_capture(cap, service, dialect, label):
    rows_buffer = []
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        seconds_remaining = config.BURST_DURATION_SECONDS - elapsed

        if seconds_remaining <= 0:
            break
        else:
            pass

        success, frame = cap.read()

        if not success:
            print("[WARNING] Frame read failed during burst. Skipping frame.")
            continue
        else:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        detection_result = service.detect(frame)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                frame = draw_hand_landmarks(frame, hand_landmarks, w, h)

            feature_vector = extract_two_hand_feature_vector(detection_result)
            rows_buffer.append(feature_vector)
        else:
            # No hand this frame — explicitly skip, do NOT record a garbage row.
            pass

        frame = draw_recording_overlay(frame, label, seconds_remaining)
        cv2.imshow("PSL Data Collector", frame)
        cv2.waitKey(1)

    csv_path = get_csv_path(dialect, label)
    append_rows_to_csv(csv_path, rows_buffer)
    print(f"[INFO] Saved {len(rows_buffer)} frames to {csv_path}")


def main():
    dialect = input("Enter dialect number (e.g. 1 or 2): ").strip()
    label = input("Enter label to record (e.g. A, B, 5): ").strip()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    else:
        print("[INFO] Webcam initialized.")

    service = HandLandmarkerService()
    print(f"Ready. Recording dialect_{dialect}/{label}")
    print("Press 'c' to start a 5-second burst. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue
        else:
            frame = cv2.flip(frame, 1)

        cv2.imshow("PSL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            run_burst_capture(cap, service, dialect, label)
        elif key == ord('q'):
            print("[INFO] Quitting.")
            break
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()
    service.close()


if __name__ == "__main__":
    main()