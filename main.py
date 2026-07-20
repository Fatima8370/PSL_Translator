"""
Entry point. Owns the webcam loop and OpenCV window lifecycle.
Delegates detection to HandLandmarkerService and rendering to drawing.py —
this file should stay thin; it's just the conductor.
"""

import cv2
import config
from landmarker_service import HandLandmarkerService
from drawing import draw_hand_landmarks
from feature_extraction import extract_normalized_landmarks



def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX in config.py.")
        return
    else:
        print("[INFO] Webcam initialized successfully.")

    service = HandLandmarkerService()
    print("Camera running. Press 'q' to close window.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("[WARNING] Failed to read frame. Skipping this iteration.")
            continue
        else:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        detection_result = service.detect(frame)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                frame = draw_hand_landmarks(frame, hand_landmarks, w, h)

                feature_vector = extract_normalized_landmarks(hand_landmarks)
                print(feature_vector.shape, feature_vector[24:27])
                            
                
        else:
            # No hand detected this frame — nothing to draw, loop continues normally.
            pass

        cv2.imshow('PSL Translator - Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed. Shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()
    service.close()


if __name__ == "__main__":
    main()