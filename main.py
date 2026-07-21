import cv2
import config
from landmarker_service import HandLandmarkerService
from drawing import draw_hand_landmarks, draw_prediction_box
from prediction_service import PredictionService


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX in config.py.")
        return
    else:
        print("[INFO] Webcam initialized successfully.")

    service = HandLandmarkerService()
    predictor = PredictionService(config.MODEL_PATH_CLASSIFIER)
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
        else:
            pass

        predicted_label, confidence = predictor.predict_from_detection(detection_result)
        frame = draw_prediction_box(frame, predicted_label, confidence)

        cv2.imshow('PSL Translator - Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed. Shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()
    service.close()


if __name__ == "__main__":
    main()