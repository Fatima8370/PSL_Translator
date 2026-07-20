import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

base_options = python.BaseOptions(model_asset_path = "hand_landmaker.task")

options = vision.HandLandmarkerOptions(
    base_options = base_options
    hands = 2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.8,    
    running_mode = vision.RunningMode.VIDEO 
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (9, 10), (10, 11), (11, 12),           # Middle finger
    (13, 14), (14, 15), (15, 16),          # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base connects
]

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    print("Camera running. Press 'q' to close window.") 
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Format image colors to MediaPipe's framework standards
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Analyze frame data arrays
        timestamp_ms = int (time.time() * 100)
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # 4. Parse arrays and draw directly to screen
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                # Draw connections (bone lines)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    pt1 = hand_landmarks[start_idx]
                    pt2 = hand_landmarks[end_idx]
                    
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw nodes (knuckle points)
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Modern Tasks Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


