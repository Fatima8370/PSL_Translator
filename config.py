"""
Central place for every tunable constant in the project.
Change thresholds/paths here — never hunt through logic files for a magic number.
"""

MODEL_PATH = "hand_landmarker.task"
MODEL_PATH_CLASSIFIER = "psl_classifier.joblib"
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.6

CAMERA_INDEX = 0

DATASET_ROOT = "dataset"
BURST_DURATION_SECONDS = 5

# Skeleton connections — index pairs matching MediaPipe's 21-point hand model.
# Kept here (not hardcoded in a drawing function) so future changes to hand
# topology or a different landmark model don't require touching draw logic.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (9, 10), (10, 11), (11, 12),           # Middle finger
    (13, 14), (14, 15), (15, 16),          # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base connects
]