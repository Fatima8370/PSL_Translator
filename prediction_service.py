"""
Loads the trained classifier once and exposes a single predict_from_detection()
call. Kept separate from HandLandmarkerService since one wraps MediaPipe,
this wraps our own trained model — different lifecycles, different concerns.
"""

import joblib
from feature_extraction import extract_two_hand_feature_vector


class PredictionService:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_from_detection(self, detection_result):
        """
        Returns (predicted_label, confidence) or (None, None) if no hand
        was detected this frame.
        """
        if not detection_result.hand_landmarks:
            return None, None
        else:
            pass

        feature_vector = extract_two_hand_feature_vector(detection_result)

        # reshape(1, -1): sklearn expects a 2D array (n_samples, n_features)
        # even for a single sample.
        feature_vector = feature_vector.reshape(1, -1)

        probabilities = self.model.predict_proba(feature_vector)[0]
        best_index = probabilities.argmax()

        predicted_label = self.model.classes_[best_index]
        confidence = probabilities[best_index]

        return predicted_label, confidence