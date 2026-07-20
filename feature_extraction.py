import numpy as np


def extract_two_hand_feature_vector(detection_result):
    """
    Builds a fixed-size 128-length vector representing up to two hands,
    ordered by HANDEDNESS LABEL (not detection order), so the same sign
    always produces the same vector layout regardless of which hand
    MediaPipe happened to list first that frame.

    Layout:
        [0:63]   -> left hand landmarks (zero-filled if absent)
        [63:126] -> right hand landmarks (zero-filled if absent)
        [126]    -> left_hand_present flag (0.0 or 1.0)
        [127]    -> right_hand_present flag (0.0 or 1.0)
    """
    left_vector = np.zeros(63, dtype=np.float32)
    right_vector = np.zeros(63, dtype=np.float32)
    left_present = 0.0
    right_present = 0.0

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness_label = detection_result.handedness[idx][0].category_name

            single_vector = extract_normalized_landmarks(hand_landmarks)

            if handedness_label == "Left":
                left_vector = single_vector
                left_present = 1.0
            elif handedness_label == "Right":
                right_vector = single_vector
                right_present = 1.0
            else:
                # Unexpected/unclassified label — skip rather than guess
                # which slot it belongs in.
                pass
    else:
        pass

    feature_vector = np.concatenate(
        [left_vector, right_vector, np.array([left_present, right_present], dtype=np.float32)]
    )
    return feature_vector