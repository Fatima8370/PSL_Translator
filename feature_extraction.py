"""
Converts raw MediaPipe hand_landmarks into a normalized, flat feature vector
suitable for ML training. This is the shared logic used by BOTH the dataset
collector (writing labeled rows to disk) and the live predictor (feeding a
single vector into the trained model for inference).

Keeping this in one place guarantees training and inference see data
processed IDENTICALLY — a mismatch here is one of the most common silent
bugs in landmark-based classifiers.
"""

import numpy as np


def extract_normalized_landmarks(hand_landmarks):
    """
    hand_landmarks: one hand's list of 21 landmark objects, each with .x, .y, .z
    (this is a single element from detection_result.hand_landmarks).

    Returns: a NumPy array of shape (63,) — 21 points * 3 coordinates,
    translated so the wrist is the origin and scaled by hand size.
    """
    # Step 1: Pull raw coordinates into a NumPy array for vectorized math.
    # Shape: (21, 3)
    raw_points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # --- Explicit check instead of trusting the input blindly ---
    if raw_points.shape != (21, 3):
        # This should never happen if MediaPipe returned a valid hand, but
        # if it does, we want a loud, specific failure — not a silent
        # shape mismatch three layers downstream in model training.
        raise ValueError(
            f"Expected 21 landmarks with 3 coords each, got shape {raw_points.shape}"
        )
    else:
        pass

    # Step 2: Translation invariance — subtract the wrist (landmark 0)
    # from every point. Wrist becomes (0, 0, 0); everything else is now
    # expressed relative to it.
    wrist = raw_points[0]
    translated_points = raw_points - wrist

    # Step 3: Scale invariance — normalize by the distance between the
    # wrist (0) and the middle finger MCP joint (9). This joint is a
    # reliable, stable reference regardless of finger pose.
    reference_distance = np.linalg.norm(translated_points[9])

    if reference_distance == 0:
        # Degenerate case: wrist and middle-MCP landmarks perfectly overlap.
        # Practically near-impossible with a real hand, but we guard
        # explicitly rather than risk a divide-by-zero crash.
        reference_distance = 1e-6
    else:
        pass

    normalized_points = translated_points / reference_distance

    # Step 4: Flatten (21, 3) -> (63,) for feeding into a classifier.
    feature_vector = normalized_points.flatten()

    return feature_vector


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