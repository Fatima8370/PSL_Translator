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