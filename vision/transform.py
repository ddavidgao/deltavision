"""
Similarity-transform detection for motion-compensated delta observation.

Detects pan, zoom, and rotate as 4-DOF similarity transforms using ORB
keypoints + RANSAC. If confirmed, warps t0 to align with t1 so the diff
engine sees only genuinely new content (the residual), not content that
merely moved or rescaled.

This is the same principle as H.264 P-frame inter-prediction: factor out
motion, transmit the residual. Applied here to agent observation instead
of video compression.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class TransformResult:
    detected: bool         # True if a similarity transform was confirmed
    inlier_ratio: float    # fraction of matched keypoints fitting the model
    warped_t0: Image.Image | None  # t0 warped to align with t1 (if detected)


def detect_similarity(
    t0: Image.Image,
    t1: Image.Image,
    min_inlier_ratio: float = 0.5,
) -> TransformResult:
    """
    Detect if t0 → t1 is a similarity transform (pan, zoom, rotate).

    Uses ORB keypoints + BFMatcher + RANSAC-fitted estimateAffinePartial2D
    (4-DOF: tx, ty, rotation, uniform scale). Excludes shearing and
    perspective warp — only transforms that preserve shape and angles.

    Returns warped t0 aligned to t1 when inlier_ratio >= min_inlier_ratio.
    Returns detected=False when frames have insufficient features (e.g. solid
    color screens) or when the inlier ratio is too low (genuinely new content).
    """
    img0 = np.array(t0.convert("L"))
    img1 = np.array(t1.convert("L"))

    orb = cv2.ORB_create(500)
    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)

    if des0 is None or des1 is None or len(kp0) < 10 or len(kp1) < 10:
        return TransformResult(detected=False, inlier_ratio=0.0, warped_t0=None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des0, des1)

    if len(matches) < 8:
        return TransformResult(detected=False, inlier_ratio=0.0, warped_t0=None)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    if M is None or inliers is None:
        return TransformResult(detected=False, inlier_ratio=0.0, warped_t0=None)

    inlier_ratio = float(inliers.sum()) / len(matches)
    if inlier_ratio < min_inlier_ratio:
        return TransformResult(detected=False, inlier_ratio=inlier_ratio, warped_t0=None)

    h, w = img1.shape
    warped = cv2.warpAffine(np.array(t0.convert("RGB")), M, (w, h))
    return TransformResult(
        detected=True,
        inlier_ratio=inlier_ratio,
        warped_t0=Image.fromarray(warped),
    )
