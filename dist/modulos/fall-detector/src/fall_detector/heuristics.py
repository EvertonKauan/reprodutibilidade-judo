import numpy as np
from typing import List

from .config import (
    LOWER_BODY_KEYPOINTS,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
)
from .geometry import euclidean_distance

def has_lower_body_keypoints(keypoints, conf_thresh=0.4, min_points=2) -> bool:
    count = 0
    for idx in LOWER_BODY_KEYPOINTS:
        _, _, conf = keypoints[idx]
        if conf >= conf_thresh:
            count += 1
        if count >= min_points:
            return True
    return False

def analyze_keypoints_for_fall(
    keypoints_xyc: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    pose_conf_threshold: float,
    min_lower_body_kpts: int,
) -> List[bool]:
    falls: List[bool] = []

    for i in range(len(keypoints_xyc)):
        kp = keypoints_xyc[i]

        if not has_lower_body_keypoints(
            kp,
            conf_thresh=pose_conf_threshold,
            min_points=min_lower_body_kpts
        ):
            continue

        x1, y1, x2, y2 = bbox_xyxy[i].astype(int)

        sL, sR = kp[LEFT_SHOULDER][:2], kp[RIGHT_SHOULDER][:2]
        hL, hR = kp[LEFT_HIP][:2], kp[RIGHT_HIP][:2]

        shoulder_center = (sL + sR) / 2
        hip_center = (hL + hR) / 2

        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

        height = abs(y2 - y1)
        width = abs(x2 - x1)

        if height <= 0:
            falls.append(False)
            continue

        dist_sh_hip = np.mean([
            euclidean_distance(sL, hL),
            euclidean_distance(sR, hR)
        ])

        # PASSO 1 — remove cabeças
        if dist_sh_hip < height * 0.15:
            falls.append(False)
            continue

        # PASSO 2 — remove bbox inflada
        pose_y = kp[:, 1]
        pose_height = np.max(pose_y) - np.min(pose_y)
        if pose_height < height * 0.4:
            falls.append(False)
            continue

        is_horizontal = width > height * 1.25
        is_angled = angle > 40
        is_compressed = dist_sh_hip < height * 0.55

        falls.append(bool(is_horizontal and is_angled and is_compressed))

    return falls
