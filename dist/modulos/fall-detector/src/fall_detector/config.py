from dataclasses import dataclass

# COCO indices (iguais)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

LOWER_BODY_KEYPOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles

@dataclass(frozen=True)
class FallDetectorConfig:
    global_cooldown: float = 8.0
    max_frames_without_lying: int = 100
    ground_zone_expansion_factor: float = 1.5

    pose_conf_threshold: float = 0.4
    min_lower_body_kpts: int = 5
