from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .config import FallDetectorConfig
from .heuristics import analyze_keypoints_for_fall, has_lower_body_keypoints
from .geometry import is_bbox_horizontal, bbox_intersect

BBoxXYXY = np.ndarray  # (N,4)
KpsXYC = np.ndarray    # (N,17,3)

@dataclass
class FallState:
    active_zone: bool = False
    red_zone: Optional[Tuple[int, int, int, int]] = None     # (x,y,w,h)
    yellow_zone: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)
    frames_without_lying: int = 0
    last_fall_frame: int = -10**12

@dataclass
class StepOutput:
    detected: bool
    person_index: Optional[int] = None  # quem caiu (mesmo do loop original)
    red_zone: Optional[Tuple[int,int,int,int]] = None
    yellow_zone: Optional[Tuple[int,int,int,int]] = None

class FallDetector:
    """
    Stateful library. Preserves behavior.
    gate_fn: optional, called ONLY when a fall is detected.
            receives the person's keypoints (17,3) and must return True (accept) / False (reject)
    """

    def __init__(self, cfg: Optional[FallDetectorConfig] = None):
        self.cfg = cfg or FallDetectorConfig()
        self.state = FallState()

    def process_frame(self, **kwargs):
        return self.step(**kwargs)

    def step(
        self,
        *,
        kps: Optional[KpsXYC],
        boxes: Optional[BBoxXYXY],
        frame_idx: int,
        fps: float,
        gate_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> StepOutput:
        if kps is None or boxes is None or len(kps) == 0 or len(boxes) == 0:
            return StepOutput(False)

        if not self.state.active_zone:
            return self._detect_new_fall(
                kps=kps, boxes=boxes, frame_idx=frame_idx, fps=fps, gate_fn=gate_fn
            )

        # zona ativa: atualiza (igual ao seu update_ground_fight_zone_video_time)
        assert self.state.yellow_zone is not None
        active, frames_without = self._update_ground_fight_zone(
            kps=kps, boxes=boxes, yellow_zone=self.state.yellow_zone
        )
        self.state.active_zone = active
        self.state.frames_without_lying = frames_without

        if not active:
            self.state.red_zone = None
            self.state.yellow_zone = None

        return StepOutput(False)

    def _detect_new_fall(
        self,
        *,
        kps: KpsXYC,
        boxes: BBoxXYXY,
        frame_idx: int,
        fps: float,
        gate_fn: Optional[Callable[[np.ndarray], bool]],
    ) -> StepOutput:
        cooldown_frames = int(self.cfg.global_cooldown * fps)

        falls = analyze_keypoints_for_fall(
            kps,
            boxes,
            pose_conf_threshold=self.cfg.pose_conf_threshold,
            min_lower_body_kpts=self.cfg.min_lower_body_kpts,
        )

        for i, is_fall in enumerate(falls):
            if is_fall and (frame_idx - self.state.last_fall_frame) > cooldown_frames:

                if gate_fn is not None and not gate_fn(kps[i]):
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                red_zone = (x1, y1, x2 - x1, y2 - y1)

                red_w, red_h = x2 - x1, y2 - y1
                new_w = int(red_w * self.cfg.ground_zone_expansion_factor)
                new_h = int(red_h * self.cfg.ground_zone_expansion_factor)

                dx = (new_w - red_w) // 2
                dy = (new_h - red_h) // 2
                yellow_zone = (x1 - dx, y1 - dy, new_w, new_h)

                # atualiza estado (igual)
                self.state.last_fall_frame = frame_idx
                self.state.active_zone = True
                self.state.red_zone = red_zone
                self.state.yellow_zone = yellow_zone
                self.state.frames_without_lying = 0

                return StepOutput(True, i, red_zone, yellow_zone)

        return StepOutput(False)

    def _update_ground_fight_zone(
        self,
        *,
        kps: KpsXYC,
        boxes: BBoxXYXY,
        yellow_zone: Tuple[int,int,int,int],
    ):
        someone_lying = False

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if not is_bbox_horizontal(x1, y1, x2, y2):
                continue
            if not bbox_intersect((x1, y1, x2, y2), yellow_zone):
                continue
            if not has_lower_body_keypoints(
                kps[i],
                conf_thresh=self.cfg.pose_conf_threshold,
                min_points=self.cfg.min_lower_body_kpts,
            ):
                continue

            someone_lying = True
            break

        if someone_lying:
            return True, 0

        frames_without = self.state.frames_without_lying + 1
        if frames_without >= self.cfg.max_frames_without_lying:
            return False, 0

        return True, frames_without
