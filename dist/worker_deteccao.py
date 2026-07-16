"""Worker que roda a deteccao de verdade (chamado como subprocesso do Python
do SISTEMA, nao do executavel congelado). Precisa de torch/ultralytics/cv2/
numpy instalados no Python que for usado pra chamar este arquivo.

Nao e pra ser executado pelo usuario diretamente -- e chamado internamente
por detector_quedas_app.py via subprocess, passando os mesmos argumentos.

NOTA: esta versao NAO usa o tatame_guard (validacao de que a queda ocorreu
dentro do tatame) -- removido a pedido, pra simplificar o pacote distribuido.
Sem essa validacao, toda queda detectada pela pose e reportada, mesmo que
tenha ocorrido fora do tatame (plateia, arbitro, etc.) -- risco maior de
falso positivo. Ver README.md.
"""
import argparse
import json
import os
import sys
import time

import cv2
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "modulos", "fall-detector", "src"))

from fall_detector import FallDetector, FallDetectorConfig, format_seconds_to_mmss  # noqa: E402

GLOBAL_COOLDOWN = 8.0
MAX_FRAMES_WITHOUT_LYING = 100
GROUND_ZONE_EXPANSION_FACTOR = 1.5
POSE_CONF_THRESHOLD = 0.4
MIN_LOWER_BODY_KPTS = 5


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0

    detector = FallDetector(FallDetectorConfig(
        global_cooldown=GLOBAL_COOLDOWN,
        max_frames_without_lying=MAX_FRAMES_WITHOUT_LYING,
        ground_zone_expansion_factor=GROUND_ZONE_EXPANSION_FACTOR,
        pose_conf_threshold=POSE_CONF_THRESHOLD,
        min_lower_body_kpts=MIN_LOWER_BODY_KPTS,
    ))

    fall_timestamps = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        results = model(frame, verbose=False, device=device)

        kps = None
        boxes = None
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            kps = r.keypoints.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            break

        # Sem gate_fn (tatame_guard removido): toda queda detectada pela pose
        # e aceita, sem validar se ocorreu dentro do tatame.
        out = detector.process_frame(
            kps=kps, boxes=boxes, frame_idx=frame_index, fps=fps,
        )
        if out.detected:
            t = frame_index / fps
            fall_timestamps.append(format_seconds_to_mmss(t))

    cap.release()
    return fall_timestamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--videos-base", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.list, encoding="utf-8") as f:
        items = json.load(f)

    # "yolo11n-pose.pt" e o modelo de pose OFICIAL da Ultralytics (nao treinado
    # por este projeto) -- passado so pelo nome (sem caminho), o proprio
    # ultralytics baixa e faz cache automaticamente na primeira execucao, sem
    # a gente precisar redistribuir o peso deles.
    model = YOLO("yolo11n-pose.pt")

    results = []
    for i, item in enumerate(items):
        video_path = os.path.join(args.videos_base, item["arquivo"])
        t0 = time.time()
        print(f"[{i+1}/{len(items)}] processando {item['arquivo']} ...", flush=True)
        falls = process_video(model, video_path)
        elapsed = time.time() - t0
        print(f"  -> {len(falls)} quedas detectadas: {falls} ({elapsed:.1f}s)", flush=True)
        results.append({
            "video": item["id"],
            "arquivo": item["arquivo"],
            "falls_mm_ss": falls,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRelatorio salvo em: {args.output}")


if __name__ == "__main__":
    main()
