import sys
import time
import cv2
import numpy as np

from src.config import (
    FACE_LANDMARKER_URL,
    FACE_LANDMARKER_PATH,
    GESTURE_RECOGNIZER_URL,
    GESTURE_RECOGNIZER_PATH,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PANEL_WIDTH,
)
from src.utils import download_model_if_missing, load_reaction_images
from src.detectors import VisionPipeline
from src.classifier import ExpressionClassifier
from src.ui import (
    draw_hud,
    draw_debug_overlay,
    draw_mesh_points,
    create_placeholder_panel,
    create_reaction_panel,
)


def initialize_camera(preferred_idx: int = 0) -> cv2.VideoCapture:
    candidates = [preferred_idx] + [i for i in [0, 1, 2] if i != preferred_idx]

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                print(f"Webcam iniciada com sucesso (indice {idx})")
                return cap
            cap.release()

    if sys.platform.startswith("win"):
        for idx in [0, 1]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    print(f"Webcam iniciada com DirectShow no Windows (indice {idx})")
                    return cap
                cap.release()

    print("[Erro] Nenhuma webcam conectada ou acessivel.", file=sys.stderr)
    print("Dicas de solucao:")
    print(" 1. Verifique se a camera nao esta sendo utilizada por outro app (Zoom, Teams, Meet).")
    print(" 2. Verifique as permissoes de camera do sistema operacional.")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Reconhecimento Facial e Gestos - Detector de Expressões")
    print("=" * 60)

    download_model_if_missing(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    download_model_if_missing(GESTURE_RECOGNIZER_URL, GESTURE_RECOGNIZER_PATH)

    reactions = load_reaction_images()
    print(f"Imagens carregadas: {list(reactions.keys())}")

    print("Inicializando detectores MediaPipe...")
    pipeline = VisionPipeline()
    classifier = ExpressionClassifier()

    cap = initialize_camera()

    window_name = "Face Recognition & Reaction"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    show_debug = True
    show_mesh = False
    calibrated_feedback_timer = 0.0

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 30.0

    print("\nCâmera em execução!")
    print("Atalhos:")
    print("  [C] Calibrar postura neutra (olhe normalmente para a câmera e pressione C)")
    print("  [D] Alternar métricas em tempo real (debug)")
    print("  [M] Alternar visualização da malha facial")
    print("  [Q] Sair\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            landmarks, blendshapes, roll_angle, gestures = pipeline.process_frame(frame)

            if show_mesh and landmarks:
                draw_mesh_points(frame, landmarks)

            active_exp, confidence, info_text, metrics = classifier.evaluate(
                blendshapes, roll_angle, gestures
            )

            fps_frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now

            is_showing_calibrated = (now - calibrated_feedback_timer) < 1.5
            draw_hud(frame, info_text, fps, show_debug, show_mesh, is_showing_calibrated)
            if show_debug:
                draw_debug_overlay(frame, metrics)

            if active_exp in reactions:
                reaction_panel = create_reaction_panel(
                    reactions[active_exp], active_exp, confidence, PANEL_WIDTH, frame_h
                )
            else:
                reaction_panel = create_placeholder_panel(PANEL_WIDTH, frame_h)

            combined = np.hstack([frame, reaction_panel])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("c"), ord("C")):
                classifier.calibrate_neutral(roll_angle, blendshapes)
                calibrated_feedback_timer = now
                print(f"[Calibração] Postura neutra definida! Ângulo base: {roll_angle:.1f}°")
            elif key in (ord("d"), ord("D")):
                show_debug = not show_debug
                print(f"Painel de métricas (debug): {'Ligado' if show_debug else 'Desligado'}")
            elif key in (ord("m"), ord("M")):
                show_mesh = not show_mesh
                print(f"Malha facial: {'Ligada' if show_mesh else 'Desligada'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Aplicação encerrada com sucesso.")


if __name__ == "__main__":
    main()
