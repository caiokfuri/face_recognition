import os
import sys
import urllib.request
import cv2
import numpy as np
from src.config import IMAGES_DIR


def download_model_if_missing(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    filename = os.path.basename(dest_path)
    print(f"Baixando modelo {filename}...")

    def progress_callback(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 25
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
        sys.stdout.write("\n")
        print(f"Modelo {filename} baixado com sucesso.")
    except Exception as e:
        sys.stdout.write("\n")
        print(f"Erro ao baixar modelo {filename}: {e}", file=sys.stderr)
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise


def load_reaction_images() -> dict:
    image_files = {
        "surprised": "surprised.jpg",
        "yawn": "yawn.jpg",
        "confused": "confused.jpg",
        "like": "like.jpg",
        "nerd": "nerd.jpg",
    }
    loaded = {}
    for key, filename in image_files.items():
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                loaded[key] = img
            else:
                print(f"[Aviso] Falha ao ler imagem: {path}")
        else:
            print(f"[Aviso] Arquivo de imagem não encontrado: {path}")
    return loaded


def resize_and_pad(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def calculate_head_tilt(landmarks, frame_w: int = 640, frame_h: int = 480) -> float:
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = (right_eye.x - left_eye.x) * frame_w
    dy = (right_eye.y - left_eye.y) * frame_h
    angle = np.degrees(np.arctan2(dy, dx))
    return float(angle)
