import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
GESTURE_RECOGNIZER_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

FACE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
GESTURE_RECOGNIZER_PATH = os.path.join(MODELS_DIR, "gesture_recognizer.task")

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PANEL_WIDTH = 440

EXPRESSION_PERSIST_SECONDS = 0.8
SMOOTHING_WINDOW_SIZE = 5

THRESHOLDS = {
    "yawn_jaw_open": 0.40,
    "surprised_jaw_open": 0.22,
    "surprised_brow_up": 0.35,
    "surprised_eye_wide": 0.22,
    "confused_max_jaw_open": 0.22,
    "confused_head_tilt_deg": 18.0,
    "confused_brow_asymmetry": 0.24,
    "confused_single_brow_min": 0.32,
    "confused_brow_down": 0.45,
    "like_smile": 0.40,
    "like_max_jaw_open": 0.30,
    "nerd_squint": 0.38,
    "nerd_smile": 0.28,
    "min_confidence": 0.35,
}
