import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.config import FACE_LANDMARKER_PATH, GESTURE_RECOGNIZER_PATH
from src.utils import calculate_head_tilt


class VisionPipeline:

    def __init__(self):
        face_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        gesture_base_options = python.BaseOptions(model_asset_path=GESTURE_RECOGNIZER_PATH)
        gesture_options = vision.GestureRecognizerOptions(
            base_options=gesture_base_options,
            num_hands=2,
        )
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    def process_frame(self, frame_bgr):
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = self.face_landmarker.detect(mp_image)
        gesture_result = self.gesture_recognizer.recognize(mp_image)

        gestures = []
        if gesture_result.gestures:
            for gesture_list in gesture_result.gestures:
                for g in gesture_list:
                    gestures.append(g.category_name)

        face_landmarks = None
        blendshapes = {}
        roll_angle = 0.0

        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face_landmarks = face_result.face_landmarks[0]
            roll_angle = calculate_head_tilt(face_landmarks, frame_w, frame_h)

            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                blendshapes = {b.category_name: b.score for b in face_result.face_blendshapes[0]}

        return face_landmarks, blendshapes, roll_angle, gestures
