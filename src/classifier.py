import time
from collections import deque, Counter
from src.config import THRESHOLDS, EXPRESSION_PERSIST_SECONDS, SMOOTHING_WINDOW_SIZE


class ExpressionClassifier:

    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
        self.last_valid_expression = "neutral"
        self.last_expression_time = 0.0

        self.baseline_tilt = 0.0
        self.baseline_asymmetry = 0.0
        self.calibrated = False

    def calibrate_neutral(self, roll_angle: float, blendshapes: dict):
        self.baseline_tilt = roll_angle
        brow_outer_up_left = blendshapes.get("browOuterUpLeft", 0.0)
        brow_outer_up_right = blendshapes.get("browOuterUpRight", 0.0)
        self.baseline_asymmetry = abs(brow_outer_up_left - brow_outer_up_right)
        self.calibrated = True

    def evaluate(self, blendshapes: dict, roll_angle: float, gestures: list):
        if not self.calibrated and blendshapes:
            self.calibrate_neutral(roll_angle, blendshapes)

        jaw_open = blendshapes.get("jawOpen", 0.0)
        brow_inner_up = blendshapes.get("browInnerUp", 0.0)
        brow_outer_up_left = blendshapes.get("browOuterUpLeft", 0.0)
        brow_outer_up_right = blendshapes.get("browOuterUpRight", 0.0)
        avg_brow_outer_up = (brow_outer_up_left + brow_outer_up_right) / 2.0
        max_single_brow_up = max(brow_outer_up_left, brow_outer_up_right)

        brow_down_left = blendshapes.get("browDownLeft", 0.0)
        brow_down_right = blendshapes.get("browDownRight", 0.0)
        avg_brow_down = (brow_down_left + brow_down_right) / 2.0

        raw_asymmetry = abs(brow_outer_up_left - brow_outer_up_right)
        effective_asymmetry = max(0.0, raw_asymmetry - self.baseline_asymmetry)

        effective_tilt = roll_angle - self.baseline_tilt

        eye_wide_left = blendshapes.get("eyeWideLeft", 0.0)
        eye_wide_right = blendshapes.get("eyeWideRight", 0.0)
        avg_eye_wide = (eye_wide_left + eye_wide_right) / 2.0

        mouth_smile_left = blendshapes.get("mouthSmileLeft", 0.0)
        mouth_smile_right = blendshapes.get("mouthSmileRight", 0.0)
        avg_smile = (mouth_smile_left + mouth_smile_right) / 2.0

        eye_squint_left = blendshapes.get("eyeSquintLeft", 0.0)
        eye_squint_right = blendshapes.get("eyeSquintRight", 0.0)
        avg_squint = (eye_squint_left + eye_squint_right) / 2.0

        has_thumb_up = "Thumb_Up" in gestures
        has_pointing_up = "Pointing_Up" in gestures

        scores = {
            "yawn": 0.0,
            "surprised": 0.0,
            "confused": 0.0,
            "like": 0.0,
            "nerd": 0.0,
        }

        if has_thumb_up:
            scores["like"] = 0.98
        if has_pointing_up:
            scores["nerd"] = 0.98

        is_mouth_open = jaw_open >= THRESHOLDS["surprised_jaw_open"]

        if is_mouth_open:
            is_surprised_brows = (
                brow_inner_up >= THRESHOLDS["surprised_brow_up"]
                or avg_brow_outer_up >= THRESHOLDS["surprised_brow_up"]
                or avg_eye_wide >= THRESHOLDS["surprised_eye_wide"]
            )

            if is_surprised_brows:
                s_jaw = min(1.0, jaw_open / 0.45)
                s_brow = min(1.0, max(brow_inner_up, avg_brow_outer_up) / 0.45)
                s_eye = min(1.0, avg_eye_wide / 0.30)
                scores["surprised"] = 0.4 * s_jaw + 0.4 * s_brow + 0.2 * s_eye
            elif jaw_open >= THRESHOLDS["yawn_jaw_open"]:
                scores["yawn"] = min(1.0, (jaw_open - THRESHOLDS["yawn_jaw_open"]) / 0.30 + 0.65)

        if jaw_open < THRESHOLDS["confused_max_jaw_open"]:
            tilt_thresh = THRESHOLDS["confused_head_tilt_deg"]
            asym_thresh = THRESHOLDS["confused_brow_asymmetry"]
            single_brow_thresh = THRESHOLDS["confused_single_brow_min"]
            down_thresh = THRESHOLDS["confused_brow_down"]

            tilt_score = 0.0
            if abs(effective_tilt) >= tilt_thresh:
                tilt_score = min(1.0, (abs(effective_tilt) - tilt_thresh) / 12.0 + 0.55)

            brow_asym_score = 0.0
            if effective_asymmetry >= asym_thresh and max_single_brow_up >= single_brow_thresh:
                brow_asym_score = min(1.0, effective_asymmetry / 0.35 + 0.55)

            frown_score = 0.0
            if avg_brow_down >= down_thresh and 0.15 <= avg_smile <= 0.45:
                frown_score = min(1.0, avg_brow_down / 0.55 + 0.50)

            scores["confused"] = max(tilt_score, brow_asym_score, frown_score)
        else:
            scores["confused"] = 0.0

        if scores["like"] == 0.0 and jaw_open < THRESHOLDS["like_max_jaw_open"]:
            if avg_smile >= THRESHOLDS["like_smile"]:
                scores["like"] = min(1.0, (avg_smile - THRESHOLDS["like_smile"]) / 0.30 + 0.55)

        if scores["nerd"] == 0.0 and jaw_open < 0.25:
            if avg_squint >= THRESHOLDS["nerd_squint"] and avg_smile >= THRESHOLDS["nerd_smile"]:
                scores["nerd"] = min(1.0, (avg_squint + avg_smile) / 1.1)

        min_conf = THRESHOLDS["min_confidence"]
        best_exp = "neutral"
        best_score = 0.0

        for exp, sc in scores.items():
            if sc >= min_conf and sc > best_score:
                best_score = sc
                best_exp = exp

        self.history.append(best_exp)
        counts = Counter(self.history)
        most_common, freq = counts.most_common(1)[0]

        now = time.time()
        if most_common != "neutral" and freq >= 2:
            active_expression = most_common
            self.last_valid_expression = active_expression
            self.last_expression_time = now
            confidence = scores.get(active_expression, best_score)
        elif (now - self.last_expression_time) < EXPRESSION_PERSIST_SECONDS:
            active_expression = self.last_valid_expression
            confidence = scores.get(active_expression, 0.7)
        else:
            active_expression = "neutral"
            confidence = 0.0

        descriptions = {
            "surprised": "SURPRESO: Espanto detectado! 😮",
            "yawn": "BOCEJO: Boca bem aberta! 🥱",
            "confused": "CONFUSO: Cabeca inclinada ou duvida! 🤨",
            "like": "LIKE: Positivo / Sorriso! 👍",
            "nerd": "NERD: Dedo apontado ou expressao nerd! 🤓",
            "neutral": "Neutro / Aguardando expressao...",
        }
        description = descriptions.get(active_expression, descriptions["neutral"])

        raw_metrics = {
            "Boca Aberta": jaw_open,
            "Sorriso": avg_smile,
            "Sobrancelha Cima": max(brow_inner_up, avg_brow_outer_up),
            "Inclinacao Relat.": abs(effective_tilt),
            "Gesto Detectado": gestures[0] if gestures else "Nenhum",
            "Score Yawn": scores["yawn"],
            "Score Surprised": scores["surprised"],
            "Score Confused": scores["confused"],
            "Score Like": scores["like"],
            "Score Nerd": scores["nerd"],
            "Calibrado": self.calibrated,
        }

        return active_expression, confidence, description, raw_metrics
