import cv2
import numpy as np
from src.utils import resize_and_pad


def draw_hud(
    frame: np.ndarray,
    info_text: str,
    fps: float,
    show_debug: bool,
    show_mesh: bool,
    calibrated_feedback: bool = False,
):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if calibrated_feedback:
        display_text = "Postura neutra calibrada com sucesso!"
        text_color = (0, 255, 120)
    else:
        display_text = info_text
        text_color = (255, 255, 255)

    cv2.putText(
        frame,
        display_text,
        (15, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        text_color,
        1,
        cv2.LINE_AA,
    )

    flags_text = f"{int(fps)} FPS"
    if show_debug:
        flags_text += " | [DEBUG]"
    if show_mesh:
        flags_text += " | [MALHA]"

    cv2.putText(
        frame,
        flags_text,
        (w - 200, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )


def draw_debug_overlay(frame: np.ndarray, metrics: dict):
    h, w = frame.shape[:2]
    box_w = 280
    box_h = 190
    start_x = 12
    start_y = h - box_h - 12

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + box_w, start_y + box_h),
        (15, 15, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(
        frame,
        (start_x, start_y),
        (start_x + box_w, start_y + box_h),
        (70, 70, 90),
        1,
    )

    cv2.putText(
        frame,
        "METRICAS EM TEMPO REAL",
        (start_x + 10, start_y + 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.45,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    meters = [
        ("Boca", metrics.get("Boca Aberta", 0.0), 1.0),
        ("Sorriso", metrics.get("Sorriso", 0.0), 1.0),
        ("Sobrancelha", metrics.get("Sobrancelha Cima", 0.0), 1.0),
        ("Inclinacao", metrics.get("Inclinacao Relat.", 0.0), 30.0),
    ]

    curr_y = start_y + 44
    bar_width = 110
    bar_height = 9

    for name, val, max_val in meters:
        cv2.putText(
            frame,
            name,
            (start_x + 10, curr_y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        bar_x = start_x + 95
        cv2.rectangle(
            frame,
            (bar_x, curr_y),
            (bar_x + bar_width, curr_y + bar_height),
            (45, 45, 55),
            -1,
        )

        ratio = min(1.0, max(0.0, val / max_val))
        fill_w = int(bar_width * ratio)
        color = (0, 255, 120) if ratio > 0.4 else (0, 180, 255)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x, curr_y),
                (bar_x + fill_w, curr_y + bar_height),
                color,
                -1,
            )

        val_str = f"{int(val * 100)}%" if max_val == 1.0 else f"{int(val)} deg"
        cv2.putText(
            frame,
            val_str,
            (bar_x + bar_width + 8, curr_y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        curr_y += 22

    sc_yawn = metrics.get("Score Yawn", 0.0)
    sc_conf = metrics.get("Score Confused", 0.0)
    score_txt = f"Scores: Yawn={int(sc_yawn*100)}% | Conf={int(sc_conf*100)}%"
    cv2.putText(
        frame,
        score_txt,
        (start_x + 10, curr_y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    gesto_txt = f"Gesto: {metrics.get('Gesto Detectado', 'Nenhum')}"
    cv2.putText(
        frame,
        gesto_txt,
        (start_x + 10, curr_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 215, 0),
        1,
        cv2.LINE_AA,
    )


def draw_mesh_points(frame: np.ndarray, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks[::5]:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 1, (0, 255, 120), -1)


def create_placeholder_panel(panel_w: int, panel_h: int) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (28, 26, 32)

    cv2.rectangle(panel, (0, 0), (panel_w, 60), (40, 38, 48), -1)
    cv2.putText(
        panel,
        "GUIA DE EXPRESSOES",
        (22, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (255, 215, 0),
        2,
        cv2.LINE_AA,
    )

    items = [
        ("Surpreso (surprised):", "Abra a boca e arregale os olhos"),
        ("Bocejo (yawn):", "Abra bem a boca bocejando"),
        ("Confuso (confused):", "Incline a cabeca de lado (boca fechada)"),
        ("Like (like):", "Faca sinal de joinha ou sorria"),
        ("Nerd (nerd):", "Aponte o indicador para cima"),
    ]

    y_pos = 98
    for title, desc in items:
        cv2.putText(
            panel,
            title,
            (22, y_pos),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            desc,
            (22, y_pos + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (195, 195, 195),
            1,
            cv2.LINE_AA,
        )
        y_pos += 62

    cv2.line(panel, (20, panel_h - 48), (panel_w - 20, panel_h - 48), (55, 52, 65), 1)
    cv2.putText(
        panel,
        "[C] Calibrar | [D] Debug | [M] Malha | [Q] Sair",
        (15, panel_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return panel


def create_reaction_panel(
    reaction_img: np.ndarray, expression: str, confidence: float, panel_w: int, panel_h: int
) -> np.ndarray:
    panel = resize_and_pad(reaction_img, panel_w, panel_h)

    badge_title = f"REACAO: {expression.upper()} ({int(confidence * 100)}%)"
    cv2.rectangle(panel, (0, 0), (panel_w, 46), (15, 15, 18), -1)
    cv2.putText(
        panel,
        badge_title,
        (18, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 120), 2)
    return panel
