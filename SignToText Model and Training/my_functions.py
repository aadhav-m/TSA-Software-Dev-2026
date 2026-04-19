import cv2
import numpy as np
import mediapipe as mp
from typing import Any

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

hand_point_style = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2)
hand_connection_style = mp_drawing.DrawingSpec(color=(0, 120, 200), thickness=1)


def draw_landmarks(frame: np.ndarray, mediapipe_results: Any) -> None:
    for hand in (mediapipe_results.left_hand_landmarks, mediapipe_results.right_hand_landmarks):
        if hand:
            mp_drawing.draw_landmarks(frame, hand, mp_holistic.HAND_CONNECTIONS, hand_point_style, hand_connection_style)


def process_frame(frame: np.ndarray, mediapipe_model: Any) -> Any:
    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mediapipe_results = mediapipe_model.process(rgb_frame)
    frame.flags.writeable = True
    return mediapipe_results


def extract_keypoints(mediapipe_results: Any) -> np.ndarray:
    def hand_to_flat_array(hand_landmarks):
        if hand_landmarks is None:
            return np.zeros(63, dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

    left_hand = hand_to_flat_array(mediapipe_results.left_hand_landmarks)
    right_hand = hand_to_flat_array(mediapipe_results.right_hand_landmarks)
    return np.concatenate([left_hand, right_hand])


def draw_centered_text(frame: np.ndarray, text: str, y_pos: int, font_scale: float = 0.8, color: tuple = (255, 255, 255), thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x_pos = max(0, (frame.shape[1] - text_width) // 2)
    cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
