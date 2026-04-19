import argparse
import os
import sys
from itertools import product
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from my_functions import draw_landmarks, process_frame, extract_keypoints, draw_centered_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions",   nargs="+", default=["a", "b"])
    parser.add_argument("--sequences", type=int, default=30)
    parser.add_argument("--frames",    type=int, default=10)
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--camera",    type=int, default=0)
    return parser.parse_args()


def create_dataset_dirs(data_dir, sign_labels, num_sequences):
    for sign, seq_idx in product(sign_labels, range(num_sequences)):
        Path(data_dir, sign, str(seq_idx)).mkdir(parents=True, exist_ok=True)


def open_camera(device_index):
    camera = cv2.VideoCapture(device_index)
    if not camera.isOpened():
        sys.exit(f"Cannot open camera {device_index}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def is_window_closed(window_name):
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1


def wait_for_spacebar(camera, holistic_model, sign_label, seq_idx, window_name):
    while True:
        _, frame = camera.read()
        mediapipe_results = process_frame(frame, holistic_model)
        draw_landmarks(frame, mediapipe_results)
        cv2.rectangle(frame, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        draw_centered_text(frame, f'Recording "{sign_label}" — sequence {seq_idx + 1}', y_pos=frame.shape[0] - 65, font_scale=0.65, color=(0, 200, 255))
        draw_centered_text(frame, "Press SPACE when ready", y_pos=frame.shape[0] - 28, font_scale=0.9)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q') or is_window_closed(window_name):
            return False


def main():
    args = parse_args()
    sign_labels  = args.actions
    num_sequences = args.sequences
    num_frames    = args.frames
    data_dir      = args.data_dir
    window_name   = "SignFlow — Data Collection"

    create_dataset_dirs(data_dir, sign_labels, num_sequences)
    camera = open_camera(args.camera)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic_model:
        for sign_label, seq_idx in product(sign_labels, range(num_sequences)):
            if not wait_for_spacebar(camera, holistic_model, sign_label, seq_idx, window_name):
                break

            for frame_idx in range(num_frames):
                _, frame = camera.read()
                mediapipe_results = process_frame(frame, holistic_model)
                draw_landmarks(frame, mediapipe_results)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(frame, f'{sign_label}  |  seq {seq_idx + 1}/{num_sequences}  |  frame {frame_idx + 1}/{num_frames}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 1, cv2.LINE_AA)

                if frame_idx == 0:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 6)

                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

                if is_window_closed(window_name):
                    break

                keypoints = extract_keypoints(mediapipe_results)
                np.save(os.path.join(data_dir, sign_label, str(seq_idx), str(frame_idx)), keypoints)

            if is_window_closed(window_name):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
