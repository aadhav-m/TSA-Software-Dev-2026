import base64
import json
import os
import string
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

from my_functions import draw_landmarks, process_frame, extract_keypoints

MODEL_PATH           = os.getenv("SIGNFLOW_MODEL", "my_model")
LABELS_PATH          = os.getenv("SIGNFLOW_LABELS", "my_model_labels.json")
CONFIDENCE_THRESHOLD = float(os.getenv("SIGNFLOW_CONF", "0.85"))
MAX_SENTENCE_LENGTH  = int(os.getenv("SIGNFLOW_SENT_LEN", "10"))
KEYPOINT_BUFFER_SIZE = int(os.getenv("SIGNFLOW_FRAMES", "10"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def load_sign_labels():
    if Path(LABELS_PATH).exists():
        with open(LABELS_PATH) as f:
            return json.load(f)
    data_dir = Path("data")
    if data_dir.exists():
        return sorted(os.listdir(data_dir))
    return []

sign_model    = load_model(MODEL_PATH)
sign_labels   = np.array(load_sign_labels())
holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)


class SignSession:
    def __init__(self):
        self.keypoint_buffer: deque = deque(maxlen=KEYPOINT_BUFFER_SIZE)
        self.current_sentence: list[str] = []
        self.previous_prediction: str = ""

    def reset(self):
        self.keypoint_buffer.clear()
        self.current_sentence.clear()
        self.previous_prediction = ""

    def add_keypoints_and_predict(self, keypoints: np.ndarray) -> Optional[str]:
        self.keypoint_buffer.append(keypoints)
        if len(self.keypoint_buffer) < KEYPOINT_BUFFER_SIZE:
            return None
        prediction_scores = sign_model.predict(np.array(self.keypoint_buffer)[np.newaxis, :, :], verbose=0)[0]
        top_confidence = float(np.amax(prediction_scores))
        predicted_label = sign_labels[np.argmax(prediction_scores)]
        if top_confidence < CONFIDENCE_THRESHOLD or predicted_label == self.previous_prediction:
            return None
        self.previous_prediction = predicted_label
        return predicted_label

    def append_to_sentence(self, new_token: str) -> None:
        self.current_sentence.append(new_token)
        if (len(self.current_sentence) >= 2
                and self.current_sentence[-1] in string.ascii_letters
                and (self.current_sentence[-2] in string.ascii_letters
                     or (self.current_sentence[-2] not in sign_labels
                         and self.current_sentence[-2] not in [s.capitalize() for s in sign_labels]))):
            self.current_sentence[-2] = self.current_sentence[-2] + self.current_sentence[-1]
            self.current_sentence.pop()
        if self.current_sentence:
            self.current_sentence[0] = self.current_sentence[0].capitalize()
        if len(self.current_sentence) > MAX_SENTENCE_LENGTH:
            self.current_sentence = self.current_sentence[-MAX_SENTENCE_LENGTH:]

    def to_response(self, confidence: float = 0.0, latest_token: str = "") -> dict:
        return {
            "sentence": self.current_sentence,
            "text": " ".join(self.current_sentence),
            "latest_token": latest_token,
            "confidence": round(confidence, 3),
        }


@app.get("/status")
async def get_status():
    return JSONResponse({"status": "ok", "classes": sign_labels.tolist(), "buffer_size": KEYPOINT_BUFFER_SIZE})


@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()
    session = SignSession()

    try:
        while True:
            raw_message = await ws.receive_text()
            message = json.loads(raw_message)

            if message.get("type") == "reset":
                session.reset()
                await ws.send_text(json.dumps({"type": "reset"}))
                continue

            if message.get("type") != "frame":
                continue

            frame_bytes = base64.b64decode(message["data"])
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            mediapipe_results = process_frame(frame, holistic_model)
            keypoints = extract_keypoints(mediapipe_results)
            new_token = session.add_keypoints_and_predict(keypoints)
            top_confidence = 0.0
            is_new_token = False

            if new_token is not None:
                session.append_to_sentence(new_token)
                is_new_token = True
                prediction_scores = sign_model.predict(np.array(session.keypoint_buffer)[np.newaxis, :, :], verbose=0)[0]
                top_confidence = float(np.amax(prediction_scores))

            draw_landmarks(frame, mediapipe_results)
            _, jpeg_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            annotated_b64 = base64.b64encode(jpeg_buffer.tobytes()).decode()

            await ws.send_text(json.dumps({
                "type": "prediction",
                "annotated_frame": annotated_b64,
                "is_new_token": is_new_token,
                **session.to_response(top_confidence, new_token or ""),
            }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
