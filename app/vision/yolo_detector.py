"""
YOLO Detector Module
-------------------
Responsibilities:
- Load YOLOv11 model
- Run inference on frames
- Return structured detections

Designed for BlueOS Extension runtime.
"""

import cv2
import time
from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        inference_stride: int = 5,
        input_size: int = 640,
    ):
        """
        Args:
            model_path: Path to YOLOv11 .pt file
            conf_threshold: Minimum confidence
            inference_stride: Run inference every N frames
            input_size: Resize shorter side to this before inference
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.inference_stride = inference_stride
        self.input_size = input_size
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print("YOLO running on device: ", self.device)

        self.frame_count = 0
        self.last_detections = []

        print(f"[YOLO] Model loaded: {model_path}")
        print(f"[YOLO] Confidence threshold: {conf_threshold}")
        print(f"[YOLO] Inference stride: {inference_stride}")

    def _preprocess(self, frame):
        """
        Resize frame for faster inference.
        """
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def detect(self, frame):
        """
        Run YOLO inference conditionally (based on stride).

        Returns:
            List of detections:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": int,
                    "class_name": str,
                    "confidence": float
                },
                ...
            ]
        """
        self.frame_count += 1

        # Skip inference to save CPU
        if self.frame_count % self.inference_stride != 0:
            return self.last_detections

        input_frame = self._preprocess(frame)
        
        try:
            results = self.model.predict(
                input_frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            print("[YOLO] Inference error:", e)
            return self.last_detections

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                })

        if detections:
            self.last_detections = detections

        return self.last_detections
