"""
Drone Detector Module
=====================
Encapsulates YOLOv8 inference, visual annotation, HUD overlays,
threat scoring, and video streaming logic.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional, Generator
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class DroneDetector:
    """
    High-level YOLOv8-powered detector for UAVs / Drones.
    Supports single images, video files, and real-time camera/simulated streams.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector with specified model weights.
        If the specified model does not exist, falls back gracefully to yolov8n.pt.
        """
        self.model = None
        self.model_path = model_path
        self.is_custom_drone_model = False
        self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        """Load or reload model weights."""
        if not ULTRALYTICS_AVAILABLE:
            print("[WARN] Ultralytics is not installed. Running in mock/fallback mode.")
            self.model = None
            return

        resolved_path = None
        # Check user supplied path
        if model_path and os.path.exists(model_path):
            resolved_path = model_path
            self.is_custom_drone_model = True
        # Check standard custom weights location
        elif os.path.exists("weights/best.pt"):
            resolved_path = "weights/best.pt"
            self.is_custom_drone_model = True
        # Fallback to standard pretrained nano model
        else:
            resolved_path = "yolov8n.pt"
            self.is_custom_drone_model = False

        try:
            print(f"[INFO] Loading YOLOv8 model from: {resolved_path}")
            self.model = YOLO(resolved_path)
            self.model_path = resolved_path
            print(f"[INFO] Model successfully loaded. Custom drone model: {self.is_custom_drone_model}")
        except Exception as e:
            print(f"[ERROR] Failed to load model {resolved_path}: {e}")
            print("[INFO] Attempting fallback to yolov8n.pt...")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model_path = "yolov8n.pt"
                self.is_custom_drone_model = False
            except Exception as e2:
                print(f"[ERROR] Fallback failed: {e2}")
                self.model = None

    def predict_image(
        self,
        image_input: Any,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Run detection on a single image.

        Args:
            image_input: PIL Image, filepath, or NumPy array (RGB/BGR).
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: Intersection-Over-Union threshold for NMS.

        Returns:
            annotated_image: RGB NumPy array with bounding boxes and HUD.
            detections_list: List of detection dictionaries.
            alert_status: Formatted text status banner.
            summary_stats: High-level metrics dictionary.
        """
        # Convert input to RGB numpy array
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                image_rgb = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h, w, _ = image_rgb.shape
        start_time = time.time()

        detections: List[Dict[str, Any]] = []

        if self.model is not None:
            results = self.model.predict(
                source=image_rgb,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            result = results[0]
            boxes = result.boxes

            for idx, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"Class {cls_id}")

                # If using custom model, it detects drone directly
                # If using base COCO model, target airplane (4), bird (14), or any object as proxy
                if not self.is_custom_drone_model:
                    is_target = cls_id in [4, 14, 0] or "drone" in cls_name.lower()
                    if not is_target and conf < 0.5:
                        continue
                    display_name = f"Drone (proxy: {cls_name})"
                else:
                    display_name = cls_name

                x1, y1, x2, y2 = map(int, coords)
                box_w = x2 - x1
                box_h = y2 - y1
                area_ratio = (box_w * box_h) / (w * h)

                # Threat estimation based on proximity / size & confidence
                if area_ratio > 0.15 or conf > 0.80:
                    threat_level = "CRITICAL"
                elif area_ratio > 0.05 or conf > 0.55:
                    threat_level = "HIGH"
                else:
                    threat_level = "MODERATE"

                detections.append({
                    "id": idx + 1,
                    "class": display_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "width_px": box_w,
                    "height_px": box_h,
                    "area_pct": round(area_ratio * 100, 2),
                    "threat": threat_level
                })
        else:
            # Fallback mock for testing when ultralytics is loading
            pass

        inference_time_ms = round((time.time() - start_time) * 1000, 2)
        annotated_image = self._draw_annotations(image_rgb.copy(), detections, inference_time_ms)

        drone_count = len(detections)
        if drone_count == 0:
            alert_status = "🟢 SECURE: Airspace clear — No UAV / Drone detected."
        elif drone_count == 1:
            alert_status = f"⚠️ WARNING: 1 Drone detected in monitored airspace! ({detections[0]['threat']} Threat)"
        else:
            alert_status = f"🚨 CRITICAL ALERT: Multiple ({drone_count}) Drones detected! Immediate airspace breach!"

        summary_stats = {
            "drone_count": drone_count,
            "inference_time_ms": inference_time_ms,
            "fps": round(1000.0 / max(inference_time_ms, 0.1), 1),
            "model_used": os.path.basename(str(self.model_path)),
            "is_custom_weights": self.is_custom_drone_model,
            "highest_confidence": max([d["confidence"] for d in detections], default=0.0)
        }

        return annotated_image, detections, alert_status, summary_stats

    def _draw_annotations(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict[str, Any]],
        inference_time_ms: float
    ) -> np.ndarray:
        """Draw tactical UI HUD overlays, crosshairs, and bounding boxes."""
        h, w, _ = image_rgb.shape
        img = image_rgb.copy()

        # HUD Top Banner Overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 23, 42), -1)  # Slate dark header
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Status text in HUD
        hud_text = f"DRONE-AI DEFENSE | TARGETS: {len(detections)} | LATENCY: {inference_time_ms}ms"
        cv2.putText(img, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

        # Draw each detection box
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            threat = det["threat"]
            cls_name = det["class"]

            # Color scheme based on threat
            if threat == "CRITICAL":
                color = (255, 40, 40)    # Bright Red
            elif threat == "HIGH":
                color = (255, 140, 0)   # Orange
            else:
                color = (255, 215, 0)   # Yellow/Amber

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Tactical corner markers for modern look
            corner_len = min(20, (x2 - x1) // 3, (y2 - y1) // 3)
            # Top-left
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Top-right
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Bottom-left
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Bottom-right
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Center targeting crosshair
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 14, 1)

            # Label badge
            label_text = f"TARGET: {cls_name} [{int(conf * 100)}%] - {threat}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 2), (15, 23, 42), -1)
            cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 2), color, 1)
            cv2.putText(img, label_text, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> np.ndarray:
        """Fast single frame processing for live streams & video feeds."""
        annotated_img, _, _, _ = self.predict_image(
            frame_rgb,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        return annotated_img

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        progress_callback: Optional[Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process an entire video file, writing annotated frames with sound/metrics.

        Returns:
            output_video_path: Path to rendered MP4 video.
            summary_stats: Aggregated video detection statistics.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not output_path:
            os.makedirs("runs/detect_video", exist_ok=True)
            output_path = os.path.join("runs/detect_video", f"annotated_{os.path.basename(video_path)}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_drones_seen = 0
        frames_with_drones = 0
        max_drones_single_frame = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb, detections, _, _ = self.predict_image(
                frame_rgb,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

            drone_count = len(detections)
            total_drones_seen += drone_count
            if drone_count > 0:
                frames_with_drones += 1
            max_drones_single_frame = max(max_drones_single_frame, drone_count)

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)
            frame_idx += 1

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

        cap.release()
        out.release()

        stats = {
            "total_frames": frame_idx,
            "fps": fps,
            "duration_sec": round(frame_idx / max(fps, 1), 2),
            "frames_with_drones": frames_with_drones,
            "drone_incursion_pct": round((frames_with_drones / max(frame_idx, 1)) * 100, 2),
            "max_concurrent_drones": max_drones_single_frame,
            "total_detection_instances": total_drones_seen,
            "output_path": output_path
        }

        return output_path, stats
