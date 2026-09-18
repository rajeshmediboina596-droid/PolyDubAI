"""
Visual Face Detection and Gender Classification Engine.
Uses OpenCV face detection and Vision AI neural network (rizvandwiki/gender-classification-2)
to accurately analyze whether the speaking person on screen is a boy/man or girl/woman.
"""

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from dubber.config import find_ffmpeg_executable


class VisualGenderClassifier:
    """
    Computer vision classifier that samples video frames during speech segments,
    detects active human faces using OpenCV, and classifies speaker gender with Vision AI.
    """

    def __init__(self, model_name: str = "rizvandwiki/gender-classification-2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._classifier_pipeline = None
        self._face_cascade = None

    def _ensure_face_detector(self):
        """Lazily initialize OpenCV Haar Cascade face detector."""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.isfile(cascade_path):
                raise FileNotFoundError(f"OpenCV face cascade not found at: {cascade_path}")
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def _ensure_vision_model(self):
        """Lazily load Hugging Face image classification pipeline with offline cache priority."""
        if self._classifier_pipeline is None:
            try:
                from transformers import pipeline
                device_idx = 0 if self.device == "cuda" else -1
                hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
                cached_model_dir = os.path.join(hub_cache, "models--" + self.model_name.replace("/", "--"))
                local_only = os.path.isdir(cached_model_dir)
                if local_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                self._classifier_pipeline = pipeline(
                    "image-classification",
                    model=self.model_name,
                    device=device_idx,
                    model_kwargs={"local_files_only": local_only} if local_only else {},
                )
            except Exception:
                try:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    from transformers import pipeline
                    device_idx = 0 if self.device == "cuda" else -1
                    self._classifier_pipeline = pipeline(
                        "image-classification",
                        model=self.model_name,
                        device=device_idx,
                    )
                except Exception:
                    self._classifier_pipeline = False

    def extract_frame(
        self,
        video_path: str,
        timestamp: float,
        output_image_path: str,
        ffmpeg_path: Optional[str] = None,
    ) -> bool:
        """Extract a single frame from video at the given timestamp in seconds."""
        if not ffmpeg_path:
            ffmpeg_path = find_ffmpeg_executable()

        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        cmd = [
            ffmpeg_path,
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-y", output_image_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0 and os.path.isfile(output_image_path)

    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect face in an image file, crop it with margin padding, and predict gender.
        If no face is detected, returns face_detected: False.
        """
        self._ensure_face_detector()
        self._ensure_vision_model()

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"face_detected": False, "gender": "unknown", "confidence": 0.0, "box": None, "predictions": []}

        h_img, w_img = img_bgr.shape[:2]
        min_dim = min(h_img, w_img)
        if min_dim >= 720:
            min_s = int(min_dim * 0.18)
        elif min_dim >= 360:
            min_s = int(min_dim * 0.14)
        else:
            min_s = 40

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_s, min_s),
        )
        # Filter out extreme corner watermark logos / channel avatar icons (top 12% corner)
        valid_faces = []
        for f in faces:
            x, y, w, h = f
            is_top_corner = (y / max(1, h_img) < 0.12) and (x / max(1, w_img) < 0.15 or (x + w) / max(1, w_img) > 0.85)
            if is_top_corner and (w < w_img * 0.28):
                continue
            valid_faces.append(f)

        if len(valid_faces) > 0:
            # Pick the largest face (closest to camera / most prominent speaker)
            x, y, w, h = max(valid_faces, key=lambda f: f[2] * f[3])
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            face_crop = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            face_detected = True
            box = (int(x), int(y), int(w), int(h))
        else:
            face_detected = False
            box = None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

        if not self._classifier_pipeline:
            return {
                "face_detected": face_detected,
                "gender": "unknown",
                "confidence": 0.0,
                "box": box,
                "predictions": [],
            }

        preds = self._classifier_pipeline(pil_img)
        top_pred = preds[0] if preds else {"label": "male", "score": 0.5}

        gender = "female" if "female" in top_pred["label"].lower() else "male"
        confidence = float(top_pred.get("score", 0.5))

        return {
            "face_detected": face_detected,
            "gender": gender,
            "confidence": confidence,
            "box": box,
            "predictions": preds,
        }

    def classify_segment(
        self,
        video_path: str,
        start: float,
        end: float,
        temp_dir: str,
        segment_idx: int = 0,
        ffmpeg_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-frame sampling across speech segment to reliably find active speaker face.
        Samples up to 3 candidate points (0.25x, 0.50x, 0.75x) to eliminate missing faces during head turns.
        """
        dur = max(0.05, end - start)
        if dur >= 1.2:
            timestamps = [start + 0.25 * dur, start + 0.50 * dur, start + 0.75 * dur]
        else:
            timestamps = [(start + end) / 2.0]

        best_detection = {
            "face_detected": False,
            "gender": "unknown",
            "confidence": 0.0,
            "box": None,
            "predictions": [],
        }

        for idx, t in enumerate(timestamps):
            frame_filename = os.path.join(temp_dir, f"vis_seg_{segment_idx}_{idx}_{t:.2f}s.jpg")
            success = self.extract_frame(
                video_path=video_path,
                timestamp=t,
                output_image_path=frame_filename,
                ffmpeg_path=ffmpeg_path,
            )
            if not success or not os.path.isfile(frame_filename):
                continue

            try:
                res = self.classify_image(frame_filename)
                if res.get("face_detected"):
                    conf = res.get("confidence", 0.0)
                    if not best_detection["face_detected"] or conf > best_detection["confidence"]:
                        best_detection = res
                    if conf >= 0.85:
                        break
            finally:
                if os.path.isfile(frame_filename):
                    try:
                        os.remove(frame_filename)
                    except Exception:
                        pass

        return best_detection
