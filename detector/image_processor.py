import cv2 as cv
from cv2.typing import MatLike
from ultralytics import YOLO
import torch
import logging

from detector.app import App
from detector.video_manager import VideoManager


CONFIDENCE_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageProcessor:
    def __init__(self, parent_app: App, video_manager: VideoManager) -> None:
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

        self._parent_app = parent_app
        self._video_manager = video_manager

        self._detector = YOLO('yolo_models/yolov8n-pretrained-default.pt')


    def process_frame(self, max_frame_width, max_frame_height) -> None:
        frame = self._video_manager.get_frame()
        if frame is None:
            return None
        
        output_width, output_height = self._fitting_dimensions(frame, max_frame_width, max_frame_height)
        frame = cv.resize(frame, (output_width, output_height), interpolation=cv.INTER_LINEAR)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        detections = self._detect_people(frame)
        frame = self._draw_rectangles(frame, detections)
        
        return frame


    def _detect_people(self, frame: MatLike):
        return self._detector(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE
        )[0].boxes


    def _draw_rectangles(self, frame: MatLike, detections, confidence_threshold=0.5):
        print(len(detections))

        for box in detections:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            if cls == 0 and conf >= confidence_threshold:
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        return frame


    def _fitting_dimensions(self, frame, max_width, max_height):
        height, width = frame.shape[:2]

        if width <= max_width and height <= max_height:
            return width, height
        
        scale_x = max_width / width
        scale_y = max_height / height

        scale = min(scale_x, scale_y)

        new_width = int(width * scale)
        new_height = int(height * scale)

        return new_width, new_height
        