import cv2 as cv
from cv2.typing import MatLike
from ultralytics import YOLO

from app import App
from video_manager import VideoManager

class ImageProcessor:
    def __init__(self, parent_app: App, video_manager: VideoManager) -> None:
        self._parent_app = parent_app
        self._video_manager = video_manager
        self._detector = YOLO('yolo11n.pt')

    def process_frame(self, max_frame_width, max_frame_height) -> None:
        frame = self._video_manager.get_frame()
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        resized_frame = self._fit_frame(frame, width, height, max_frame_width, max_frame_height)

        results = self._detector(resized_frame)

        detections = results[0].boxes

        confidence_threshold = 0.5

        for box in detections:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            if cls == 0 and conf >= confidence_threshold:
                cv.rectangle(resized_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        return resized_frame
            

    def _fit_frame(self, frame, width, height, max_frame_width, max_frame_height):
        if width > max_frame_width or height > max_frame_height:
            scale_x = max_frame_width / width
            scale_y = max_frame_height / height

            scale = min(scale_x, scale_y)

            new_width = int(width * scale)
            new_height = int(height * scale)

            return cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
        
        return frame 
    