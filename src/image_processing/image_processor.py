import cv2 as cv
from cv2.typing import MatLike

from app import App
from video_manager import VideoManager

class ImageProcessor:
    def __init__(self, parent_app: App, video_manager: VideoManager) -> None:
        self._parent_app = parent_app
        self._video_manager = video_manager

        self._window_width = None
        self._window_height = None

    def set_window_dimensions(self, width, height):
        self._window_height = width
        self._window_height = height

    def process_frame(self, max_frame_width, max_frame_height) -> None:
        frame = self._video_manager.get_frame()
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        resized_frame = self._fit_frame(frame, width, height, max_frame_width, max_frame_height)

        return resized_frame
    

    def _fit_frame(self, frame, width, height, max_frame_width, max_frame_height):
        if width > max_frame_width or height > max_frame_height:
            scale_x = max_frame_width / width
            scale_y = max_frame_height / height

            scale = min(scale_x, scale_y)
            scale = 0.4

            new_width = int(width * scale)
            new_height = int(height * scale)

            return cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
        
        return frame 
    