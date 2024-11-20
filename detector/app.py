from playsound import playsound
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from cv2.typing import MatLike
import os

from detector.video_capture import VideoCapture, NO_VIDEO, VIDEO_FILE
from detector.image_processor import ImageProcessor
from detector.video_processing_engine import VideoProcessingEngine
from detector.consts import ALERT_SOUND


class App:
    def __init__(self) -> None:
        from detector.interface import GUI

        self._play_alert_executor = ThreadPoolExecutor(max_workers=1)

        self._alert_event = threading.Event()
        self._alert_event.clear()

        self._video_capture = VideoCapture()
        self._image_processor = ImageProcessor()
        self._video_processing_engine = VideoProcessingEngine(self._video_capture, self._image_processor, self._notify_user)
        self._gui = GUI(self)


    def run(self) -> None:
        self._gui.show()
        self.shutdown_video_processing_engine()

    
    def shutdown_video_processing_engine(self):
        self._video_processing_engine.shutdown()


    def set_video_source(self, source_id: int) -> None:
        if source_id == NO_VIDEO:
            self._video_processing_engine.remove_video_source()
        elif source_id == VIDEO_FILE:
            path = self._gui.select_video_file()
            if os.path.exists(path) and os.path.isfile(path):
                self._video_processing_engine.set_video_source(source=path)
            else:
                self._gui.set_video_source(source_id)
        else:
            self._video_processing_engine.set_video_source(source=source_id)

    
    def get_latest_frame(self) -> Tuple[Optional[bool], Optional[MatLike]]:
        return self._video_processing_engine.get_latest_frame()


    def set_frame_area_dimensions(self, max_width, max_height, min_width, min_height) -> None:
        self._video_processing_engine.set_window_dimensions(max_width, max_height, min_width, min_height)


    def _play_alert(self):
        self._alert_event.set()
        playsound(ALERT_SOUND)
        self._alert_event.clear()


    def _notify_user(self):
        if not self._alert_event.is_set():
            self._play_alert_executor.submit(self._play_alert)