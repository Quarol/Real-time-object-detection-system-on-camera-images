from playsound import playsound
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from cv2.typing import MatLike

import os
import time

from detector.video_capture import VideoCapture, NO_VIDEO, VIDEO_FILE
from detector.image_processor import ImageProcessor
from detector.video_processing_engine import VideoProcessingEngine
from detector.yolo_settings import yolo_inference_config

ALERT_SOUND = 'assets/alert.wav'

class App:
    def __init__(self) -> None:
        from detector.interface import GUI

        self._play_alert_executor = ThreadPoolExecutor(max_workers=1)

        self._alert_event = threading.Event()
        self._alert_event.clear()

        self._video_capture = VideoCapture()
        self._image_processor = ImageProcessor()
        self._video_processing_engine = VideoProcessingEngine(self._video_capture, self._image_processor,
                                                              self._notify_user, self._display_frame)
        
        frame_display_scaling_factor = 1.0
        self._gui = GUI(self, frame_display_scaling_factor)


    def run(self) -> None:
        self._video_processing_engine.run()
        self._gui.show()
        self._video_processing_engine.shutdown()

    
    def _display_frame(self, frame: MatLike) -> None:
        self._gui._show_frame(frame)


    def set_video_source(self, source_id: int) -> None:
        if source_id == NO_VIDEO:
            self._video_processing_engine.remove_video_source()

        elif source_id == VIDEO_FILE:
            path = self._gui.select_video_file()
            if os.path.exists(path) and os.path.isfile(path):
                self._video_processing_engine.set_video_source(source=path)
            else:
                self._gui.set_video_source(source_id)
                self._video_processing_engine.remove_video_source()
                
        else:
            self._video_processing_engine.set_video_source(source=source_id)

    
    def get_latest_frame(self) -> Tuple[bool, Optional[MatLike]]:
        return self._video_processing_engine.get_processed_frame()


    def set_frame_area_dimensions(self, max_width: int, max_height: int) -> None:
        self._video_processing_engine.set_window_dimensions(max_width, max_height)


    def get_available_classes(self) -> list[str]:
        return yolo_inference_config.get_available_classes()


    def add_detected_class(self, class_index: int) -> None:
        yolo_inference_config.add_detected_class(class_index)

    
    def remove_detected_class(self, class_index: int) -> None:
        yolo_inference_config.remove_detected_class(class_index)

    
    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        yolo_inference_config.set_confidence_threshold(confidence_threshold)

    
    def get_detected_classes(self) -> list[int]:
        return yolo_inference_config.classes


    def get_confidence_threshold(self) -> float:
        return yolo_inference_config.confidence_threshold


    def get_available_sources(self) -> dict[str, int]:
        return VideoCapture.get_available_sources()
    

    def _play_alert(self) -> None:
        self._alert_event.set()
        playsound(ALERT_SOUND)
        self._alert_event.clear()


    def _notify_user(self) -> None:
        if not self._alert_event.is_set():
            self._play_alert_executor.submit(self._play_alert)