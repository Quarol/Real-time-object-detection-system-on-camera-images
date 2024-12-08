from playsound import playsound
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from cv2.typing import MatLike

import os
import time

from detector.video_capture import VideoCapture, NO_VIDEO, VIDEO_FILE
from detector.image_processor import ImageProcessor

ALERT_SOUND = 'assets/alert.wav'

class App:
    def __init__(self) -> None:
        from detector.interface import GUI
        from detector.video_processing_engine import VideoProcessingEngine

        self._play_alert_executor = ThreadPoolExecutor(max_workers=1)

        self._alert_event = threading.Event()
        self._alert_event.clear()

        self._video_capture = VideoCapture()
        self._image_processor = ImageProcessor()
        self._video_processing_engine = VideoProcessingEngine(self._video_capture, self._image_processor, self)
        
        frame_display_scaling_factor = 0.8
        self._gui = GUI(self, frame_display_scaling_factor)


    def run(self) -> None:
        self._video_processing_engine.run()
        self._gui.show()
        self._video_processing_engine.shutdown()


    def set_video_source(self, source_id: int) -> bool:
        if source_id == NO_VIDEO:
            self._video_processing_engine.remove_video_source()
            return False

        elif source_id == VIDEO_FILE:
            path = self._gui.select_video_file()
            if os.path.exists(path) and os.path.isfile(path):
                self._video_processing_engine.set_video_source(source=path)
                return True
            else:
                self._video_processing_engine.remove_video_source()
                return False

        else:
            self._video_processing_engine.set_video_source(source=source_id)
            return True

    
    def get_processed_frame(self) -> Tuple[bool, Optional[MatLike]]:
        return self._video_processing_engine.get_processed_frame()


    def set_max_display_dimention(self, max_width: int, max_height: int) -> None:
        self._video_processing_engine.set_max_frame_dimension(max_width, max_height)


    def get_available_classes(self) -> list[str]:
        return self._image_processor.get_available_classes()

        
    def get_detected_classes(self) -> list[int]:
        return self._image_processor.get_detected_classes()
    

    def add_detected_class(self, class_index: int) -> None:
        self._image_processor.add_detected_class(class_index)


    def remove_detected_class(self, class_index: int) -> None:
        self._image_processor.remove_detected_class(class_index)

    
    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self._image_processor.set_confidence_threshold(confidence_threshold)


    def get_confidence_threshold(self) -> float:
        return self._image_processor.get_confidence_threshold()


    def get_available_sources(self) -> dict[str, int]:
        return VideoCapture.get_available_sources()
    

    def _play_alert(self) -> None:
        self._alert_event.set()
        playsound(ALERT_SOUND)
        self._alert_event.clear()


    def play_audio_alert(self) -> None:
        if not self._alert_event.is_set():
            self._play_alert_executor.submit(self._play_alert)