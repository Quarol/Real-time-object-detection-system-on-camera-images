import threading
import time
from cv2.typing import MatLike

from detector.image_processor import ImageProcessor
from detector.video_capture import VideoCapture

class FrameProcessor:
    def __init__(self, video_capture: VideoCapture, image_processor: ImageProcessor) -> None:
        self._video_capture = video_capture
        self._image_processor = image_processor
        self._lock = threading.Lock()
        self._processing_thread = None
        self._is_processing = False

        self._latest_frame = None
        self._ret = False

    
    def set_max_frame_size(self, width, height) -> None:
        self._max_frame_width = width
        self._max_frame_height = height


    def start_processing(self) -> None:
        self.stop_processing()

        self._is_processing = True
        self._processing_thread = threading.Thread(target=self._process_frames)
        self._processing_thread.start()


    def stop_processing(self) -> None:
        with self._lock:
            self._is_processing = False

        if self._processing_thread is not None:
            self._processing_thread.join()
            self._processing_thread = None

    
    def remove_video_source(self) -> None:
        self.stop_processing()
        self._video_capture.end_capture()
        self._ret = False


    def set_video_source(self, source: int|str) -> None:
        self.stop_processing()
        self._video_capture.end_capture()
        self._video_capture.start_capture(source)
        self._ret = True
        self.start_processing()


    def _process_frames(self) -> None:
        while self._is_processing:
            ret, frame = self._video_capture.get_frame()

            if ret == False:
                with self._lock:
                    self._ret = False
                    self._latest_frame = None
                    self._is_processing = False
                break

            if frame is None:
                time.sleep(0.001)
                continue
            
            processed_frame = self._image_processor.process_frame(
                frame,
                self._max_frame_width,
                self._max_frame_height
            )

            with self._lock:
                self._latest_frame = processed_frame

            time.sleep(0.001)


    def naive(self) -> MatLike:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        
        return self._ret, frame


    def get_latest_frame(self) -> MatLike:
        return self.naive()

        while True:
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                    return frame
            
            time.sleep(0.001)
