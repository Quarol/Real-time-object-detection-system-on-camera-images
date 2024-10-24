import threading
import time
from collections import deque
from cv2.typing import MatLike

from detector.image_processor import ImageProcessor
from detector.video_capture import VideoCapture

CAPTURED_FRAMES_QUEUE_SIZE = 10  # Define a limit for frame queue size

class FrameProcessor:
    def __init__(self, video_capture: VideoCapture, image_processor: ImageProcessor) -> None:
        self._video_capture = video_capture
        self._image_processor = image_processor

        self._frame_queue = deque(maxlen=CAPTURED_FRAMES_QUEUE_SIZE)

        self._lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._lock)
        self._queue_not_full = threading.Condition(self._lock)

        self._capture_thread = None
        self._is_capturing = False

        self._processing_thread = None
        self._is_processing = False

        self._latest_frame = None
        self._ret = False

        self._camera_seconds_per_frame = None


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
        self._end_capture()
        self._video_capture.end_capture()
        self._ret = False


    def set_video_source(self, source: int|str) -> None:
        self.stop_processing()
        self._end_capture()
        self._video_capture.end_capture()

        self._video_capture.start_capture(source)
        capture_fps = self._video_capture.get_fps()

        if capture_fps is None:
            return
        
        self._camera_seconds_per_frame = 1 / capture_fps
        self._ret = True
        self._start_capture(source)
        self.start_processing()


    def _start_capture(self, source: int|str):
        self._is_capturing = True
        self._capture_thread = threading.Thread(target=self._capture_frames)
        self._capture_thread.start()


    def _end_capture(self):
        with self._lock:
            self._is_capturing = False
            self._frame_queue.clear()
            self._queue_not_empty.notify_all()
            self._queue_not_full.notify_all()

        if self._capture_thread is not None:
            self._capture_thread.join()
            self._capture_thread = None


    def _capture_frames(self) -> None:
        self._ret = True

        while self._is_capturing:
            is_capture_on, newest_frame = self._video_capture.get_frame()

            if not is_capture_on:
                with self._lock:
                    self._is_capturing = False
                    self._is_processing = False
                    self._ret = False
                break

            if newest_frame is None:
                continue
            
            with self._queue_not_full:
                while len(self._frame_queue) == CAPTURED_FRAMES_QUEUE_SIZE:
                    self._queue_not_full.wait() 

                self._frame_queue.append(newest_frame)
                self._queue_not_empty.notify()


    def _process_frames(self) -> None:
        while self._is_processing:
            with self._queue_not_empty:
                while not self._frame_queue:
                    self._queue_not_empty.wait()  

                frame = self._frame_queue.popleft()
                self._queue_not_full.notify()


            processed_frame = frame
            processed_frame = self._image_processor.process_frame(
                frame,
                self._max_frame_width,
                self._max_frame_height
            )


            with self._lock:
                self._latest_frame = processed_frame
            

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
