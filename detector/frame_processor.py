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

        self._latest_frame = None
        self._ret = False

        self._camera_seconds_per_frame = None

        self._max_frame_width = 1920
        self._max_frame_height = 1080

        self._frame_queue = deque(maxlen=CAPTURED_FRAMES_QUEUE_SIZE)

        self._lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._lock)
        self._queue_not_full = threading.Condition(self._lock)
        self._capture_event = threading.Event()
        self._process_event = threading.Event()

        self._continue_process_loop = True
        self._continue_capture_loop = True

        self._processing_thread = threading.Thread(target=self._process_frames)
        self._capture_thread = threading.Thread(target=self._capture_frames)

        self._processing_thread.start()
        self._capture_thread.start()


    def set_max_frame_size(self, width, height) -> None:
        self._max_frame_width = width
        self._max_frame_height = height

    
    def shutdown(self):
        self._continue_process_loop = False
        self._continue_capture_loop = False
        
        with self._lock:
            self._queue_not_empty.notify()
            self._queue_not_full.notify()
        
        self._capture_thread.join()
        self._processing_thread.join()


    def start_processing(self) -> None:
        self.stop_processing()
        self._process_event.set()


    def stop_processing(self) -> None:
        self._process_event.clear()


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
        self._capture_event.set()


    def _end_capture(self):
        self._capture_event.clear()

        with self._lock:
            self._frame_queue.clear()
            self._queue_not_empty.notify_all()
            self._queue_not_full.notify_all()


    def _capture_frames(self) -> None:
        self._ret = True

        while self._continue_capture_loop:
            if not self._capture_event.is_set():
                self._capture_event.wait()

            capture_time_begin = time.perf_counter()
            is_capture_on, frame = self._video_capture.get_frame()
            pure_capture_end = time.perf_counter()

            if not is_capture_on:
                with self._lock:
                    self._ret = False
                break

            if frame is None:
                continue

            frame = self._image_processor.fit_frame_into_screen(frame, self._max_frame_width, self._max_frame_height)
            
            with self._queue_not_full:
                while len(self._frame_queue) == CAPTURED_FRAMES_QUEUE_SIZE:
                    self._queue_not_full.wait() 

                self._frame_queue.append(frame)
                self._queue_not_empty.notify()

            capture_time_end = time.perf_counter()
            capture_duration = capture_time_end - capture_time_begin
            sleep_time = self._camera_seconds_per_frame - capture_duration
            print(pure_capture_end - capture_time_begin, capture_duration, sleep_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        

    def _process_frames(self) -> None:
        while self._continue_process_loop:
            if not self._process_event.is_set():
                self._process_event.wait()

            with self._queue_not_empty:
                while not self._frame_queue:
                    self._queue_not_empty.wait()  

                if not self._continue_process_loop:
                    break

                frame = self._frame_queue.popleft()
                self._queue_not_full.notify()

            detections = self._image_processor.detect_objects(frame)
            frame, areTherePeople = self._image_processor.visualize_people_presence(frame, detections)

            with self._lock:
                self._latest_frame = frame
       

    def get_latest_frame(self) -> MatLike:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None

        return self._ret, frame
