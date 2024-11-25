import threading
from collections import deque
from cv2.typing import MatLike
from typing import Tuple, Optional, Callable
import time

from detector.image_processor import ImageProcessor
from detector.video_capture import VideoCapture
from detector.timer import Timer

CAPTURED_FRAMES_QUEUE_SIZE = 1

class VideoProcessingEngine:
    def __init__(self, video_capture: VideoCapture, image_processor: ImageProcessor,
                 notification_function: Callable[[], None]) -> None:
        self._video_capture = video_capture
        self._image_processor = image_processor
        self._notification_function = notification_function

        self._latest_frame = None
        self._is_capture_on = False

        self._max_frame_width = 1920
        self._max_frame_height = 1080
        self._min_frame_width = self._max_frame_width * 0.5
        self._min_frame_height = self._max_frame_height * 0.5

        self._frame_queue = deque(maxlen=CAPTURED_FRAMES_QUEUE_SIZE)

        self._frame_set_lock = threading.Lock()
        self._video_capture_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._queue_lock)
        self._queue_not_full = threading.Condition(self._queue_lock)
        self._capture_event = threading.Event()
        self._process_event = threading.Event()

        self._continue_thread_loop = True

        self._processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)

        self._capture_event.clear()
        self._process_event.clear()


    def run(self) -> None:
        self._processing_thread.start()
        self._capture_thread.start()


    def set_window_dimensions(self, max_width: int, max_height: int, min_width: int, min_height: int) -> None:
        self._max_frame_width = max_width
        self._max_frame_height = max_height

        self._min_frame_width = min_width
        self._min_frame_height = min_height

    
    def shutdown(self) -> None:
        print('Begin shutdown of video processing engine')
        self._continue_thread_loop = False
        self._capture_event.clear()
        self._process_event.clear()

        with self._queue_lock:
            self._queue_not_empty.notify_all()
            self._queue_not_full.notify_all()
        
        self.remove_video_source()

        print('Cleanup variables')
        self.remove_video_source()

        print('End of cleanup, waiting for main thread to shut down deamon threads')

    
    def _start_processing(self) -> None:
        self._stop_processing()
        self._process_event.set()


    def _stop_processing(self) -> None:
        self._process_event.clear()


    def remove_video_source(self) -> None:
        self._is_capture_on = False
        self._stop_processing()

        with self._video_capture_lock:
            self._end_capture()
            self._video_capture.end_capture()


    def set_video_source(self, source: int|str) -> None:
        self.remove_video_source()

        self._video_capture.start_capture(source)
        self._is_capture_on = True

        self._start_capture()
        self._start_processing()


    def _start_capture(self) -> None:
        self._capture_event.set()


    def _end_capture(self) -> None:
        self._capture_event.clear()

        with self._queue_lock:
            self._frame_queue.clear()
            self._queue_not_empty.notify_all()
            self._queue_not_full.notify_all()


    def _capture_frames(self) -> None:
        while self._continue_thread_loop:
            if not self._capture_event.is_set():
                self._capture_event.wait()
                # In case shutdown happened: end thread
                if not self._continue_thread_loop:
                    return

            with self._video_capture_lock:
                is_capture_on, frame = self._video_capture.get_frame()

            if not is_capture_on:
                self.remove_video_source()
                continue

            if frame is None:
                continue

            frame = self._image_processor.fit_frame_into_screen(frame, 
                                                                self._max_frame_width, self._max_frame_height,
                                                                self._min_frame_width, self._min_frame_height)
            
            with self._queue_not_full:
                while len(self._frame_queue) == CAPTURED_FRAMES_QUEUE_SIZE:
                    self._queue_not_full.wait()
                    # In case shutdown happened: end thread
                    if not self._continue_thread_loop:
                        return

                self._frame_queue.append(frame)
                self._queue_not_empty.notify()


    def _process_frames(self) -> None:
        while self._continue_thread_loop:
            if not self._process_event.is_set():
                self._process_event.wait()
                # In case shutdown happened: end thread
                if not self._continue_thread_loop:
                    return

            with self._queue_not_empty:
                while not self._frame_queue: # If queue is empty
                    self._queue_not_empty.wait()
                    # In case shutdown happened: end thread
                    if not self._continue_thread_loop:  
                        return

                frame = self._frame_queue.popleft()
                self._queue_not_full.notify()


            start = Timer.get_current_time()
            detections, are_there_objects = self._image_processor.detect_objects(frame)
            stop_detect = Timer.get_current_time()
            frame = self._image_processor.visualize_objects_presence(frame, detections)
            stop = Timer.get_current_time()

            total_duration = Timer.get_duration(start, stop)
            detection_duration = Timer.get_duration(start, stop_detect)
            draw_duration = Timer.get_duration(stop_detect, stop)

            if are_there_objects:
                self._notification_function()
       
            with self._frame_set_lock:
                self._latest_frame = frame
            

    def get_latest_frame(self) -> Tuple[bool, Optional[MatLike]]:
        with self._frame_set_lock:
            is_capture_on = self._is_capture_on
            frame = self._latest_frame
            self._latest_frame = None

        return is_capture_on, frame
