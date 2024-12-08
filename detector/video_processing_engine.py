import threading
from cv2.typing import MatLike
from typing import Tuple, Optional, Callable
import time

from detector.image_processor import ImageProcessor
from detector.video_capture import VideoCapture
from detector.app import App


class VideoProcessingEngine:
    def __init__(self, video_capture: VideoCapture, image_processor: ImageProcessor, audio_alarm: App) -> None:
        self._video_capture = video_capture
        self._image_processor = image_processor
        self._audio_alarm = audio_alarm

        self._max_frame_width = 1920 # Default assumed width
        self._max_frame_height = 1080 # Default assumed height

        self._frame_buffer = None
        self._processed_frame_buffer = None
        self._is_capture_on = False

        self._frame_set_lock = threading.Lock()
        self._video_capture_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._buffer_not_empty = threading.Condition(self._buffer_lock)
        self._capture_event = threading.Event()
        self._process_event = threading.Event()

        self._continue_thread_loop = True

        self._processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)

        self._capture_event.clear()
        self._process_event.clear()
    

    def _is_buffer_empty(self) -> bool:
        return self._frame_buffer is None 
    
    
    def _set_buffer(self, frame: MatLike) -> None:
        self._frame_buffer = frame

    
    def _fetch_buffer(self) -> MatLike|None:
        frame = self._frame_buffer
        self._frame_buffer = None
        return frame
    

    def _set_processed_frame_buffer(self, frame: MatLike) -> None:
        self._processed_frame_buffer = frame


    def _fetch_processed_frame_buffer(self) -> MatLike:
        frame = self._processed_frame_buffer
        self._processed_frame_buffer = None

        return frame


    def run(self) -> None:
        self._processing_thread.start()
        self._capture_thread.start()


    def set_max_frame_dimension(self, max_width: int, max_height: int) -> None:
        self._max_frame_width = max_width
        self._max_frame_height = max_height

    
    def shutdown(self) -> None:
        print('Begin shutdown of video processing engine')
        self._continue_thread_loop = False
        self._capture_event.clear()
        self._process_event.clear()

        with self._buffer_lock:
            self._buffer_not_empty.notify_all()
        
        self.remove_video_source()

        print('Cleanup variables')
        self.remove_video_source()

        print('End of cleanup, waiting for main thread to shut down deamon threads')

    
    def _start_processing(self) -> None:
        self._stop_processing()
        self._process_event.set()


    def _stop_processing(self) -> None:
        self._process_event.clear()
        with self._buffer_lock:
            self._buffer_not_empty.notify_all()


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

        with self._buffer_lock:
            self._frame_buffer = None
            self._buffer_not_empty.notify_all()


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
                                                                self._max_frame_width, self._max_frame_height)
            
            with self._buffer_lock:
                self._set_buffer(frame)
                self._buffer_not_empty.notify()


    def _process_frames(self) -> None:
        while self._continue_thread_loop:
            if not self._process_event.is_set():
                self._process_event.wait()
                # In case shutdown happened: end thread
                if not self._continue_thread_loop:
                    return

            with self._buffer_not_empty:
                while self._is_buffer_empty():
                    self._buffer_not_empty.wait()
                    # In case shutdown happened: end thread
                    if not self._continue_thread_loop:  
                        return

                frame = self._fetch_buffer() 

            detections, are_there_objects = self._image_processor.detect_objects(frame)
            frame = self._image_processor.visualize_objects_presence(frame, detections)

            if are_there_objects:
                self._audio_alarm.play_audio_alert()
       
            with self._frame_set_lock:
                self._set_processed_frame_buffer(frame)
    
    
    def get_processed_frame(self) -> Tuple[bool, Optional[MatLike]]:
        with self._frame_set_lock:
            is_capture_on = self._is_capture_on
            frame = self._fetch_processed_frame_buffer()

        return is_capture_on, frame
