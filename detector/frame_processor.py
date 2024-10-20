import threading
import time


class FrameProcessor:
    def __init__(self, video_manager, image_processor):
        self._video_manager = video_manager
        self._image_processor = image_processor
        self._latest_frame = None
        self._lock = threading.Lock()
        self._processing_thread = None
        self._is_processing = False
        self._processing_duration = None

    
    def set_max_frame_size(self, width, height):
        self._max_frame_width = width
        self._max_frame_height = height


    def start_processing(self):
        self.stop_processing()

        self._is_processing = True
        self._processing_thread = threading.Thread(target=self._process_frames)
        self._processing_thread.start()


    def stop_processing(self):
        self._is_processing = False
        if self._processing_thread is not None:
            self._processing_thread.join()

        self._processing_duration = None


    def _process_frames(self):
        while self._is_processing:
            frame = self._video_manager.get_frame()
            if frame is None:
                continue

            processed_frame = self._image_processor.process_frame(
                frame,
                self._max_frame_width,
                self._max_frame_height
            )

            with self._lock:
                self._latest_frame = processed_frame

            time.sleep(0.001)
            


    def get_latest_frame(self):
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame
