import cv2 as cv
from cv2.typing import MatLike
import time

MAX_NUMBER_OF_CAMERAS = 10

NO_VIDEO = -2
VIDEO_FILE = -1
# camera: >= 0

class VideoCapture:
    max_number_of_cameras = None

    def __init__(self) -> None:
        self._video_capture = None


    def start_capture(self, source: int|str) -> None:
        self._video_capture = cv.VideoCapture(source)


    def end_capture(self) -> None:
        if self._video_capture is None:
            return
        self._video_capture.release()
        

    def is_capture_on(self) -> bool:
        return self._video_capture is not None and self._video_capture.isOpened()
    

    def get_frame(self):
        if self._video_capture is None:
            return None, None
        ret, frame = self._video_capture.read()
        
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        else:
            print(self._video_capture.isOpened())

        return ret, frame
    

    def get_fps(self):
        if self._video_capture is None:
            return None
        return self._video_capture.get(cv.CAP_PROP_FPS)


    @staticmethod
    def get_available_sources():
        sources = {
            'No video': NO_VIDEO,
            'Video file': VIDEO_FILE
        }

        for i in range(MAX_NUMBER_OF_CAMERAS):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                sources[f'Camera {i}'] = i
                cap.release()
        return sources