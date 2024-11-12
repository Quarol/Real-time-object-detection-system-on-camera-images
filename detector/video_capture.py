import cv2 as cv
from cv2.typing import MatLike
from typing import Tuple, Optional, Dict, Union

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
        self._video_capture = None
    

    def get_frame(self) -> Tuple[Optional[bool], Optional[MatLike]]:
        if self._video_capture is None:
            return None, None
        ret, frame = self._video_capture.read()
        
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return ret, frame
    

    def get_fps(self) -> float:
        if self._video_capture is None:
            return None
        return self._video_capture.get(cv.CAP_PROP_FPS)


    @staticmethod
    def get_available_sources() -> Dict[str, int]:
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