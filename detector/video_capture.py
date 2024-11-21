import cv2 as cv
from cv2.typing import MatLike
from typing import Tuple, Optional, Dict

NO_VIDEO = -2
VIDEO_FILE = -1
# camera: >= 0

class VideoCapture:
    max_number_of_cameras = None

    def __init__(self) -> None:
        self._video_capture = None


    def start_capture(self, source: int|str) -> None:
        self.end_capture()
        self._video_capture = cv.VideoCapture(source)


    def end_capture(self) -> None:
        if self._video_capture is None:
            return
        self._video_capture.release()
        self._video_capture = None
    

    def get_frame(self) -> Tuple[bool, Optional[MatLike]]:
        if self._video_capture is None:
            return False, None
        is_capture_on, frame = self._video_capture.read()
        
        if is_capture_on:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        else:
            self.end_capture()

        return is_capture_on, frame
    

    def get_fps(self) -> float:
        if self._video_capture is None:
            return None
        
        fps = self._video_capture.get(cv.CAP_PROP_FPS)
        if fps == 0:
            return None
        return fps


    @staticmethod
    def get_available_sources() -> Dict[str, int]:
        sources = {
            'No video': NO_VIDEO,
            'Video file': VIDEO_FILE
        }

        i = 0
        while True:
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                sources[f'Camera {i}'] = i
                cap.release()
            else:
                break
            i += 1
            
        return sources