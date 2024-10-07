import cv2 as cv
from cv2.typing import MatLike

MAX_NUMBER_OF_CAMERAS = 10

NO_VIDEO = -2
VIDEO_FILE = -1
# camera: >= 0

class VideoManager:
    max_number_of_cameras = None

    def __init__(self) -> None:
        self._video_capture = None


    def start_capture(self, source: int|str):
        self.end_capture()
        self._video_capture = cv.VideoCapture(source)


    def end_capture(self):
        if self._video_capture is None:
            return
        self._video_capture.release()


    def get_frame(self) -> MatLike|None:
        if self._video_capture is None:
            return None
        _, frame = self._video_capture.read()
        return frame


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