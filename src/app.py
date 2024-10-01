
from video_manager import VideoManager, NO_VIDEO, VIDEO_FILE

class App:
    def __init__(self) -> None:
        from interface import GUI
        from image_processing.image_processor import ImageProcessor

        self._video_manager = VideoManager()
        self._image_processor = ImageProcessor(self, self._video_manager)
        self._gui = GUI(self, self._image_processor)


    def run(self):
        self._gui.show()
        self._video_manager.end_capture()

    def set_video_source(self, source_id: int):
        if source_id == NO_VIDEO:
            self._video_manager.end_capture()
        elif source_id == VIDEO_FILE:
            path = self._gui.select_video_file()
            self._video_manager.start_capture(path)
        else:
            self._video_manager.start_capture(source_id)