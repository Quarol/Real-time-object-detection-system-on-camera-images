from detector.video_capture import VideoCapture, NO_VIDEO, VIDEO_FILE
from detector.frame_processor import FrameProcessor


class App:
    def __init__(self) -> None:
        from detector.interface import GUI
        from detector.image_processor import ImageProcessor

        self._video_manager = VideoCapture()
        self._image_processor = ImageProcessor(self)
        self._frame_processor = FrameProcessor(self._video_manager, self._image_processor)
        self._gui = GUI(self, self._frame_processor)


    def run(self) -> None:
        self._gui.show()
        self._frame_processor.stop_processing()
        self._video_manager.end_capture()


    def set_video_source(self, source_id: int) -> None:
        if source_id == NO_VIDEO:
            self._frame_processor.stop_processing()
            self._video_manager.end_capture()

        elif source_id == VIDEO_FILE:
            self._frame_processor.stop_processing()
            
            path = self._gui.select_video_file()
            self._video_manager.start_capture(path)
            
            self._frame_processor.start_processing()

        else:
            self._frame_processor.stop_processing()
            self._video_manager.start_capture(source_id)
            self._frame_processor.start_processing()


    def change_model(self, path: str) -> None:
        self._frame_processor.stop_processing()
        # code here
        self._frame_processor.start_processing()