from detector.video_capture import VideoCapture, NO_VIDEO, VIDEO_FILE

class App:
    def __init__(self) -> None:
        from detector.interface import GUI
        from detector.image_processor import ImageProcessor
        from detector.frame_processor import FrameProcessor


        self._video_capture = VideoCapture()
        self._image_processor = ImageProcessor(self)
        self._frame_processor = FrameProcessor(self._video_capture, self._image_processor)
        self._gui = GUI(self, self._frame_processor)


    def run(self) -> None:
        self._gui.show()
        self._frame_processor.remove_video_source()


    def set_video_source(self, source_id: int) -> None:
        if source_id == NO_VIDEO:
            self._frame_processor.remove_video_source()
        elif source_id == VIDEO_FILE:
            path = self._gui.select_video_file()
            self._frame_processor.set_video_source(source=path)
        else:
            self._frame_processor.set_video_source(source=source_id)


    def change_model(self, path: str) -> None:
        self._frame_processor.stop_processing()
        # code here
        self._frame_processor.start_processing()