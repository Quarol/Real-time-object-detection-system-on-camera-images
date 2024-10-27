import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from cv2.typing import MatLike
import numpy as np
import math

from detector.timer import Timer
from detector.app import App
from detector.frame_processor import FrameProcessor
from detector.video_capture import VideoCapture, NO_VIDEO
from detector.image_processor import ImageProcessor
from detector.consts import MILLISECONDS_PER_FRAME

VIDEO_FRAME_MARGIN = 10
AFTER_DELAY = 1

class GUI:
    def __init__(self, parent_app: App, video_source: FrameProcessor) -> None:
        self._parent_app = parent_app
        self._video_source = video_source

        self._selected_video_source_id = None
        self._initialize_gui()

        self._no_video_image = self._generate_black_image()
        self._frame_counter = 0
        self._time_before_frame = Timer.get_current_time()

        self._video_source.set_max_frame_size(self._max_frame_width, self._max_frame_height)


    def _initialize_gui(self) -> None:
        self._root = tk.Tk()

        self._root.title('Pedestrian detector')
        self._root.state('zoomed')
        self._root.resizable(False, False)

        self._menubar = tk.Menu(master=self._root)
        self._root.config(menu=self._menubar)
        self._initialize_video_source_menu()
        self._initialize_video_player()


    def _initialize_video_source_menu(self) -> None:
        self._source_menu = tk.Menu(master=self._menubar, tearoff=0)
        self._selected_video_source_id = tk.IntVar()

        sources = VideoCapture.get_available_sources()
        for source_name in sources:
            self._source_menu.add_radiobutton(
                label=source_name,
                variable=self._selected_video_source_id,
                value=sources[source_name],
                command=lambda source_index=sources[source_name]: self._parent_app.set_video_source(source_index)
            )

        self._selected_video_source_id.set(NO_VIDEO)
        self._menubar.add_cascade(menu=self._source_menu, label='Video source')


    def _initialize_video_player(self) -> None:
        self._video_frame = tk.Frame(self._root)
        self._video_frame.pack(
            padx=VIDEO_FRAME_MARGIN,
            pady=VIDEO_FRAME_MARGIN,
            fill=tk.BOTH,
            expand=True
        )

        self._display_frame = tk.Label(self._video_frame)

        self._display_frame.grid(row=0, column=0)

        self._video_frame.grid_rowconfigure(0, weight=1)
        self._video_frame.grid_columnconfigure(0, weight=1)

        self._root.update()

        self._max_frame_width = self._video_frame.winfo_width()
        self._max_frame_height = self._video_frame.winfo_height()


    def select_video_file(self) -> str:
        filetypes = [
            ('Video files', '.mp4 *.avi *.mkv *.mov *.wmv')
        ]
        file_path = filedialog.askopenfilename(title='Select a recorded video', filetypes=filetypes)

        return file_path


    def show(self) -> None:
        self._show_frame(self._no_video_image)
        self._time_before_frame = Timer.get_current_time()
        self._update_frame()
        self._root.mainloop()


    def _measure_time(self, func, *args, **kwargs):
        time1 = Timer.get_current_time()
        result = func(*args, **kwargs)
        time2 = Timer.get_current_time()
        duration = time2 - time1
        fps = 1 / duration if duration != 0 else 'inf'

        print(f'Duration: {duration}s', end='')
        print(f', FPS: {fps}')

        return result


    def _show_frame(self, frame: MatLike) -> None:
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self._display_frame.imgtk = imgtk
        self._display_frame.configure(image=imgtk)


    def _update_frame(self) -> None:
        ret, frame = self._video_source.get_latest_frame()

        if ret == False:
            self._show_frame(self._no_video_image)
        else:
            if frame is not None:
                self._show_frame(frame)
                self._frame_counter += 1

        self._count_and_update_fps()
        self._video_frame.after(AFTER_DELAY, self._update_frame)
    

    def _generate_black_image(self) -> MatLike:
        black_image = np.zeros((self._max_frame_height, self._max_frame_width, 3), dtype=np.uint8)
        return black_image
    

    def _count_and_update_fps(self) -> None:
        time_after_frame = Timer.get_current_time()
        duration = time_after_frame - self._time_before_frame

        if duration >= 1:
            real_fps = self._frame_counter / duration if duration != 0 else 'inf'
            #print(f'real_fps: {real_fps}')

            self._time_before_frame = time_after_frame
            self._frame_counter = 0
