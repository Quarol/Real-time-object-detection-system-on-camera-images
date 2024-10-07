import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
from cv2.typing import MatLike
import numpy as np

from app import App
from video_manager import VideoManager, NO_VIDEO
from image_processing.image_processor import ImageProcessor
from consts import MILLISECONDS_PER_FRAME

VIDEO_FRAME_MARGIN = 10

class GUI:
    def __init__(self, parent_app: App, video_source: ImageProcessor) -> None:
        self._parent_app = parent_app
        self._video_source = video_source
        self._initialize_gui()
        self._selected_video_source_id = None


    def _initialize_gui(self):
        self._root = tk.Tk()

        self._root.title('Pedestrian detector')
        self._root.state('zoomed')
        self._root.resizable(False, False)

        self._menubar = tk.Menu(master=self._root)
        self._root.config(menu=self._menubar)
        self._initialize_video_source_menu()
        self._initialize_video_player()


    def _initialize_video_source_menu(self):
        self._source_menu = tk.Menu(master=self._menubar, tearoff=0)
        self._selected_video_source_id = tk.IntVar()

        sources = VideoManager.get_available_sources()
        for source_name in sources:
            self._source_menu.add_radiobutton(
                label=source_name,
                variable=self._selected_video_source_id,
                value=sources[source_name],
                command=lambda source_index=sources[source_name]: self._parent_app.set_video_source(source_index)
            )

        self._selected_video_source_id.set(NO_VIDEO)
        self._menubar.add_cascade(menu=self._source_menu, label='Video source')


    def _initialize_video_player(self):
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


    def select_video_file(self):
        filetypes = [
            ('Video files', '.mp4 *.avi *.mkv *.mov *.wmv')
        ]
        file_path = filedialog.askopenfilename(title='Select a recorded video', filetypes=filetypes)

        return file_path


    def show(self):
        self._update_frame(self._video_source.process_frame(self._max_frame_width, self._max_frame_height))
        self._root.mainloop()


    def _update_frame(self, frame: MatLike):
        if frame is None:
            frame = self._generate_black_image()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self._display_frame.imgtk = imgtk
        self._display_frame.configure(image=imgtk)

        self._video_frame.after(
            MILLISECONDS_PER_FRAME,
            lambda frame=self._video_source.process_frame(self._max_frame_width, self._max_frame_height): 
                          self._update_frame(frame)
        )

    def _generate_black_image(self) -> MatLike:
        black_image = np.zeros((self._max_frame_height, self._max_frame_width, 3), dtype=np.uint8)
        return black_image
