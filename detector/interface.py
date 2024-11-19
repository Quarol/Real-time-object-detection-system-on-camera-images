import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageGrab
from cv2.typing import MatLike
import numpy as np

from detector.timer import Timer
from detector.app import App
from detector.video_processor import VideoProcessor
from detector.yolo_settings import yolo_inference_config
import detector.yolo_settings as yolo_settings
from detector.video_capture import VideoCapture, NO_VIDEO
from detector.image_processor import ImageProcessor

VIDEO_FRAME_MARGIN = 10
AFTER_DELAY = 1

class GUI:
    def __init__(self, parent_app: App, video_source: VideoProcessor) -> None:
        self._parent_app = parent_app
        self._video_source = video_source

        self._selected_video_source_id = None
        self._initialize_gui()

        self._no_video_image = self._generate_black_image()
        self._frame_counter = 0
        self._time_before_frame = Timer.get_current_time()

        self._video_source.set_window_dimensions(self._max_frame_width, self._max_frame_height,
                                                 self._min_frame_width, self._min_frame_height)


    def _initialize_gui(self) -> None:
        self._root = tk.Tk()

        self._root.title('Object detector')
        self._root.resizable(False, False)

        screen_space_factor = 0.70
        screen = ImageGrab.grab()
        screen_width, screen_height = screen.size

        window_width = int(screen_width * screen_space_factor)
        window_height = int(screen_height * screen_space_factor)

        self._root.geometry(f'{window_width}x{window_height}') # Set window size
        self._root.geometry(f'+{0}+{0}') # Set window position

        self._menubar = tk.Menu(master=self._root)
        self._root.config(menu=self._menubar)
        self._initialize_video_source_menu()
        self._initialize_detector_parameters_menu()
        self._initialize_video_player()

        self._root_width = window_width
        self._root_height = window_height


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

    
    def _initialize_detector_parameters_menu(self):
        self._detector_menu = tk.Menu(master=self._menubar, tearoff=0)
        self._menubar.add_cascade(menu=self._detector_menu, label='Detector Parameters')

        self._detector_menu.add_command(
            label="Show detector parameters", 
            command=self._show_detector_parameters_window
        )


    def _show_detector_parameters_window(self):
        # Check if the window already exists
        if hasattr(self, '_detector_parameters_window') and self._detector_parameters_window.winfo_exists():
            self._detector_parameters_window.lift() 
            return
        
        # Create a slider for confidence threshold
        self._detector_parameters_window = tk.Toplevel(self._root)
        self._detector_parameters_window.title('Detector Parameters')

        width_scaling_factor = 0.36
        height_scaling_factor = 0.5
        window_width = int(self._root_width * width_scaling_factor)
        window_height = int(self._root_height * height_scaling_factor)

        self._detector_parameters_window.geometry(f'{window_width}x{window_height}')
        self._detector_parameters_window.resizable(False, False)

        confidence_threshold_label = tk.Label(self._detector_parameters_window, text='Confidence threshold:')
        confidence_threshold_label.pack(pady=1)

        confidence_threshold_slider = tk.Scale(
            self._detector_parameters_window,
            from_=0.00,
            to=1.00,
            resolution=0.01,
            orient='horizontal',
            command=self._update_confidence_threshold
        )
        confidence_threshold_slider.set(yolo_inference_config.confidence_threshold)
        confidence_threshold_slider.pack(pady=10)

        # Create classes of object that will be detected:
        classes_label = tk.Label(self._detector_parameters_window, text='Classes of objects:')
        classes_label.pack(pady=1)

        checkbox_canvas = tk.Canvas(self._detector_parameters_window)
        checkbox_canvas.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(self._detector_parameters_window, orient='vertical', command=checkbox_canvas.yview)
        scrollbar.pack(side='right', fill='y')

        checkbox_canvas.configure(yscrollcommand=scrollbar.set)

        checkbox_frame = tk.Frame(checkbox_canvas)
        checkbox_canvas.create_window((0, 0), window=checkbox_frame, anchor='nw')

        checkbox_vars = {}
        row = 0
        col = 0
        max_items_per_row = 4
        for index, class_name in yolo_settings.YOLO_CLASSES.items():
            var = tk.BooleanVar(value=(index in yolo_inference_config.classes))
            checkbox = tk.Checkbutton(
                checkbox_frame,
                text=class_name,
                variable=var,
                command=lambda idx=index, v=var: self._update_classes(idx, v)
            )

            # Place the checkbox in the grid (3 per row)
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)

            checkbox_vars[index] = var

            col += 1
            if col == max_items_per_row:
                col = 0
                row += 1

        checkbox_frame.update_idletasks()
        checkbox_canvas.config(scrollregion=checkbox_canvas.bbox("all")) 

        # Add moouse scroll event to object selection menu
        def on_mouse_wheel(event):
            if event.delta < 0:
                checkbox_canvas.yview_scroll(1, 'units')
            else:
                checkbox_canvas.yview_scroll(-1, 'units')
        self._detector_parameters_window.bind_all("<MouseWheel>", on_mouse_wheel)


    def _update_confidence_threshold(self, value):
        yolo_inference_config.confidence_threshold = float(value)


    def _update_classes(self, class_index, var):
        if var.get():
            yolo_inference_config.add_detected_class(class_index)
        else:
            yolo_inference_config.remove_detected_class(class_index)


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

        min_frame_scale_factor = 0.45 
        self._max_frame_width = self._video_frame.winfo_width()
        self._max_frame_height = self._video_frame.winfo_height()
        self._min_frame_width = self._max_frame_width * min_frame_scale_factor
        self._min_frame_height = self._max_frame_height * min_frame_scale_factor


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
